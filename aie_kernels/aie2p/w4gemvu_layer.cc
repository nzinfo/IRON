// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define NOCPP

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>

// P28 layer-v2: the WHOLE transformer layer rides ONE task group per
// worker, persistent-worker style (P28-1 model, P28-3 ring proof). Eight
// workers on tiles (0,2..5)+(1,2..5) form a serpentine ring; each worker
// owns a 256-row j-slice of every projection and the three all-gathers
// (o-out, swiglu, down-out) traverse core<->core ObjectFifos instead of
// the C->DDR->window round trips that made the task group the atom of
// scheduling cost (P27-4: 129 groups x ~75us = the entire device-side
// gap to FLM; groups can only be REMOVED by designing away cross-group
// dependencies).
//
// Per exec per worker the A fifo delivers (in this exact order):
//   K=0    X element: [0,2048) attn int8 q (host-quantized g32),
//          [6144,6400) d bf16[64], [6400,6404) worker id u32,
//          K header @18552. Stages the replicated A-ops + d and resets
//          ALL per-exec state (the persistent worker's exec boundary).
//   K=100  xn element: [0,512) x_n chunk bf16 (256 = this worker's slice
//          of the residual) -> lv_xn.
//   K=2048 x16 o blocks (phase O): partials -> lv_o.
//   ---- ring 1 (r13 fifo, 512B elements): fr1/fw1/st1 gather x' ----
//   K=101  w2 element: [0,4096) ln2 weight bf16. rms(x' full) -> quantize
//          -> replicated arena + lv_dA; resets gate/up counters.
//   K=103  x48 gate blocks -> lv_gate (768 bf16).
//   K=104  x48 up blocks -> lv_upwin (2-block window); every 2nd block
//          runs one swiglu group k (sig -> amax -> quant) with global
//          group g = p*24 + k: q int8 -> lv_sw[g*32], scale ->
//          lv_sw[6144 + 2g].
//   ---- ring 2 (r2 fifo, 832B elements): fr2/fw2/st2 gather sw ----
//   K=105  x48 down chunk-blocks (c-major: c outer 0..2, b inner 0..15;
//          chunk word @18556 selects the sw slice). Arena rebuilt from
//          lv_sw[2048c..] only when c changes (3 rebuilds, not 48).
//          Each partial (bf16-rounded) accumulates f32 into dacc.
//   ---- ring 3 (REUSES the r13 fifo): fr3/st3 gather xn1 (fr3
//   recomputes x' from lv_xn+lv_o — see the lv_shared overlay note) ----
//   K=102  w1 element: rms(xn1 full) -> quantize -> arena + lv_dA;
//          phase = QKV.
//   K=2048 x24 qkv blocks (phase QKV) -> REAL C elements.
//   lv_cxn x16: drain this worker's xn1 chunk (256 bf16) as C elements
//          (the next layer's residual, host-refilled as the next exec's
//          K=100/xn and used for host attention).
//
// Only the qkv (24) and cxn (16) calls touch the C fifo — 40 C elements
// = 1280 B per worker per exec. Everything else stays in L1/.bss, which
// is why TWO extern "C" entries exist: MLIR cannot pass a null memref,
// so the no-C flavors go through the 4-arg w4gemvu_layer_a and only the
// qkv flavor uses the 5-arg w4gemvu_layer_bf16 (the mha design's 8
// kernels sharing one bin_name is the proven multi-Kernel pattern; the
// Worker collects bin_names in a set so the single .o links once).
//
// Ring position arithmetic: every ring edge must join unit-step
// neighbor tiles — core<->core objectFifos only lower to shared memory
// between mem-affine tiles; anything farther splits into a core mem-DMA
// MM2S channel and npu2 cores have just 2 output channels (the N=16
// serpentine wrap (3,2)->(0,2) busted worker 12 with 3; see
// design_layerv2.HAM16). N=8 keeps the two-column serpentine
// (order 0,1,2,3,7,6,5,4): worker w at tile (col c = w>>2, row w&3),
// p = w on even c and p = 8c + 3 - w on odd c. N=16 uses the
// Hamiltonian cycle order 0,1,2,3,7,6,5,9,10,11,15,14,13,12,8,4 —
// the piecewise inverse lives in lv_xelem. In round r (r = 1..N-1) a
// worker receives the chunk that started at ring position
// (p - r) & (N-1) — stored to slot (p - r) & (N-1) of the gather
// buffer, and forwarded on rounds 1..N-2 only (the last round's chunk
// would return to its origin). Worker w's global j-slices are defined
// by p, never by w.
//
// NUMERICS: every rounding step mirrors the v5/fused golden chain
// (P17/P19): x' = bf16(f32(x_n) + f32(o)) [fr1]; rms in f32 over the
// bf16 residual with inv = fused_rsqrt(sumsq/2048 + 1e-5), xn NOT
// rounded to bf16 before quantize; amax = integer max of (bits &
// 0x7fffffff); d = bf16(amax/127); q = magic-add RNE then clip +-127;
// sigmoid via hw exp2<bfloat16> (P19d measured ~2e-4 actual
// differential, tanh fallback not needed); sw rounded to bf16 before
// quantize; down partials bf16-rounded then f32-accumulated in c order;
// xn1 = bf16(f32(x') + dacc) [fr3].
//
// STACK LAW (P16/P17): the peano linker's single paddxm covers the
// WHOLE call chain; the deepest path here is entry -> lv_body ->
// flavor -> lv_compute (the proven 0x340 hot shape) — keep flavors
// lean and re-check the ELF frame after any structural change.
// PMEM: 16 KB incl. 3052 B glue (P19b law); all scalar loops stay
// ROLLED (#pragma clang loop unroll(disable)).

constexpr uint32_t kBlockBytes = 18560; // A fifo element (v4/v5 ABI)
constexpr uint32_t kKMax = 6144;
constexpr uint32_t kTileRows = 16;
constexpr uint32_t kTileK = 2048;
constexpr uint32_t kGroups = kTileK / 32; // 64
constexpr uint32_t kM1 = 2048;            // model hidden width

// ---- P28-6 ring widening: the worker count N is RUNTIME state, read
// from the X element ([6404,6408) u32) at every exec boundary -- one .o
// serves the 8/16/32-worker designs. Derived per-worker geometry (all
// exact for power-of-2 N <= 128):
//   rows = 2048/N residual rows   (N=8: 256, 16: 128, 32: 64)
//   jpw  = 6144/N gate/up rows    (768 / 384 / 192, all multiples of 32)
//   grp  = jpw/32 quant groups    (24 / 12 / 6)
// The ring masks use & (N-1). lv_xn/lv_o/lv_dacc stay allocated at the
// N=8 maximum (512/512/1024 B) -- smaller N just uses less of them.
// The ring2 SCALE copies go through SCALAR u16 stores (lv_fr2/lv_st2):
// the per-slot scale stride grp*2 is 48/24/12 B and only the 48B case is
// 16B-aligned; the AIE ALIGNMENT LAW (P28-4) makes vector ops at
// 16-mod-32 addresses undefined, and 24B would hit it on odd slots.

// ---- .bss state block (per worker; arrays as raw bytes to keep
// bfloat16 ctors out of the AIE's ctor-less startup). lv_xn/lv_o/lv_dacc
// are sized for the N=8 MAXIMUM (256 rows); smaller N uses less of them. ----
constexpr uint32_t kRowsMax = kM1 / 8; // 256
static uint8_t lv_arena[kGroups * 128] __attribute__((aligned(64))); // 8192: replicated A-ops
static uint8_t lv_sw[6144 + 384] __attribute__((aligned(64)));       // 6528: int8 q global-j + scales @6144
static uint8_t lv_xn[kRowsMax * 2] __attribute__((aligned(64)));     // 512: rows bf16
static uint8_t lv_o[kRowsMax * 2] __attribute__((aligned(64)));      // 512: rows bf16
static float lv_dacc[kRowsMax] __attribute__((aligned(64)));         // 1024: f32 down partials
static uint8_t lv_shared[kM1 * 2] __attribute__((aligned(64)));      // 4096: N ring slots (rows*2B each)
static uint8_t lv_dA[kGroups * 2] __attribute__((aligned(64)));      // 128: 64 bf16
static uint8_t lv_temp[128] __attribute__((aligned(64)));            // per-group f32 staging
static uint8_t lv_qscratch[128] __attribute__((aligned(64)));        // quant div lanes (PMEM law)
static uint8_t lv_ctr[24] __attribute__((aligned(64)));

// Phase-overlaid windows INSIDE lv_shared (L1 law). After K=101's rms
// consumes the gathered x', all 4096 B are dead until ring3 rewrites
// the slots with xn1 — the gate window (1536 B), the up window (64 B)
// and the dead c_out alias for the 4-arg entry live in that gap.
// This is not optional: the ObjectFifo home rule (home tile = FIRST
// worker in the design's workers list that references the fifo) means
// one tile always carries BOTH its ring edges' depth-2 buffers for
// BOTH rings — the worst tile is 37120 (A) + 2048 + 3328 (rings) +
// 64 (C) + 21184 (.bss) + 1024 (stack) = 64768 < 65536. The touch
// graph is a cycle on the workers, so no list ordering can spread the
// double-home; reordering only moves it.
#define lv_gate (lv_shared)
#define lv_upwin (lv_shared + 1536)
#define lv_dummy (lv_shared + 1600)

// P28-12 attention-phase overlays (zero net .bss — everything lives in
// storage that is dead during attention): worker state (q/acc/m,l/kv)
// overlays lv_sw (the sw int8 arena, written only in the gate/up phase
// AFTER attention); cos/sin + q/k-norm weights overlay lv_shared (x'
// gather image, dead between the qkv gather and ring3); the finalized
// 256-row output overlays the SAME lv_shared region (cos/sin and the
// norm weights are consumed at init, before any output is produced).
#define lv_attn_q (lv_sw)                // 2 heads x 128 bf16 (512B)
#define lv_attn_acc (lv_sw + 512)        // 2 heads x 128 f32 (1024B)
#define lv_attn_ml (lv_sw + 1536)        // m0,m1,l0,l1 f32 (16B)
#define lv_attn_kv (lv_sw + 1560)        // staged k_cur,v_cur bf16 x128
#define lv_attn_cs (lv_shared)           // cos 128 f32 | sin 128 f32
#define lv_attn_nw (lv_shared + 1024)    // qn 128 bf16 | kn 128 bf16
#define lv_attn_out (lv_shared)          // 256 bf16, written at finalize

// lv_ctr words
constexpr uint32_t cO = 0;      // o block index (0..15)
constexpr uint32_t cGate = 1;   // gate block index (0..47)
constexpr uint32_t cUpG = 2;    // swiglu group index (0..23)
constexpr uint32_t cUpH = 3;    // up window half (0/1)
constexpr uint32_t cChunk = 4;  // last arena chunk built (255 = none)
constexpr uint32_t cXnI = 5;    // cxn drain index (0..15)
constexpr uint32_t cPhase = 6;  // 0 = O, 1 = QKV
constexpr uint32_t cR1 = 7;     // ring1 rounds received
constexpr uint32_t cR2 = 8;     // ring2 rounds received
constexpr uint32_t cR3 = 9;     // ring3 rounds received
constexpr uint32_t cW = 10;     // worker id
constexpr uint32_t cDown = 11;  // down element index (0..N_DOWN-1)
constexpr uint32_t cP = 12;     // ring position
// P28-6 runtime geometry (set by lv_xelem from the X element's N word):
constexpr uint32_t cN = 13;     // worker count (8/16/32)
constexpr uint32_t cRows = 14;  // 2048/N residual rows per worker
constexpr uint32_t cGrp = 15;   // (6144/N)/32 quant groups per worker
constexpr uint32_t cMask = 16;  // N-1 (ring slot mask; N power of 2)
constexpr uint32_t cJpw = 17;   // 6144/N int8 q bytes per ring2 slot
constexpr uint32_t cDMask = 18; // (rows/16)-1: down row-block-in-chunk mask
constexpr uint32_t cR2E = 19;   // ring2 element bytes = jpw + align32(grp*2)
// P28-12 device-side attention state (standalone vehicle for layer-v3):
constexpr uint32_t cAttnS = 20; // total attended positions incl current
constexpr uint32_t cAttnJ = 21; // kvhist elements consumed
constexpr uint32_t cAttnO = 22; // output 16-row pieces emitted (0..15)
static volatile uint32_t khist_probe_nop; // bisect: 1 = skip attn_step

// shared per-group helpers (P19 forms, cloned from w4gemvu.cc verbatim)
static uint32_t __attribute__((noinline)) fused_sw_amax(const float *__restrict temp);
static void __attribute__((noinline)) fused_sw_quant(uint32_t m_bits,
                                                     float *__restrict temp,
                                                     uint16_t *d_out);
static void __attribute__((noinline)) fused_sw_x(const float *__restrict temp,
                                                 uint8_t *__restrict dst);

// Newton iterations ride the VECTOR unit (PMEM law, P28-4): peano has
// no scalar FPU, so scalar f32 mul is a __mulsf3 soft call -- 3
// iterations x 3 muls pulled the ~780B runtime (plus __muldi3) into
// the 16KB program memory. Vector f32 mul/sub are correctly rounded,
// so the chain stays bit-identical to the golden's np.float32 scalars.
// lv_temp is dead here (the sumsq reduce already consumed it).
static inline float fused_rsqrt(float s)
{
    union { float f; uint32_t u; } v = { s };
    v.u = 0x5F3759DFu - (v.u >> 1);
    const aie::vector<float, 32> sv = aie::broadcast<float, 32>(s);
    const aie::vector<float, 32> two = aie::broadcast<float, 32>(2.0f);
    aie::vector<float, 32> xv = aie::broadcast<float, 32>(v.f);
    float *__restrict scr = reinterpret_cast<float *__restrict>(lv_temp);
#pragma clang loop unroll(disable)
    for (int i = 0; i < 3; i++) {
        const aie::vector<float, 32> y =
            aie::mul(xv, xv).to_vector<float>();
        const aie::vector<float, 32> t =
            aie::sub(two, aie::mul(sv, y).to_vector<float>());
        xv = aie::mul(xv, t).to_vector<float>();
    }
    aie::store_v(scr, xv);
    return scr[0];
}

static inline float fused_bf16_to_f32(uint16_t b)
{
    union { float f; uint32_t u; } v = { 0.0f };
    v.u = (uint32_t)b << 16;
    return v.f;
}

static inline uint16_t fused_f32_to_bf16(float f)
{
    union { float f; uint32_t u; } v = { f };
    uint32_t rounded = v.u + 0x7FFFu + ((v.u >> 16) & 1u);
    return (uint16_t)(rounded >> 16);
}

// ---- compute core: the v5.4 dual-accumulator hot loop, verbatim except
// the A-operand/d pointers are parameters (lv_arena / lv_dA or the sw
// tail) and c_out is the caller's destination.
static void __attribute__((noinline)) lv_compute(const uint8_t *__restrict a_in,
                                                 const int8_t *__restrict As,
                                                 const bfloat16 *__restrict d,
                                                 bfloat16 *__restrict c_out)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    const int4 *__restrict nib = reinterpret_cast<const int4 *__restrict>(a_in);
    const bfloat16 *__restrict sf_t =
        reinterpret_cast<const bfloat16 *__restrict>(a_in + kTileRows * kTileK / 2);

    aie::accum<accfloat, kTileRows> acc0 = aie::zeros<accfloat, kTileRows>();
    aie::accum<accfloat, kTileRows> acc1 = aie::zeros<accfloat, kTileRows>();

    for (uint32_t g = 0; g < kGroups; g += 2) {
        aie::vector<int8, 64> A0 = aie::load_v<64>(As + g * 128);
        aie::vector<int8, 64> A1 = aie::load_v<64>(As + g * 128 + 64);
        aie::vector<int4, 256> B0 = aie::load_v<256>(nib + g * 256);
        aie::vector<int4, 256> B1 = aie::load_v<256>(nib + g * 256 + 128);
        aie::mmul<4, 16, 16, int8, int4> mm;
        mm.mac(A0, B0);
        mm.mac(A1, B1);
        aie::vector<int32, 16> r0 = mm.to_vector<int32>().extract<16>(0);
        aie::vector<float, 16> rf = aie::to_float<float>(r0);
        aie::vector<bfloat16, 16> sfb = aie::load_v<16>(sf_t + g * kTileRows);
        aie::vector<bfloat16, 16> db = aie::broadcast<bfloat16, 16>(d[g]);
        aie::vector<float, 16> sfd = aie::mul(sfb, db).to_vector<float>();
        acc0 = aie::mac(acc0, rf, sfd);

        aie::vector<int8, 64> A2 = aie::load_v<64>(As + g * 128 + 128);
        aie::vector<int8, 64> A3 = aie::load_v<64>(As + g * 128 + 192);
        aie::vector<int4, 256> B2 = aie::load_v<256>(nib + g * 256 + 256);
        aie::vector<int4, 256> B3 = aie::load_v<256>(nib + g * 256 + 384);
        aie::mmul<4, 16, 16, int8, int4> mm1;
        mm1.mac(A2, B2);
        mm1.mac(A3, B3);
        aie::vector<int32, 16> r1 = mm1.to_vector<int32>().extract<16>(0);
        aie::vector<float, 16> rf1 = aie::to_float<float>(r1);
        aie::vector<bfloat16, 16> sfb1 = aie::load_v<16>(sf_t + g * kTileRows + kTileRows);
        aie::vector<bfloat16, 16> db1 = aie::broadcast<bfloat16, 16>(d[g + 1]);
        aie::vector<float, 16> sfd1 = aie::mul(sfb1, db1).to_vector<float>();
        acc1 = aie::mac(acc1, rf1, sfd1);
    }
    aie::vector<float, kTileRows> ones = aie::broadcast<float, kTileRows>(1.0f);
    aie::accum<accfloat, kTileRows> acc =
        aie::mac(acc0, ones, acc1.template to_vector<float>());
    aie::vector<bfloat16, kTileRows> out = acc.template to_vector<bfloat16>();
    aie::store_v(c_out, out);
}

// int8 q (64 groups x 32) -> replicated [4x16] A-ops in lv_arena.
// The K=0 staging body (attn) and the K=105 chunk rebuild (sw) share it.
static void __attribute__((noinline)) lv_build_arena(const int8_t *__restrict x8)
{
#pragma clang loop unroll(disable)
    for (uint32_t g = 0; g < kGroups; g++) {
        const int8_t *xg = x8 + g * 32;
        aie::vector<int8, 16> x0 = aie::load_v<16>(xg);
        aie::vector<int8, 16> x1 = aie::load_v<16>(xg + 16);
        aie::vector<int8, 32> x0l = aie::concat(x0, x0);
        aie::vector<int8, 32> x1l = aie::concat(x1, x1);
        int8_t *dst = reinterpret_cast<int8_t *__restrict>(lv_arena) + g * 128;
        aie::store_v(dst, aie::concat(x0l, x0l));
        aie::store_v(dst + 64, aie::concat(x1l, x1l));
    }
}

// one group's rms-weight pass: temp = (h2 * inv) * wgt in f32, UNROUNDED
// (stage2b semantics — verified against the golden chain in P17).
static void __attribute__((noinline)) lv_wgroup(const bfloat16 *__restrict h2g,
                                                const bfloat16 *__restrict wgtg,
                                                float inv,
                                                float *__restrict temp)
{
    const aie::vector<bfloat16, 32> ones_bf =
        aie::broadcast<bfloat16, 32>((bfloat16)1.0f);
    const aie::accum<accfloat, 32> z = aie::zeros<accfloat, 32>();
    const aie::vector<float, 32> hf =
        aie::mul(aie::load_v<32>(h2g), ones_bf).to_vector<float>();
    const aie::vector<float, 32> wf =
        aie::mul(aie::load_v<32>(wgtg), ones_bf).to_vector<float>();
    const aie::vector<float, 32> t =
        aie::mac(z, hf, aie::broadcast<float, 32>(inv)).to_vector<float>();
    const aie::vector<float, 32> xn = aie::mac(z, t, wf);
    aie::store_v(temp, xn);
}

// ---- K flavors ----

// K=0 X element: stage attn operands + reset ALL per-exec state (incl.
// the P28-6 runtime geometry). Generalized serpentine ring position:
// worker w sits at tile (col c = w>>2, row w&3); even columns walk rows
// ascending, odd columns descending, so p = w on even c and
// p = 8c + 3 - w on odd c (N=8: p = (w<4)?w:11-w, the P28-3 form).
static void __attribute__((noinline)) lv_xelem(const uint8_t *__restrict a_in)
{
    lv_build_arena(reinterpret_cast<const int8_t *__restrict>(a_in));
    const uint32_t d_words = (kGroups * 2) / 4;
    const uint32_t *__restrict ds =
        reinterpret_cast<const uint32_t *__restrict>(a_in + kKMax);
    uint32_t *__restrict dd =
        reinterpret_cast<uint32_t *__restrict>(lv_dA);
    for (uint32_t i = 0; i < d_words; i++)
        dd[i] = ds[i];
    const uint32_t w = *(const uint32_t *__restrict)(a_in + 6400);
    // N at [6404,6408). Only 16/32 change the geometry; anything else
    // (incl. the legacy all-zero X the old builders produced) is N=8.
    uint32_t n = *(const uint32_t *__restrict)(a_in + 6404);
    if (n != 16 && n != 32)
        n = 8;
    const uint32_t n0 = n; // the shift loop below consumes n — keep the
                           // original for the ring mask (P28-6 near-miss:
                           // storing the post-shift n made cMask always 7)
    // rows = 2048/N without a __ctzsi2 call: seed at the N=8 value
    // (kRowsMax) and halve once per halving of n past 8. Seeding at kM1
    // was the P28-6 board hang: N=8 never entered the loop, rows stayed
    // 2048, and the dacc-zero loop wrote 8KB over lv_shared/lv_dA/lv_ctr
    // — the zeroed cRows/cMask then skipped every ring loop and the
    // workers deadlocked on fifo acquire (ERT_CMD_STATE_TIMEOUT).
    uint32_t rows = kRowsMax; // 256
    while (n > 8) {
        n >>= 1;
        rows >>= 1;
    }
    const uint32_t jpw = rows * 3;   // 6144/N (6144 = 3*2048)
    const uint32_t grp = jpw >> 5;
    lv_ctr[cN] = n0;
    lv_ctr[cRows] = rows;
    lv_ctr[cGrp] = grp;
    lv_ctr[cMask] = n0 - 1;
    lv_ctr[cJpw] = jpw;
    lv_ctr[cDMask] = (rows >> 4) - 1;
    lv_ctr[cR2E] = jpw + ((grp * 2 + 31) & ~31u);
    // zero the f32 down accumulators once per exec (first partial += )
    uint32_t *__restrict da =
        reinterpret_cast<uint32_t *__restrict>(lv_dacc);
#pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < rows; i++)
        da[i] = 0;
    const uint32_t col = w >> 2;
    lv_ctr[cW] = w;
    // Ring position. N=8: the two-column serpentine (p = w on even
    // columns, 8c+3-w on odd ones). N=16: the all-adjacent Hamiltonian
    // cycle (CORE<->CORE OBJECTFIFO LAW, P28-6 -- non-adjacent
    // core<->core edges split into a core mem-DMA MM2S channel and npu2
    // cores have only 2 output channels; the serpentine wrap (3,2)->
    // (0,2) needed 3 on worker 12). Cycle order 0,1,2,3,7,6,5,9,10,11,
    // 15,14,13,12,8,4 (design_layerv2.HAM16) has no closed-form inverse,
    // so the position rides this piecewise arithmetic (mirrored by the
    // packer/golden pos()): col 0 = r, col 3 = 13-r, col 1/2 bottom
    // worker = 15/14, else col 1 = 7-r, col 2 = 6+r.
    uint32_t p;
    if (n0 != 16) {
        p = (col & 1) ? (8 * col + 3 - w) : w;
    } else {
        const uint32_t r = w & 3;
        p = (col == 0) ? r
          : (col == 3) ? (13 - r)
          : (r == 0) ? ((col == 1) ? 15 : 14)
          : ((col == 1) ? (7 - r) : (6 + r));
    }
    lv_ctr[cP] = p;
    lv_ctr[cO] = 0;
    lv_ctr[cGate] = 0;
    lv_ctr[cUpG] = 0;
    lv_ctr[cUpH] = 0;
    lv_ctr[cChunk] = 255;
    lv_ctr[cXnI] = 0;
    lv_ctr[cPhase] = 0;
    lv_ctr[cDown] = 0;
    lv_ctr[cR1] = 0;
    lv_ctr[cR2] = 0;
    lv_ctr[cR3] = 0;
}

// K=100 xn element: this worker's x_n chunk (rows bf16).
static void __attribute__((noinline)) lv_xnelem(const uint8_t *__restrict a_in)
{
    const uint16_t *__restrict src =
        reinterpret_cast<const uint16_t *__restrict>(a_in);
    uint16_t *__restrict dst =
        reinterpret_cast<uint16_t *__restrict>(lv_xn);
    const uint32_t rows = lv_ctr[cRows];
#pragma clang loop unroll(disable)
    for (uint32_t r = 0; r < rows; r += 16)
        aie::store_v(dst + r, aie::load_v<16>(src + r));
}

// K=101 (w2, is_w1=0) / K=102 (w1, is_w1=1): rms over the gathered
// vector in lv_shared, quantize g32, rebuild the replicated arena + d.
static void __attribute__((noinline)) lv_rms_elem(const uint8_t *__restrict a_in,
                                                  uint32_t is_w1)
{
    const bfloat16 *__restrict h2 =
        reinterpret_cast<const bfloat16 *__restrict>(lv_shared);
    float *__restrict temp =
        reinterpret_cast<float *__restrict>(lv_temp);

    const aie::vector<bfloat16, 32> ones_bf =
        aie::broadcast<bfloat16, 32>((bfloat16)1.0f);
    aie::accum<accfloat, 32> sq = aie::zeros<accfloat, 32>();
    for (uint32_t g = 0; g < kM1 / 32; g++) {
        const aie::vector<bfloat16, 32> hb = aie::load_v<32>(h2 + g * 32);
        const aie::vector<float, 32> hf = aie::mul(hb, ones_bf).to_vector<float>();
        sq = aie::mac(sq, hf, hf);
    }
    aie::store_v(temp, sq.to_vector<float>());
    float sumsq = 0.0f;
#pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < 32; i++)
        sumsq += temp[i];

    // sumsq/2048 rides the vector unit too: LLVM legally rewrites the
    // division-by-power-of-2 as a mul by the exact reciprocal 2^-11,
    // and that ONE soft mul was the last __mulsf3 call site. The
    // vector mul is exact for every normal (and the epsilon add washes
    // out any denormal tail). lv_temp is dead here (sumsq consumed it).
    aie::store_v(
        temp,
        aie::mul(aie::broadcast<float, 32>(sumsq),
                 aie::broadcast<float, 32>(0x1p-11f)).to_vector<float>());
    const float inv = fused_rsqrt(temp[0] + 1e-5f);
    const bfloat16 *__restrict wgt =
        reinterpret_cast<const bfloat16 *__restrict>(a_in);
    uint16_t *__restrict dv =
        reinterpret_cast<uint16_t *__restrict>(lv_dA);
    for (uint32_t g = 0; g < kM1 / 32; g++) {
        lv_wgroup(h2 + 32 * g, wgt + 32 * g, inv, temp);
        fused_sw_quant(fused_sw_amax(temp), temp, dv + g);
        fused_sw_x(temp, lv_arena + 128 * g);
    }
    lv_ctr[cChunk] = 255;
    if (is_w1) {
        lv_ctr[cPhase] = 1;
    } else {
        lv_ctr[cGate] = 0;
        lv_ctr[cUpG] = 0;
        lv_ctr[cUpH] = 0;
    }
}

// K=103 gate block.
static void __attribute__((noinline)) lv_gateelem(const uint8_t *__restrict a_in)
{
    bfloat16 *dst = reinterpret_cast<bfloat16 *__restrict>(lv_gate) +
                    lv_ctr[cGate] * 16;
    lv_ctr[cGate]++;
    lv_compute(a_in,
               reinterpret_cast<const int8_t *__restrict>(lv_arena),
               reinterpret_cast<const bfloat16 *__restrict>(lv_dA), dst);
}

// K=105 down chunk-block: rebuild arena on chunk change, accumulate the
// bf16-rounded partial into dacc in f32 (c-ascending = bit-exact).
static void __attribute__((noinline)) lv_downelem(const uint8_t *__restrict a_in)
{
    const uint32_t c = *(const uint32_t *__restrict)(a_in + kBlockBytes - 4);
    if (c != lv_ctr[cChunk]) {
        lv_build_arena(reinterpret_cast<const int8_t *__restrict>(lv_sw) +
                       c * kTileK);
        lv_ctr[cChunk] = c;
    }
    bfloat16 *part = reinterpret_cast<bfloat16 *__restrict>(lv_temp);
    lv_compute(a_in,
               reinterpret_cast<const int8_t *__restrict>(lv_arena),
               reinterpret_cast<const bfloat16 *__restrict>(lv_sw + 6144 +
                                                            c * 128),
               part);
    float *__restrict dp = lv_dacc + (lv_ctr[cDown] & lv_ctr[cDMask]) * 16;
    // SCALAR accumulate (P28-7 bisect verdict): both vector forms tried
    // so far are wrong on board -- plain aie::add lowers to a bare vadd.f
    // (single mismatch), and the mul->mac(acc,ones,v) form corrupts
    // wholesale. The scalar loop stays until a unit-tested vector form
    // exists; the accumulator-domain float semantics have sharp edges
    // (P17's "bare acc+acc illegal" was a warning).
    const uint16_t *__restrict pu =
        reinterpret_cast<const uint16_t *__restrict>(part);
#pragma clang loop unroll(disable)
    for (uint32_t r = 0; r < 16; r++)
        dp[r] += fused_bf16_to_f32(pu[r]);
    lv_ctr[cDown]++;
}

// K=2048: phase O -> lv_o slice; phase QKV -> the real C element.
static void __attribute__((noinline)) lv_k2048(const uint8_t *__restrict a_in,
                                               bfloat16 *__restrict c_out)
{
    bfloat16 *dst;
    if (lv_ctr[cPhase] != 0) {
        dst = c_out;
    } else {
        dst = reinterpret_cast<bfloat16 *__restrict>(lv_o) + lv_ctr[cO] * 16;
        lv_ctr[cO]++;
    }
    lv_compute(a_in,
               reinterpret_cast<const int8_t *__restrict>(lv_arena),
               reinterpret_cast<const bfloat16 *__restrict>(lv_dA), dst);
}

// one group's swiglu: sw = g * sigmoid(g) * u (bf16-rounded), staged f32
// (P19 fused_sw_sig, verbatim).
static void __attribute__((noinline)) fused_sw_sig(const bfloat16 *__restrict gp,
                                                   const bfloat16 *__restrict upp,
                                                   float *__restrict temp)
{
    const aie::vector<bfloat16, 32> ones_bf =
        aie::broadcast<bfloat16, 32>((bfloat16)1.0f);
    const aie::vector<float, 32> ones_f = aie::broadcast<float, 32>(1.0f);
    const aie::accum<accfloat, 32> z = aie::zeros<accfloat, 32>();
    const aie::vector<float, 32> negL =
        aie::broadcast<float, 32>(-1.4426950408889634f);
    const aie::accum<accfloat, 32> one = aie::mac(z, ones_f, ones_f);

    const aie::vector<float, 32> gv =
        aie::mul(aie::load_v<32>(gp), ones_bf).to_vector<float>();
    const aie::vector<float, 32> uv =
        aie::mul(aie::load_v<32>(upp), ones_bf).to_vector<float>();
    const aie::vector<float, 32> xarg =
        aie::mac(z, gv, negL).to_vector<float>();
    const aie::vector<bfloat16, 32> t = aie::exp2(xarg);
    const aie::vector<float, 32> tf =
        aie::mul(t, ones_bf).to_vector<float>();
    const aie::vector<float, 32> den =
        aie::mac(one, tf, ones_f).to_vector<float>();
    const aie::vector<float, 32> sig = aie::div(ones_f, den);
    const aie::vector<float, 32> gs = aie::mac(z, gv, sig).to_vector<float>();
    const aie::vector<bfloat16, 32> swb =
        aie::mac(z, gs, uv).to_vector<bfloat16>();
    aie::store_v(temp, aie::mul(swb, ones_bf).to_vector<float>());
}

// integer amax of the staged sw — P28-7 vector form. The scalar loop
// compares MASKED BITS (bits & 0x7fffffff) as integers; float is
// sign-magnitude, and the mask flip is exactly a wraparound add of
// INT_MIN: signed lanes become [b (positives) | (b+2^31)&mask
// (negatives)] -- so the pair (reduce_max(t), reduce_max(t+INT_MIN))
// always contains the masked-bit max, and their scalar max IS it.
// Proof-checked in numpy over 200k random + adversarial patterns
// (denormals, +-0, NaN payloads, +-inf, all-zero groups): 0 mismatches.
// (The obvious max(b, -b) is WRONG -- -signed(b) is the two's-complement
// 2^31-m, not the sign-magnitude m; that form was the first board try
// and failed 10/10.) vadd.32 + vmax-tree are real vector ops; the
// aie::abs/bit_and routes scalarize on 32b lanes.
static uint32_t __attribute__((noinline)) fused_sw_amax(const float *__restrict temp)
{
    const aie::vector<int32_t, 32> t = aie::load_v<32>(
        reinterpret_cast<const int32_t *__restrict>(temp));
    const aie::vector<int32_t, 32> t2 =
        aie::add(t, aie::broadcast<int32_t, 32>((int32_t)0x80000000));
    const int32_t m1 = aie::reduce_max(t);
    const int32_t m2 = aie::reduce_max(t2);
    return (uint32_t)(m1 > m2 ? m1 : m2);
}

// d + invd + the RNE magic add, one group (v24e form: broadcasts at
// function scope, no loops). The two divisions ride the vector unit's
// aie::div (mul by the hw reciprocal, ~1-2 ulp off IEEE): they were the
// only __divsf3 callers, and with them gone __divsf3+__muldi3 (~1.2KB)
// drop out of the 16KB program memory. The ulp drift is invisible at
// the golden bands (db is bf16 = 2^-8 granularity; invd feeds a scale
// multiply). lv_qscratch because temp is LIVE across this function.
static void __attribute__((noinline)) fused_sw_quant(uint32_t m_bits,
                                                     float *__restrict temp,
                                                     uint16_t *d_out)
{
    float invd = 0.0f;
    uint16_t db = 0;
    if (m_bits != 0) {
        union { uint32_t u; float f; } amax;
        amax.u = m_bits;
        float *__restrict scr =
            reinterpret_cast<float *__restrict>(lv_qscratch);
        // aie::div returns an ACCUM (it is mul(a, inv(b)) and the mul
        // yields one) -- store_v's template deduction needs the explicit
        // .to_vector (fused_sw_sig only compiles because assignment
        // converts implicitly).
        aie::store_v(scr, aie::div(aie::broadcast<float, 32>(amax.f),
                                   aie::broadcast<float, 32>(127.0f))
                               .to_vector<float>());
        db = fused_f32_to_bf16(scr[0]);
        aie::store_v(scr, aie::div(aie::broadcast<float, 32>(1.0f),
                                   aie::broadcast<float, 32>(
                                       fused_bf16_to_f32(db)))
                               .to_vector<float>());
        invd = scr[0];
    }
    *d_out = db;
    const aie::vector<float, 32> ones_f = aie::broadcast<float, 32>(1.0f);
    const aie::accum<accfloat, 32> z = aie::zeros<accfloat, 32>();
    const aie::accum<accfloat, 32> sc =
        aie::mac(z, aie::load_v<32>(temp),
                 aie::broadcast<float, 32>(invd));
    aie::store_v(temp,
                 aie::mac(sc, ones_f, aie::broadcast<float, 32>(12582912.0f))
                     .to_vector<float>());
}

// q extraction (round-then-clip) + the replicated A-operand store
// (P19 strided form) — used by the rms pipeline (K=101/102).
// P28-7 final vector form. Every op class here is individually
// board-proven: the double-pack narrowing (lv_sw_x32, bisect round 5),
// the 32B int8 store (same), and the load_v<16>(+16)/concat/32B-store
// replication is lv_build_arena's exact body (P19, board-proven) --
// hand-inlined with a dst parameter because that kernel writes a
// fixed base. Rounds 4/6 failed on the UNPROVEN classes: 64B concat
// stores, and 16B stores at 16-mod-32 offsets (the P28-4 ALIGNMENT
// LAW -- dst+16/48/80/112 -- scalar byte stores have no such limit,
// which is why the original loop was safe).
static void __attribute__((noinline)) fused_sw_x(const float *__restrict temp,
                                                 uint8_t *__restrict dst)
{
    const aie::vector<int32_t, 32> t = aie::load_v<32>(
        reinterpret_cast<const int32_t *__restrict>(temp));
    const aie::vector<int32_t, 32> qs =
        aie::sub(t, aie::broadcast<int32_t, 32>((int32_t)0x4B400000));
    const aie::vector<int32_t, 32> qc = aie::min(
        aie::max(qs, aie::broadcast<int32_t, 32>(-127)),
        aie::broadcast<int32_t, 32>(127));
    // lv_qscratch is dead here (fused_sw_quant consumed it before x in
    // the same group chain) -- reuse it as the 32B staging buffer.
    uint8_t *scr = lv_qscratch;
    aie::store_v(reinterpret_cast<int8_t *__restrict>(scr),
                 qc.pack<int16_t>().pack<int8_t>());
    // lv_build_arena's body, g=0, dst-parameterized (all 32B-aligned):
    // [q0 q0 | q0 q0] then [q1 q1 | q1 q1].
    const int8_t *x8 = reinterpret_cast<const int8_t *__restrict>(scr);
    int8_t *d = reinterpret_cast<int8_t *__restrict>(dst);
    const aie::vector<int8, 16> q0 = aie::load_v<16>(x8);
    const aie::vector<int8, 16> q1 = aie::load_v<16>(x8 + 16);
    const aie::vector<int8, 32> r0 = aie::concat(q0, q0);
    const aie::vector<int8, 32> r0b = aie::concat(q0, q0);
    const aie::vector<int8, 32> r1 = aie::concat(q1, q1);
    const aie::vector<int8, 32> r1b = aie::concat(q1, q1);
    aie::store_v(d, r0);
    aie::store_v(d + 32, r0b);
    aie::store_v(d + 64, r1);
    aie::store_v(d + 96, r1b);
}

// LINEAR 32B q extraction (ring2's global-j layout — no replication;
// down's K=105 rebuild does the replication later).
// P28-7 bisect round 5: vector WITHOUT the concat replication — if this
// passes, the round-4 corruption lived in fused_sw_x's replication; if
// it fails, the double-pack narrowing itself is guilty.
static void __attribute__((noinline)) lv_sw_x32(const float *__restrict temp,
                                                uint8_t *__restrict dst)
{
    const aie::vector<int32_t, 32> t = aie::load_v<32>(
        reinterpret_cast<const int32_t *__restrict>(temp));
    const aie::vector<int32_t, 32> qs =
        aie::sub(t, aie::broadcast<int32_t, 32>((int32_t)0x4B400000));
    const aie::vector<int32_t, 32> qc = aie::min(
        aie::max(qs, aie::broadcast<int32_t, 32>(-127)),
        aie::broadcast<int32_t, 32>(127));
    const aie::vector<int8_t, 32> q = qc.pack<int16_t>().pack<int8_t>();
    aie::store_v(reinterpret_cast<int8_t *__restrict>(dst), q);
}

// K=104 up block: window half fill; every completed 32-value pair runs
// one swiglu group with global index g = p*24 + k.
static void __attribute__((noinline)) lv_upelem(const uint8_t *__restrict a_in)
{
    bfloat16 *dst = reinterpret_cast<bfloat16 *__restrict>(lv_upwin) +
                    lv_ctr[cUpH] * 16;
    lv_compute(a_in,
               reinterpret_cast<const int8_t *__restrict>(lv_arena),
               reinterpret_cast<const bfloat16 *__restrict>(lv_dA), dst);
    if (lv_ctr[cUpH] == 1) {
        const uint32_t g = lv_ctr[cP] * lv_ctr[cGrp] + lv_ctr[cUpG];
        float *__restrict temp =
            reinterpret_cast<float *__restrict>(lv_temp);
        fused_sw_sig(reinterpret_cast<const bfloat16 *__restrict>(lv_gate) +
                         lv_ctr[cUpG] * 32,
                     reinterpret_cast<const bfloat16 *__restrict>(lv_upwin),
                     temp);
        fused_sw_quant(fused_sw_amax(temp), temp,
                       reinterpret_cast<uint16_t *__restrict>(lv_sw + 6144 +
                                                              2 * g));
        lv_sw_x32(temp, lv_sw + 32 * g);
        lv_ctr[cUpG]++;
        lv_ctr[cUpH] = 0;
    } else {
        lv_ctr[cUpH] = 1;
    }
}

// ---- P28-12 device-side attention flavors ----
// Standalone verification vehicle for layer-v3 (all inference on NPU):
// worker p (ring position) computes Q heads 2p, 2p+1 against KV head
// p/2 (GQA 16Q/4KV), streaming the KV history one position per element
// with an ONLINE softmax (no K-history storage — L1 cannot hold it).
// Numerics mirror the golden chain (rope_pairs / qk_rms_bits /
// attention_bits: f32 scores, softmax, PV; bf16 boundaries) with the
// online-rescale and vector-rounding drift absorbed by the test
// tolerance (4e-2 rel — the same band the mha kernel passes with).
//
// Element ABI (offsets in the 18560B element):
//   K=210 init: [0,4096) q_full 2048 bf16 | [4096,5120) k_cur 512 bf16
//               | [5120,6144) v_cur 512 bf16 | [6144,7168) cos 128 f32
//               | [7168,8192) sin 128 f32 | [8192,8448) qn 128 bf16
//               | [8448,8704) kn 128 bf16 | [8704,8708) S u32
//   K=211 hist: [0,2048) k 4x128 bf16 | [2048,4096) v 4x128 bf16
//   K=212 out:  first call finalizes (consumes the staged current k/v,
//               normalizes, bf16-converts); every call copies the next
//               16 output rows into the C element.

// rotate-half rope on one 128-dim head: pairs (i, i+64) share
// (cos[i], sin[i]); f32 math, bf16 out (golden rope_pairs form).
// out_lo = lo*c - hi*s (sub: fused_rsqrt-proven); out_hi = hi*c + lo*s
// via mul->acc then mac(acc, ones, prod) — the lv_fr1-proven add.
static void __attribute__((noinline)) attn_rope(const bfloat16 *__restrict x,
                                                bfloat16 *__restrict dst)
{
    const float *__restrict cs = reinterpret_cast<const float *__restrict>(lv_attn_cs);
    const aie::vector<bfloat16, 32> ones_bf =
        aie::broadcast<bfloat16, 32>((bfloat16)1.0f);
    const aie::vector<float, 32> ones_f = aie::broadcast<float, 32>(1.0f);
#pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < 64; i += 32) {
        const aie::vector<float, 32> lo =
            aie::mul(aie::load_v<32>(x + i), ones_bf).to_vector<float>();
        const aie::vector<float, 32> hi =
            aie::mul(aie::load_v<32>(x + 64 + i), ones_bf).to_vector<float>();
        const aie::vector<float, 32> c = aie::load_v<32>(cs + i);
        const aie::vector<float, 32> s = aie::load_v<32>(cs + 128 + i);
        const aie::vector<float, 32> v_lo =
            aie::sub(aie::mul(lo, c).to_vector<float>(),
                     aie::mul(hi, s).to_vector<float>());
        aie::accum<accfloat, 32> a_hi = aie::mul(hi, c);
        a_hi = aie::mac(a_hi, ones_f, aie::mul(lo, s).to_vector<float>());
        aie::store_v(dst + i, aie::mul(v_lo, ones_f).to_vector<bfloat16>());
        aie::store_v(dst + 64 + i, a_hi.to_vector<bfloat16>());
    }
}

// per-head rms over 128 AFTER rope, x*inv*w, bf16 out (qk_rms_bits
// golden: SEQUENTIAL scalar sumsq — mirrored; rsqrt via the vector
// Newton form; the /128 rides the vector mul like K=101's /2048).
static void __attribute__((noinline)) attn_qknorm(const bfloat16 *__restrict x,
                                                  const bfloat16 *__restrict w,
                                                  bfloat16 *__restrict dst)
{
    const aie::vector<bfloat16, 32> ones_bf =
        aie::broadcast<bfloat16, 32>((bfloat16)1.0f);
    float *__restrict temp = reinterpret_cast<float *__restrict>(lv_temp);
    aie::accum<accfloat, 32> sq = aie::zeros<accfloat, 32>();
#pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < 128; i += 32) {
        const aie::vector<float, 32> hf =
            aie::mul(aie::load_v<32>(x + i), ones_bf).to_vector<float>();
        sq = aie::mac(sq, hf, hf);
    }
    aie::store_v(temp, sq.to_vector<float>());
    float ms = 0.0f;
#pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < 32; i++)
        ms += temp[i];
    aie::store_v(temp, aie::mul(aie::broadcast<float, 32>(ms),
                                aie::broadcast<float, 32>(0x1p-7f)).to_vector<float>());
    const float inv = fused_rsqrt(temp[0] + 1e-5f);
    const aie::accum<accfloat, 32> z = aie::zeros<accfloat, 32>();
#pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < 128; i += 32) {
        const aie::vector<float, 32> hf =
            aie::mul(aie::load_v<32>(x + i), ones_bf).to_vector<float>();
        const aie::vector<float, 32> wf =
            aie::mul(aie::load_v<32>(w + i), ones_bf).to_vector<float>();
        const aie::vector<float, 32> t =
            aie::mul(hf, aie::broadcast<float, 32>(inv)).to_vector<float>();
        aie::accum<accfloat, 32> o = aie::mac(z, t, wf);
        aie::store_v(dst + i, o.to_vector<bfloat16>());
    }
}

// e^x via the vector unit (peano has no scalar FPU): broadcast, scale
// by log2e ON VECTORS, exp2 (bf16 result — the mha.cc-proven precision
// class), widen, read lane 0.
static __attribute__((noinline)) float attn_exp(float x)
{
    // fused_sw_sig's EXACT forms (mac-built argument, exp2, bf16->f32
    // widen, store then memory readback) — only the log2e sign differs
    // (we want e^x, sigmoid wants e^-x).
    const aie::vector<bfloat16, 32> ones_bf =
        aie::broadcast<bfloat16, 32>((bfloat16)1.0f);
    const aie::vector<float, 32> ones_f = aie::broadcast<float, 32>(1.0f);
    const aie::accum<accfloat, 32> z = aie::zeros<accfloat, 32>();
    const aie::vector<float, 32> posL =
        aie::broadcast<float, 32>(1.4426950408889634f);
    const aie::vector<float, 32> xv = aie::broadcast<float, 32>(x);
    const aie::vector<float, 32> xarg = aie::mac(z, xv, posL).to_vector<float>();
    const aie::vector<bfloat16, 32> t = aie::exp2(xarg);
    const aie::vector<float, 32> tf = aie::mul(t, ones_bf).to_vector<float>();
    (void)ones_f;
    aie::store_v(reinterpret_cast<float *__restrict>(lv_temp), tf);
    return reinterpret_cast<const float *__restrict>(lv_temp)[0];
}

// online-softmax update for both heads against one KV position:
// s = dot(q_h, k)/sqrt(128) (f32; the /sqrt rides aie::div like the
// quant path); m/l/acc rescale in f32 with mul->acc + mac(acc, e, v).
// STACK LAW (P16/P17): peano gives each core a 0x400 stack window with
// the A-fifo buffers directly above it — attn_step with everything
// inlined hit 0x580 and overwrote the fifo stream (first-exec hang,
// the P17 M6 signature). Keep the flavor frames small by splitting the
// per-head compute into noinline helpers.
static float __attribute__((noinline)) attn_dot(const bfloat16 *__restrict qh,
                                                const bfloat16 *__restrict k_j)
{
    const aie::vector<bfloat16, 32> ones_bf =
        aie::broadcast<bfloat16, 32>((bfloat16)1.0f);
    float *__restrict temp = reinterpret_cast<float *__restrict>(lv_temp);
    aie::accum<accfloat, 32> acc = aie::zeros<accfloat, 32>();
#pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < 128; i += 32) {
        const aie::vector<float, 32> qf =
            aie::mul(aie::load_v<32>(qh + i), ones_bf).to_vector<float>();
        const aie::vector<float, 32> kf =
            aie::mul(aie::load_v<32>(k_j + i), ones_bf).to_vector<float>();
        acc = aie::mac(acc, qf, kf);
    }
    aie::store_v(temp, acc.to_vector<float>());
    float dot = 0.0f;
#pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < 32; i++)
        dot += temp[i];
    const aie::vector<float, 32> d_v = aie::broadcast<float, 32>(dot);
    const aie::vector<float, 32> r_v = aie::broadcast<float, 32>(11.3137085f);
    aie::store_v(temp, aie::div(d_v, r_v).to_vector<float>());
    return temp[0]; // q.k / sqrt(128)
}

static void __attribute__((noinline)) attn_acc_update(float *__restrict ah,
                                                      const bfloat16 *__restrict v_j,
                                                      float alpha, float e)
{
    const aie::vector<bfloat16, 32> ones_bf =
        aie::broadcast<bfloat16, 32>((bfloat16)1.0f);
    const aie::vector<float, 32> alpha_v = aie::broadcast<float, 32>(alpha);
    const aie::vector<float, 32> e_v = aie::broadcast<float, 32>(e);
#pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < 128; i += 32) {
        const aie::vector<float, 32> af = aie::load_v<32>(ah + i);
        const aie::vector<float, 32> vf =
            aie::mul(aie::load_v<32>(v_j + i), ones_bf).to_vector<float>();
        aie::accum<accfloat, 32> r = aie::mul(af, alpha_v);
        r = aie::mac(r, e_v, vf);
        aie::store_v(ah + i, r.to_vector<float>());
    }
}

static void __attribute__((noinline)) attn_step(const bfloat16 *__restrict k_j,
                                                const bfloat16 *__restrict v_j)
{
    float *__restrict temp = reinterpret_cast<float *__restrict>(lv_temp);
    float *__restrict ml = reinterpret_cast<float *__restrict>(lv_attn_ml);
#pragma clang loop unroll(disable)
    for (uint32_t h = 0; h < 2; h++) {
        const bfloat16 *__restrict qh =
            reinterpret_cast<const bfloat16 *__restrict>(lv_attn_q) + h * 128;
        const float s = attn_dot(qh, k_j);
        if (s != s) return; // BISECT: stop after dot+div
        const float m_old = ml[h];
        const float m_new = s > m_old ? s : m_old;
        // EXP2 DOMAIN LAW: aie::exp2 traps/stalls on huge-negative args
        // (the shipped call sites only ever see |x| < ~100 sigmoid
        // arguments; the online-softmax's first step feeds -1e30).
        // Clamp to -80: exp(-80) ~ 1.8e-35 is numerically the zero the
        // math wants anyway.
        float ea = m_old - m_new;
        if (ea < -40.0f)
            ea = -40.0f;
        float earg = s - m_new;
        if (earg < -40.0f)
            earg = -40.0f;
        const float alpha = attn_exp(ea);
        const float e = attn_exp(earg);
        ml[h] = m_new;
        // l_new = l*alpha + e on the vector unit (P17: NO scalar f32 mul
        // — it would relink __mulsf3 into the 16KB PMEM).
        const aie::vector<float, 32> l_old = aie::broadcast<float, 32>(ml[2 + h]);
        const aie::vector<float, 32> al_v = aie::broadcast<float, 32>(alpha);
        const aie::vector<float, 32> ones_f = aie::broadcast<float, 32>(1.0f);
        const aie::vector<float, 32> e_v2 = aie::broadcast<float, 32>(e);
        aie::store_v(temp,
                     aie::mac(aie::mul(l_old, al_v), ones_f, e_v2).to_vector<float>());
        ml[2 + h] = temp[0];
        attn_acc_update(
            reinterpret_cast<float *__restrict>(lv_attn_acc) + h * 128,
            v_j, alpha, e);
    }
}

// K=210: stage cos/sin + norm weights, rope+qk-norm q (heads 2p, 2p+1)
// and k_cur (head p/2), zero the online state.
static void __attribute__((noinline)) lv_attninit(const uint8_t *__restrict a_in)
{
    // CR-STATE LAW (P28-12): every shipped aie::exp2 call site runs
    // AFTER some flavor set conv_even (lv_compute's residue) — exp2
    // traps/stalls under the default CR state. The attention path
    // touches no set_rounding flavor before its first exp2, so set it
    // here, once.
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    const uint32_t p = lv_ctr[cP];
    lv_ctr[cAttnS] = *(const uint32_t *__restrict)(a_in + 8704);
    lv_ctr[cAttnJ] = 0;
    lv_ctr[cAttnO] = 0;
    const uint32_t *__restrict cs32 =
        reinterpret_cast<const uint32_t *__restrict>(a_in + 6144);
    uint32_t *__restrict cd32 =
        reinterpret_cast<uint32_t *__restrict>(lv_attn_cs);
#pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < 256; i++)
        cd32[i] = cs32[i]; // cos 128 f32 | sin 128 f32
    const bfloat16 *__restrict src =
        reinterpret_cast<const bfloat16 *__restrict>(a_in);
    bfloat16 *__restrict nw =
        reinterpret_cast<bfloat16 *__restrict>(lv_attn_nw);
#pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < 128; i += 16) {
        aie::store_v(nw + i, aie::load_v<16>(src + 8192 / 2 + i));       // qn
        aie::store_v(nw + 128 + i, aie::load_v<16>(src + 8448 / 2 + i)); // kn
    }
    bfloat16 *__restrict qd =
        reinterpret_cast<bfloat16 *__restrict>(lv_attn_q);
    bfloat16 *__restrict kvs =
        reinterpret_cast<bfloat16 *__restrict>(lv_attn_kv);
    const uint32_t h0 = 2 * p;
    attn_rope(src + h0 * 128, qd);
    attn_rope(src + (h0 + 1) * 128, qd + 128);
    attn_qknorm(qd, nw, qd);
    attn_qknorm(qd + 128, nw, qd + 128);
    const uint32_t kvh = p / 2;
    attn_rope(src + 2048 + kvh * 128, kvs);
    attn_qknorm(kvs, nw + 128, kvs);
#pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < 128; i += 16)
        aie::store_v(kvs + 128 + i,
                     aie::load_v<16>(src + 2560 + kvh * 128 + i)); // v_cur
    float *__restrict ml = reinterpret_cast<float *__restrict>(lv_attn_ml);
    float *__restrict accp = reinterpret_cast<float *__restrict>(lv_attn_acc);
#pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < 4; i++)
        ml[i] = -1e30f;
#pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < 256; i++)
        accp[i] = 0.0f;
}

// K=211: one history position — worker p/2's KV slices.
static void __attribute__((noinline)) lv_attnhist(const uint8_t *__restrict a_in)
{
    const uint32_t p = lv_ctr[cP];
    const uint32_t kvh = p / 2;
    const bfloat16 *__restrict b =
        reinterpret_cast<const bfloat16 *__restrict>(a_in);
    attn_step(b + kvh * 128, b + 1024 + kvh * 128);
    lv_ctr[cAttnJ]++;
}

// K=212: first call processes the staged current k/v and finalizes
// (out = acc/l, bf16); every call drains 16 output rows to the C
// element.
static void __attribute__((noinline)) lv_attnout(bfloat16 *__restrict c_out)
{
    const bfloat16 *__restrict kvs =
        reinterpret_cast<const bfloat16 *__restrict>(lv_attn_kv);
    if (lv_ctr[cAttnO] == 0) {
        attn_step(kvs, kvs + 128); // current position = last entry
        float *__restrict ml = reinterpret_cast<float *__restrict>(lv_attn_ml);
            aie::store_v(reinterpret_cast<float *__restrict>(lv_temp),
                     aie::div(aie::broadcast<float, 32>(1.0f),
                              aie::broadcast<float, 32>(ml[2]))
                         .to_vector<float>()); // 1/l for head 0
        const float inv0 = reinterpret_cast<const float *__restrict>(lv_temp)[0];
        aie::store_v(reinterpret_cast<float *__restrict>(lv_temp),
                     aie::div(aie::broadcast<float, 32>(1.0f),
                              aie::broadcast<float, 32>(ml[3]))
                         .to_vector<float>()); // 1/l for head 1
        const float inv1 = reinterpret_cast<const float *__restrict>(lv_temp)[0];
        bfloat16 *__restrict out =
            reinterpret_cast<bfloat16 *__restrict>(lv_attn_out);
        const float *__restrict accp =
            reinterpret_cast<const float *__restrict>(lv_attn_acc);
#pragma clang loop unroll(disable)
        for (uint32_t h = 0; h < 2; h++) {
            const float inv = h == 0 ? inv0 : inv1;
            const aie::vector<float, 32> inv_v = aie::broadcast<float, 32>(inv);
#pragma clang loop unroll(disable)
            for (uint32_t i = 0; i < 128; i += 32) {
                const aie::vector<float, 32> af =
                    aie::load_v<32>(accp + h * 128 + i);
                aie::store_v(out + h * 128 + i,
                             aie::mul(af, inv_v).to_vector<bfloat16>());
            }
        }
    }
    const uint32_t o = lv_ctr[cAttnO]++;
    // DEBUG MODE (S word bit 31): the 16 C elements stream the POST-
    // rope/qknorm q rows (256 bf16) instead of the attention output —
    // segment-by-segment bring-up.
    const bfloat16 *__restrict out =
        (lv_ctr[cAttnS] & 0x80000000u)
            ? reinterpret_cast<const bfloat16 *__restrict>(lv_attn_q)
            : reinterpret_cast<const bfloat16 *__restrict>(lv_attn_out);
    aie::store_v(c_out, aie::load_v<16>(out + o * 16));
}

// ---- dispatcher (P16 stack law: tiny, tail-calls noinline flavors) ----
static void lv_body(const uint8_t *__restrict a_in, bfloat16 *c_out)
{
    const uint32_t k = *(const uint32_t *__restrict)(a_in + kBlockBytes - 8);
    if (k == 2049) {
        // P28-7 PERF-ONLY floor discriminator: the pack rewrites every W
        // element's K header to 2049 (tools/lv2_floor_pack.py), the full
        // lv_compute runs into the dead lv_dummy alias with zero glue and
        // zero state -- T(full) - T(floor) IS the glue exposure the 6f-9
        // duty-cycle model predicted (never directly measured before).
        lv_compute(a_in,
                   reinterpret_cast<const int8_t *__restrict>(lv_arena),
                   reinterpret_cast<const bfloat16 *__restrict>(lv_dA),
                   reinterpret_cast<bfloat16 *__restrict>(lv_dummy));
        return;
    }
    if (k == kTileK) {
        lv_k2048(a_in, c_out);
        return;
    }
    if (k == 0) {
        lv_xelem(a_in);
        return;
    }
    if (k == 100) {
        lv_xnelem(a_in);
        return;
    }
    if (k == 101) {
        lv_rms_elem(a_in, 0);
        return;
    }
    if (k == 102) {
        lv_rms_elem(a_in, 1);
        return;
    }
    if (k == 103) {
        lv_gateelem(a_in);
        return;
    }
    if (k == 104) {
        lv_upelem(a_in);
        return;
    }
    if (k == 105) {
        lv_downelem(a_in);
        return;
    }
    if (k == 210) {
        lv_attninit(a_in);
        return;
    }
    if (k == 211) {
        lv_attnhist(a_in);
        return;
    }
    if (k == 212) {
        lv_attnout(c_out);
        return;
    }
    // unknown K: no-op (the fill never produces one)
}

extern "C" {

// C-producing entry: the qkv flavor (and any future C-producing
// flavor). Signature-trimmed (PMEM law): m/group_size/tile_idx were
// ignored — and every trimmed arg is a wrapper-side constant the MLIR
// call site materializes in program memory.
void w4gemvu_layer_bf16(const uint8_t *__restrict a_in,
                        bfloat16 *__restrict c_out)
{
    lv_body(a_in, c_out);
}

// No-C entry: every flavor that acquires no C element (MLIR cannot
// pass a null memref; c_out routes to dead scratch and only the QKV
// phase of K=2048 would ever dereference it — which this entry never
// receives by construction).
void w4gemvu_layer_a(const uint8_t *__restrict a_in)
{
    lv_body(a_in, reinterpret_cast<bfloat16 *__restrict>(lv_dummy));
}

// P28-12 standalone attention entries: micro-dispatchers (PMEM
// isolation — the lv_body chain would link ~20KB of flavors into a
// 16KB program memory). Handles the X init (K=0, worker id/state) and
// the three attention flavors; anything else is a no-op.
void w4gemvu_attn_a(const uint8_t *__restrict a_in)
{
    const uint32_t k = *(const uint32_t *__restrict)(a_in + kBlockBytes - 8);
    if (k == 0) {
        lv_xelem(a_in);
        return;
    }
    if (k == 210) {
        lv_attninit(a_in);
        return;
    }
    if (k == 211) {
        lv_attnhist(a_in);
        return;
    }
}

void w4gemvu_attn_bf16(const uint8_t *__restrict a_in,
                       bfloat16 *__restrict c_out)
{
    const uint32_t k = *(const uint32_t *__restrict)(a_in + kBlockBytes - 8);
    if (k == 212) {
        lv_attnout(c_out);
        return;
    }
    w4gemvu_attn_a(a_in);
}

// ---- ring helpers (fifo acquire/release live in the design's worker
// body; these are the data moves between them) ----

// ring1 make: x'_p = bf16(f32(x_n) + f32(o)) for the OWN chunk, into
// the outgoing element AND the own slot of lv_shared.
void lv_fr1(uint8_t *__restrict ob)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    const aie::vector<bfloat16, 32> ones_bf =
        aie::broadcast<bfloat16, 32>((bfloat16)1.0f);
    const aie::vector<float, 32> ones_f = aie::broadcast<float, 32>(1.0f);
    const uint32_t p = lv_ctr[cP];
    const uint32_t rows = lv_ctr[cRows];
    const bfloat16 *__restrict xn =
        reinterpret_cast<const bfloat16 *__restrict>(lv_xn);
    const bfloat16 *__restrict oo =
        reinterpret_cast<const bfloat16 *__restrict>(lv_o);
    bfloat16 *__restrict sh =
        reinterpret_cast<bfloat16 *__restrict>(lv_shared) + p * rows;
    bfloat16 *__restrict dst = reinterpret_cast<bfloat16 *__restrict>(ob);
#pragma clang loop unroll(disable)
    for (uint32_t g = 0; g < rows / 32; g++) {
        aie::accum<accfloat, 32> a =
            aie::mul(aie::load_v<32>(xn + g * 32), ones_bf);
        a = aie::mac(a, ones_f,
                     aie::mul(aie::load_v<32>(oo + g * 32), ones_bf)
                         .to_vector<float>());
        const aie::vector<bfloat16, 32> v = a.to_vector<bfloat16>();
        aie::store_v(dst + g * 32, v);
        aie::store_v(sh + g * 32, v);
    }
    lv_ctr[cR1] = 1;
}

// ring1 forward (rows*2B) — also ring3's forward (same r13 fifo, same
// element size).
void lv_fw1(const uint8_t *__restrict s, uint8_t *__restrict ob)
{
    const uint16_t *__restrict src =
        reinterpret_cast<const uint16_t *__restrict>(s);
    uint16_t *__restrict dst =
        reinterpret_cast<uint16_t *__restrict>(ob);
    const uint32_t rows = lv_ctr[cRows];
#pragma clang loop unroll(disable)
    for (uint32_t r = 0; r < rows; r += 16)
        aie::store_v(dst + r, aie::load_v<16>(src + r));
}

// ring1 store: slot (p - r1r) & (N-1) of lv_shared.
void lv_st1(const uint8_t *__restrict s)
{
    const uint32_t slot = (lv_ctr[cP] - lv_ctr[cR1]) & lv_ctr[cMask];
    const uint32_t rows = lv_ctr[cRows];
    const uint16_t *__restrict src =
        reinterpret_cast<const uint16_t *__restrict>(s);
    uint16_t *__restrict dst =
        reinterpret_cast<uint16_t *__restrict>(lv_shared) + slot * rows;
#pragma clang loop unroll(disable)
    for (uint32_t r = 0; r < rows; r += 16)
        aie::store_v(dst + r, aie::load_v<16>(src + r));
    lv_ctr[cR1]++;
}

// ring2 make: own jpw int8 q + grp u16 scales -> R2B element (q region
// and every 32B boundary stay aligned; the trailing pad is left stale).
void lv_fr2(uint8_t *__restrict ob)
{
    const uint32_t p = lv_ctr[cP];
    const uint32_t jpw = lv_ctr[cJpw];
    const uint8_t *__restrict q = lv_sw + p * jpw;
#pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < jpw; i += 32) {
        const auto v = aie::load_v<32>(q + i);
        aie::store_v(ob + i, v);
    }
    // AIE ALIGNMENT LAW (P28-4): the per-position scale stride is
    // grp*2 = 48/24/12B for N=8/16/32 and only 48B is even 16B-aligned,
    // so the P28-4 three-16B-op fix does not generalize. SCALAR u16
    // copies have no alignment constraint -- grp (6-24) iterations of
    // a rolled loop, invisible next to the jpw-byte q copy.
    const uint32_t grp = lv_ctr[cGrp];
    const uint16_t *__restrict sc =
        reinterpret_cast<const uint16_t *__restrict>(lv_sw + 6144 +
                                                     p * grp * 2);
    uint16_t *__restrict d2 =
        reinterpret_cast<uint16_t *__restrict>(ob + jpw);
#pragma clang loop unroll(disable)
    for (uint32_t k = 0; k < grp; k++)
        d2[k] = sc[k];
    lv_ctr[cR2] = 1;
}

// ring2 forward: EXACTLY cR2E bytes in 32B chunks — never overrun into
// the adjacent depth-2 fifo buffer (element size is a multiple of 32).
void lv_fw2(const uint8_t *__restrict s, uint8_t *__restrict ob)
{
    const uint32_t r2e = lv_ctr[cR2E];
#pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < r2e; i += 32) {
        const auto v = aie::load_v<32>(s + i);
        aie::store_v(ob + i, v);
    }
}

// ring2 store: int8 -> slot*jpw, scales -> 6144 + slot*grp*2.
void lv_st2(const uint8_t *__restrict s)
{
    const uint32_t slot = (lv_ctr[cP] - lv_ctr[cR2]) & lv_ctr[cMask];
    const uint32_t jpw = lv_ctr[cJpw];
    uint8_t *__restrict q = lv_sw + slot * jpw;
#pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < jpw; i += 32) {
        const auto v = aie::load_v<32>(s + i);
        aie::store_v(q + i, v);
    }
    // Scalar scale copies — same alignment reasoning as lv_fr2.
    const uint32_t grp = lv_ctr[cGrp];
    const uint16_t *__restrict sc =
        reinterpret_cast<const uint16_t *__restrict>(s + jpw);
    uint16_t *__restrict d2 =
        reinterpret_cast<uint16_t *__restrict>(lv_sw + 6144 + slot * grp * 2);
#pragma clang loop unroll(disable)
    for (uint32_t k = 0; k < grp; k++)
        d2[k] = sc[k];
    lv_ctr[cR2]++;
}

// ring3 make: xn1 = bf16(f32(x') + dacc), into the outgoing element AND
// the own lv_shared slot. x' itself is RECOMPUTED from the retained
// lv_xn/lv_o chunks (t = bf16(f32(xn) + f32(o)) — the identical rounding
// fr1 applied) because lv_shared's x' image is dead after K=101's rms:
// the gate/up windows alias that space (see the overlay note above).
void lv_fr3(uint8_t *__restrict ob)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    const aie::vector<bfloat16, 32> ones_bf =
        aie::broadcast<bfloat16, 32>((bfloat16)1.0f);
    const aie::vector<float, 32> ones_f = aie::broadcast<float, 32>(1.0f);
    const aie::accum<accfloat, 32> z = aie::zeros<accfloat, 32>();
    const uint32_t p = lv_ctr[cP];
    const uint32_t rows = lv_ctr[cRows];
    const bfloat16 *__restrict xn =
        reinterpret_cast<const bfloat16 *__restrict>(lv_xn);
    const bfloat16 *__restrict oo =
        reinterpret_cast<const bfloat16 *__restrict>(lv_o);
    bfloat16 *__restrict sh =
        reinterpret_cast<bfloat16 *__restrict>(lv_shared) + p * rows;
    const float *__restrict da = lv_dacc;
    bfloat16 *__restrict dst = reinterpret_cast<bfloat16 *__restrict>(ob);
#pragma clang loop unroll(disable)
    for (uint32_t g = 0; g < rows / 32; g++) {
        aie::accum<accfloat, 32> a =
            aie::mul(aie::load_v<32>(xn + g * 32), ones_bf);
        a = aie::mac(a, ones_f,
                     aie::mul(aie::load_v<32>(oo + g * 32), ones_bf)
                         .to_vector<float>());
        const aie::vector<float, 32> tf =
            aie::mul(a.to_vector<bfloat16>(), ones_bf).to_vector<float>();
        aie::accum<accfloat, 32> b = aie::mac(z, ones_f, tf);
        b = aie::mac(b, ones_f, aie::load_v<32>(da + g * 32));
        const aie::vector<bfloat16, 32> v = b.to_vector<bfloat16>();
        aie::store_v(dst + g * 32, v);
        aie::store_v(sh + g * 32, v);
    }
    lv_ctr[cR3] = 1;
}

// ring3 store: overwrite slot's x' with xn1.
void lv_st3(const uint8_t *__restrict s)
{
    const uint32_t slot = (lv_ctr[cP] - lv_ctr[cR3]) & lv_ctr[cMask];
    const uint32_t rows = lv_ctr[cRows];
    const uint16_t *__restrict src =
        reinterpret_cast<const uint16_t *__restrict>(s);
    uint16_t *__restrict dst =
        reinterpret_cast<uint16_t *__restrict>(lv_shared) + slot * rows;
#pragma clang loop unroll(disable)
    for (uint32_t r = 0; r < rows; r += 16)
        aie::store_v(dst + r, aie::load_v<16>(src + r));
    lv_ctr[cR3]++;
}

// drain one 16-bf16 piece of the OWN xn1 chunk as a C element.
void lv_cxn(uint8_t *__restrict c)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    const uint32_t i = lv_ctr[cXnI]++;
    const uint32_t p = lv_ctr[cP];
    const uint16_t *__restrict src =
        reinterpret_cast<const uint16_t *__restrict>(lv_shared) +
        p * lv_ctr[cRows] + i * 16;
    aie::store_v(reinterpret_cast<uint16_t *__restrict>(c),
                 aie::load_v<16>(src));
}

// P28-6 discriminator kernels (dumb-gather probe, design_lv2probe
// rings=11): the r13 gather relay with ALL computation removed -- same
// ObjectFifo acquire/release/forward structure as lv_fr1/lv_fw1/lv_st1,
// dumb bodies. Board-pass => lv_* kernel N=16 data path guilty; hang =>
// fill dispatch/ordering guilty (perf-lab 6f-2). 128 = rows at N=16.
void ring_touch1(uint16_t *__restrict dst)
{
    // read-back write-back: a pure slot touch, same store_v/load_v forms
    // as lv_cxn (no zeros<> overload games).
    ::aie::store_v(dst, ::aie::load_v<16>(dst));
}

void ring_copy128_bf16(const uint16_t *__restrict src,
                       uint16_t *__restrict dst)
{
    for (uint32_t i = 0; i < 128; i += 16) {
        ::aie::store_v(dst + i, aie::load_v<16>(src + i));
    }
}

// rings=15 probe: copy128 padded with a spin matching lv_fw1's
// rolled-loop latency -- same timing signature, none of its codegen
// (no lv_ctr read, no memory-trip ZOL). Hang => the N=16 relay deadlock
// is a TIMING RACE in the fifo/lock protocol; pass => lv_fw1's codegen.
static volatile uint32_t lv_probe_spin;

void ring_copy128_slow(const uint16_t *__restrict src,
                       uint16_t *__restrict dst)
{
    uint32_t acc = 0;
#pragma clang loop unroll(disable)
    for (uint32_t j = 0; j < 64; j++)
        acc ^= j;
    lv_probe_spin = acc;
    for (uint32_t i = 0; i < 128; i += 16) {
        ::aie::store_v(dst + i, aie::load_v<16>(src + i));
    }
}

// rings=16 probe: lv_fw1's body VERBATIM under a different symbol --
// separates body-content (hangs) from symbol/placement (passes).
void lv_fw1_clone(const uint8_t *__restrict s, uint8_t *__restrict ob)
{
    const uint16_t *__restrict src =
        reinterpret_cast<const uint16_t *__restrict>(s);
    uint16_t *__restrict dst =
        reinterpret_cast<uint16_t *__restrict>(ob);
    const uint32_t rows = lv_ctr[cRows];
#pragma clang loop unroll(disable)
    for (uint32_t r = 0; r < rows; r += 16)
        ::aie::store_v(dst + r, ::aie::load_v<16>(src + r));
}

// rings=17 probe: lv_fw1's body with a compile-time 128 bound (no
// lv_ctr read, no memory-trip ZOL) -- pass => the memory-loaded trip
// count is the culprit; N=16-only by construction.
void ring_fw_const(const uint8_t *__restrict s, uint8_t *__restrict ob)
{
    const uint16_t *__restrict src =
        reinterpret_cast<const uint16_t *__restrict>(s);
    uint16_t *__restrict dst =
        reinterpret_cast<uint16_t *__restrict>(ob);
#pragma clang loop unroll(disable)
    for (uint32_t r = 0; r < 128; r += 16)
        ::aie::store_v(dst + r, ::aie::load_v<16>(src + r));
}

} // extern "C"
