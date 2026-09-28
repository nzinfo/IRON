// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#define NOCPP

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>

// Fused INT4-dequant GEMV, UNIVERSAL — the M3b engine kernel, layout v5
// (v4 = P11 matrix-unit mmul; P12 removed the B stream: the activation
// rides the A fifo as a K=0 element). Same signed-int4 weight ABI as
// v2/v3 (per-group-32 symmetric, scale = amax/7 bf16). ONE PDI serves
// every projection shape: blocks are UNIFORM (16 rows x 2048 k, chunk id
// in the header), so the device side is shape-free and only the ctrl
// code differs.
//
// Why v4 (P11): the v3 A/B proved the fp inner loop is ISSUE-bound
// (~10 vector ops per 32 MACs, 7.3us/block shape-independent) while DDR
// sat at 15/45 GB/s. The aie2p matrix unit does mac_4x16_16x16 =
// 1024 MACs per instruction — an on-chip probe measured ~100 GMAC/s per
// core, 32x the fp path. v4 formulates the tile as mmul<4,16,16,int8,int4>:
//   y[n] = sum_g sf[n][g] * d[g] * ( int32 dot of W[n][g,*] x[g,*] )
//
// Block layout v4 (one fifo element = 18560 bytes = ONE 16-row x 2048 tile):
//   [0 .. 16384)   nibbles, GROUP-major: group g holds 256 nibbles laid out
//                  k-major [k][n] (element g*256 + k*16 + n; n = tile row) —
//                  the B-operand order of mac_4x16_16x16. 16-byte aligned
//                  (load_v streams cannot start misaligned — P10 lesson).
//   [16384 .. 18432)  sf_t: bf16[64][16], row n's group-g scale at
//                  g*16 + n (TRANSPOSED vs v3 so one 32B load feeds the
//                  per-group vector scale step).
//   [18552 .. 18556)  K as u32 (2048 = compute, 0 = activation element,
//                    anything else = stale/garbage -> zero rows)
//   [18556 .. 18560)  chunk as u32 (v5: which 2048-wide x slice this block
//                    consumes — self-describing, block order no longer
//                    constrained; 0 for K=2048 ops)
//
// v5 activation element (SAME 18560-B fifo element, K header = 0):
//   [0 .. 6144)    x int8, ALL chunks (chunk c at c*2048)
//   [6144 .. 6528) d bf16[192], ALL chunks (chunk c's 64 at +c*128)
//   [6528 .. 18552) padding; [18552..18556) K = 0
// One activation element precedes each op's weight blocks ON THE SAME
// FIFO (fill order): fifo order IS the per-op barrier — a parked core
// cannot reach an op's blocks before consuming that op's activation
// (the v4 in-loop B acquire served exactly this role, P12 removed the
// B stream). It emits one zero C element the host skips. x/d are staged
// to a core-local buffer so the weight blocks read x without any B-slot
// fifo mechanics in the hot loop.
// The kernel builds each 16-byte x chunk's [4x16] A operand (M=4 is the
// only aie2p int8xint4 shape; the 4 A rows are redundant, C row 0 carries
// the 16 real dots) ONCE per op, at activation staging: the replicated
// operands are identical for every block of the op, and rebuilding them
// per group (2 x-loads + 6 concats) made the v5 hot loop COMPUTE-bound
// (~3.8us/block vs the 2.77us fill — P13: B removal exposed it; the v5s
// zero-rows probe matched the pure-stream rate exactly). The hot loop is
// then per group: 2 A loads (prebuilt), 2 B loads (128B), 2 matrix macs,
// one int32->f32 row-0 extract scaled by d[g] (broadcast) and sf_t[g][*]
// (vector) into an f32 accumulator — the f32 domain is REQUIRED: group
// partials reach 2^15 and bf16 would round them to death before the sum.
//
// STACK BUDGET — the peano linker reserves only 0x400 bytes of stack and
// the placer puts the neighbor fifo buffer directly above it (notes §12).
// This loop keeps no arrays and matches w4gemv2's frame shape; re-check
// `paddxm [sp], #imm` in the ELF after any structural change.

constexpr uint32_t kBlockBytes = 18560; // ELEM v4: one 16-row x 2048 tile.
    // MUST be 64-byte aligned end to end: depth-2 fifo element buffers
    // sit at base and base+ELEM, and on aie2p (arch 21) a 1024-bit
    // load_v stream — our load_v<256> int4 B operand — needs 64B
    // alignment (ld_st.hpp: >256-bit vectors align 64). ELEM % 64 != 0
    // made every odd element buffer compute from garbage nibbles (P11
    // fingerprint: even 16-row tiles exact, odd tiles wrong; at %32 !=
    // 0 the sf loads went junk too and values hit 1e18).
constexpr uint32_t kKMax = 6144;
constexpr uint32_t kTileRows = 16;
constexpr uint32_t kTileK = 2048;
constexpr uint32_t kGroups = kTileK / 32; // 64 groups per chunk

// v5.1 activation staging (25088 B): the PREBUILT A operands + d + the
// P19 swiglu per-group temp.
//   [c*8192 + g*128 .. +64)   A0 = [x0 x0 x0 x0] (int8, 64B, 64B-aligned)
//   [c*8192 + g*128 + 64)     A1 = [x1 x1 x1 x1]
//   [24576 + c*128 ..)        d bf16[64] (chunk c)
//   [24960 .. 25088)          P19 K=5 per-group temp (f32 x32)
// Group stride 128 B keeps every A operand 64-B aligned (aie2p >256-bit
// load_v rule). L1 budget: 2x18560 (A fifo) + 25088 + 64 (C fifo) +
// stack ~= 63.4 KB of the 64-KB tile (P12's 74-KB two-fifo probe is the
// known-over watermark). Filled by the K=0 activation element; weight
// blocks read it by chunk id. .bss (uninitialized) keeps it out of .data;
// the peano/placer core-private accounting covers it like the stack
// (verify in the .map after edits).
static uint8_t x_stage[3 * kGroups * 128 + 192 * 2 + 128] __attribute__((aligned(64)));
constexpr uint32_t kAStageBytes = 3 * kGroups * 128; // 24576
constexpr uint32_t kDStageOff = kAStageBytes;        // d follows A ops
constexpr uint32_t kSwTempOff = kDStageOff + 192 * 2; // 24960: K=5 group temp
// P19 quad extras: the phase flag (K=5 sets, the re-read K=1 consumes) and
// the re-read's o-partial staging offset.
constexpr uint32_t kQuadFlagOff = kSwTempOff + 124;
constexpr uint32_t kOPartOff = 4096;

// M6 fused-rms flavors (P16). The layer's ops are fused ONE exec per
// pair: op1 (gemv) -> [await] -> rms window -> w/glue element -> op2
// (gemv, whose A-operands the kernel prebuilds DIRECTLY — op2 has no X
// fill at all). BOTH task groups carry exactly the proven v5 BD shape
// (4 fills + 2 drains per shim): the ctrl generator's S2MM drain
// queue-value is pinned to the channel's slots 4/5, so a group with
// more than 4 preceding fills on a shim desyncs value from descriptor
// address and hangs the runlist (first fused attempt, P16).
//   K=1 element (tg2's first fill): the C-window [res | headers] —
//     rows [8 sections | residual M1 bf16 | pad | K=1 u32 @ELEM-8,
//     blocks1 u32 @ELEM-4]. Sections were just drained there (tg1's
//     drains are awaited before tg2 issues — DPU program order is the
//     cross-op barrier); the residual is host-written into the C BO's
//     free rows each token. Stages COMPACTED chunk-major partials +
//     the residual into the dead chunk-1/2 A-op region.
//   K=3 element (head of op2's weight fill — fifo order guarantees it
//     runs before every op2 block): [rms weight M1 bf16 | pad | K=3
//     @ELEM-8, blocks1 @ELEM-4]. Runs the glue with the HOST's exact
//     semantics (f32 math, RNE bf16 steps, ties-even int8, d =
//     bf16(amax/127)), then prebuilds chunk-0 A operands + d — the
//     SAME contract the K=0 path leaves.
// P17: the glue is VECTORIZED — the scalar soft-float version measured
// 1176 us of a 1673-us exec (every f32 scalar op on peano is a soft
// call; ~15/row * 2048 rows/core), while the machinery alone (probe
// with the glue flavors disabled) runs 497 us. The vectorized glue then
// broke the 0x400 STACK LAW again (monolith = 0x480 frame, overflow
// trashed the A-fifo buffer staged above sp -> op2 first-blocks/late-
// columns corruption). P17b bisected peano frame costs op by op:
// aie::to_fixed<int32> ~0x600 (FORBIDDEN), abs/reduce_max +0x140,
// in-function scalar reads of vector stores +0x140, int min/max +0x80,
// vector-only functions 0x0 — so the glue is a CHAIN of small noinline
// stages (2a..2f + 0x40 sequencer) of only proven-lowering forms. amax
// = integer max of (bits & 0x7fffffff), EXACT f32 max|x|; int8
// conversion = scale by invd then ADD magic 1.5*2^23 (ulp=1 there ->
// conv_even addition rounds RNE to the parked integer), q = bits -
// 0x4B400000 + scalar clip (round-then-clip, the host order). Two
// compile hazards found on the way: a u32-loop MIXED with soft-float
// calls (stage2c v22) made clang chew >10 min, and broadcasting a
// runtime-loaded scalar in a load_v/mac loop likewise — int and float
// live in separate stages, per-group invd is staged REPLICATED so the
// scale pass is pure load_v/mac/store_v.
// Scratch map (x_stage, 24960 B): during tg1's K=1: compacted partials
// [kSecStageOff .. +chunks1*4096) + residual at kResStageOff (both
// dead after stage2a); during K=3: h2 bf16 [0..4096), sumsq f32 word
// at 4224, amax-bits u32[64] at 4352, xn f32 [8192..16384), replicated
// invd f32[64][32] [16384..24576) (over the consumed residual), q int8
// [6144..8192), prebuilt A-ops overwrite [0..8192) last; d at
// kDStageOff.
// M1 is the model hidden width 2048 (both fused pairs: o->gateup and
// down->qkv).
constexpr uint32_t kFusedM1 = 2048;
constexpr uint32_t kFusedRowsPerCol = kFusedM1 / 8; // 256
constexpr uint32_t kSecStageOff = 8192; // compacted partials (chunk-major)
constexpr uint32_t kResStageOff = kAStageBytes - 2 * kFusedM1; // 20480

// P19 per-group helpers, shared by the K=5 swiglu chain AND the stage2c/
// stage2f glue passes (identical 32-element jobs) — see the P19 block
// below for the rationale (16-KB program memory).
static uint32_t __attribute__((noinline)) fused_sw_amax(const float *__restrict temp);
static void __attribute__((noinline)) fused_sw_x(const float *__restrict temp,
                                                 uint8_t *__restrict dst);
static void __attribute__((noinline)) fused_sw_quant(uint32_t m_bits,
                                                     float *__restrict temp,
                                                     uint16_t *d_out);

static inline float fused_rsqrt(float s)
{
    // 3-iteration magic-constant Newton (NOCPP: no libm sqrtf). ~1e-7
    // relative — far under the bf16 half-ULP the flavor rounds to.
    union { float f; uint32_t u; } v = { s };
    v.u = 0x5F3759DFu - (v.u >> 1);
    for (int i = 0; i < 3; i++)
        v.f = v.f * (2.0f - s * v.f * v.f);
    return v.f;
}

static inline float fused_bf16_to_f32(uint16_t b)
{
    union { float f; uint32_t u; } v = { 0.0f };
    v.u = (uint32_t)b << 16;
    return v.f;
}

static inline uint16_t fused_f32_to_bf16(float f)
{
    // RNE via the classic carry trick (P1b): add 0x7FFF + LSB before
    // the shift. NaN/Inf not expected in this path.
    union { float f; uint32_t u; } v = { f };
    uint32_t rounded = v.u + 0x7FFFu + ((v.u >> 16) & 1u);
    return (uint16_t)(rounded >> 16);
}



// STACK LAW (P16, measured): the peano linker gives each core a 0x400-B
// stack window ([0x70000,0x70400) in the ld script) with the A fifo
// buffer placed DIRECTLY at its ceiling (A_L3L1_0_cons_buff_0 = 0x70400).
// The frame is allocated by the ONE prologue `paddxm [sp], #N` for the
// WHOLE call chain — N > 0x400 spills into the weight-stream fifo and
// hangs the runlist at the FIRST exec (the monolithic M6 kernel compiled
// to 0x440 and broke every design, fused or not; 0x340 is the proven
// hot shape). So: ONE tiny dispatcher reads the K header and tail-calls
// noinline callees — the hot path keeps its 0x340 frame, and each glue
// flavor gets its own frame within the same 0x400 budget (dispatcher ~
// 0x20 + callee). This is also what broke the v5.3 helper-function
// variant: helpers CALLED FROM the big-frame function stack on top of
// 0x340 — same law, inverted.

// the glue flavors' common tail: emit the (contractual) zero C element.
// Shared noinline — each flavor carrying its own unrolled 16-store copy
// was ~100 B x 5 of program memory (P19, third overflow).
static void __attribute__((noinline)) fused_zero_c(bfloat16 *__restrict c_out)
{
    for (uint32_t r = 0; r < kTileRows; r++)
        c_out[r] = static_cast<bfloat16>(0);
}

// M6 fused-rms stage 1 (the K=1 C-window element): compact-stage op1's
// partials and the residual into the dead chunk-1/2 A-op region of
// x_stage. See the M6 comment block above for the window layout. P17:
// the gather is row-contiguous in 16-wide runs (r = col*256 + t*16 +
// jj maps 16 consecutive rows to 16 consecutive u16) — one vector copy
// per (col, t, chunk). P19: the gather is ONE shared noinline helper
// (skip = leading dummy groups: 1 for the o/plain windows, 2 for win2)
// — program memory is 16 KB and three stage1 flavors each carrying
// their own copy overflowed it (first board run, _XAie_LoadProgMemSection).
static void __attribute__((noinline)) fused_gather1(const uint8_t *__restrict a_in,
                                                    uint16_t *__restrict part,
                                                    uint32_t skip)
{
    const uint32_t blocks1 =
        *(const uint32_t *__restrict)(a_in + kBlockBytes - 4);
    const uint32_t chunks1 = blocks1 / 16; // T = 16 at M1 = 2048
    const uint32_t stride_rows = (blocks1 + 2) * 16;
    const uint16_t *__restrict win =
        reinterpret_cast<const uint16_t *__restrict>(a_in);
    for (uint32_t col = 0; col < 8; col++)
        for (uint32_t t = 0; t < 16; t++)
            for (uint32_t c = 0; c < chunks1; c++)
                aie::store_v(part + c * kFusedM1 + col * 256 + t * 16,
                             aie::load_v<16>(win + col * stride_rows + skip +
                                             (c * 16 + t) * 16));
}

// stage1's residual copy (window rows right after the 8 sections)
static void __attribute__((noinline)) fused_copy_res1(const uint8_t *__restrict a_in)
{
    const uint32_t blocks1 =
        *(const uint32_t *__restrict)(a_in + kBlockBytes - 4);
    const uint32_t stride_rows = (blocks1 + 2) * 16;
    const uint16_t *__restrict win =
        reinterpret_cast<const uint16_t *__restrict>(a_in);
    uint16_t *__restrict resd = reinterpret_cast<uint16_t *__restrict>(
        x_stage + kResStageOff);
    const uint32_t c_rows1 = 8 * stride_rows;
    for (uint32_t r = 0; r < kFusedM1; r += 16)
        aie::store_v(resd + r, aie::load_v<16>(win + c_rows1 + r));
}

static void __attribute__((noinline)) fused_stage1s(const uint8_t *__restrict a_in,
                                                    bfloat16 *__restrict c_out,
                                                    uint32_t skip)
{
    fused_gather1(a_in, reinterpret_cast<uint16_t *__restrict>(
                            x_stage + kSecStageOff),
                  skip);
    fused_copy_res1(a_in);
    fused_zero_c(c_out);
}

// K=1 (win1, o window): ONE dummy group ahead of the sections (the X
// element's zero C).
static void __attribute__((noinline)) fused_stage1(const uint8_t *__restrict a_in,
                                                   bfloat16 *__restrict c_out)
{
    fused_stage1s(a_in, c_out, 16);
}

// P19 quad K=2 (win2, the down->qkv window): TWO dummy groups ahead of the
// sections (the K=4 and K=5 glue elements each emit a zero C), so the
// partial gather and the residual base move 16 rows down. Dispatched on the
// window's own header word (the win2 pad rows are host-writable, unlike the
// re-read window below). PMEM: same body as stage1 through the shared
// stage1s — a second full copy overflowed program memory (P19).
static void __attribute__((noinline)) fused_stage1b(const uint8_t *__restrict a_in,
                                                    bfloat16 *__restrict c_out)
{
    fused_stage1s(a_in, c_out, 32);
}


// P19 shared h2 pass: h2 = bf16(f32(res) + sum_c f32(part_c)) over M1
// rows, group-vectorized. stage1r (h2' = x + o_out from the win1 re-read)
// and stage2a (h2 = res + partials) ran DUPLICATE copies of this loop —
// program memory is 16 KB and the linked ELF measured 16800 (third board
// overflow), so the body is ONE helper now. All lowering forms are the
// proven stage2a set (function-scope broadcasts, load_v/mul/mac/store_v).
static void __attribute__((noinline)) fused_h2make(const bfloat16 *__restrict res,
                                                   const bfloat16 *__restrict part,
                                                   uint32_t chunks,
                                                   bfloat16 *__restrict h2_out)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    const aie::vector<bfloat16, 32> ones_bf =
        aie::broadcast<bfloat16, 32>((bfloat16)1.0f);
    const aie::vector<float, 32> ones_f = aie::broadcast<float, 32>(1.0f);
    for (uint32_t g = 0; g < kFusedM1 / 32; g++) {
        aie::accum<accfloat, 32> a =
            aie::mul(aie::load_v<32>(res + g * 32), ones_bf);
        for (uint32_t c = 0; c < chunks; c++)
            a = aie::mac(a, ones_f,
                         aie::mul(aie::load_v<32>(part + c * kFusedM1 + g * 32),
                                  ones_bf).to_vector<float>());
        aie::store_v(h2_out + g * 32, a.to_vector<bfloat16>());
    }
}

// P19 quad K=1' (win1 RE-READ, dispatched by the quad flag): residual2 =
// x'_n = x_n + o_out is NOT host-computable (o_out is produced inside this
// exec), so tg4 re-reads the win1 window (o sections + residual1 are still
// live in C after tg1's drains) and stages h2' = bf16(x_n + o_out) at
// kResStageOff — exactly where the K=3 glue reads its residual, overwriting
// the res2 copy stage1b staged one element earlier. The f32 sum of two bf16
// values rounded once to bf16 is bit-identical to the host's add_bf16
// (chunks1 == 1 here, no accumulation-order question). Flag dispatch is
// forced because the re-read window IS the tg2 window — same bytes, so its
// header still says K=1. The only setter is K=5 (quad-only) and this is the
// only consumer (it resets), so pair/plain flows always see 0.
static void __attribute__((noinline)) fused_stage1r(const uint8_t *__restrict a_in,
                                                    bfloat16 *__restrict c_out)
{
    const uint32_t blocks1 =
        *(const uint32_t *__restrict)(a_in + kBlockBytes - 4);
    const uint32_t chunks1 = blocks1 / 16;
    const uint32_t stride_rows = (blocks1 + 2) * 16;
    const uint16_t *__restrict win =
        reinterpret_cast<const uint16_t *__restrict>(a_in);
    fused_gather1(a_in, reinterpret_cast<uint16_t *__restrict>(
                            x_stage + kOPartOff),
                  16);
    fused_h2make(reinterpret_cast<const bfloat16 *__restrict>(
                     win + 8 * stride_rows),
                 reinterpret_cast<const bfloat16 *__restrict>(x_stage + kOPartOff),
                 chunks1,
                 reinterpret_cast<bfloat16 *__restrict>(x_stage + kResStageOff));
    *(uint32_t *__restrict)(x_stage + kQuadFlagOff) = 0;
    fused_zero_c(c_out);
}
// monolithic vector glue compiled a 0x480 frame and broke the 0x400
// stack window (STACK LAW) — the overflow trashed the A-fifo buffer
// staged above sp and corrupted op2's first blocks (and whole late
// columns). Pass 1 (h2 + sumsq) and pass 2 (rms + quantize + prebuild)
// run as sequential noinline halves so neither holds the other's
// vector liveness; sumsq crosses over through scratch.
static void __attribute__((noinline)) fused_stage2a(const uint8_t *__restrict a_in)
{
    const uint32_t blocks1 =
        *(const uint32_t *__restrict)(a_in + kBlockBytes - 4);
    const uint32_t chunks1 = blocks1 / 16;
    const bfloat16 *__restrict res = reinterpret_cast<const bfloat16 *__restrict>(
        x_stage + kResStageOff);
    const bfloat16 *__restrict part =
        reinterpret_cast<const bfloat16 *__restrict>(x_stage + kSecStageOff);
    bfloat16 *__restrict h2 = reinterpret_cast<bfloat16 *__restrict>(x_stage);
    float *__restrict scr_f = reinterpret_cast<float *__restrict>(x_stage + 4096);

    // pass 1: h2 (+ bf16 round, shared helper), then sumsq over the
    // ROUNDED bits — re-reading what h2make just stored is bit-identical
    // to the old fused loop's hb (same store, same values).
    fused_h2make(res, part, chunks1, h2);
    const aie::vector<bfloat16, 32> ones_bf =
        aie::broadcast<bfloat16, 32>((bfloat16)1.0f);
    aie::accum<accfloat, 32> sq = aie::zeros<accfloat, 32>();
    for (uint32_t g = 0; g < kFusedM1 / 32; g++) {
        const aie::vector<bfloat16, 32> hb = aie::load_v<32>(h2 + g * 32);
        const aie::vector<float, 32> hf = aie::mul(hb, ones_bf).to_vector<float>();
        sq = aie::mac(sq, hf, hf);
    }
    aie::store_v(scr_f, sq.to_vector<float>());
    float sumsq = 0.0f;
#pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < 32; i++)
        sumsq += scr_f[i];
    *(float *__restrict)(x_stage + 4096 + 128) = sumsq;
}

// P17b frame laws, discovered by ablation (v1-v22 offline compiles):
// to_fixed -> 0x600, abs/reduce_max -> +0x140, in-function scalar reads of
// vector stores -> +0x140, min/max int -> +0x80 — and the monolithic glue
// was 0x480, overflowing the 0x400 STACK LAW window into the A-fifo buffer
// staged above sp (corrupted op2's first blocks + late columns). The glue
// is therefore a CHAIN of small noinline stages, each with its own tiny
// frame, using only proven-lowering forms. amax is taken as the INTEGER
// max of (bits & 0x7fffffff) — for finite floats bit order = value order,
// so this is EXACTLY max|x| with zero float ops. Int->float conversion
// and the divides live in their own stage (mixing the u32 loop with the
// soft-float calls made clang chew >10 min; separated it is instant).
// invd is staged REPLICATED (f32 x32 per group) so the scale pass uses
// only load_v/mac/store_v — broadcast of a runtime-loaded scalar was the
// other compile hazard. Rounding to int8: scale by invd then ADD the
// magic 1.5*2^23 — at that exponent ulp = 1, so conv_even addition
// rounds to the nearest-even INTEGER parked in the mantissa; q = bits -
// 0x4B400000, then a scalar clip (round-then-clip, the host's order).
// Zero groups: invd = 0 so v = 0*0 = 0 -> q = 0, d = 0.
// Scratch (dead op1 regions after stage2a): h2 bf16 [0..4096), sumsq f32
// at 4224, m u32[64] at 4352, xn f32 [8192..16384) (scaled+magic in
// place by the per-group quant calls — P19 removed the replicated-invd
// staging at [16384..24576)).
static void __attribute__((noinline)) fused_stage2b(const uint8_t *__restrict a_in)
{
    const uint32_t groups = kFusedM1 / 32;
    const bfloat16 *__restrict wgt =
        reinterpret_cast<const bfloat16 *__restrict>(a_in);
    const bfloat16 *__restrict h2 =
        reinterpret_cast<const bfloat16 *__restrict>(x_stage);
    float *__restrict xn_out =
        reinterpret_cast<float *__restrict>(x_stage + kSecStageOff);

    const float inv = fused_rsqrt(
        *(const float *__restrict)(x_stage + 4096 + 128) / (float)kFusedM1 +
        1e-5f);

    const aie::vector<bfloat16, 32> ones_bf =
        aie::broadcast<bfloat16, 32>((bfloat16)1.0f);
    const aie::accum<accfloat, 32> z = aie::zeros<accfloat, 32>();
    for (uint32_t g = 0; g < groups; g++) {
        const aie::vector<float, 32> hf =
            aie::mul(aie::load_v<32>(h2 + g * 32), ones_bf).to_vector<float>();
        const aie::vector<float, 32> wf =
            aie::mul(aie::load_v<32>(wgt + g * 32), ones_bf).to_vector<float>();
        const aie::vector<float, 32> t =
            aie::mac(z, hf, aie::broadcast<float, 32>(inv)).to_vector<float>();
        const aie::vector<float, 32> xn =
            aie::mac(z, t, wf).to_vector<float>();
        aie::store_v(xn_out + g * 32, xn);
    }
}

// integer amax: max(bits & 0x7fffffff) per group — pure int, no float
// (P19: delegates to the shared per-group helper)
static void __attribute__((noinline)) fused_stage2c()
{
    const float *__restrict xn = reinterpret_cast<const float *__restrict>(
        x_stage + kSecStageOff);
    uint32_t *__restrict m_out =
        reinterpret_cast<uint32_t *__restrict>(x_stage + 4352);
    for (uint32_t g = 0; g < kFusedM1 / 32; g++)
        m_out[g] = fused_sw_amax(xn + 32 * g);
}

// d + scale + magic, one pass (P19: stage2d+stage2e collapsed into a
// driver over the shared per-group fused_sw_quant — the replicated-invd
// array the old 2d staged existed only to keep 2e's loop free of a
// runtime broadcast, and the per-group function form is the proven-safe
// alternative (broadcast at function scope, no loop — v24e). Also saves
// program memory.)
static void __attribute__((noinline)) fused_stage2d()
{
    const uint32_t *__restrict m_in =
        reinterpret_cast<const uint32_t *__restrict>(x_stage + 4352);
    uint16_t *__restrict d_out =
        reinterpret_cast<uint16_t *__restrict>(x_stage + kDStageOff);
    float *__restrict xnv =
        reinterpret_cast<float *__restrict>(x_stage + kSecStageOff);
    for (uint32_t g = 0; g < kFusedM1 / 32; g++)
        fused_sw_quant(m_in[g], xnv + 32 * g, d_out + g);
}

// integer tail: q extraction (+-127 clip) + the K=0-contract prebuild
// (P19: per-group extract+store delegated to the shared fused_sw_x —
// stage2e left the magic-added floats at kSecStageOff, the same contract
// sw_quant leaves in the K=5 temp)
static void __attribute__((noinline)) fused_stage2f(bfloat16 *__restrict c_out)
{
    for (uint32_t g = 0; g < kGroups; g++)
        fused_sw_x(reinterpret_cast<const float *__restrict>(
                       x_stage + kSecStageOff + 128 * g),
                   x_stage + 128 * g);
    fused_zero_c(c_out);
}

static void __attribute__((noinline)) fused_stage2(const uint8_t *__restrict a_in,
                                                   bfloat16 *__restrict c_out)
{
    fused_stage2a(a_in);
    fused_stage2b(a_in);
    fused_stage2c();
    fused_stage2d(); // P19: also does the scale+magic pass (old stage2e)
    fused_stage2f(c_out);
}

// P19 quad flavors (K=4 / K=5): the WHOLE layer rides one exec
// [o -> rms1 -> gateup -> swiglu -> down -> rms2 -> qkv']. The swiglu
// inputs are the gateup op's own C sections, re-read through the A fifo
// as two C-sourced window fills: K=4 = [gate cols 0..3 | padA] (padA is
// a host-owned gap so the K-header words at ELEM-8/ELEM-4 land on dead
// rows), K=5 = [up cols 4..7 | padB]. K=4 stages the gate half to
// x_stage[0..12288) bf16; K=5 consumes it with the up half from its own
// element and prebuilds ALL THREE chunks of down's A operands + d —
// down's fill carries NO X element at all (same contract the K=3 path
// leaves for its op2, extended to K=6144).
//
// Kernel-side group arithmetic (hy gateup M2 = 12288, 8 cols, 1536 rows
// per column section of which 16 dummy): gate[j] lives in C column j/1536
// at section row j%1536; the K=4 element's window starts at column 0's
// section, so gate[j] = win_u16[(j/1536)*1568 + 16 + j%1536]; the K=5
// window starts at column 4's section, so up[j] = win_u16[(j/1536)*1568 +
// 16 + j%1536] with j/1536 in 0..4.
//
// NUMERICS (phase 1, decided in notes/perf-lab.md P19): sigmoid runs on
// the AIE2P hardware elementary — aie::exp2<bfloat16>(vector<float>) —
// which P4 measured as value-biased (mean +3.25%, max +5.67%, exact at
// integer args, periodic in the fraction). The bias is mostly common-mode
// within a group and absorbed by d = amax/127; the E2E gates decide. The
// division-free tanh form (sig = 0.5 + 0.5*tanh(g/2), same hardware
// family) is the drop-in fallback if the gates fail. sw is rounded to
// bf16 before quantize, mirroring the host's swiglu_bf16 -> quantize
// order.
//
// Frame budget (v24e offline probe): stage3b 0x40 / sig 0x0 / amax 0x0 /
// quant 0x280 / x 0x40 -> chain 0x2e0 < 0x400 STACK LAW. The per-group
// split is MANDATORY, not stylistic: the monolithic vector+scalar-tail
// form hangs clang (>5 min, the P17b u32-loop/soft-float mixing law) and
// the 64-iter sigmoid LOOP form compiles a 0x400 frame (register
// pressure). Groups run DESCENDING (gi = 191..0): the A-op write
// [128*gi, +128) is then always above every unread gate byte [64*gi',
// +64) for gi' < gi (128*gi >= 64*gi + 64), so the staging area is
// consumed exactly as it is overwritten — ascending or chunk-phased
// orderings clobber unread gate (derived repeatedly; do not "simplify").

// K=4: stage the gate half (bf16) to x_stage[0..12288)
static void __attribute__((noinline)) fused_stage3a(const uint8_t *__restrict a_in,
                                                    bfloat16 *__restrict c_out)
{
    const uint16_t *__restrict win =
        reinterpret_cast<const uint16_t *__restrict>(a_in);
    uint16_t *__restrict gate =
        reinterpret_cast<uint16_t *__restrict>(x_stage);
    const uint32_t stride_rows = 1568; // (96 + 2) * 16
    // +32: gate sections carry TWO dummy groups (the K=1 and K=3 glue
    // elements' zero Cs) before the 96 real gateup rows.
    for (uint32_t col = 0; col < 4; col++)
        for (uint32_t t = 0; t < 96; t++)
            aie::store_v(gate + col * 1536 + t * 16,
                         aie::load_v<16>(win + col * stride_rows + 32 + t * 16));
    // C-ELEMENT CONTRACT: every A element emits exactly one 16-row C. The
    // first quad board run OMITTED this zero (the only flavor that did) —
    // tg3+tg3b then produced 49 Cs against the C3 drain's 50 groups, and
    // the starved S2MM served its first group from a STALE L1 fifo buffer
    // (the last gateup compute — the observed row-10816 duplicate) before
    // re-aligning: down dummy0 garbage in all 8 columns, deterministic,
    // unmodified by task-group restructuring (an ELEMENT-count bug, not a
    // BD/slot bug).
    fused_zero_c(c_out);
}

// one group's swiglu: sw = g * sigmoid(g) * u (bf16-rounded), staged f32
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

// integer amax of the staged sw — pure int (bits & 0x7fffffff). PMEM: the
// 32-iter max loop must stay ROLLED (clang unrolls it to ~0x1d0 of compare
// chains — program memory is 16 KB, second board overflow, P19).
static uint32_t __attribute__((noinline)) fused_sw_amax(const float *__restrict temp)
{
    const uint32_t *__restrict tb =
        reinterpret_cast<const uint32_t *__restrict>(temp);
    uint32_t m = 0;
#pragma clang loop unroll(disable)
    for (uint32_t j = 0; j < 32; j++) {
        const uint32_t a = tb[j] & 0x7fffffffu;
        if (a > m)
            m = a;
    }
    return m;
}

// d + invd + the RNE magic add, one group (no loops: soft-float + u32
// loops in one function is the P17b compile hang)
static void __attribute__((noinline)) fused_sw_quant(uint32_t m_bits,
                                                     float *__restrict temp,
                                                     uint16_t *d_out)
{
    float invd = 0.0f;
    uint16_t db = 0;
    if (m_bits != 0) {
        union { uint32_t u; float f; } amax;
        amax.u = m_bits;
        db = fused_f32_to_bf16(amax.f / 127.0f);
        invd = 1.0f / fused_bf16_to_f32(db);
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

// q extraction (round-then-clip) + the replicated A-operand store. PMEM:
// the flat 8-store body unrolls x16 to ~0x410 (worst text offender in the
// second PMEM overflow, P19) — the replication is a strided inner loop
// instead (k = 0..3, stride 16; q0 low half, q1 high half), same 128
// scalar stores executed ROLLED.
static void __attribute__((noinline)) fused_sw_x(const float *__restrict temp,
                                                 uint8_t *__restrict dst)
{
    const uint32_t *__restrict tb =
        reinterpret_cast<const uint32_t *__restrict>(temp);
#pragma clang loop unroll(disable)
    for (uint32_t j = 0; j < 16; j++) {
        int32_t q0 = (int32_t)tb[j] - 0x4B400000;
        int32_t q1 = (int32_t)tb[j + 16] - 0x4B400000;
        if (q0 > 127) q0 = 127;
        if (q0 < -127) q0 = -127;
        if (q1 > 127) q1 = 127;
        if (q1 < -127) q1 = -127;
        const uint8_t r0 = (uint8_t)q0;
        const uint8_t r1 = (uint8_t)q1;
        for (uint32_t k = 0; k < 4; k++) {
            dst[j + 16 * k] = r0;
            dst[64 + j + 16 * k] = r1;
        }
    }
}

// K=5: the up half + swiglu + quantize + prebuild down's 3 chunks
static void __attribute__((noinline)) fused_stage3b(const uint8_t *__restrict a_in,
                                                    bfloat16 *__restrict c_out)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    const bfloat16 *__restrict gate =
        reinterpret_cast<const bfloat16 *__restrict>(x_stage);
    const uint16_t *__restrict upw =
        reinterpret_cast<const uint16_t *__restrict>(a_in);
    float *__restrict temp =
        reinterpret_cast<float *__restrict>(x_stage + kSwTempOff);
    uint16_t *__restrict dv =
        reinterpret_cast<uint16_t *__restrict>(x_stage + kDStageOff);
    for (uint32_t gi = 6144 / 32; gi-- > 0;) {
        const uint32_t j0 = 32 * gi;
        // +32: up sections carry the same two dummy groups as gate.
        fused_sw_sig(gate + j0,
                     reinterpret_cast<const bfloat16 *__restrict>(
                         upw + (j0 / 1536) * 1568 + 32 + (j0 % 1536)),
                     temp);
        fused_sw_quant(fused_sw_amax(temp), temp, dv + gi);
        fused_sw_x(temp, x_stage + 128 * gi);
    }
    // mark the quad phase so tg4's K=1 re-read dispatches to stage1r
    *(uint32_t *__restrict)(x_stage + kQuadFlagOff) = 1;
    fused_zero_c(c_out);
}


// One group's compute (v5.2 op set), spelled out inline in the hot loop:
// calls with vector/accum references do NOT inline reliably on this
// toolchain — the v5.3 helper-function variant hung the core (stack/ABI
// break — now understood, see the STACK LAW above) and the
// lambda+sfb4-extract variant broke goldens. Keep the loop FLAT with
// every index = address arithmetic on g.

static void __attribute__((noinline)) w4gemvu_compute(uint32_t m,
                    const uint8_t *__restrict a_in,
                    bfloat16 *__restrict c_out,
                    uint32_t group_size,
                    uint32_t tile_idx)
{
    (void)m;      // always kTileRows (ctrl code carries it)
    (void)tile_idx; // one call per element (v3 ABI slot)

    ::aie::set_rounding(aie::rounding_mode::conv_even);

    // Self-describing guard: K=2048 computes, K=0 stages the activation
    // (prebuilding the replicated A operands — see x_stage), anything
    // else (stale/garbage) degrades to zero rows instead of reading
    // wild offsets.
    const uint32_t k = *(const uint32_t *__restrict)(a_in + kBlockBytes - 8);
    if (k == 0) {
        const int8_t *__restrict x8 = reinterpret_cast<const int8_t *__restrict>(a_in);
        const bfloat16 *__restrict d_in =
            reinterpret_cast<const bfloat16 *__restrict>(a_in + kKMax);
        for (uint32_t c = 0; c < kKMax / kTileK; c++) { // 3 chunks
            for (uint32_t g = 0; g < kGroups; g++) {
                const int8_t *xg = x8 + c * kTileK + g * group_size;
                aie::vector<int8, 16> x0 = aie::load_v<16>(xg);
                aie::vector<int8, 16> x1 = aie::load_v<16>(xg + 16);
                // concat needs equal-width halves: (x0+x0)|(x0+x0) -> [4x16].
                aie::vector<int8, 32> x0l = aie::concat(x0, x0);
                aie::vector<int8, 32> x1l = aie::concat(x1, x1);
                int8_t *dst = reinterpret_cast<int8_t *__restrict>(
                    x_stage + c * (kGroups * 128) + g * 128);
                aie::store_v(dst, aie::concat(x0l, x0l));
                aie::store_v(dst + 64, aie::concat(x1l, x1l));
            }
        }
        const uint32_t d_words = (192 * 2) / 4;
        const uint32_t *__restrict ds =
            reinterpret_cast<const uint32_t *__restrict>(d_in);
        uint32_t *__restrict dd =
            reinterpret_cast<uint32_t *__restrict>(x_stage + kDStageOff);
        for (uint32_t i = 0; i < d_words; i++)
            dd[i] = ds[i];
        for (uint32_t r = 0; r < kTileRows; r++)
            c_out[r] = static_cast<bfloat16>(0);
        return;
    }
    if (k != kTileK) {
        for (uint32_t r = 0; r < kTileRows; r++)
            c_out[r] = static_cast<bfloat16>(0);
        return;
    }
    const uint32_t chunk = *(const uint32_t *__restrict)(a_in + kBlockBytes - 4);

    const int8_t *__restrict As =
        reinterpret_cast<const int8_t *__restrict>(x_stage) + chunk * (kGroups * 128);
    const bfloat16 *__restrict d =
        reinterpret_cast<const bfloat16 *__restrict>(x_stage + kDStageOff) + chunk * kGroups;
    // Nibbles anchored at tile+0, transposed scales just past them (the
    // P10/P3 anchor quirk — offsets on stream bases get dropped — is
    // moot here: both bases are plain a_in offsets at aligned addresses).
    const int4 *__restrict nib = reinterpret_cast<const int4 *>(a_in);
    const bfloat16 *__restrict sf_t =
        reinterpret_cast<const bfloat16 *>(a_in + kTileRows * kTileK / 2);

    aie::accum<accfloat, kTileRows> acc0 = aie::zeros<accfloat, kTileRows>();
    aie::accum<accfloat, kTileRows> acc1 = aie::zeros<accfloat, kTileRows>();

    // FLAT loop, step 2: every per-group index is ADDRESS arithmetic on g
    // (the compiler's strong case) — a nested g4/j loop left j runtime and
    // turned extract<16>(j)/dv[g] into vshuffle-per-group + d spilled to
    // the stack with dynamic scalar loads (v5.1 regression, P13). The TWO
    // accumulators break the 64-deep serial f32 FMA chain (acc = mac(acc,
    // ...) is a loop-carried dependency at ~4-5 cycles latency — P13 left
    // ~0.5us/block of compute exposure and that chain is the prime
    // suspect); even/odd group partials sum at the end, a reordering well
    // inside the bf16 golden tolerance.
    for (uint32_t g = 0; g < kGroups; g += 2) {
        aie::vector<int8, 64> A0 = aie::load_v<64>(As + g * 128);
        aie::vector<int8, 64> A1 = aie::load_v<64>(As + g * 128 + 64);
        aie::vector<int4, 256> B0 = aie::load_v<256>(nib + g * 256);
        aie::vector<int4, 256> B1 = aie::load_v<256>(nib + g * 256 + 128);
        aie::mmul<4, 16, 16, int8, int4> mm;
        mm.mac(A0, B0);
        mm.mac(A1, B1);
        // C[4][16] int32 row-major: lanes 0..15 = the 16 rows' group dots.
        aie::vector<int32, 16> r0 = mm.to_vector<int32>().extract<16>(0);
        aie::vector<float, 16> rf = aie::to_float<float>(r0);
        // No bf16->f32 vector overload exists on this aie_api: mul two
        // bf16 vectors into an accfloat instead — the product of two
        // 8-bit-mantissa values is EXACT in f32, so sf*d needs no extra
        // rounding step.
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
    // Sum the two accumulator chains via fpmac (acc0 + 1.0*v1): a bare
    // accum+accum lowers to G_FADD <16 x s32>, which the peano backend
    // cannot legalize — fpmac with an accumulator first operand is the
    // known-legal form (this whole loop is built on it).
    aie::vector<float, kTileRows> ones = aie::broadcast<float, kTileRows>(1.0f);
    aie::accum<accfloat, kTileRows> acc =
        aie::mac(acc0, ones, acc1.template to_vector<float>());
    aie::vector<bfloat16, kTileRows> out = acc.template to_vector<bfloat16>();
    aie::store_v(c_out, out);
}

// Dispatcher (P16 stack law): read the self-describing K header and
// tail-call the right noinline callee — the hot path keeps its proven
// 0x340 frame, each glue flavor gets its own within the 0x400 budget.
template <uint32_t block_size>
void w4gemvu_matvec(uint32_t m,
                    const uint8_t *__restrict a_in,
                    bfloat16 *__restrict c_out,
                    uint32_t group_size,
                    uint32_t tile_idx)
{
    static_assert(block_size == 32, "block_size must be 32 (int4 vector width)");
    const uint32_t k = *(const uint32_t *__restrict)(a_in + kBlockBytes - 8);
    if (k == 1) {
        if (*(const uint32_t *__restrict)(x_stage + kQuadFlagOff) != 0)
            fused_stage1r(a_in, c_out); // P19 quad tg4 win1 re-read
        else
            fused_stage1(a_in, c_out);
        return;
    }
    if (k == 2) {
        fused_stage1b(a_in, c_out); // P19 quad win2 (2-dummy sections)
        return;
    }
    if (k == 3) {
        fused_stage2(a_in, c_out);
        return;
    }
    if (k == 4) {
        fused_stage3a(a_in, c_out);
        return;
    }
    if (k == 5) {
        fused_stage3b(a_in, c_out);
        return;
    }
    w4gemvu_compute(m, a_in, c_out, group_size, tile_idx);
}

extern "C" {

// Entry point for the universal design: one self-describing 16-row x 2048
// tile (or K=0 activation element) per call. m is always 16 (ctrl code);
// tile_idx is vestigial (v3 ABI slot).
void w4gemvu_matvec_bf16(uint32_t m,
                         const uint8_t *__restrict a_in,
                         bfloat16 *__restrict c_out,
                         uint32_t group_size,
                         uint32_t tile_idx)
{
    w4gemvu_matvec<32>(m, a_in, c_out, group_size, tile_idx);
}

} // extern "C"
