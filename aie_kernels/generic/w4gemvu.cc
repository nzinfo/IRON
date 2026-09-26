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

// v5.1 activation staging (24960 B): the PREBUILT A operands + d.
//   [c*8192 + g*128 .. +64)   A0 = [x0 x0 x0 x0] (int8, 64B, 64B-aligned)
//   [c*8192 + g*128 + 64)     A1 = [x1 x1 x1 x1]
//   [24576 + c*128 ..)        d bf16[64] (chunk c)
// Group stride 128 B keeps every A operand 64-B aligned (aie2p >256-bit
// load_v rule). L1 budget: 2x18560 (A fifo) + 24960 + 64 (C fifo) +
// stack ~= 63.2 KB of the 64-KB tile (P12's 74-KB two-fifo probe is the
// known-over watermark). Filled by the K=0 activation element; weight
// blocks read it by chunk id. .bss (uninitialized) keeps it out of .data;
// the peano/placer core-private accounting covers it like the stack
// (verify in the .map after edits).
static uint8_t x_stage[3 * kGroups * 128 + 192 * 2] __attribute__((aligned(64)));
constexpr uint32_t kAStageBytes = 3 * kGroups * 128; // 24576
constexpr uint32_t kDStageOff = kAStageBytes;        // d follows A ops

// One group's compute (v5.2 op set), spelled out inline in the hot loop:
// calls with vector/accum references do NOT inline reliably on this
// toolchain — the v5.3 helper-function variant hung the core (stack/ABI
// break) and the lambda+sfb4-extract variant broke goldens. Keep the
// loop FLAT with every index = address arithmetic on g.

template <uint32_t block_size>
void w4gemvu_matvec(uint32_t m,
                    const uint8_t *__restrict a_in,
                    bfloat16 *__restrict c_out,
                    uint32_t group_size,
                    uint32_t tile_idx)
{
    static_assert(block_size == 32, "block_size must be 32 (int4 vector width)");
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
