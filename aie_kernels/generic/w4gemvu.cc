// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#define NOCPP

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>

// Fused INT4-dequant GEMV, UNIVERSAL — the M3b engine kernel, layout v4
// (MATRIX-UNIT mmul, P11). Same signed-int4 weight ABI as v2/v3
// (per-group-32 symmetric, scale = amax/7 bf16). ONE PDI serves every
// projection shape: blocks are UNIFORM (16 rows x 2048 k each — K=6144 ops
// stream 3 chunk-blocks per tile, chunk-major so consecutive blocks share
// one x), so the device side is shape-free and only ctrl code differs.
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
//   [18552 .. 18556)  K as u32 (always 2048; guard only)
//
// Activation B-slot (6528 bytes, uniform):
//   [0 .. 6144)    x as int8 (per-group-32 symmetric quantization,
//                  q in [-127,127], first K bytes live)
//   [6144 .. 6528) d: bf64[192] per-group x scales (group g at 6144+2g;
//                  K=2048 uses the first 64)
// The kernel replicates each 16-byte x chunk 4x in registers to build the
// [4x16] A operand (M=4 is the only aie2p int8xint4 shape; the 4 A rows are
// redundant, C row 0 carries the 16 real dots). Per group: 2 A-builds, 2 B
// loads (128B), 2 matrix macs, one int32->f32 row-0 extract scaled by
// d[g] (broadcast) and sf_t[g][*] (vector) into an f32 accumulator — the
// f32 domain is REQUIRED: group partials reach 2^15 and bf16 would round
// them to death before the sum.
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

template <uint32_t block_size>
void w4gemvu_matvec(uint32_t m,
                    const uint8_t *__restrict a_in,
                    const uint8_t *__restrict b_in,
                    bfloat16 *__restrict c_out,
                    uint32_t group_size,
                    uint32_t tile_idx)
{
    static_assert(block_size == 32, "block_size must be 32 (int4 vector width)");
    (void)m;      // always kTileRows (ctrl code carries it)
    (void)tile_idx; // one call per block in v4

    ::aie::set_rounding(aie::rounding_mode::conv_even);

    // Self-describing guard: K is 2048 by construction; a stale/garbage
    // header degrades to zero rows instead of reading wild offsets.
    const uint32_t k = *(const uint32_t *__restrict)(a_in + kBlockBytes - 8);
    if (k != kTileK) {
        for (uint32_t r = 0; r < kTileRows; r++)
            c_out[r] = static_cast<bfloat16>(0);
        return;
    }
    const uint32_t groups = kTileK / group_size; // 64

    const int8_t *__restrict x8 = reinterpret_cast<const int8_t *>(b_in);
    const bfloat16 *__restrict d = reinterpret_cast<const bfloat16 *>(b_in + kKMax);
    // Nibbles anchored at tile+0, transposed scales just past them (the
    // P10/P3 anchor quirk — offsets on stream bases get dropped — is
    // moot here: both bases are plain a_in offsets at aligned addresses).
    const int4 *__restrict nib = reinterpret_cast<const int4 *>(a_in);
    const bfloat16 *__restrict sf_t =
        reinterpret_cast<const bfloat16 *>(a_in + kTileRows * kTileK / 2);

    aie::accum<accfloat, kTileRows> acc = aie::zeros<accfloat, kTileRows>();

    for (uint32_t g = 0; g < groups; g++) {
        aie::mmul<4, 16, 16, int8, int4> mm;
        const int8_t *xg = x8 + g * group_size;
        aie::vector<int8, 16> x0 = aie::load_v<16>(xg);
        aie::vector<int8, 16> x1 = aie::load_v<16>(xg + 16);
        // concat needs equal-width halves: (x0+x0) | (x0+x0) -> [4x16].
        aie::vector<int8, 32> x0l = aie::concat(x0, x0);
        aie::vector<int8, 32> x1l = aie::concat(x1, x1);
        aie::vector<int8, 64> A0 = aie::concat(x0l, x0l);
        aie::vector<int8, 64> A1 = aie::concat(x1l, x1l);
        aie::vector<int4, 256> B0 = aie::load_v<256>(nib + g * 256);
        aie::vector<int4, 256> B1 = aie::load_v<256>(nib + g * 256 + 128);
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
        acc = aie::add(acc, aie::mul(rf, sfd));
    }
    aie::vector<bfloat16, kTileRows> out = acc.template to_vector<bfloat16>();
    aie::store_v(c_out, out);
}

extern "C" {

// Entry point for the universal design: one self-describing 16-row x 2048
// tile per call. m is always 16 (ctrl code); tile_idx is vestigial (v3 ABI).
void w4gemvu_matvec_bf16(uint32_t m,
                         const uint8_t *__restrict a_in,
                         const uint8_t *__restrict b_in,
                         bfloat16 *__restrict c_out,
                         uint32_t group_size,
                         uint32_t tile_idx)
{
    w4gemvu_matvec<32>(m, a_in, b_in, c_out, group_size, tile_idx);
}

} // extern "C"
