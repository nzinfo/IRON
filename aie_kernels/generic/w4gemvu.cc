// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#define NOCPP

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>

// Fused INT4-dequant GEMV, UNIVERSAL (self-describing blocks) — the M3b
// engine kernel, layout v3 (compact blocks, P10). Same numerics and
// signed-int4 ABI as w4gemv2 (two interleaved mac accumulators, notes
// §11-13). ONE PDI still serves every projection shape: the block header
// carries K at runtime and the ctrl code carries only block counts and
// taps — no CU switch, no PDI reload (the ~650us/op switch cost measured
// by run-w4layer stays gone).
//
// Block layout v3 (unit = one fifo element of 13856 bytes, COMPACT):
//   [0 .. n*tile_stride)  n = 6144/K tiles back to back (n=3 K2048,
//                        n=1 K6144), each tile padded to a 16-byte
//                        stride — load_v streams cannot start misaligned
//                        (fingerprinted: tile@4616 garbage, @0/@9232 ok):
//     [0 .. m*K/2)       row-major packed int4 nibbles (row r at r*K/2)
//     [ .. +8)           8-byte junk hole
//     [ .. +m*(K/32)*2)  bf16 per-group-32 scales (row-major)
//   [ .. 13848)          pad (K=6144 blocks carry 16 B)
//   [13848..13852)       K as u32 (little-endian)
//   [13852..13856)       reserved
// v2 padded every tile to 13840 B (3x DDR waste at K=2048 — P9's biggest
// gap item); v3 packs tiles tight, killing the waste at both K. The core
// calls the kernel 3x per block (static loop — device side stays
// shape-free); calls with tile_idx >= n write zero rows so the C fifo
// accounting is K-independent (K=6144 drains 3M rows, 2/3 zeros).
//
// ANCHOR QUIRK (peano -O2, verified by unit-vector fingerprinting on
// hardware): the int4 weight stream pointer compiled from `tile_base + 8`
// reads from `tile_base + 0` — the +8 byte offset is silently dropped on
// the movs/padda streaming path — while the indexed scale loads keep it,
// and so does the scalar K load. The layout above is built for the
// EFFECTIVE anchors: weights land at tile_base+0 (the nibble start),
// scales at tile_base+8+m*K/2 (just past the hole), K at the fixed block
// tail. v3 re-fingerprinted the nonzero tile base (pytest golden compare
// per row). Do not "fix" the source anchors without re-fingerprinting.
//
// STACK BUDGET — the peano linker reserves only 0x400 bytes of stack and
// the placer puts the neighbor fifo buffer directly above it (notes §12);
// this loop shape matches w4gemv2's 2-accumulator form (0x1c0 frame),
// verified safe. Do not widen the interleave without re-checking
// `paddxm [sp], #imm` in the ELF.

constexpr uint32_t kBlockBytes = 13888; // ELEM v3: one compact block slot
constexpr uint32_t kKMax = 6144;

template <uint32_t block_size>
void w4gemvu_matvec(uint32_t m,
                    const uint8_t *__restrict a_in,
                    const bfloat16 *__restrict b_in,
                    bfloat16 *__restrict c_out,
                    uint32_t group_size,
                    uint32_t tile_idx)
{
    static_assert(block_size == 32, "block_size must be 32 (int4 vector width)");

    ::aie::set_rounding(aie::rounding_mode::conv_even);

    // Self-describing block: K comes from the fixed block tail, not the
    // ELF. A stale/garbage header degrades to zero rows (no div-by-zero).
    const uint32_t k = *(const uint32_t *__restrict)(a_in + kBlockBytes - 8);
    const uint32_t tiles_per_block = (k == 2048 || k == 6144) ? kKMax / k : 0;
    const uint32_t tile_bytes = 8 + m * k / 2 + m * (k / group_size) * 2;
    const uint32_t tile_stride = (tile_bytes + 15) & ~15;
    if (tile_idx >= tiles_per_block) {
        // Beyond the live tiles of this block: deterministic zero rows so
        // the C fifo element count is K-independent (ctrl-free accounting).
        for (uint32_t r = 0; r < m; r++)
            c_out[r] = static_cast<bfloat16>(0);
        return;
    }
    const uint8_t *tile = a_in + (size_t)tile_idx * tile_stride + 8;
    const int4 *weights_packed = reinterpret_cast<const int4 *>(tile);
    const uint8_t *scale_bytes = tile + (size_t)m * k / 2;
    const bfloat16 *scales = reinterpret_cast<const bfloat16 *>(scale_bytes);
    const uint32_t groups_per_row = k / group_size;

    for (uint32_t row = 0; row < m; row++) {
        const int4 *row_w = weights_packed + (size_t)row * k / 2;
        const bfloat16 *row_s = scales + (size_t)row * groups_per_row;
        const bfloat16 *bp = b_in;

        aie::accum<accfloat, block_size> acc0 = aie::zeros<accfloat, block_size>();
        aie::accum<accfloat, block_size> acc1 = acc0;

        // Groups in twos: g -> acc0, g+1 -> acc1.
        uint32_t g = 0;
        for (; g + 1 < groups_per_row; g += 2) {
#pragma unroll
            for (uint32_t j = 0; j < 2; j++) {
                aie::vector<int4, block_size> I0 = aie::load_v<block_size>(row_w);
                row_w += block_size / 2; // int4* arithmetic is byte-based
                aie::vector<int8, block_size> as_i8 = aie::unpack(I0);
                aie::vector<int16, block_size> as_i16 = aie::unpack(as_i8);
                aie::vector<bfloat16, block_size> w = aie::to_float<bfloat16>(as_i16, 0);
                aie::vector<bfloat16, block_size> x = aie::load_v<block_size>(bp);
                bp += block_size;
                bfloat16 sf = row_s[g + j];
                aie::vector<bfloat16, block_size> sf_broadcast =
                    aie::broadcast<bfloat16, block_size>(sf);
                aie::vector<bfloat16, block_size> wd =
                    aie::mul(w, sf_broadcast).template to_vector<bfloat16>();
                if (j == 0)
                    acc0 = aie::mac(acc0, wd, x);
                else
                    acc1 = aie::mac(acc1, wd, x);
            }
        }
        // Tail (groups_per_row % 2): depth <= 1, chain cost negligible.
        for (; g < groups_per_row; g++) {
            aie::vector<int4, block_size> I0 = aie::load_v<block_size>(row_w);
            row_w += block_size / 2;
            aie::vector<int8, block_size> as_i8 = aie::unpack(I0);
            aie::vector<int16, block_size> as_i16 = aie::unpack(as_i8);
            aie::vector<bfloat16, block_size> w = aie::to_float<bfloat16>(as_i16, 0);
            aie::vector<bfloat16, block_size> x = aie::load_v<block_size>(bp);
            bp += block_size;
            bfloat16 sf = row_s[g];
            aie::vector<bfloat16, block_size> sf_broadcast =
                aie::broadcast<bfloat16, block_size>(sf);
            aie::vector<bfloat16, block_size> wd =
                aie::mul(w, sf_broadcast).template to_vector<bfloat16>();
            acc0 = aie::mac(acc0, wd, x);
        }

        float total = aie::reduce_add(aie::add(acc0.template to_vector<float>(),
                                               acc1.template to_vector<float>()));
        *c_out++ = static_cast<bfloat16>(total);
    }
}

extern "C" {

// Entry point for the universal design: one self-describing BLOCK of
// tiles per call; tile_idx selects the sub-tile (>= live count -> zeros).
void w4gemvu_matvec_bf16(uint32_t m,
                         const uint8_t *__restrict a_in,
                         const bfloat16 *__restrict b_in,
                         bfloat16 *__restrict c_out,
                         uint32_t group_size,
                         uint32_t tile_idx)
{
    w4gemvu_matvec<32>(m, a_in, b_in, c_out, group_size, tile_idx);
}

} // extern "C"
