// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define NOCPP

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>

// Fused INT4-dequant GEMV, UNIVERSAL (self-describing tiles) — the M3b
// engine kernel. Same numerics and signed-int4 ABI as w4gemv2 (two
// interleaved mac accumulators, notes §11-13), but the contraction
// length K is read FROM THE TILE HEADER at runtime instead of being a
// compiled-in scalar, so ONE kernel/PDI serves every projection shape:
// a decode step streams qkv/o/gate_up (K=2048) and down (K=6144) tiles
// through the same workers, and only the ctrl-code (DMA bd lengths and
// taps) differs per shape — no CU switch, no PDI reload (the ~650us/op
// switch cost measured by run-w4layer disappears).
//
// Tile layout v2 (self-describing, unit = one fifo element of 13840 bytes):
//   [0 .. m*K/2)      row-major packed int4 nibbles (row r at r*K/2)
//   [ .. +8)          8-byte junk hole
//   [ .. +m*(K/32)*2) bf16 per-group-32 scales (row-major)
//   [ .. 13832)       pad (K=2048 tiles carry ~9.2 KB of DDR padding)
//   [13832..13836)    K as u32 (little-endian)
//   [13836..13840)    reserved
// Every tile is ONE element (acquire(1)): the core simply reads only K
// columns. m (rows per tile) stays a compiled-in constant (4).
//
// ANCHOR QUIRK (peano -O2, verified by unit-vector fingerprinting on
// hardware): the int4 weight stream pointer compiled from `a_in + 8`
// reads from `a_in + 0` — the +8 byte offset is silently dropped on the
// movs/padda streaming path — while the indexed scale loads keep it, and
// so does the scalar K load. The layout above is built for the EFFECTIVE
// anchors: weights land at a_in+0 (the nibble start), scales at
// a_in+8+m*K/2 (just past the hole), K at the fixed slot tail. Do not
// "fix" the source anchors without re-fingerprinting: writing a_in+0
// explicitly may get shifted again.
//
// STACK BUDGET — the peano linker reserves only 0x400 bytes of stack and
// the placer puts the neighbor fifo buffer directly above it (notes §12);
// this loop shape matches w4gemv2's 2-accumulator form (0x1c0 frame),
// verified safe. Do not widen the interleave without re-checking
// `paddxm [sp], #imm` in the ELF.

constexpr uint32_t kSlotBytes = 13840; // ELEM: one padded max-K tile slot

template <uint32_t block_size>
void w4gemvu_matvec(uint32_t m,
                    const uint8_t *__restrict a_in,
                    const bfloat16 *__restrict b_in,
                    bfloat16 *__restrict c_out,
                    uint32_t group_size)
{
    static_assert(block_size == 32, "block_size must be 32 (int4 vector width)");

    ::aie::set_rounding(aie::rounding_mode::conv_even);

    // Self-describing tile: K comes from the fixed slot tail, not the ELF
    // (a_in+0 is the nibble start now — see the anchor-quirk note above).
    const uint32_t k = *(const uint32_t *__restrict)(a_in + kSlotBytes - 8);
    const uint8_t *tile = a_in + 8;
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

// Entry point for the universal design: one self-describing tile of
// m_input rows per call (row_offset is gone — every tile starts at 0).
void w4gemvu_matvec_bf16(uint32_t m,
                         const uint8_t *__restrict a_in,
                         const bfloat16 *__restrict b_in,
                         bfloat16 *__restrict c_out,
                         uint32_t group_size)
{
    w4gemvu_matvec<32>(m, a_in, b_in, c_out, group_size);
}

} // extern "C"
