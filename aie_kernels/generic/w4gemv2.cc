// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define NOCPP

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>

// Fused INT4-dequant GEMV, v2 — same DDR tile ABI as the upstream
// aie2p/fused_dequant_gemv.cc:
//   [m * k / 2 bytes packed uint4 weights (low nibble first)]
//   [m * (k / group_size) bf16 scale factors]
// restructured to break the inner-loop dependency chains that bound the
// upstream kernel at ~2.4us/row / 2.7% MAC utilization (notes §11-12):
//   * groups alternate between TWO interleaved accumulators (g%2), so the
//     mac dependency chains are 32 deep instead of 64;
//   * one horizontal reduce per row (unchanged from upstream — the upstream
//     reduce was already per row; the chain was the issue, not the reduce).
// Measured: 2.8-3.7x upstream (8col/16tsi: 235us seq / 176us pipelined vs
// 660us), bit-exact against the dyadic test recipe.
//
// STACK BUDGET — the hard constraint on this kernel's shape. The peano
// linker reserves only 0x400 bytes of stack after the kernel image
// (_sp_start_value_DM_stack, ". += 0x400 /* stack */"), and the mlir_aie
// placer allocates the next tile buffer exactly above that region (e.g. B
// at 0x70400 with sp=0x70000). Upstream kernels are stackless; a kernel
// whose frame exceeds 0x400 spills accumulators straight into the neighbor
// ObjectFifo buffer (a 4-accumulator variant at frame 0x4c0 corrupted
// x[0..95] this way — every call read the same smashed vector and all rows
// went wrong; diagnosed by a canary kernel, notes §12). The 2-accumulator
// shape below compiles to a 0x1c0 frame and is safe. Do not raise the
// interleave width without re-checking `paddxm [sp], #imm` in the ELF.
//
// Note: peano clang 773413fb crashes in RegBankSelect at -O2 on a variant
// of this loop that multiplies each fresh product by a broadcast f32 group
// scale and adds into f32 vectors ("Why do we split?" assertion on a
// non-PHI); the mac-into-accumulators shape below compiles cleanly.

template <uint32_t block_size>
void w4gemv2_matvec(uint32_t m,
                    uint32_t k,
                    const uint8_t *__restrict a_in,
                    const bfloat16 *__restrict b_in,
                    bfloat16 *__restrict c_out,
                    uint32_t group_size)
{
    static_assert(block_size == 32, "block_size must be 32 (uint4 vector width)");

    ::aie::set_rounding(aie::rounding_mode::conv_even);

    const uint4 *weights_packed = reinterpret_cast<const uint4 *>(a_in);
    const uint8_t *scale_bytes = a_in + (size_t)m * k / 2;
    const bfloat16 *scales = reinterpret_cast<const bfloat16 *>(scale_bytes);
    const uint32_t groups_per_row = k / group_size;

    for (uint32_t row = 0; row < m; row++) {
        const uint4 *row_w = weights_packed + (size_t)row * k / 2;
        const bfloat16 *row_s = scales + (size_t)row * groups_per_row;
        const bfloat16 *bp = b_in;

        aie::accum<accfloat, block_size> acc0 = aie::zeros<accfloat, block_size>();
        aie::accum<accfloat, block_size> acc1 = acc0;

        // Groups in twos: g -> acc0, g+1 -> acc1.
        uint32_t g = 0;
        for (; g + 1 < groups_per_row; g += 2) {
#pragma unroll
            for (uint32_t j = 0; j < 2; j++) {
                aie::vector<uint4, block_size> I0 = aie::load_v<block_size>(row_w);
                row_w += block_size / 2; // uint4* arithmetic is byte-based
                // Dequant chain identical to expand.cc / the upstream kernel.
                aie::vector<uint8, block_size> as_u8 = aie::unpack(I0);
                aie::vector<uint16, block_size> as_u16 = aie::unpack(as_u8);
                aie::vector<bfloat16, block_size> w = aie::to_float<bfloat16>(as_u16, 0);
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
            aie::vector<uint4, block_size> I0 = aie::load_v<block_size>(row_w);
            row_w += block_size / 2;
            aie::vector<uint8, block_size> as_u8 = aie::unpack(I0);
            aie::vector<uint16, block_size> as_u16 = aie::unpack(as_u8);
            aie::vector<bfloat16, block_size> w = aie::to_float<bfloat16>(as_u16, 0);
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

// Entry point matching the fused GEMV signature pattern
// (m, k, row_offset, a, b, c, group_size) so the upstream operator layout
// drives it unchanged.
void w4gemv2_matvec_bf16(uint32_t m,
                         uint32_t k,
                         uint32_t row_offset,
                         const uint8_t *__restrict a_in,
                         const bfloat16 *__restrict b_in,
                         bfloat16 *__restrict c_out,
                         uint32_t group_size)
{
    c_out += row_offset;
    w4gemv2_matvec<32>(m, k, a_in, b_in, c_out, group_size);
}

} // extern "C"
