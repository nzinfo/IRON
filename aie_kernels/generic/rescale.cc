// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define NOCPP

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// q8 route-A epilogue: rescale an exact int32 GEMM accumulation back to bf16
// with per-row (activation) and per-column (weight) scales,
//   C[m][n] = bf16(f32(A[m][n]) * sa[m] * sw[n]).
// A is row-major MxN int32. The scales travel as one contiguous f32 block,
// m row scales followed by n column scales (f32 rather than bf16: the host
// widens them losslessly anyway, and it keeps the kernel on native float
// vector loads — the 32-lane bf16 from_vector emulation was returning the
// row-scale half for both pointers in bring-up).
//
// The bf16 narrowing conversion is the rounding-mode-sensitive step: the AIE
// default rounds ties toward -inf (floor), so conv_even is set explicitly —
// the same rule the mm.cc ROUND_CONV_EVEN flag exists for, applied here from
// day one (see notes/rust-drm-port-log.md §9: the add fixture predates this
// and rounds tie sums one ULP away from host RNE).

template <uint32_t r> void rescale_vectorized(uint32_t m,
                                              uint32_t n,
                                              const int32_t *__restrict a,
                                              const float *__restrict s,
                                              bfloat16 *__restrict c)
{
    const float *__restrict sa = s;
    const float *__restrict sw = s + m;
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    for (uint32_t row = 0; row < m; row++) {
        const float sar = sa[row];
        const int32_t *__restrict pA = a + (size_t)row * n;
        const int32_t *const pA_end = pA + n;
        bfloat16 *__restrict pC = c + (size_t)row * n;
        const float *__restrict pSW = sw;
        for (; pA < pA_end; pA += r, pC += r, pSW += r) {
            aie::vector<int32_t, r> a_v = aie::load_v<r>(pA);
            aie::vector<float, r> swf = aie::load_v<r>(pSW);
            // to_float is the numeric Fix2Float conversion — from_vector into
            // an FP accumulator would just reinterpret the int bits. int32 ->
            // float widens exactly for q8 magnitudes.
            aie::vector<float, r> af = aie::to_float(a_v);
            aie::vector<float, r> sarf = aie::broadcast<float, r>(sar);
            // float x float needs the accumulator tag spelled out (unlike the
            // bf16 convenience mul, AccumTag is not inferrable).
            aie::accum<accfloat, r> prod = aie::mul<accfloat>(af, sarf);
            prod = aie::mul<accfloat>(prod.template to_vector<float>(), swf);
            // Final SRS narrowing honors the conv_even rounding set above.
            aie::store_v(pC, prod.template to_vector<bfloat16>());
        }
    }
}

extern "C" {

void rescale_i32_bf16_vector(const int32_t *a_in,
                             const float *s_in,
                             bfloat16 *c_out,
                             uint32_t m,
                             uint32_t n)
{
    rescale_vectorized<32>(m, n, a_in, s_in, c_out);
}

} // extern "C"
