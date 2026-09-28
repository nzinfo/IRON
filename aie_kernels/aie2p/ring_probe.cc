// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// P28-3 ring probe kernel: plain chunk copy, no computation.
//
// Each worker calls it twice per element: (a -> ring_out) to feed the
// ring, then (ring_in -> c) to emit what traversed the neighbor core.
// The probe proves cross-column core<->core ObjectFifo placement +
// routing + lock flow on npu2 within the 2-in/2-out stream port budget
// (worker i consumes [shim A, ring_in] and produces [ring_out, shim C]).

#define NOCPP

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void ring_copy_bf16(const bfloat16 *__restrict src, bfloat16 *__restrict dst)
{
    constexpr int N = 512;
    for (int i = 0; i < N; i += 16) {
        aie::vector<bfloat16, 16> v = aie::load_v<16>(src + i);
        aie::store_v(dst + i, v);
    }
}
