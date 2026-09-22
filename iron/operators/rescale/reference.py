# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch


def generate_golden_reference(M: int, N: int, seed=42):
    """Reference for the q8 epilogue: C = bf16(f32(A_i32) * sa[M] * sw[N]).

    The scale values are exact binary fractions so the f32 multiplies stay
    inside a few roundings; matching the kernel's association order
    ((A * sa) * sw) element-for-element makes the reference bit-reproducible
    against the conv_even (RNE) kernel conversion.
    """
    torch.manual_seed(seed)
    # Magnitudes like a real q8 GEMM accumulation (i8 inputs, K in the
    # hundreds): comfortably inside int32, far from float32's exact range.
    acc = torch.randint(-16000, 16001, (M, N), dtype=torch.int32)

    # Exact binary fractions, so the f32 -> bf16 conversion below is lossless
    # (no need for ml_dtypes round-trips torch can't do).
    sa = torch.from_numpy((((np.arange(M) % 8) - 3) / 8.0).astype(np.float32)).to(
        torch.bfloat16
    )
    sw = torch.from_numpy((((np.arange(N) % 5) + 1) / 16.0).astype(np.float32)).to(
        torch.bfloat16
    )

    # Same association order as the kernel: (A * sa) * sw, each step in f32.
    step1 = acc.to(torch.float32) * sa.to(torch.float32)[:, None]
    prod = step1 * sw.to(torch.float32)[None, :]
    output = prod.to(torch.bfloat16)

    return {"input": acc, "sa": sa, "sw": sw, "output": output}
