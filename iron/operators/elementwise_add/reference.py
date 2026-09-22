# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from iron.common.utils import torch_dtype_map


def generate_golden_reference(input_length: int, dtype="bf16", seed=42):
    torch.manual_seed(seed)
    dtype_torch = torch_dtype_map[dtype]
    if dtype_torch.is_floating_point:
        val_range = 4
        input_a = torch.rand(input_length, dtype=dtype_torch) * val_range
        input_b = torch.rand(input_length, dtype=dtype_torch) * val_range
    else:
        # Integer dtypes: torch.rand is unsupported, and small values keep the
        # exact-match check meaningful (sums stay in [-120, 120], no wrap).
        input_a = torch.randint(-60, 61, (input_length,), dtype=dtype_torch)
        input_b = torch.randint(-60, 61, (input_length,), dtype=dtype_torch)
    output = input_a + input_b
    return {"A": input_a, "B": input_b, "C": output}
