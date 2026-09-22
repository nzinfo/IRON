# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Same packing + golden math as the upstream fused_dequant_gemv reference —
# the v2 kernel is numerically identical (bf16 dequant mul, f32 mac lanes,
# one conv_even narrowing), only the inner-loop scheduling changed.

from iron.operators.fused_dequant_gemv.reference import (
    generate_golden_reference,
    quantize_and_pack,
)

__all__ = ["generate_golden_reference", "quantize_and_pack"]
