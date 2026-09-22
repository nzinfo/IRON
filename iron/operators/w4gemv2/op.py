# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from ml_dtypes import bfloat16
from pathlib import Path

from iron.common import (
    AIEOperatorBase,
    AIEOperatorConstraintError,
    XclbinArtifact,
    InstsBinArtifact,
    KernelObjectArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
)


class AIEW4GEMV2(AIEOperatorBase):
    """W4 fused dequant GEMV, v2 — our kernel (two interleaved mac
    accumulators break the dependency chains that bound the upstream
    fused_dequant_gemv at ~2.4us/row; see notes/rust-drm-port-log.md §11-12
    for the shape's stack-budget constraint).

    Identical DDR tile ABI and host contract to AIEFusedDequantGEMV:
    packed uint4 weights + per-group bf16 scales in, C = W_dequant @ x out.
    """

    def __init__(
        self,
        M,
        K,
        num_aie_columns=4,
        tile_size_input=1,
        tile_size_output=None,
        group_size=32,
        context=None,
    ):
        if tile_size_output is None:
            tile_size_output = M // num_aie_columns

        if tile_size_output % tile_size_input != 0 or tile_size_output < tile_size_input:
            raise AIEOperatorConstraintError(
                "tile_size_output must be a multiple of tile_size_input"
            )
        if K % group_size != 0:
            raise AIEOperatorConstraintError("K must be a multiple of group_size")
        if group_size != 32:
            # The v2 tight loop handles exactly one vector block per group.
            raise AIEOperatorConstraintError("w4gemv2 kernel requires group_size == 32")
        if M % num_aie_columns != 0:
            raise AIEOperatorConstraintError("M must be a multiple of num_aie_columns")

        self.M = M
        self.K = K
        self.num_aie_columns = num_aie_columns
        self.tile_size_input = tile_size_input
        self.tile_size_output = tile_size_output
        self.group_size = group_size

        self.xclbin_artifact = None
        self.insts_artifact = None

        AIEOperatorBase.__init__(self, context=context)

    def _packed_buffer_size(self):
        num_groups_per_row = self.K // self.group_size
        packed_tile_bytes = (
            self.tile_size_input * self.K // 2
            + self.tile_size_input * num_groups_per_row * 2
        )
        rows_per_col = self.M // self.num_aie_columns
        tiles_per_col = rows_per_col // self.tile_size_input
        return self.num_aie_columns * tiles_per_col * packed_tile_bytes

    def set_up_artifacts(self):
        operator_dir = Path(__file__).parent
        file_name_base = (
            f"w4gemv2_{self.M}x{self.K}"
            f"_{self.tile_size_input}tsi"
            f"_{self.tile_size_output}tso"
            f"_{self.num_aie_columns}col"
            f"_g{self.group_size}"
        )

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_w4gemv2",
            callback_args=[
                self.context.device_manager.device_type,
                self.num_aie_columns,
                self.M,
                self.K,
                self.tile_size_input,
                self.tile_size_output,
                self.group_size,
            ],
        )

        xclbin_artifact = XclbinArtifact.new(
            f"{file_name_base}.xclbin",
            depends=[
                mlir_artifact,
                KernelObjectArtifact.new(
                    "w4gemv2.o",
                    depends=[
                        SourceArtifact.new(
                            self.context.base_dir
                            / "aie_kernels"
                            / "generic"
                            / "w4gemv2.cc"
                        )
                    ],
                ),
            ],
        )

        insts_artifact = InstsBinArtifact.new(
            f"{file_name_base}.bin", depends=[mlir_artifact]
        )

        self.xclbin_artifact = xclbin_artifact
        self.insts_artifact = insts_artifact
        self.add_artifacts([xclbin_artifact, insts_artifact])

    def set_up_runtime(self):
        self.add_buffer("packed_weights", self._packed_buffer_size(), dtype=np.uint8)
        self.add_buffer("vector", self.K, dtype=bfloat16)
        self.add_buffer("output", self.M, dtype=bfloat16)
        self.add_kernel(
            "w4gemv2",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_to_runlist("w4gemv2", "packed_weights", "vector", "output")

    def forward(self, vector, packed_weights=None):
        """vector: bf16 (K,); packed_weights: optional packed uint8 buffer."""
        if vector.shape[-1] != self.K or vector.dtype != torch.bfloat16:
            raise AIEOperatorConstraintError(
                f"AIEW4GEMV2: expected bf16 vector of length {self.K}, "
                f"got shape {tuple(vector.shape)} dtype {vector.dtype}"
            )
        if packed_weights is not None:
            self.write_buffer("packed_weights", packed_weights)
        self.write_buffer("vector", vector)
        self.run_runlist()
        return self.read_buffer_as_torch("output", (self.M,))
