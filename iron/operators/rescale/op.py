# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from ml_dtypes import bfloat16
import logging
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


class AIERescale(AIEOperatorBase):
    """AIE-accelerated q8 epilogue: bf16 = i32 accumulation x per-row x per-column scale."""

    def __init__(
        self,
        M,
        N,
        tile_m=None,
        context=None,
    ):
        if tile_m is None:
            tile_m = 32
        if M % tile_m != 0 or N % 32 != 0:
            raise AIEOperatorConstraintError(
                f"AIERescale: M={M} must be a multiple of tile_m={tile_m} and "
                f"N={N} a multiple of 32 (kernel vector width)."
            )
        self.M = M
        self.N = N
        self.tile_m = tile_m
        self.num_blocks = M // tile_m

        self.xclbin_artifact = None
        self.insts_artifact = None

        AIEOperatorBase.__init__(self, context=context)

    def set_up_artifacts(self):
        operator_dir = Path(__file__).parent
        file_name_base = f"rescale_{self.M}x{self.N}_{self.tile_m}t"

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_rescale",
            callback_args=[
                self.context.device_manager.device_type,
                self.M,
                self.N,
                self.tile_m,
                0,
            ],
        )

        xclbin_artifact = XclbinArtifact.new(
            f"{file_name_base}.xclbin",
            depends=[
                mlir_artifact,
                KernelObjectArtifact.new(
                    "rescale.o",
                    depends=[
                        SourceArtifact.new(
                            self.context.base_dir / "aie_kernels" / "generic" / "rescale.cc"
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
        # Scales travel concatenated per row chunk ([tile_m row | N col]) in
        # f32 — the bf16 values widened losslessly — one chunk per kernel
        # block, staying within the tile's DMA channel budget (2 in / 1 out,
        # like the elementwise ops).
        self.add_buffer("input", self.M * self.N, dtype=np.int32)
        self.add_buffer("scales", (self.tile_m + self.N) * self.num_blocks, dtype=np.float32)
        self.add_buffer("output", self.M * self.N, dtype=bfloat16)
        self.add_kernel(
            "rescale",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_to_runlist("rescale", "input", "scales", "output")

    def forward(self, acc, sa, sw):
        """acc: [M, N] int accumulation; sa: [M] bf16; sw: [N] bf16."""
        if acc.shape != (self.M, self.N) or sa.shape != (self.M,) or sw.shape != (self.N,):
            raise AIEOperatorConstraintError(
                f"AIERescale: got {tuple(acc.shape)}/{tuple(sa.shape)}/{tuple(sw.shape)}, "
                f"expected ({self.M},{self.N})/({self.M},)/({self.N},)"
            )
        return self._execute_aie_operation(acc, sa, sw)

    def _execute_aie_operation(self, acc, sa, sw):
        self.write_buffer("input", acc.reshape(-1))
        # One concatenated f32 scale chunk per row block:
        # [sa rows of this block | full sw column scales].
        sa_np = sa.detach().numpy().astype(np.float32)
        sw_np = sw.detach().numpy().astype(np.float32)
        chunks = [
            np.concatenate([sa_np[b * self.tile_m : (b + 1) * self.tile_m], sw_np])
            for b in range(self.num_blocks)
        ]
        self.write_buffer("scales", np.concatenate(chunks))
        self.write_buffer("output", np.zeros(self.M * self.N, dtype=bfloat16))
        self.run_runlist()
        result = self.read_buffer_as_torch("output", shape=(self.M * self.N,), dtype=bfloat16)
        return result.reshape(self.M, self.N)
