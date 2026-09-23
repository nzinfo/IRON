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

ELEM = 13840
ELEMS_PER_TILE = 1
K_MAX = 6144


class AIEW4GEMVU(AIEOperatorBase):
    """Universal W4 fused dequant GEMV — ONE PDI for every projection
    shape (notes §13-14): the kernel reads K from a self-describing tile
    header at runtime, so all (M, K) variants compile a bit-identical
    device side and differ only in ctrl-code. This is the decode-step
    answer to the ~650us-per-op PDI-reload cost run-w4layer measured:
    the whole projection chain runs on a single CU.

    Tile layout and fixed fifo geometry: see design.py.
    """

    def __init__(
        self,
        M,
        K,
        num_aie_columns=8,
        group_size=32,
        context=None,
    ):
        if K not in (2048, 6144):
            raise AIEOperatorConstraintError(
                "w4gemvu variants cover K in {2048, 6144} (MiniCPM5 shapes)"
            )
        if group_size != 32:
            raise AIEOperatorConstraintError("w4gemvu kernel requires group_size == 32")
        if M % (num_aie_columns * 4) != 0:
            raise AIEOperatorConstraintError(
                "M must be a multiple of num_aie_columns * 4 rows/tile"
            )

        self.M = M
        self.K = K
        self.num_aie_columns = num_aie_columns
        self.group_size = group_size

        self.xclbin_artifact = None
        self.insts_artifact = None

        AIEOperatorBase.__init__(self, context=context)

    def _packed_buffer_size(self):
        tiles_per_col = self.M // self.num_aie_columns // 4
        return self.num_aie_columns * tiles_per_col * ELEMS_PER_TILE * ELEM

    def set_up_artifacts(self):
        operator_dir = Path(__file__).parent
        file_name_base = f"w4gemvu_{self.M}x{self.K}"

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_w4gemvu",
            callback_args=[
                self.context.device_manager.device_type,
                self.num_aie_columns,
                self.M,
                self.K,
                self.group_size,
            ],
        )

        xclbin_artifact = XclbinArtifact.new(
            f"{file_name_base}.xclbin",
            depends=[
                mlir_artifact,
                KernelObjectArtifact.new(
                    "w4gemvu.o",
                    depends=[
                        SourceArtifact.new(
                            self.context.base_dir
                            / "aie_kernels"
                            / "generic"
                            / "w4gemvu.cc"
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
        # The B fifo slot is K_MAX wide for every variant; activations for
        # K=2048 occupy the first 2048 elements and zero-pad the rest.
        self.add_buffer("vector", K_MAX, dtype=bfloat16)
        self.add_buffer("output", self.M, dtype=bfloat16)
        self.add_kernel(
            "w4gemvu",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_to_runlist("w4gemvu", "packed_weights", "vector", "output")

    def forward(self, vector, packed_weights=None):
        """vector: bf16 (K,) — the first K entries are the activation."""
        if vector.shape[-1] != self.K or vector.dtype != torch.bfloat16:
            raise AIEOperatorConstraintError(
                f"AIEW4GEMVU: expected bf16 vector of length {self.K}, "
                f"got shape {tuple(vector.shape)} dtype {vector.dtype}"
            )
        if packed_weights is not None:
            self.write_buffer("packed_weights", packed_weights)
        self.write_buffer("vector", vector)
        self.run_runlist()
        return self.read_buffer_as_torch("output", (self.M,))
