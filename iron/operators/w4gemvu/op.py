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

    def _b_reps(self):
        """F: activation copies in the DDR vector buffer (one per B fifo
        element). A single multi-element fill keeps the shim BD count at
        one per channel — per-element fills would need F BDs and gate_up
        (F=24) exhausts the allocator's 16."""
        tiles_per_col = self.M // self.num_aie_columns // 4
        return tiles_per_col // 16

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
        # The B stream is one multi-element fill: the vector buffer holds
        # F copies of the activation back to back (F = self._b_reps()).
        # Each B fifo slot is K_MAX wide; K=2048 activations zero-pad.
        self.add_buffer("vector", self._b_reps() * K_MAX, dtype=bfloat16)
        self.add_buffer("output", self.M, dtype=bfloat16)
        self.add_kernel(
            "w4gemvu",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_to_runlist("w4gemvu", "packed_weights", "vector", "output")

    def replicate_vector(self, vector):
        """Expand a (K,) activation into the F-slot DDR vector buffer
        (each K_MAX-wide B slot holds x zero-padded). forward() and any
        direct write_buffer("vector", ...) caller must use this — a bare
        (K,) tensor only covers the first slot and leaves the rest stale.
        """
        if vector.shape[-1] != self.K or vector.dtype != torch.bfloat16:
            raise AIEOperatorConstraintError(
                f"AIEW4GEMVU: expected bf16 vector of length {self.K}, "
                f"got shape {tuple(vector.shape)} dtype {vector.dtype}"
            )
        reps = self._b_reps()
        if self.K == K_MAX:
            return vector.repeat(reps)
        vb = torch.zeros(reps * K_MAX, dtype=torch.bfloat16)
        for r in range(reps):
            vb[r * K_MAX : r * K_MAX + self.K] = vector
        return vb

    def forward(self, vector, packed_weights=None):
        """vector: bf16 (K,) — the activation; replicated F times internally."""
        if packed_weights is not None:
            self.write_buffer("packed_weights", packed_weights)
        self.write_buffer("vector", self.replicate_vector(vector))
        self.run_runlist()
        return self.read_buffer_as_torch("output", (self.M,))
