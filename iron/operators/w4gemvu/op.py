# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

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
from iron.operators.w4gemvu import reference

ELEM = reference.ELEM  # 18560: one 16-row x 2048 MATRIX-UNIT tile (v4)
ELEMS_PER_TILE = 1
K_MAX = reference.K_MAX
TILE_ROWS = reference.TILE_ROWS


class AIEW4GEMVU(AIEOperatorBase):
    """Universal W4 fused dequant GEMV — ONE PDI for every projection
    shape (notes §13-14; v4 = P11 matrix-unit tiles; v5 = P12 B-stream
    removal): every block is a uniform 16-row x 2048-k tile computed by
    mmul<4,16,16,int8,int4> (mac_4x16_16x16, 1024 MACs/instr). The
    activation rides the A fifo as a K=0 element staged to core-local
    memory (no B stream — 8 fill channels reach the ~55 GB/s device
    wall). K=6144 ops stream 3 chunk-blocks per tile (chunk-major) and
    the host sums the partials. The activation is quantized int8
    per-group-32 (numerics ABI change vs v2/v3 — goldens bake it in).

    Block layout and fixed fifo geometry: see design.py.
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
        # v5 block ABI: 16-row tiles / 8 cols; K=6144 chunk-major sections
        # — one uniform bound.
        if M % 256 != 0:
            raise AIEOperatorConstraintError(
                "v5 ABI: M must be a multiple of 256 (16-row tiles x 8 cols, "
                "K=6144 chunk sections)"
            )

        self.M = M
        self.K = K
        self.num_aie_columns = num_aie_columns
        self.group_size = group_size
        self.blocks_per_col = reference.blocks_per_col(M, K, TILE_ROWS, num_aie_columns)

        self.xclbin_artifact = None
        self.insts_artifact = None

        AIEOperatorBase.__init__(self, context=context)

    def _packed_buffer_size(self):
        return self.num_aie_columns * self.blocks_per_col * ELEMS_PER_TILE * ELEM

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
        # v5: ONE ELEM-sized activation element (K=0), shared by every
        # column's X fill — replaces the F-slot vector buffer entirely.
        self.add_buffer("vector", ELEM, dtype=np.uint8)
        # Every C row is live except each column's leading x-element
        # zeros: K=6144 carries the 3 chunk partials in chunk-major
        # M-row sections — use forward() to sum them.
        self.add_buffer("output", self.output_rows(), dtype=bfloat16)
        self.add_kernel(
            "w4gemvu",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_to_runlist("w4gemvu", "packed_weights", "vector", "output")

    def output_rows(self):
        return reference.output_rows(self.M, self.K)

    def replicate_vector(self, vector):
        """Expand a (K,) bf16 activation into the ONE-ELEM activation
        element (x all chunks at 0, d at K_MAX, K header 0). forward()
        and any direct write_buffer("vector", ...) caller must use this
        — a bare (K,) tensor leaves the K header stale.
        """
        if vector.shape[-1] != self.K or vector.dtype != torch.bfloat16:
            raise AIEOperatorConstraintError(
                f"AIEW4GEMVU: expected bf16 vector of length {self.K}, "
                f"got shape {tuple(vector.shape)} dtype {vector.dtype}"
            )
        q, d, _ = reference.quantize_vector(vector, self.group_size)
        return reference.build_activation_element(q, d, self.K)

    def forward(self, vector, packed_weights=None):
        """vector: bf16 (K,) — the activation; quantized and packed into
        the activation element internally. Returns the (M,) real output
        (K=6144 chunk partials summed)."""
        if packed_weights is not None:
            self.write_buffer("packed_weights", packed_weights)
        self.write_buffer("vector", self.replicate_vector(vector))
        self.run_runlist()
        raw = self.read_buffer_as_torch("output", (self.output_rows(),))
        return reference.unshuffle_output(raw, self.M, self.K)
