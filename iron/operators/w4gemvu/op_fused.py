#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
import torch
from ml_dtypes import bfloat16
from pathlib import Path
import struct

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
from iron.operators.w4gemvu.reference import ELEM, TILE_ROWS

FUSED_M1 = 2048
RMS_ELEM_ROWS = ELEM // 2  # 9280
# Probe switches (P16 hang bisection): set both to 0 to make the kernel
# treat the rms window / w element as plain K=0 activations — the K=1/K=3
# flavors (and their soft-float calls) never execute, isolating the ctrl
# machinery (two task groups + RMS fill from the C tensor) from the kernel.
PROBE_K_HDR_WINDOW = 1
PROBE_K_HDR_W = 3


class AIEW4GEMVUFused(AIEOperatorBase):
    """M6 fused rms-pair W4 GEMV (see design_fused.py): op1 ->
    add+rms+quantize (kernel K=1/K=3 flavors) -> op2, ONE exec. The rms
    window reads op1's C back through the A fifo (a fill sourced from
    the C tensor after tg1's drains are awaited) and carries the
    residual in its free rows; the rms weight rides a K=3 element at
    the HEAD of packed2 so fifo order runs the glue before op2's
    blocks. Op2 ships no X element at all.
    """

    def __init__(self, M1, K1, M2, num_aie_columns=8, group_size=32, context=None):
        if M1 != FUSED_M1:
            raise AIEOperatorConstraintError("fused pairs assume M1 == 2048 (hidden)")
        if K1 not in (2048, 6144):
            raise AIEOperatorConstraintError("K1 in {2048, 6144}")
        if M2 % 256 != 0:
            raise AIEOperatorConstraintError("M2 % 256 == 0 (v5 ABI; K2 is 2048)")
        if group_size != 32:
            raise AIEOperatorConstraintError("group_size == 32")

        self.M1, self.K1, self.M2 = M1, K1, M2
        self.num_aie_columns = num_aie_columns
        self.group_size = group_size
        self.blocks1 = reference.blocks_per_col(M1, K1, TILE_ROWS, num_aie_columns)
        self.blocks2 = reference.blocks_per_col(M2, 2048, TILE_ROWS, num_aie_columns)
        self.bytes1_per_col = self.blocks1 * ELEM
        self.bytes2_per_col = (1 + self.blocks2) * ELEM
        self.c_rows1 = num_aie_columns * (self.blocks1 + 2) * TILE_ROWS
        self.section2_rows = (self.blocks2 + 2) * TILE_ROWS
        self.c_total_rows = RMS_ELEM_ROWS + num_aie_columns * self.section2_rows

        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    def set_up_artifacts(self):
        operator_dir = Path(__file__).parent
        file_name_base = f"w4gemvuf_{self.M1}x{self.K1}_{self.M2}x2048"

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_base}.mlir",
            import_path=operator_dir / "design_fused.py",
            callback_fn="my_w4gemvu_fused",
            callback_args=[
                self.context.device_manager.device_type,
                self.num_aie_columns,
                self.M1,
                self.K1,
                self.M2,
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
        self.add_buffer(
            "packed1", self.num_aie_columns * self.bytes1_per_col, dtype=np.uint8
        )
        self.add_buffer(
            "packed2", self.num_aie_columns * self.bytes2_per_col, dtype=np.uint8
        )
        self.add_buffer("vector", ELEM, dtype=np.uint8)
        self.add_buffer("output", self.c_total_rows, dtype=bfloat16)
        self.add_kernel(
            "w4gemvuf",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        # rt.sequence order: A1, A2, X, C
        self.add_to_runlist("w4gemvuf", "packed1", "packed2", "vector", "output")

    def build_packed2(self, packed_blocks, weight):
        """Plain v5 packed blocks (cols*blocks2*ELEM) + the (M1,) bf16
        rms weight -> per column [w element | that column's blocks].
        The w element: [weight | pad | K=3 at ELEM-8 | blocks1 at ELEM-4]
        (every column carries the same element)."""
        assert tuple(packed_blocks.shape) == (
            self.num_aie_columns * self.blocks2 * ELEM,
        )
        assert weight.shape[-1] == self.M1
        w_elem = np.zeros(ELEM, dtype=np.uint8)
        w_elem[0 : 2 * self.M1] = weight.view(torch.uint16).numpy().view(np.uint8)
        w_elem[ELEM - 8 : ELEM - 4] = np.frombuffer(
            struct.pack("<I", PROBE_K_HDR_W), dtype=np.uint8
        )
        w_elem[ELEM - 4 : ELEM] = np.frombuffer(
            struct.pack("<I", self.blocks1), dtype=np.uint8
        )
        out = np.zeros(self.num_aie_columns * self.bytes2_per_col, dtype=np.uint8)
        for col in range(self.num_aie_columns):
            base = col * self.bytes2_per_col
            out[base : base + ELEM] = w_elem
            out[base + ELEM : base + self.bytes2_per_col] = packed_blocks[
                col * self.blocks2 * ELEM : (col + 1) * self.blocks2 * ELEM
            ]
        return torch.from_numpy(out)

    def build_c_init(self, residual):
        """The C tensor pre-run state: the rms window's host-written
        region — residual at rows [c_rows1..c_rows1+M1) and the header
        words (K=1 as u32 at ELEM-8, blocks1 as u32 at ELEM-4; i.e.
        u16 rows 9276..9279 = [1, 0, blocks1, 0]) — plus zeros. The
        drains never touch any of these rows."""
        assert residual.shape[-1] == self.M1
        c = torch.zeros(self.c_total_rows, dtype=torch.bfloat16)
        c[self.c_rows1 : self.c_rows1 + self.M1] = residual.to(torch.bfloat16)
        bits = c.view(torch.uint16)
        bits[9276] = PROBE_K_HDR_WINDOW
        bits[9277] = 0
        bits[9278] = self.blocks1 & 0xFFFF
        bits[9279] = 0
        return c

    def replicate_vector(self, vector):
        """bf16 (K1,) op1 activation -> its K=0 element (host quantize,
        exactly the unfused path)."""
        if vector.shape[-1] != self.K1:
            raise AIEOperatorConstraintError("vector length must be K1")
        q, d, _ = reference.quantize_vector(vector, self.group_size)
        return reference.build_activation_element(q, d, self.K1)
