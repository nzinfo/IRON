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
PAD_ROWS = 3008


class AIEW4GEMVUQuad(AIEOperatorBase):
    """M6 QUAD whole-layer W4 GEMV (P19, see design_quad.py): o ->
    rms1 -> gateup -> swiglu -> down -> rms2 -> qkv(n+1) in ONE exec.
    The swiglu inputs are gateup's own C sections re-read as two C-sourced
    window fills (K=4 gate half, K=5 up half — header words parked in the
    host-owned padA/padB gap rows); residual2 = x + o_out is computed on
    device by a tg4 re-read of the win1 window (quad flag dispatch), so
    the host writes ONLY residual1 per token. rt.sequence order:
    A1, A2, A3, A4, C (the X activation element rides the head of
    packed1 — the NPU ctrl kernel signature caps at 5 buffer args).
    """

    def __init__(
        self,
        M1,
        K1,
        M2,
        M3,
        K3,
        M4,
        num_aie_columns=8,
        group_size=32,
        context=None,
    ):
        if (M1, K1, M2, M3, K3, M4) != (FUSED_M1, 2048, 12288, FUSED_M1, 6144, 3072):
            raise AIEOperatorConstraintError(
                "quad is the fixed hy-mt2 layer shape "
                "(2048x2048, 12288x2048, 2048x6144, 3072x2048)"
            )
        if group_size != 32:
            raise AIEOperatorConstraintError("group_size == 32")

        self.M1, self.K1, self.M2 = M1, K1, M2
        self.M3, self.K3, self.M4 = M3, K3, M4
        self.num_aie_columns = num_aie_columns
        self.group_size = group_size
        self.blocks1 = reference.blocks_per_col(M1, K1, TILE_ROWS, num_aie_columns)
        self.blocks2 = reference.blocks_per_col(M2, 2048, TILE_ROWS, num_aie_columns)
        self.blocks3 = reference.blocks_per_col(M3, K3, TILE_ROWS, num_aie_columns)
        self.blocks4 = reference.blocks_per_col(M4, 2048, TILE_ROWS, num_aie_columns)
        self.bytes1_per_col = (1 + self.blocks1) * ELEM  # X (K=0) element first
        self.bytes2_per_col = (1 + self.blocks2) * ELEM  # K=3 w1 element first
        self.bytes3_per_col = self.blocks3 * ELEM  # no X, no head — pure blocks
        self.bytes4_per_col = (1 + self.blocks4) * ELEM  # K=3 w2 element first
        self.c_rows1 = num_aie_columns * (self.blocks1 + 2) * TILE_ROWS  # 2304
        self.sec_gu = (self.blocks2 + 2) * TILE_ROWS  # 1568
        self.sec_dn = (self.blocks3 + 2) * TILE_ROWS  # 800
        self.sec_q = (self.blocks4 + 4) * TILE_ROWS  # 448: 3 dummies + pad
        self.gate_off = RMS_ELEM_ROWS  # 9280
        self.up_off = self.gate_off + 4 * self.sec_gu + PAD_ROWS  # 18560
        self.win2_off = self.up_off + 4 * self.sec_gu + PAD_ROWS  # 27840
        self.qkv_off = self.win2_off + RMS_ELEM_ROWS  # 37120
        self.c_total_rows = self.qkv_off + num_aie_columns * self.sec_q  # 40704

        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    def set_up_artifacts(self):
        operator_dir = Path(__file__).parent
        file_name_base = f"w4gemvuq_{self.M1}x{self.K1}_{self.M2}_{self.M3}x{self.K3}_{self.M4}"

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_base}.mlir",
            import_path=operator_dir / "design_quad.py",
            callback_fn="my_w4gemvu_quad",
            callback_args=[
                self.context.device_manager.device_type,
                self.num_aie_columns,
                self.M1,
                self.K1,
                self.M2,
                self.M3,
                self.K3,
                self.M4,
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
        self.add_buffer(
            "packed3", self.num_aie_columns * self.bytes3_per_col, dtype=np.uint8
        )
        self.add_buffer(
            "packed4", self.num_aie_columns * self.bytes4_per_col, dtype=np.uint8
        )
        self.add_buffer("output", self.c_total_rows, dtype=bfloat16)
        self.add_kernel(
            "w4gemvuq",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        # rt.sequence order: A1, A2, A3, A4, C — the X element rides the
        # head of packed1 (5-BO ctrl-kernel signature cap, see design).
        self.add_to_runlist(
            "w4gemvuq", "packed1", "packed2", "packed3", "packed4", "output"
        )

    def build_packed1(self, packed_blocks, activation_elem):
        """Per column [X element | that column's 16 o blocks] — the P12
        pattern (activation rides the A stream head), so the op needs no
        separate vector BO (5-BO ctrl-kernel cap)."""
        blocks = len(packed_blocks) // (self.num_aie_columns * ELEM)
        assert blocks == self.blocks1
        bytes_per_col = (1 + blocks) * ELEM
        out = np.zeros(self.num_aie_columns * bytes_per_col, dtype=np.uint8)
        for col in range(self.num_aie_columns):
            base = col * bytes_per_col
            out[base : base + ELEM] = activation_elem.numpy()
            out[base + ELEM : base + bytes_per_col] = packed_blocks[
                col * blocks * ELEM : (col + 1) * blocks * ELEM
            ]
        return torch.from_numpy(out)

    def build_packed_w(self, packed_blocks, weight, head_blocks1):
        """Plain v5 packed blocks + the (M1,) bf16 rms weight -> per column
        [w element | that column's blocks]. The w element: [weight | pad |
        K=3 at ELEM-8 | head_blocks1 at ELEM-4] — head_blocks1 is the block
        count of the op whose partials the glue consumes (o: 16 for A2,
        down: 48 for A4)."""
        blocks = len(packed_blocks) // (self.num_aie_columns * ELEM)
        assert blocks in (self.blocks2, self.blocks4)
        assert weight.shape[-1] == self.M1
        w_elem = np.zeros(ELEM, dtype=np.uint8)
        w_elem[0 : 2 * self.M1] = weight.view(torch.uint16).numpy().view(np.uint8)
        w_elem[ELEM - 8 : ELEM - 4] = np.frombuffer(
            struct.pack("<I", 3), dtype=np.uint8
        )
        w_elem[ELEM - 4 : ELEM] = np.frombuffer(
            struct.pack("<I", head_blocks1), dtype=np.uint8
        )
        bytes_per_col = (1 + blocks) * ELEM
        out = np.zeros(self.num_aie_columns * bytes_per_col, dtype=np.uint8)
        for col in range(self.num_aie_columns):
            base = col * bytes_per_col
            out[base : base + ELEM] = w_elem
            out[base + ELEM : base + bytes_per_col] = packed_blocks[
                col * blocks * ELEM : (col + 1) * blocks * ELEM
            ]
        return torch.from_numpy(out)

    def build_c_init(self, residual1):
        """C pre-run state: residual1 at rows [2304..4352) and the four
        windows' header words — win1 K=1/blocks1=16 (rows 9276..9279),
        padA K=4 (rows 18556..18559), padB K=5 (rows 27836..27839), win2
        K=2/blocks1=48 (rows 37116..37119). residual2 rows are left ZERO:
        the device computes h2' = x + o_out itself (the stage1b copy of
        those rows is dead — stage1r overwrites the staging slot)."""
        assert residual1.shape[-1] == self.M1
        c = torch.zeros(self.c_total_rows, dtype=torch.bfloat16)
        c[self.c_rows1 : self.c_rows1 + self.M1] = residual1.to(torch.bfloat16)
        bits = c.view(torch.uint16)

        def hdr(row, k, blocks1):
            bits[row] = k & 0xFFFF
            bits[row + 1] = 0
            bits[row + 2] = blocks1 & 0xFFFF
            bits[row + 3] = 0

        hdr(9276, 1, self.blocks1)  # win1 (o sections + residual1)
        hdr(18556, 4, 0)  # padA: K=4 gate window
        hdr(27836, 5, 0)  # padB: K=5 up window
        hdr(37116, 2, self.blocks3)  # win2 (down sections)
        return c

    def replicate_vector(self, vector):
        """bf16 (K1,) o activation -> its K=0 element (host quantize,
        exactly the unfused path)."""
        if vector.shape[-1] != self.K1:
            raise AIEOperatorConstraintError("vector length must be K1")
        q, d, _ = reference.quantize_vector(vector, self.group_size)
        return reference.build_activation_element(q, d, self.K1)
