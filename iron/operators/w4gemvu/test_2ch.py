#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""P12 probe: channel scaling of the weight stream (B removed).

PERF ONLY — the kernel's b_in is fed the A element itself and the K
header reads garbage, so every call takes the zero-rows path (outputs are
garbage; identical wire bytes). Compare latencies against the production
test and across channels:

    production (8ch A + 8ch B) vs 1ch (8ch A, no B) vs 2ch (16ch A, no B)

- 1ch gain over production = B-removal / fewer fills
- 2ch gain over 1ch        = true per-channel scaling
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from iron.common import (
    AIEOperatorBase,
    XclbinArtifact,
    InstsBinArtifact,
    KernelObjectArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
)
from iron.operators.w4gemvu import reference
from iron.common.test_utils import run_test


class AIEW4GEMVUCH(AIEOperatorBase):
    def __init__(self, M, K, num_aie_columns=8, channels=2, context=None):
        self.M = M
        self.K = K
        self.num_aie_columns = num_aie_columns
        self.channels = channels
        AIEOperatorBase.__init__(self, context=context)

    def set_up_artifacts(self):
        operator_dir = Path(__file__).parent
        file_name_base = f"w4gemvuch{self.channels}_{self.M}x{self.K}"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_base}.mlir",
            import_path=operator_dir / "design_2ch.py",
            callback_fn="my_w4gemvu_2ch",
            callback_args=[
                self.context.device_manager.device_type,
                self.num_aie_columns,
                self.M,
                self.K,
                32,
                self.channels,
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
                            self.context.base_dir / "aie_kernels" / "generic" / "w4gemvu.cc"
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
        from ml_dtypes import bfloat16

        blocks_per_col = reference.blocks_per_col(
            self.M, self.K, reference.TILE_ROWS, self.num_aie_columns
        )
        self.add_buffer(
            "packed_weights",
            self.num_aie_columns * blocks_per_col * reference.ELEM,
            dtype=np.uint8,
        )
        # Probe C layout: one 16-row element per A element per fifo —
        # blocks_per_col * 2 / channels halves per column.
        elements_per_fifo = blocks_per_col * 2 // self.channels
        self.add_buffer(
            "output",
            self.num_aie_columns * elements_per_fifo * reference.TILE_ROWS,
            dtype=bfloat16,
        )
        self.add_kernel(
            "w4gemvuch",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_to_runlist("w4gemvuch", "packed_weights", "output")


# The perf comparison set: the two extreme real shapes (smallest and
# largest M) plus gateup/down, at 1 and 2 channels per column.
params = [
    (2048, 2048, 1),
    (12288, 2048, 1),
    (2048, 6144, 1),
    (121088, 2048, 1),
    (2048, 2048, 2),
    (12288, 2048, 2),
    (2048, 6144, 2),
    (121088, 2048, 2),
]
names = [f"w4gemvu{ch}ch_{M}x{K}" for M, K, ch in params]
all_params = [pytest.param(*p, id=n) for p, n in zip(params, names)]


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
)
@pytest.mark.parametrize("M,K,channels", all_params)
def test_w4gemvu_2ch(M, K, channels, aie_context):
    golden_ref = reference.generate_golden_reference(M=M, K=K)
    operator = AIEW4GEMVUCH(
        M=M, K=K, num_aie_columns=8, channels=channels, context=aie_context
    )

    input_buffers = {"packed_weights": torch.from_numpy(golden_ref["packed_weights"])}
    bpc = reference.blocks_per_col(M, K, reference.TILE_ROWS, 8)
    out_rows = 8 * (bpc * 2 // channels) * reference.TILE_ROWS
    output_buffers = {
        "output": torch.zeros(out_rows, dtype=golden_ref["output_raw"].dtype)
    }

    # (run_test computes errors against garbage — ignored, PERF ONLY)
    _, latency_us, _ = run_test(operator, input_buffers, output_buffers)
    stream_mb = len(golden_ref["packed_weights"]) / 1e6
    print(
        f"\n[{channels}ch {M}x{K}] Latency (us): {latency_us:.1f}, "
        f"{stream_mb / latency_us * 1e3:.2f} GB/s weights"
    )
