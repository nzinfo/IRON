#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""P28-3 ring probe: cross-column core<->core ObjectFifo placement proof.

8 persistent workers, serpentine ring (see design_ring.py), one task
group with one fill + one drain per worker per exec. The golden is
worker-identifying constants (A chunk w is filled with the value w+1),
so C[w] must equal pred(w)+1: a mismatch's VALUE names which worker's
data actually arrived, and a hang names the edge that failed to route.
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from ml_dtypes import bfloat16

from iron.common import (
    AIEOperatorBase,
    XclbinArtifact,
    InstsBinArtifact,
    KernelObjectArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
)
from iron.common.test_utils import run_test

CHUNK = 512
SUCC = {0: 1, 1: 2, 2: 3, 3: 7, 7: 6, 6: 5, 5: 4, 4: 0}  # mirror of design_ring
PRED = {v: k for k, v in SUCC.items()}


class AIERingProbe(AIEOperatorBase):
    def __init__(self, cols=8, chunk=CHUNK, context=None):
        self.cols = cols
        self.chunk = chunk
        AIEOperatorBase.__init__(self, context=context)

    def set_up_artifacts(self):
        operator_dir = Path(__file__).parent
        file_name_base = f"ringprobe{self.cols}"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_base}.mlir",
            import_path=operator_dir / "design_ring.py",
            callback_fn="my_ring_probe",
            callback_args=[
                self.context.device_manager.device_type,
                self.cols,
                self.chunk,
            ],
        )
        xclbin_artifact = XclbinArtifact.new(
            f"{file_name_base}.xclbin",
            depends=[
                mlir_artifact,
                KernelObjectArtifact.new(
                    "ring_probe.o",
                    depends=[
                        SourceArtifact.new(
                            self.context.base_dir
                            / "aie_kernels"
                            / "aie2p"
                            / "ring_probe.cc"
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
        self.add_buffer("input", self.cols * self.chunk, dtype=bfloat16)
        self.add_buffer("output", self.cols * self.chunk, dtype=bfloat16)
        self.add_kernel(
            "ringprobe",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_to_runlist("ringprobe", "input", "output")


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
)
def test_ring_probe(aie_context):
    cols, chunk = 8, CHUNK
    operator = AIERingProbe(cols=cols, chunk=chunk, context=aie_context)

    a = torch.zeros(cols * chunk, dtype=torch.bfloat16)
    expected = torch.zeros(cols * chunk, dtype=torch.bfloat16)
    for w in range(cols):
        a[w * chunk : (w + 1) * chunk] = w + 1
        p = PRED[w]
        expected[w * chunk : (w + 1) * chunk] = p + 1

    input_buffers = {"input": a}
    output_buffers = {"output": expected}

    errors, latency_us, _ = run_test(operator, input_buffers, output_buffers)

    print(f"\nLatency (us): {latency_us:.1f}")
    assert not errors, f"ring probe mismatch: {errors}"
