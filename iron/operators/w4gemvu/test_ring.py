#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""P28-3/P28-6 ring probe: cross-column core<->core ObjectFifo proof.

N persistent workers (8/16; serpentine / all-adjacent Hamiltonian ring,
see design_ring.py), one task group with one fill + one drain per worker
per exec. The golden is worker-identifying constants (A chunk w is
filled with the value w+1), so C[w] must equal pred(w)+1: a mismatch's
VALUE names which worker's data actually arrived, and a hang names the
edge that failed to route.
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

from iron.operators.w4gemvu.design_ring import succ_tables

CHUNK = 512


class AIERingProbe(AIEOperatorBase):
    def __init__(self, cols=8, chunk=CHUNK, laps=1, context=None):
        self.cols = cols
        self.chunk = chunk
        self.laps = laps
        AIEOperatorBase.__init__(self, context=context)

    def set_up_artifacts(self):
        operator_dir = Path(__file__).parent
        file_name_base = f"ringprobe{self.cols}_l{self.laps}"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_base}.mlir",
            import_path=operator_dir / "design_ring.py",
            callback_fn="my_ring_probe",
            callback_args=[
                self.context.device_manager.device_type,
                self.cols,
                self.chunk,
                self.laps,
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
@pytest.mark.parametrize(
    "cols,laps",
    [(8, 1), (16, 1), (8, 7), (16, 7), (16, 15)],
)
def test_ring_probe(aie_context, cols, laps):
    chunk = CHUNK
    _, PRED = succ_tables(cols)
    operator = AIERingProbe(cols=cols, chunk=chunk, laps=laps, context=aie_context)

    a = torch.zeros(cols * chunk, dtype=torch.bfloat16)
    expected = torch.zeros(cols * chunk, dtype=torch.bfloat16)
    for w in range(cols):
        a[w * chunk : (w + 1) * chunk] = w + 1
        # laps>1: the drained chunk originated laps hops upstream, so the
        # golden is pred^laps(w), not pred(w) -- the first grid run
        # asserted pred(w) for every laps and "failed" correct data.
        p = w
        for _ in range(laps):
            p = PRED[p]
        expected[w * chunk : (w + 1) * chunk] = p + 1

    input_buffers = {"input": a}
    output_buffers = {"output": expected}

    errors, latency_us, _ = run_test(operator, input_buffers, output_buffers)

    print(f"\nLatency (us): {latency_us:.1f}")
    assert not errors, f"ring probe mismatch: {errors}"
