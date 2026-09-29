#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""P28-6 bisect probe run: layerv2 with gathers removed (design_lv2probe).

COMPLETION is the signal -- the C image is garbage by construction (see
the design docstring); we only assert the runlist finishes and print the
latency, which also gives a first N=16 A/C plumbing timing point.
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

HIDDEN, INTER, QKV_M = 2048, 6144, 3072


def geom(n):
    rows = HIDDEN // n
    N_O = rows // 16
    N_GATE = (INTER // n) // 16
    N_QKV = (QKV_M // n) // 16
    return {
        "n": n, "rows": rows, "N_O": N_O, "N_GATE": N_GATE,
        "N_DOWN": 3 * N_O, "N_QKV": N_QKV, "N_CXN": N_O,
        "N_WELEM": N_O + 2 * N_GATE + 3 * N_O + N_QKV + 2,
    }


class AIELv2Probe(AIEOperatorBase):
    def __init__(self, cols=16, rings=0, context=None):
        self.cols = cols
        self.rings = rings
        AIEOperatorBase.__init__(self, context=context)

    def set_up_artifacts(self):
        operator_dir = Path(__file__).parent
        file_name_base = f"lv2probe{self.cols}_r{self.rings}"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_base}.mlir",
            import_path=operator_dir / "design_lv2probe.py",
            callback_fn="my_lv2probe",
            callback_args=[
                self.context.device_manager.device_type,
                self.cols,
                self.rings,
            ],
        )
        xclbin_artifact = XclbinArtifact.new(
            f"{file_name_base}.xclbin",
            depends=[
                mlir_artifact,
                KernelObjectArtifact.new(
                    "w4gemvu_layer.o",
                    depends=[
                        SourceArtifact.new(
                            self.context.base_dir
                            / "aie_kernels"
                            / "aie2p"
                            / "w4gemvu_layer.cc"
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
        g = geom(self.cols)
        ELEM = 18560
        self.add_buffer("W", self.cols * g["N_WELEM"] * ELEM, dtype=np.uint8)
        self.add_buffer("X", self.cols * ELEM, dtype=np.uint8)
        self.add_buffer("XN", self.cols * ELEM, dtype=np.uint8)
        self.add_buffer(
            "C", self.cols * (g["N_QKV"] + g["N_CXN"]) * 16, dtype=bfloat16)
        self.add_kernel(
            "lv2probe",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_to_runlist("lv2probe", "W", "X", "XN", "C")


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
)
@pytest.mark.parametrize("cols,rings", [(16, 17), (16, 16)])
def test_lv2probe(aie_context, cols, rings):
    import struct

    g = geom(cols)
    operator = AIELv2Probe(cols=cols, rings=rings, context=aie_context)

    ELEM = 18560

    def u32(v):
        return np.frombuffer(struct.pack("<I", v), dtype=np.uint8)

    # X elements: worker id + N words; q left zero (arena zeros), d zero
    x = np.zeros(cols * ELEM, dtype=np.uint8)
    for w in range(cols):
        x[w * ELEM + 6400 : w * ELEM + 6404] = u32(w)
        x[w * ELEM + 6404 : w * ELEM + 6408] = u32(cols)
    # XN elements: K=100 header so lv_xnelem runs (rows bf16 zeros)
    xn = np.zeros(cols * ELEM, dtype=np.uint8)
    for w in range(cols):
        xn[w * ELEM + ELEM - 8 : w * ELEM + ELEM - 4] = u32(100)
    # W: real K headers in fill order [o xN_O | w2 | gate | up | down |
    # w1 | qkv] with all-zero weights -> every flavor executes at N=16
    # geometry with exact-zero partials (amax=0 -> d=0 -> q=0 chains)
    ks = ([2048] * g["N_O"] + [101] + [103] * g["N_GATE"] +
          [104] * g["N_GATE"] + [105] * g["N_DOWN"] + [102] +
          [2048] * g["N_QKV"])
    assert len(ks) == g["N_WELEM"]
    wbuf = np.zeros(cols * g["N_WELEM"] * ELEM, dtype=np.uint8)
    for c in range(cols):
        base = c * g["N_WELEM"] * ELEM
        for i, k in enumerate(ks):
            off = base + i * ELEM
            wbuf[off + ELEM - 8 : off + ELEM - 4] = u32(k)

    input_buffers = {"W": torch.from_numpy(wbuf),
                     "X": torch.from_numpy(x),
                     "XN": torch.from_numpy(xn)}
    output_buffers = {"C": torch.zeros(
        cols * (g["N_QKV"] + g["N_CXN"]) * 16, dtype=torch.bfloat16)}

    errors, latency_us, _ = run_test(operator, input_buffers, output_buffers)

    print(f"\n[lv2probe w{cols} rings={rings}] Latency (us): {latency_us:.1f}")
    assert not errors, f"lv2 probe mismatch: {errors}"
