#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 nzinfo. All rights reserved.
# SPDX-License-Identifier: Apache-2.0.

"""P28-12 device-side attention board test (design_lv2attn).

Feeds the w4gemvu_attn_* flavors a random GQA fixture and compares the
drained attention output against the golden chain (rope_pairs /
qk_rms_bits / attention_bits — the SAME mirror the hy E2E host glue
uses). The kernel's online softmax + bf16 exp2 drift against the
golden's two-pass f32 softmax is absorbed by the tolerance (4e-2 rel /
1.5e-1 abs — the band the mha kernel passes with).
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
from iron.common.test_utils import run_test, torch_to_numpy


def std_out_dbg():
    import os

    return os.environ.get("LV2ATTN_DBG") is not None

ELEM = 18560
HEAD_DIM, HEADS, KV_HEADS = 128, 16, 4
EPS = 1e-5
ROPE_BASE = 11158840.0
K_OUT = 16  # C elements per worker


def _f32(b):
    return (np.asarray(b, dtype=np.uint32) << 16).view(np.float32).copy()


def _bf16(f32):
    u = np.asarray(f32, dtype=np.float32).view(np.uint32)
    rounded = u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def pos_of(w):
    return w if w < 4 else 11 - w


DBG = False  # P28-12 bring-up: stream post-rope/qknorm q rows via C


def golden_attention(rng, S):
    """Random q/k/v + weights -> per-worker golden output rows (u16)."""
    q = _bf16(rng.standard_normal(HEADS * HEAD_DIM).astype(np.float32))
    k_all = _bf16(rng.standard_normal(KV_HEADS * HEAD_DIM).astype(np.float32))
    v_all = _bf16(rng.standard_normal(KV_HEADS * HEAD_DIM).astype(np.float32))
    qn = _bf16(rng.standard_normal(HEAD_DIM).astype(np.float32))
    kn = _bf16(rng.standard_normal(HEAD_DIM).astype(np.float32))
    p = S - 1  # decode position of the current token
    j = np.arange(HEAD_DIM // 2, dtype=np.float32)
    ang = np.float32(p) * (
        np.float32(ROPE_BASE) ** (-j / np.float64(HEAD_DIM // 2))
    ).astype(np.float32)
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

    def rope(x_bits):
        xf = _f32(x_bits).reshape(-1, HEAD_DIM)
        out = np.empty_like(xf)
        out[:, :64] = xf[:, :64] * cos - xf[:, 64:] * sin
        out[:, 64:] = xf[:, 64:] * cos + xf[:, :64] * sin
        return _bf16(out).reshape(-1)

    def qknorm(x_bits, w_bits):
        x = _f32(x_bits).reshape(-1, HEAD_DIM)
        w = _f32(w_bits)
        out = np.zeros_like(x)
        for h in range(x.shape[0]):
            v = x[h]
            ms = np.float32(0)
            for t in v:
                ms = np.float32(ms + np.float32(t * t))
            inv = np.float32(1.0) / np.sqrt(np.float32(ms / HEAD_DIM + EPS))
            out[h] = v * inv * w
        return _bf16(out).reshape(-1)

    # history: S-1 random positions + the current one (rope'd + normed)
    kcache = _bf16(rng.standard_normal((S - 1, KV_HEADS * HEAD_DIM)).astype(np.float32))
    vcache = _bf16(rng.standard_normal((S - 1, KV_HEADS * HEAD_DIM)).astype(np.float32))
    k_cur = qknorm(rope(k_all), kn).reshape(KV_HEADS, HEAD_DIM)
    v_cur = v_all.reshape(KV_HEADS, HEAD_DIM)
    kc = np.concatenate([kcache, k_cur.reshape(1, -1)], axis=0)  # (S, 512)
    vc = np.concatenate([vcache, v_cur.reshape(1, -1)], axis=0)
    qr = qknorm(rope(q), qn).reshape(HEADS, HEAD_DIM)

    group = HEADS // KV_HEADS
    out = np.zeros(HEADS * HEAD_DIM, dtype=np.uint16)
    for h in range(HEADS):
        kk = _f32(kc.reshape(S, KV_HEADS, HEAD_DIM)[:, h // group])
        vv = _f32(vc.reshape(S, KV_HEADS, HEAD_DIM)[:, h // group])
        scores = kk @ _f32(qr[h].astype(np.uint16).copy()) / np.float32(
            HEAD_DIM**0.5
        )
        scores = scores - scores.max()
        e = np.exp(scores)
        out[h * HEAD_DIM : (h + 1) * HEAD_DIM] = _bf16((e / e.sum()) @ vv)

    # per-worker slice: worker p emits heads 2p, 2p+1 (256 rows)
    img = np.zeros(8 * 256, dtype=np.uint16)
    for w in range(8):
        p = pos_of(w)
        img[w * 256 : (w + 1) * 256] = np.concatenate(
            [out[2 * p * HEAD_DIM : (2 * p + 1) * HEAD_DIM],
             out[(2 * p + 1) * HEAD_DIM : (2 * p + 2) * HEAD_DIM]]
        )
    return q, k_all, v_all, qn, kn, cos, sin, kcache, vcache, img


def build_w_stream(rng, S, parts):
    q, k_all, v_all, qn, kn, cos, sin, kcache, vcache, _ = parts
    khist = S - 1
    per_w = khist + 18
    W = np.zeros(8 * per_w * ELEM, dtype=np.uint8)

    def u32(w, i, off, val):
        base = (w * per_w + i) * ELEM
        W[base + off : base + off + 4] = np.frombuffer(
            np.uint32(val).tobytes(), dtype=np.uint8
        )

    def put(w, i, off, arr_u16):
        base = (w * per_w + i) * ELEM
        W[base + off : base + off + arr_u16.nbytes] = np.frombuffer(
            arr_u16.tobytes(), dtype=np.uint8
        )

    def putf(w, i, off, arr_f32):
        base = (w * per_w + i) * ELEM
        W[base + off : base + off + arr_f32.nbytes] = np.frombuffer(
            arr_f32.astype(np.float32).tobytes(), dtype=np.uint8
        )

    for w in range(8):
        # X element: zeros + worker id + N=8 + K=0
        u32(w, 0, 6400, w)
        u32(w, 0, 6404, 8)
        u32(w, 0, ELEM - 8, 0)
        # attn-init (K=210)
        put(w, 1, 0, q)
        put(w, 1, 4096, k_all)
        put(w, 1, 5120, v_all)
        putf(w, 1, 6144, cos)
        putf(w, 1, 6400, sin)
        put(w, 1, 8192, qn)
        put(w, 1, 8448, kn)
        u32(w, 1, 8704, S | 0x80000000 if DBG else S)
        u32(w, 1, ELEM - 8, 210)
        # kvhist (K=211): one position per element, history first
        for j in range(khist):
            put(w, 2 + j, 0, kcache[j])
            put(w, 2 + j, 2048, vcache[j])
            u32(w, 2 + j, ELEM - 8, 211)
        # out elements (K=212)
        for i in range(K_OUT):
            u32(w, 2 + khist + i, ELEM - 8, 212)
    return W


class AIELv2Attn(AIEOperatorBase):
    def __init__(self, khist, context=None):
        self.khist = khist
        AIEOperatorBase.__init__(self, context=context)

    def set_up_artifacts(self):
        operator_dir = Path(__file__).parent
        file_name_base = f"lv2attn_h{self.khist}"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_base}.mlir",
            import_path=operator_dir / "design_lv2attn.py",
            callback_fn="my_lv2attn",
            callback_args=[
                self.context.device_manager.device_type,
                8,
                self.khist,
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
        self.add_buffer("weights", 8 * (self.khist + 18) * ELEM, dtype=np.uint8)
        self.add_buffer(
            "output", 8 * K_OUT * 16, dtype=bfloat16
        )
        self.add_kernel(
            "lv2attn",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        # rt.sequence(W, C)
        self.add_to_runlist("lv2attn", "weights", "output")


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
)
@pytest.mark.parametrize("S", [48, 96])
def test_lv2attn(S, aie_context):
    print(f"\nlv2attn: S={S} (GQA 16Q/4KV d=128, online softmax on NPU)")
    global DBG
    DBG = False
    rng = np.random.default_rng(42)
    parts = golden_attention(rng, S)
    W = build_w_stream(rng, S, parts)
    # debug golden: worker p's 256 rows = post-rope/qknorm q heads 2p, 2p+1
    q, k_all, v_all, qn, kn, cos, sin, kcache, vcache, _ = parts
    def rope(x_bits):
        xf = _f32(x_bits).reshape(-1, HEAD_DIM)
        out = np.empty_like(xf)
        out[:, :64] = xf[:, :64] * cos - xf[:, 64:] * sin
        out[:, 64:] = xf[:, 64:] * cos + xf[:, :64] * sin
        return _bf16(out).reshape(-1)
    def qknorm(x_bits, w_bits):
        x = _f32(x_bits).reshape(-1, HEAD_DIM)
        w = _f32(w_bits)
        out = np.zeros_like(x)
        for h in range(x.shape[0]):
            v = x[h]
            ms = np.float32(0)
            for t in v:
                ms = np.float32(ms + np.float32(t * t))
            out[h] = v * np.float32(1.0) / np.sqrt(np.float32(ms / HEAD_DIM + EPS)) * w
        return _bf16(out).reshape(-1)
    golden = parts[-1]  # full attention output per worker

    operator = AIELv2Attn(khist=S - 1, context=aie_context)
    input_buffers = {"weights": torch.from_numpy(W)}
    # golden bits -> torch.bfloat16 (float-level comparison, not bit
    # patterns) via the repo's u16-view idiom
    output_buffers = {"output": torch.from_numpy(golden.copy()).view(torch.bfloat16)}

    errors, latency_us, _ = run_test(
        operator, input_buffers, output_buffers, rel_tol=4.0e-2, abs_tol=1.5e-1
    )
    n = len(golden)
    bad = len(errors.get("output", []))
    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"({bad} errors out of {n} values)")
    import struct

    raw = operator.read_buffer("output", (8 * 256,), dtype=np.uint16)

    def _bf(v):
        return struct.unpack("<f", struct.pack("<I", int(v) << 16))[0]

    print(
        f"INSTR w0: l(h0)={_bf(raw[240]):.4f} J={_bf(raw[241]):.1f} "
        f"q0..3={_bf(raw[0]):.4f},{_bf(raw[1]):.4f},{_bf(raw[2]):.4f},{_bf(raw[3]):.4f}"
    )
    print("TRACE step  s      m      alpha  e      l")
    for k in range(0, 250, 5):
        vals = [round(_bf(raw[k + t]), 3) for t in range(5)]
        print(f"  {k//5:3d}  " + " ".join(f"{v:7.3f}" for v in vals))
    print("TRAJ l:", [round(_bf(raw[128 + k]), 3) for k in range(0, 48, 4)])
    print("TRAJ m:", [round(_bf(raw[192 + k]), 3) for k in range(0, 48, 4)])
    assert bad <= int(n * 0.005), f"test failed with {bad}/{n}"
