#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""P28 layer-v2 test: the WHOLE transformer layer in ONE task group per
exec on 8 persistent ring workers (design_layerv2.py + w4gemvu_layer.cc).

The golden mirrors the kernel's flavor chain end to end (the P16/P19
bit-path mirrors, reused from test_quad): o -> ring1 gather x' -> rms2 ->
gate/up -> swiglu -> ring2 gather sw -> down -> ring3 gather xn1 -> rms1
-> qkv, with the drain tensor carrying per-worker [qkv 384 | xn1 256]
rows that the test reassembles by ring position.

Weight elements: the standard packer tiles, except gate/up/down blocks
carry K=103/104/105 headers (the packer writes 2048 — the fixture
patches them) so the kernel's dispatcher routes each block to its
flavor; o/qkv blocks keep 2048 and ride the phase word.
"""

import struct
from pathlib import Path

import pytest
import numpy as np
import torch

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
from iron.operators.w4gemvu import reference
from iron.operators.w4gemvu.reference import ELEM, TILE_K

COLS = 8
SUCC = {0: 1, 1: 2, 2: 3, 3: 7, 7: 6, 6: 5, 5: 4, 4: 0}  # mirror of design_layerv2


def pos(w):
    """Worker index -> ring position (serpentine)."""
    return w if w < 4 else 11 - w


N_O, N_GATE, N_UP, N_DOWN, N_QKV = 16, 48, 48, 48, 24
N_WELEM = N_O + N_GATE + N_UP + N_DOWN + N_QKV + 2  # + w2 + w1 = 186
OUT_ROWS = (N_QKV + 16) * 16  # 40 C elements per worker = 640 rows


def _f32(bf16_bits):
    return (bf16_bits.astype(np.uint32) << 16).view(np.float32)


def _bf16(f32):
    u = f32.astype(np.float32).view(np.uint32)
    rounded = u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def _kernel_rsqrt(s):
    v = np.float32(s)
    magic = np.array(0x5F3759DF, dtype=np.uint32)
    x = (magic - (v.view(np.uint32) >> np.uint32(1))).view(np.float32)
    for _ in range(3):
        x = np.float32(x * (np.float32(2.0) - v * x * x))
    return x


def _sumsq_sequential(h2f):
    sumsq = np.float32(0)
    for v in h2f:
        sumsq = np.float32(sumsq + np.float32(v * v))
    return sumsq


def _quantize_kernel(xn_f32, m):
    v32 = xn_f32.reshape(m // 32, 32)
    amax = np.abs(v32).max(axis=1)
    d_bits = _bf16((amax / np.float32(127.0)).astype(np.float32))
    invd = (np.float32(1.0) / _f32(d_bits.astype(np.uint32)))[:, None]
    v = v32 * invd
    v = np.where(amax[:, None] == 0, 0.0, v)
    q = np.round(v)
    q = np.clip(q, -127, 127).astype(np.int8).reshape(m)
    return q, d_bits


def _dequant(q, d_bits):
    return q.astype(np.float32) * np.repeat(_f32(d_bits.astype(np.uint32)), 32)


def _u32(x):
    return np.frombuffer(struct.pack("<I", x), dtype=np.uint8)


def _deq_f32(W, group_size=32):
    """Device-faithful weight dequant for the golden chain: the kernel's
    mmul computes int4 x f32-scale exactly and never rounds q*scale to
    bf16, but reference.quantize_and_pack's W_dequant does (a torch
    convenience). Over FOUR chained quantize stages that extra rounding
    compounds to ~270 absolute drift on xn1 -- pure model error, not
    hardware error (P28-4: the device matched the f32 chain bit-exactly
    at every echo point while golden drifted). Mirrors test_fused's
    kernel-glue-faithful golden; same group math as the packer minus
    the final bf16 round."""
    M, K = W.shape
    Wg = torch.from_numpy(np.ascontiguousarray(W)).to(torch.float32)
    Wg = Wg.reshape(M * K // group_size, group_size)
    amax = Wg.abs().amax(dim=1, keepdim=True)
    scale = (amax / 7.0).to(torch.bfloat16).to(torch.float32)
    q = torch.where(amax == 0, torch.zeros_like(Wg), torch.round(Wg / scale))
    q = torch.clamp(q, -8, 7)
    return (q * scale).reshape(M, K)


def _norm_element(wgt_bf16, k):
    """K=101/102 element: ln weight bf16[2048] at [0,4096), K at ELEM-8."""
    e = np.zeros(ELEM, dtype=np.uint8)
    e[0:4096] = wgt_bf16.view(torch.uint16).numpy().view(np.uint8)
    e[ELEM - 8 : ELEM - 4] = _u32(k)
    return e


def _patch_k(elem_bytes, k):
    elem_bytes[ELEM - 8 : ELEM - 4] = _u32(k)


def build_worker_weights(p, packed_o, packed_g, packed_u, packed_d, packed_q, wgt2, wgt1):
    """Worker at ring position p: its 186-element A stream in fill order
    [o x16 | w2 | gate x48 | up x48 | down x48 | w1 | qkv x24]. The
    packer's column p tiles ARE position p's rows; gate/up/down get their
    flavor K headers patched in."""
    w = np.zeros(N_WELEM * ELEM, dtype=np.uint8)
    blk = ELEM

    def put(idx, src):
        w[idx * blk : (idx + 1) * blk] = src

    o0 = p * N_O * blk
    for i in range(N_O):
        put(i, packed_o[o0 + i * blk : o0 + (i + 1) * blk])
    put(N_O, _norm_element(wgt2, 101))
    g0 = p * N_GATE * blk
    for i in range(N_GATE):
        e = packed_g[g0 + i * blk : g0 + (i + 1) * blk].copy()
        _patch_k(e, 103)
        put(N_O + 1 + i, e)
    u0 = p * N_UP * blk
    for i in range(N_UP):
        e = packed_u[u0 + i * blk : u0 + (i + 1) * blk].copy()
        _patch_k(e, 104)
        put(N_O + 1 + N_GATE + i, e)
    d0 = p * N_DOWN * blk
    for i in range(N_DOWN):
        e = packed_d[d0 + i * blk : d0 + (i + 1) * blk].copy()
        _patch_k(e, 105)
        put(N_O + 1 + N_GATE + N_UP + i, e)
    put(N_O + 1 + N_GATE + N_UP + N_DOWN, _norm_element(wgt1, 102))
    q0 = p * N_QKV * blk
    for i in range(N_QKV):
        put(N_WELEM - N_QKV + i, packed_q[q0 + i * blk : q0 + (i + 1) * blk])
    return w


def build_x_element(q1, d1, worker_id):
    """K=0: attn int8 q at [0,2048), d bf16[64] at [6144,6400), worker id
    u32 at [6400,6404) (the kernel derives ring position from it)."""
    e = np.zeros(ELEM, dtype=np.uint8)
    e[0:2048] = q1.numpy().view(np.uint8)
    e[6144:6272] = d1.view(torch.uint16).numpy().view(np.uint8)
    e[6400:6404] = _u32(worker_id)
    e[ELEM - 8 : ELEM - 4] = _u32(0)
    return e


def build_xn_element(x_n_bf16, p):
    """K=100: this position's residual chunk (256 bf16) at [0,512)."""
    e = np.zeros(ELEM, dtype=np.uint8)
    chunk = x_n_bf16[p * 256 : (p + 1) * 256]
    e[0:512] = chunk.view(torch.uint16).numpy().view(np.uint8)
    e[ELEM - 8 : ELEM - 4] = _u32(100)
    return e


def generate_layerv2_reference(seed=42):
    torch.manual_seed(seed)
    W_o = (torch.rand(2048, 2048, dtype=torch.float32) * 2 - 1).numpy()
    W_g = (torch.rand(6144, 2048, dtype=torch.float32) * 2 - 1).numpy()
    W_u = (torch.rand(6144, 2048, dtype=torch.float32) * 2 - 1).numpy()
    W_d = (torch.rand(2048, 6144, dtype=torch.float32) * 2 - 1).numpy()
    W_q = (torch.rand(3072, 2048, dtype=torch.float32) * 2 - 1).numpy()
    x_attn = (torch.rand(2048, dtype=torch.float32) * 2 - 1).to(torch.bfloat16)
    x_n = (torch.rand(2048, dtype=torch.float32) * 2 - 1).to(torch.bfloat16)
    wgt2 = (torch.rand(2048, dtype=torch.float32) * 2 - 1).to(torch.bfloat16)
    wgt1 = (torch.rand(2048, dtype=torch.float32) * 2 - 1).to(torch.bfloat16)

    packed_o, _ = reference.quantize_and_pack(W_o)
    packed_g, _ = reference.quantize_and_pack(W_g)
    packed_u, _ = reference.quantize_and_pack(W_u)
    packed_d, _ = reference.quantize_and_pack(W_d)
    packed_q, _ = reference.quantize_and_pack(W_q)
    W_o_f, W_g_f = _deq_f32(W_o), _deq_f32(W_g)
    W_u_f, W_d_f, W_q_f = _deq_f32(W_u), _deq_f32(W_d), _deq_f32(W_q)
    q1, d1, x_deq1 = reference.quantize_vector(x_attn)

    # ---- o (attn arena from the X element) ----
    o_out = (W_o_f @ x_deq1).to(torch.bfloat16)

    # ---- ring1 gather + K=101 rms2 (x' = bf16(x_n + o)) ----
    xp_f = _f32(x_n.view(torch.uint16).numpy().astype(np.uint32)) + _f32(
        o_out.view(torch.uint16).numpy().astype(np.uint32)
    )
    xp_bits = _bf16(xp_f)
    xp_r = _f32(xp_bits.astype(np.uint32))
    inv1 = _kernel_rsqrt(
        np.float32(_sumsq_sequential(xp_r) / np.float32(2048) + np.float32(1e-5))
    )
    xn_f = xp_r * inv1 * _f32(wgt2.view(torch.uint16).numpy().astype(np.uint32))
    q2, d2_bits = _quantize_kernel(xn_f, 2048)
    x_deq2 = torch.from_numpy(_dequant(q2, d2_bits))

    # ---- gate/up (ACCURATE sigmoid below — kernel runs hw exp2) ----
    g_out = (W_g_f @ x_deq2).to(torch.bfloat16)
    u_out = (W_u_f @ x_deq2).to(torch.bfloat16)
    g_f = _f32(g_out.view(torch.uint16).numpy().astype(np.uint32))
    u_f = _f32(u_out.view(torch.uint16).numpy().astype(np.uint32))
    sig = np.float32(1.0) / (np.float32(1.0) + np.exp(-g_f.astype(np.float32)))
    sw_bits = _bf16(g_f * sig * u_f)
    q3, d3_bits = _quantize_kernel(_f32(sw_bits.astype(np.uint32)), 6144)
    x_deq3 = _dequant(q3, d3_bits)

    # ---- down: 3 chunk partials (bf16 each) accumulated f32 c-ascending ----
    W_df = W_d_f
    p_dn = torch.empty(3, 2048, dtype=torch.bfloat16)
    for c in range(3):
        p_dn[c] = (
            W_df[:, c * TILE_K : (c + 1) * TILE_K]
            @ torch.from_numpy(x_deq3[c * TILE_K : (c + 1) * TILE_K])
        ).to(torch.bfloat16)

    # ---- ring3 gather: xn1 = bf16(f32(x') + dacc) ----
    xn1_bits = _bf16(
        xp_r + _f32(p_dn.view(torch.uint16).numpy().astype(np.uint32)).sum(axis=0)
    )
    xn1_r = _f32(xn1_bits.astype(np.uint32))

    # ---- K=102 rms1 over xn1 -> qkv ----
    inv2 = _kernel_rsqrt(
        np.float32(_sumsq_sequential(xn1_r) / np.float32(2048) + np.float32(1e-5))
    )
    xn2_f = xn1_r * inv2 * _f32(wgt1.view(torch.uint16).numpy().astype(np.uint32))
    q4, d4_bits = _quantize_kernel(xn2_f, 2048)
    x_deq4 = torch.from_numpy(_dequant(q4, d4_bits))
    qkv = (W_q_f @ x_deq4).to(torch.bfloat16)

    # ---- assemble the tensors the exec consumes/produces ----
    weights = np.zeros(COLS * N_WELEM * ELEM, dtype=np.uint8)
    x_elems = np.zeros(COLS * ELEM, dtype=np.uint8)
    xn_elems = np.zeros(COLS * ELEM, dtype=np.uint8)
    out = torch.zeros(COLS * OUT_ROWS, dtype=torch.bfloat16)
    for w in range(COLS):
        p = pos(w)
        weights[w * N_WELEM * ELEM : (w + 1) * N_WELEM * ELEM] = build_worker_weights(
            p, packed_o, packed_g, packed_u, packed_d, packed_q, wgt2, wgt1
        )
        x_elems[w * ELEM : (w + 1) * ELEM] = build_x_element(q1, d1, w)
        xn_elems[w * ELEM : (w + 1) * ELEM] = build_xn_element(x_n, p)
        base = w * OUT_ROWS
        out[base : base + 384] = qkv[p * 384 : (p + 1) * 384]
        out[base + 384 : base + OUT_ROWS] = torch.from_numpy(
            xn1_bits[p * 256 : (p + 1) * 256].view(np.uint16)
        ).view(torch.bfloat16)

    return {
        "weights": weights,
        "x": x_elems,
        "xn": xn_elems,
        "output": out,
        "qkv": qkv,
        "xn1_bits": xn1_bits,
    }


class AIELayerV2(AIEOperatorBase):
    def __init__(self, cols=COLS, context=None):
        self.cols = cols
        AIEOperatorBase.__init__(self, context=context)

    def set_up_artifacts(self):
        operator_dir = Path(__file__).parent
        file_name_base = f"w4gemvu_layerv2_{self.cols}"
        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_base}.mlir",
            import_path=operator_dir / "design_layerv2.py",
            callback_fn="my_layerv2",
            callback_args=[
                self.context.device_manager.device_type,
                self.cols,
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
        self.add_buffer("weights", self.cols * N_WELEM * ELEM, dtype=np.uint8)
        self.add_buffer("x", self.cols * ELEM, dtype=np.uint8)
        self.add_buffer("xn", self.cols * ELEM, dtype=np.uint8)
        self.add_buffer("output", self.cols * OUT_ROWS, dtype=bfloat16)
        self.add_kernel(
            "w4gemvu_layerv2",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        # BO order must match design_layerv2's rt.sequence(W, X, XN, C).
        self.add_to_runlist("w4gemvu_layerv2", "weights", "x", "xn", "output")


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
)
def test_layerv2(aie_context):
    golden = generate_layerv2_reference()

    operator = AIELayerV2(context=aie_context)

    input_buffers = {
        "weights": torch.from_numpy(golden["weights"]),
        "x": torch.from_numpy(golden["x"]),
        "xn": torch.from_numpy(golden["xn"]),
    }
    output_buffers = {"output": golden["output"]}

    errors, latency_us, _ = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.08, abs_tol=0.8
    )

    # xn1 rows (the +256..640 slice of each worker's 640-row drain)
    # carry the hw-sigmoid deviation through the down partials — same
    # phase-1 band as test_quad: rel 0.08 + abs 200 at term scale; the
    # qkv rows stay strict (rms1 renormalization absorbs the common
    # mode, measured <= 0.75 absolute in P19b).
    LOOSE_REL, LOOSE_ABS = 0.08, 200.0
    if errors:
        act_bits = operator.read_buffer(
            "output", (COLS * OUT_ROWS,), dtype=np.uint16
        )
        exp_bits = golden["output"].view(torch.uint16).numpy()
        af, ef = _f32(act_bits), _f32(exp_bits)
        strict_fail = []
        for r in errors["output"]:
            in_xn1 = (r % OUT_ROWS) >= 384
            if in_xn1 and abs(af[r] - ef[r]) <= max(LOOSE_REL * abs(ef[r]), LOOSE_ABS):
                continue
            strict_fail.append(r)
        errors = {"output": strict_fail} if strict_fail else {}

    mb = (len(golden["weights"]) + 2 * len(golden["x"])) / 1e6
    print(
        f"\n[layerv2 whole-layer] Latency (us): {latency_us:.1f}, "
        f"{mb / latency_us * 1e3:.2f} GB/s weights"
    )

    assert not errors, f"Test failed with errors: {errors}"
