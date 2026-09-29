#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""P28 layer-v2 test: the WHOLE transformer layer in ONE task group per
exec on N persistent ring workers (design_layerv2.py + w4gemvu_layer.cc).

P28-6: N is a test PARAMETER (8/16/32) -- the kernel reads the worker
count from the X element ([6404,6408) u32) so one .o serves every width.
N=8 is the regression anchor (expected bit-identical to P28-4); 16/32
light up the widened ring.

The golden mirrors the kernel's flavor chain end to end (the P16/P19
bit-path mirrors, reused from test_quad): o -> ring1 gather x' -> rms2 ->
gate/up -> swiglu -> ring2 gather sw -> down -> ring3 gather xn1 -> rms1
-> qkv, with the drain tensor carrying per-worker [qkv 3072/N | xn1
2048/N] rows that the test reassembles by ring position.

Weight elements: the standard packer tiles, except gate/up/down blocks
carry K=103/104/105 headers (the packer writes 2048 -- the fixture
patches them) so the kernel's dispatcher routes each block to its
flavor; o/qkv blocks keep 2048 and ride the phase word.

SUB-COLUMN SLICING LAW (P28-6): the v4 packs are 8-column (a column
spans M/8 rows), so a position's row range only coincides with a whole
column when N=8. For N=16/32 the position's 16-row blocks are the
t-subrange [t0, t0+span/16) of pack column col = r0 // (M/8), per chunk
for K=6144 matrices -- sliced explicitly, never as a flat p*count run
(which silently scrambles gate/up/down for N>8; a position's span
always divides the per-column row count, so runs never straddle).
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
from iron.operators.w4gemvu.design_layerv2 import ring_tables
from iron.operators.w4gemvu.reference import ELEM, TILE_K

HIDDEN, INTER, QKV_M = 2048, 6144, 3072


def geom(n):
    """Per-N geometry mirror of design_layerv2.my_layerv2."""
    rows = HIDDEN // n
    N_O = rows // 16
    N_GATE = (INTER // n) // 16
    N_QKV = (QKV_M // n) // 16
    g = {
        "n": n, "rows": rows, "jpw": INTER // n,
        "N_O": N_O, "N_GATE": N_GATE,
        "N_UP": N_GATE, "N_DOWN": 3 * N_O, "N_QKV": N_QKV, "N_CXN": N_O,
        "qkv_rows": QKV_M // n,
        "N_WELEM": N_O + N_GATE + N_GATE + 3 * N_O + N_QKV + 2,
    }
    # ring order straight from the design (serpentine N=8, all-adjacent
    # Hamiltonian cycle N=16 -- the CORE<->CORE OBJECTFIFO LAW in
    # design_layerv2.ring_tables; one source of truth, no mirror to rot)
    order, _ = ring_tables(n)
    g["order"] = order
    g["pos"] = order.index  # worker id -> ring position
    return g


G8 = geom(8)  # default geometry for module-level helpers


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


def _sub_blocks(packed, M, p, rows_per_pos, chunks=1, chunk=0):
    """Position p's 16-row blocks of an 8-column (M, 2048) pack (or of
    chunk `chunk` of an (M, 6144) pack), where a position owns
    rows_per_pos rows: the contiguous t-run [t0, t0+rows_per_pos/16) of
    pack column col -- see the SUB-COLUMN SLICING LAW above. A column
    spans M/8 ROWS no matter the K-chunking (K=6144 tiles are stored 3
    chunk-major blocks per 16 rows, empirically pinned: block b -> col
    b//48, chunk (b%48)//16, tile (b%48)%16), so rows_per_col is NOT
    divided by chunks -- chunks multiplies the block stride, not the row
    span. Never straddles a column (rows_per_pos divides rows_per_col
    for every matrix here). Returns the byte slice."""
    rows_per_col = M // 8
    r0 = p * rows_per_pos
    col, t0 = r0 // rows_per_col, (r0 % rows_per_col) // 16
    T = rows_per_col // 16
    base = (col * chunks + chunk) * T + t0
    cnt = rows_per_pos // 16
    return packed[base * ELEM : (base + cnt) * ELEM], cnt


def build_worker_weights(g, p, packed_o, packed_g, packed_u, packed_d, packed_q, wgt2, wgt1):
    """Worker at ring position p: its N_WELEM-element A stream in fill
    order [o xN_O | w2 | gate xN_GATE | up xN_UP | down xN_DOWN | w1 |
    qkv xN_QKV]. All matrices are sliced by the pack sub-column law."""
    rows, jpw, N_O, N_GATE = g["rows"], g["jpw"], g["N_O"], g["N_GATE"]
    w = np.zeros(g["N_WELEM"] * ELEM, dtype=np.uint8)
    blk = ELEM

    def put(idx, src):
        w[idx * blk : (idx + 1) * blk] = src

    run, _ = _sub_blocks(packed_o, 2048, p, rows)
    for i in range(N_O):
        put(i, run[i * blk : (i + 1) * blk])
    put(N_O, _norm_element(wgt2, 101))
    run, _ = _sub_blocks(packed_g, 6144, p, jpw)
    for i in range(N_GATE):
        e = run[i * blk : (i + 1) * blk].copy()
        _patch_k(e, 103)
        put(N_O + 1 + i, e)
    run, _ = _sub_blocks(packed_u, 6144, p, jpw)
    for i in range(N_GATE):
        e = run[i * blk : (i + 1) * blk].copy()
        _patch_k(e, 104)
        put(N_O + 1 + N_GATE + i, e)
    # down: c-major chunk blocks, each chunk's run sliced by the same law
    N_DOWN = g["N_DOWN"]
    for c in range(3):
        run, _ = _sub_blocks(packed_d, 2048, p, rows, chunks=3, chunk=c)
        for i in range(N_O):
            e = run[i * blk : (i + 1) * blk].copy()
            _patch_k(e, 105)
            put(N_O + 1 + 2 * N_GATE + c * N_O + i, e)
    put(N_O + 1 + 2 * N_GATE + N_DOWN, _norm_element(wgt1, 102))
    run, _ = _sub_blocks(packed_q, 3072, p, g["qkv_rows"])
    N_QKV = g["N_QKV"]
    for i in range(N_QKV):
        put(g["N_WELEM"] - N_QKV + i, run[i * blk : (i + 1) * blk])
    return w


def build_x_element(q1, d1, worker_id, n):
    """K=0: attn int8 q at [0,2048), d bf16[64] at [6144,6400), worker id
    u32 at [6400,6404), worker count N u32 at [6404,6408) (the kernel
    derives ring position + all geometry from these two words)."""
    e = np.zeros(ELEM, dtype=np.uint8)
    e[0:2048] = q1.numpy().view(np.uint8)
    e[6144:6272] = d1.view(torch.uint16).numpy().view(np.uint8)
    e[6400:6404] = _u32(worker_id)
    e[6404:6408] = _u32(n)
    e[ELEM - 8 : ELEM - 4] = _u32(0)
    return e


def build_xn_element(x_n_bf16, p, rows):
    """K=100: this position's residual chunk (rows bf16) at [0,rows*2)."""
    e = np.zeros(ELEM, dtype=np.uint8)
    chunk = x_n_bf16[p * rows : (p + 1) * rows]
    e[0 : rows * 2] = chunk.view(torch.uint16).numpy().view(np.uint8)
    e[ELEM - 8 : ELEM - 4] = _u32(100)
    return e


def generate_layerv2_reference(seed=42, n=8):
    g = geom(n)
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
    rows, qkv_rows = g["rows"], g["qkv_rows"]
    out_rows = (g["N_QKV"] + g["N_CXN"]) * 16  # qkv_rows + rows
    weights = np.zeros(n * g["N_WELEM"] * ELEM, dtype=np.uint8)
    x_elems = np.zeros(n * ELEM, dtype=np.uint8)
    xn_elems = np.zeros(n * ELEM, dtype=np.uint8)
    out = torch.zeros(n * out_rows, dtype=torch.bfloat16)
    for w in range(n):
        p = g["pos"](w)
        weights[w * g["N_WELEM"] * ELEM : (w + 1) * g["N_WELEM"] * ELEM] = (
            build_worker_weights(g, p, packed_o, packed_g, packed_u, packed_d,
                                 packed_q, wgt2, wgt1))
        x_elems[w * ELEM : (w + 1) * ELEM] = build_x_element(q1, d1, w, n)
        xn_elems[w * ELEM : (w + 1) * ELEM] = build_xn_element(x_n, p, rows)
        base = w * out_rows
        out[base : base + qkv_rows] = qkv[p * qkv_rows : (p + 1) * qkv_rows]
        out[base + qkv_rows : base + out_rows] = torch.from_numpy(
            xn1_bits[p * rows : (p + 1) * rows].view(np.uint16)
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
    def __init__(self, cols=G8["n"], context=None):
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
        g = geom(self.cols)
        out_rows = (g["N_QKV"] + g["N_CXN"]) * 16
        self.add_buffer("weights", self.cols * g["N_WELEM"] * ELEM, dtype=np.uint8)
        self.add_buffer("x", self.cols * ELEM, dtype=np.uint8)
        self.add_buffer("xn", self.cols * ELEM, dtype=np.uint8)
        self.add_buffer("output", self.cols * out_rows, dtype=bfloat16)
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
# SHIM CHANNEL LAW (P28-6): each shim tile carries 2 MM2S + 2 S2MM
# channels and npu2 has 8 shims -> the whole device can host at most 16
# inbound + 16 outbound streams. One A + one C fifo per worker means
# N=16 EXACTLY saturates the channel budget (SequentialPlacer: every
# shim, both channels, both directions) and N=32 cannot place at all
# (ValueError "no tile matching column 5" -- shims exhausted). It also
# has no upside: N=8 already runs ~4.3 GB/s per channel = 33.8 GB/s
# aggregate, consume-limited BELOW the ~55 GB/s DDR wall, so N=16 is the
# width that reaches the wall (16 x 4.3 = 69 -> capped ~55). Reaching
# N=32 would need link-split stream sharing -- same aggregate, pointless.
@pytest.mark.parametrize("workers", [8, 16])
def test_layerv2(aie_context, workers):
    golden = generate_layerv2_reference(n=workers)
    g = geom(workers)
    out_rows = (g["N_QKV"] + g["N_CXN"]) * 16

    operator = AIELayerV2(cols=workers, context=aie_context)

    input_buffers = {
        "weights": torch.from_numpy(golden["weights"]),
        "x": torch.from_numpy(golden["x"]),
        "xn": torch.from_numpy(golden["xn"]),
    }
    output_buffers = {"output": golden["output"]}

    errors, latency_us, _ = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.08, abs_tol=0.8
    )

    # xn1 rows (the trailing rows slice of each worker's drain) carry the
    # hw-sigmoid deviation through the down partials — same phase-1 band
    # as test_quad: rel 0.08 + abs 200 at term scale; the qkv rows stay
    # strict (rms1 renormalization absorbs the common mode, measured
    # <= 0.75 absolute in P19b).
    LOOSE_REL, LOOSE_ABS = 0.08, 200.0
    if errors:
        act_bits = operator.read_buffer(
            "output", (workers * out_rows,), dtype=np.uint16
        )
        exp_bits = golden["output"].view(torch.uint16).numpy()
        af, ef = _f32(act_bits), _f32(exp_bits)
        strict_fail = []
        for r in errors["output"]:
            in_xn1 = (r % out_rows) >= g["qkv_rows"]
            if in_xn1 and abs(af[r] - ef[r]) <= max(LOOSE_REL * abs(ef[r]), LOOSE_ABS):
                continue
            strict_fail.append(r)
        errors = {"output": strict_fail} if strict_fail else {}

    mb = (len(golden["weights"]) + 2 * len(golden["x"])) / 1e6
    print(
        f"\n[layerv2 whole-layer w{workers}] Latency (us): {latency_us:.1f}, "
        f"{mb / latency_us * 1e3:.2f} GB/s weights"
    )

    assert not errors, f"Test failed with errors: {errors}"
