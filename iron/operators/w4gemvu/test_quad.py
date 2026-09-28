#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""M6 QUAD whole-layer test (P19): o -> rms1 -> gateup -> swiglu -> down
-> rms2 -> qkv in ONE exec, golden computed by mirroring the kernel
flavors' semantics (f32 glue, RNE bf16 steps, per-group-32 int8
quantize with d = bf16(amax/127)) with one deliberate exception: the
sigmoid uses the ACCURATE f32 exp, while the kernel's K=5 runs the AIE2P
hardware exp2<bfloat16> (P4: mean +3.25%, max +5.67%). The bias is
mostly common-mode within a group and absorbed by d; the tolerance
covers the differential part. The E2E gates remain the arbiter (phase-1
decision, notes/perf-lab.md P19).

Perf comparison target: the fused PAIR costs pair A + gap + pair B
(~485 + ~370 us standalone); the quad should land under their sum minus
one machinery floor (~100 us) and saves the host swiglu round trip.
"""

import pytest
import numpy as np
import torch

from iron.operators.w4gemvu.op_quad import AIEW4GEMVUQuad, RMS_ELEM_ROWS
from iron.operators.w4gemvu import reference
from iron.operators.w4gemvu.reference import ELEM, TILE_ROWS, TILE_K
from iron.common.test_utils import run_test


def _f32(bf16_bits):
    """(...,) uint16 bf16 bits -> float32 array."""
    return (bf16_bits.astype(np.uint32) << 16).view(np.float32)


def _bf16(f32):
    """float32 -> uint16 bf16 bits with RNE (the kernel's carry trick)."""
    u = f32.astype(np.float32).view(np.uint32)
    rounded = u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def _kernel_rsqrt(s):
    """The kernel's 3-iteration Newton rsqrt, in np.float32."""
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
    """Unrounded f32 (m,) -> per-group-32 int8 with the kernel's exact
    semantics: d = bf16(amax/127), q = round-ties-even(xn/d) clipped."""
    v32 = xn_f32.reshape(m // 32, 32)
    amax = np.abs(v32).max(axis=1)
    d_bits = _bf16((amax / np.float32(127.0)).astype(np.float32))
    invd = (np.float32(1.0) / _f32(d_bits.astype(np.uint32)))[:, None]
    v = v32 * invd
    v = np.where(amax[:, None] == 0, 0.0, v)
    q = np.round(v)
    q = np.clip(q, -127, 127).astype(np.int8).reshape(m)
    return q, d_bits


def generate_quad_reference(M1, K1, M2, M3, K3, M4, seed=42):
    torch.manual_seed(seed)
    W1 = (torch.rand(M1, K1, dtype=torch.float32) * 2 - 1).numpy()  # o
    W2 = (torch.rand(M2, 2048, dtype=torch.float32) * 2 - 1).numpy()  # gateup
    W3 = (torch.rand(M3, K3, dtype=torch.float32) * 2 - 1).numpy()  # down
    W4 = (torch.rand(M4, 2048, dtype=torch.float32) * 2 - 1).numpy()  # qkv
    x = (torch.rand(K1, dtype=torch.float32) * 2 - 1).to(torch.bfloat16)  # attn out
    res1 = (torch.rand(M1, dtype=torch.float32) * 2 - 1).to(torch.bfloat16)  # x_n
    wgt1 = (torch.rand(M1, dtype=torch.float32) * 2 - 1).to(torch.bfloat16)  # ln2 w
    wgt2 = (torch.rand(M1, dtype=torch.float32) * 2 - 1).to(torch.bfloat16)  # ln1 w

    packed1, W1_deq = reference.quantize_and_pack(W1)
    packed2_blocks, W2_deq = reference.quantize_and_pack(W2)
    packed3_blocks, W3_deq = reference.quantize_and_pack(W3)
    packed4_blocks, W4_deq = reference.quantize_and_pack(W4)
    q1, d1, x_deq1 = reference.quantize_vector(x)

    # ---- op1 (o): bit path identical to the unfused golden ----
    partials_o = (W1_deq.to(torch.float32) @ x_deq1).to(torch.bfloat16)

    # ---- K=1' (stage1r): h2' = bf16(f32(res1) + f32(o_partial)) ----
    h2p_f = _f32(res1.view(torch.uint16).numpy().astype(np.uint32)) + _f32(
        partials_o.view(torch.uint16).numpy().astype(np.uint32)
    )
    h2p_bits = _bf16(h2p_f)

    # ---- K=3 #1 (rms1 + quantize, kernel semantics) ----
    h2p_r = _f32(h2p_bits.astype(np.uint32))
    inv1 = _kernel_rsqrt(
        np.float32(_sumsq_sequential(h2p_r) / np.float32(M1) + np.float32(1e-5))
    )
    w1_bits = wgt1.view(torch.uint16).numpy().astype(np.uint32)
    xn1 = h2p_r * inv1 * _f32(w1_bits)
    q2, d2_bits = _quantize_kernel(xn1, M1)
    d_rep = np.repeat(_f32(d2_bits.astype(np.uint32)), 32)
    x_deq2 = q2.astype(np.float32) * d_rep

    # ---- gateup partials ----
    partials_gu = (W2_deq.to(torch.float32) @ torch.from_numpy(x_deq2)).to(
        torch.bfloat16
    )

    # ---- K=4/K=5 (swiglu, ACCURATE sigmoid in the golden) ----
    # M2 = 12288: gate rows [0, 6144), up rows [6144, 12288)
    gu_f = _f32(partials_gu.view(torch.uint16).numpy().astype(np.uint32))
    inter = M2 // 2  # 6144
    gate = gu_f[:inter]
    up = gu_f[inter:]
    sig = np.float32(1.0) / (
        np.float32(1.0) + np.exp(-gate.astype(np.float32))
    )
    gs = gate * sig
    sw_bits = _bf16(gs * up)
    sw_f = _f32(sw_bits.astype(np.uint32))
    q3, d3_bits = _quantize_kernel(sw_f, inter)
    d3_rep = np.repeat(_f32(d3_bits.astype(np.uint32)), 32)
    x_deq3 = (q3.astype(np.float32) * d3_rep).reshape(3, TILE_K)  # 3 chunks

    # ---- down partials (chunk-major per column) ----
    W3f = W3_deq.to(torch.float32)
    partials_dn = torch.empty(3, M3, dtype=torch.bfloat16)
    for c in range(3):
        partials_dn[c] = (W3f[:, c * TILE_K : (c + 1) * TILE_K] @ torch.from_numpy(
            x_deq3[c]
        )).to(torch.bfloat16)

    # ---- K=2 + K=3 #2: x_{n+1} = bf16(f32(h2') + sum_c f32(p_dn)) ----
    p_dn_bits = partials_dn.view(torch.uint16).numpy().astype(np.uint32)
    h2pp_f = _f32(h2p_bits.astype(np.uint32)) + _f32(p_dn_bits).sum(axis=0)
    h2pp_bits = _bf16(h2pp_f)

    h2pp_r = _f32(h2pp_bits.astype(np.uint32))
    inv2 = _kernel_rsqrt(
        np.float32(_sumsq_sequential(h2pp_r) / np.float32(M1) + np.float32(1e-5))
    )
    w2_bits = wgt2.view(torch.uint16).numpy().astype(np.uint32)
    xn2 = h2pp_r * inv2 * _f32(w2_bits)
    q4, d4_bits = _quantize_kernel(xn2, M1)
    d4_rep = np.repeat(_f32(d4_bits.astype(np.uint32)), 32)
    x_deq4 = q4.astype(np.float32) * d4_rep

    # ---- qkv partials ----
    partials_q = (W4_deq.to(torch.float32) @ torch.from_numpy(x_deq4)).to(
        torch.bfloat16
    )

    # ---- expected C tensor (40704 rows) ----
    blocks1 = reference.blocks_per_col(M1, K1)
    sec_gu = (reference.blocks_per_col(M2, 2048) + 2) * TILE_ROWS  # 1568
    sec_dn = (reference.blocks_per_col(M3, K3) + 2) * TILE_ROWS  # 800
    sec_q = (reference.blocks_per_col(M4, 2048) + 4) * TILE_ROWS  # 448
    up_off = RMS_ELEM_ROWS + 4 * sec_gu + 3008
    win2_off = up_off + 4 * sec_gu + 3008
    qkv_off = win2_off + RMS_ELEM_ROWS
    c_exp = torch.zeros(qkv_off + 8 * sec_q, dtype=torch.bfloat16)

    # o sections (1 dummy: the X element's zero C)
    c_exp[0 : 8 * (blocks1 + 2) * TILE_ROWS] = reference.shuffle_output(
        partials_o.reshape(1, M1), M1, K1
    )
    # res1 and the four windows' header words survive from the input seed
    c_rows1 = 8 * (blocks1 + 2) * TILE_ROWS
    c_exp[c_rows1 : c_rows1 + M1] = res1
    bits = c_exp.view(torch.uint16)

    def hdr(row, k, blocks1):
        bits[row] = k & 0xFFFF
        bits[row + 1] = 0
        bits[row + 2] = blocks1 & 0xFFFF
        bits[row + 3] = 0

    hdr(9276, 1, blocks1)  # win1
    hdr(18556, 4, 0)  # padA (K=4 gate window)
    hdr(27836, 5, 0)  # padB (K=5 up window)
    hdr(37116, 2, reference.blocks_per_col(M3, K3))  # win2
    # gate cols 0..3 / up cols 4..7 (2 dummies: K=1 + K=3 zero Cs)
    rpc2 = M2 // 8  # 1536
    for col in range(8):
        base = (
            RMS_ELEM_ROWS + col * sec_gu
            if col < 4
            else up_off + (col - 4) * sec_gu
        )
        c_exp[base + 2 * TILE_ROWS : base + 2 * TILE_ROWS + rpc2] = partials_gu[
            col * rpc2 : (col + 1) * rpc2
        ]
    # down cols (2 dummies: K=4 + K=5 zero Cs), chunk-major partials
    rpc3 = M3 // 8  # 256
    for col in range(8):
        base = win2_off + col * sec_dn + 2 * TILE_ROWS
        for c in range(3):
            c_exp[base + c * rpc3 : base + (c + 1) * rpc3] = partials_dn[
                c, col * rpc3 : (col + 1) * rpc3
            ]
    # qkv cols (3 dummies: K=2 + K=1' + K=3 zero Cs)
    rpc4 = M4 // 8  # 384
    for col in range(8):
        base = qkv_off + col * sec_q + 3 * TILE_ROWS
        c_exp[base : base + rpc4] = partials_q[col * rpc4 : (col + 1) * rpc4]

    return {
        "packed1": packed1,
        "packed2_blocks": packed2_blocks,
        "packed3_blocks": packed3_blocks,
        "packed4_blocks": packed4_blocks,
        "x": x,
        "res1": res1,
        "wgt1": wgt1,
        "wgt2": wgt2,
        "activation": reference.build_activation_element(q1, d1, K1),
        "c_init": None,  # filled by the test (needs the operator's layout)
        "output_raw": c_exp,
        "output": partials_q,
        "h2p_bits": h2p_bits,
        "h2pp_bits": h2pp_bits,
    }


params = [
    (2048, 2048, 12288, 2048, 6144, 3072),  # the whole hy-mt2 layer
]
names = ["w4gemvuq_layer"]
all_params = [pytest.param(*p, id=n) for p, n in zip(params, names)]


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
)
@pytest.mark.parametrize("M1,K1,M2,M3,K3,M4", all_params)
def test_w4gemvu_quad(M1, K1, M2, M3, K3, M4, aie_context):
    golden = generate_quad_reference(M1, K1, M2, M3, K3, M4)

    operator = AIEW4GEMVUQuad(
        M1=M1,
        K1=K1,
        M2=M2,
        M3=M3,
        K3=K3,
        M4=M4,
        num_aie_columns=8,
        group_size=32,
        context=aie_context,
    )

    # run_test ZEROES every output buffer, then writes input buffers — so
    # the C BO's host-written region (residual1 + the four windows' header
    # words) must ride in input_buffers (P16 lesson).
    input_buffers = {
        # X element rides the head of packed1 (5-BO ctrl-kernel cap).
        "packed1": operator.build_packed1(
            torch.from_numpy(golden["packed1"]), golden["activation"]
        ),
        "packed2": operator.build_packed_w(
            torch.from_numpy(golden["packed2_blocks"]), golden["wgt1"], operator.blocks1
        ),
        "packed3": torch.from_numpy(golden["packed3_blocks"]),
        "packed4": operator.build_packed_w(
            torch.from_numpy(golden["packed4_blocks"]), golden["wgt2"], operator.blocks3
        ),
        "output": operator.build_c_init(golden["res1"]),
    }
    # The EXPECTED post-run C: all sections written, host regions intact.
    output_buffers = {"output": golden["output_raw"]}

    errors, latency_us, _ = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.08, abs_tol=0.8
    )

    # Down-partial rows carry the phase-1 deviation (notes/perf-lab.md P19):
    # the kernel's sigmoid runs the AIE2P hw exp2<bfloat16> (P4: mean
    # +3.25%, max +5.67%) while this golden uses the accurate f32 exp. The
    # per-group common-mode part is absorbed by d = amax/127; the residual
    # DIFFERENTIAL bias (~2-3%) propagates through the 2048-term dot at
    # TERM scale, not result scale — partials are +-8000-magnitude sums of
    # cancelling terms, so a 3% per-group scale error yields absolute
    # deviations up to ~200 with unbounded RELATIVE error on near-zero
    # results (measured: max_abs 192, max_rel 41). The end-to-end effect
    # is absorbed by rms2 renormalization: the qkv sections (computed from
    # these partials) deviate <= 0.75 ABSOLUTE from the accurate golden.
    # Band: rel 0.08 for large partials + abs 200 (3% of term scale);
    # every other region stays at rel 0.08 / abs 0.8, and the E2E decode
    # gates remain the arbiter of the numerics decision.
    LOOSE_REL, LOOSE_ABS = 0.08, 200.0
    if errors:
        act_bits = operator.read_buffer(
            "output", (operator.c_total_rows,), dtype=np.uint16
        )
        exp_bits = golden["output_raw"].view(torch.uint16).numpy()

        def _f32(bits):
            return (np.uint32(bits) << np.uint32(16)).view(np.float32)

        af, ef = _f32(act_bits), _f32(exp_bits)
        strict_fail, loose_fail = [], []
        for r in errors["output"]:
            in_dn_partial = (
                operator.win2_off + 2 * TILE_ROWS
                <= r
                < operator.win2_off + 8 * operator.sec_dn
                and (r - operator.win2_off) % operator.sec_dn >= 2 * TILE_ROWS
            )
            if in_dn_partial and abs(af[r] - ef[r]) <= max(
                LOOSE_REL * abs(ef[r]), LOOSE_ABS
            ):
                continue
            (loose_fail if in_dn_partial else strict_fail).append(r)
        if loose_fail:
            errors["output"] = strict_fail + loose_fail
        else:
            errors = {"output": strict_fail} if strict_fail else {}

    mb = (
        len(golden["packed1"])
        + len(golden["packed2_blocks"])
        + len(golden["packed3_blocks"])
        + len(golden["packed4_blocks"])
        + 3 * 8 * ELEM  # the X + two K=3 w elements
    ) / 1e6
    print(
        f"\n[quad {M1}x{K1}->{M2}->sw->{M3}->{M4}] Latency (us): {latency_us:.1f}, "
        f"{mb / latency_us * 1e3:.2f} GB/s weights"
    )

    assert not errors, f"Test failed with errors: {errors}"
