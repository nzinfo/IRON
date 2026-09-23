# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from pathlib import Path

from iron.common import (
    AIEOperatorBase,
    AIEOperatorConstraintError,
    XclbinArtifact,
    InstsBinArtifact,
    KernelObjectArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
)
from iron.operators.flowkv_decode.reference import interleave_kv_cache


def pack_q_with_angles(q, angles, group_size, num_kv_heads, seq_len_cur=None):
    """Pack Q vectors and RoPE angles into the DDR layout for DMA.

    DDR layout: [Q_group0 (gs*hd) | angles (hd) | hdr(16) | Q_group1 ...]
    The 16-element header rides at the END of each group block: the kernel's
    RoPE base pointers must stay identical to the upstream (headerless)
    layout or peano -O2 drops the constant offset from the angles load base
    (streaming-pointer anchor bug, notes §13/§16). The header carries the
    runtime sequence length S as a u32 split across the first two bf16 bit
    patterns (must match FLOWKV_Q_HDR_ELEMS in aie_kernels/aie2p/flowkv.cc
    and Q_HDR_ELEMS in design.py).

    Args:
        q:       Query vectors, shape (num_heads, head_dim) in bf16.
        angles:  RoPE angles, shape (head_dim,) in bf16.
                 Interleaved [cos0, sin0, cos1, sin1, ...].
        group_size: Number of query heads per KV group.
        num_kv_heads: Number of KV heads.
        seq_len_cur: Live cache positions S (1 <= S <= compiled seq_len).
                     None means S = the full compiled seq_len. Cache rows at
                     positions >= S are neutralized by the kernel (they still
                     stream; V rows there must be finite, zero-fill is fine).

    Returns:
        Packed 1D tensor for the Q DDR buffer.
    """
    head_dim = q.shape[1]
    hdr = torch.zeros(16, dtype=torch.bfloat16)
    if seq_len_cur is not None:
        hv = hdr.view(torch.uint16)
        hv[0] = seq_len_cur & 0xFFFF
        hv[1] = seq_len_cur >> 16
    chunks = []
    for kv_h in range(num_kv_heads):
        start = kv_h * group_size
        end = start + group_size
        chunks.append(q[start:end].reshape(-1))
        chunks.append(angles)
        chunks.append(hdr)
    return torch.cat(chunks)


class AIEFlowKVDecode(AIEOperatorBase):
    """AIE-accelerated FlowKV decode attention operator with fused RoPE.

    Implements streaming decode attention with online softmax using a 2-tile
    pipeline per KV head group. RoPE is applied to Q in-register on the score
    tile before computing attention scores, eliminating a separate RoPE
    operator invocation. K in the cache is assumed to be already rotated.

    Computes for each query head h:
        O[h] = softmax(RoPE(Q[h]) @ K[kv_h]^T / sqrt(d)) @ V[kv_h]

    where kv_h = h // group_size is the corresponding KV head index.

    This implements exact FlashAttention semantics via online softmax in a
    single streaming pass over the KV cache. The K and V caches are streamed
    in chunks, with score computation and value accumulation pipelined across
    two tiles per KV head group.

    DDR buffer layout:
        KV cache:  interleaved K and V rows per head per position.
                   Shape: (num_kv_heads, seq_len, 2, head_dim) flattened.
        Q:         query heads + RoPE angles packed per KV group.
                   Layout: [Q_group0 (gs*hd) | angles (hd) | Q_group1 ...].
        Output:    attention output. Shape: (num_heads, head_dim) flattened.

    Use `interleave_kv_cache(k_cache, v_cache)` from the reference module to
    create the interleaved DDR layout.
    """

    def __init__(
        self,
        num_heads,
        num_kv_heads,
        head_dim,
        seq_len,
        chunk_size=32,
        num_cols=4,
        context=None,
    ):
        assert (
            num_heads % num_kv_heads == 0
        ), "num_heads must be divisible by num_kv_heads"
        assert seq_len % chunk_size == 0, "seq_len must be divisible by chunk_size"
        assert (
            num_kv_heads % num_cols == 0
        ), "num_kv_heads must be divisible by num_cols"
        assert head_dim in (64, 128), "Only head_dim=64 and head_dim=128 are supported"
        assert (
            num_heads // num_kv_heads <= 16 and head_dim <= 128
        ), "kernel static buffers cap group_size at 16 and head_dim at 128"

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.num_cols = num_cols
        self.group_size = num_heads // num_kv_heads

        self.xclbin_artifact = None
        self.insts_artifact = None

        AIEOperatorBase.__init__(self, context=context)

    def set_up_artifacts(self):
        operator_dir = Path(__file__).parent
        file_name_base = (
            f"flowkv_decode_{self.num_heads}h_{self.num_kv_heads}kv_"
            f"{self.head_dim}d_{self.seq_len}s_{self.chunk_size}cs_"
            f"{self.num_cols}col"
        )

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_flowkv_decode",
            callback_args=[
                self.context.device_manager.device_type,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.seq_len,
                self.chunk_size,
                self.num_cols,
            ],
        )

        xclbin_artifact = XclbinArtifact.new(
            f"{file_name_base}.xclbin",
            depends=[
                mlir_artifact,
                KernelObjectArtifact.new(
                    "flowkv.o",
                    depends=[
                        SourceArtifact.new(
                            self.context.base_dir
                            / "aie_kernels"
                            / "aie2p"
                            / "flowkv.cc"
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
        # KV cache buffer: interleaved K and V. seq_len here is the compiled
        # cache CAPACITY: the runtime live length S rides in the Q header.
        kv_size = self.num_kv_heads * self.seq_len * 2 * self.head_dim
        self.add_buffer("kv_cache", kv_size)

        # Q buffer: query heads + RoPE angles + header packed per KV group
        # Layout: [Q_group0 (gs*hd) | angles (hd) | hdr(16) | Q_group1 ...]
        q_group_stride = self.group_size * self.head_dim + self.head_dim + 16
        q_size = self.num_kv_heads * q_group_stride
        self.add_buffer("queries", q_size)

        # Output buffer: attention result
        o_size = self.num_heads * self.head_dim
        self.add_buffer("output", o_size)

        self.add_kernel(
            "flowkv_decode",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_to_runlist("flowkv_decode", "kv_cache", "queries", "output")

    def forward(self, q, k_cache, v_cache, q_angles, seq_len_cur=None):
        """Run FlowKV decode attention with fused RoPE on Q.

        RoPE is applied to Q in-register on the score tile before computing
        attention scores. K in the cache is assumed to be already rotated
        (standard practice: K is rotated before being stored in the KV cache).

        Args:
            q:        Unrotated query vectors, shape (num_heads, head_dim) bf16.
            k_cache:  K cache (already rotated), shape
                      (num_kv_heads, seq_len, head_dim) in bf16 — seq_len is
                      the compiled cache capacity.
            v_cache:  V cache, shape (num_kv_heads, seq_len, head_dim) in bf16.
                      Rows at positions >= seq_len_cur must be finite
                      (zero-fill): they still stream and get multiplied by
                      F_c = 0.
            q_angles: RoPE angles for the current decode position, shape
                      (head_dim,) in bf16.  Interleaved [cos0, sin0, cos1, ...].
            seq_len_cur: Live cache positions S (1 <= S <= seq_len). None
                      means the full compiled seq_len. Rides in the Q header.

        Returns:
            Attention output, shape (num_heads, head_dim) in bf16.
        """
        if seq_len_cur is None:
            seq_len_cur = self.seq_len
        if not 1 <= seq_len_cur <= self.seq_len:
            raise AIEOperatorConstraintError(
                f"seq_len_cur must be in [1, {self.seq_len}], got {seq_len_cur}"
            )
        # Validate shapes
        if q.shape != (self.num_heads, self.head_dim):
            raise AIEOperatorConstraintError(
                f"Expected Q shape ({self.num_heads}, {self.head_dim}), "
                f"got {q.shape}"
            )
        if k_cache.shape != (
            self.num_kv_heads,
            self.seq_len,
            self.head_dim,
        ):
            raise AIEOperatorConstraintError(
                f"Expected K_cache shape "
                f"({self.num_kv_heads}, {self.seq_len}, {self.head_dim}), "
                f"got {k_cache.shape}"
            )
        if v_cache.shape != (
            self.num_kv_heads,
            self.seq_len,
            self.head_dim,
        ):
            raise AIEOperatorConstraintError(
                f"Expected V_cache shape "
                f"({self.num_kv_heads}, {self.seq_len}, {self.head_dim}), "
                f"got {v_cache.shape}"
            )
        if q_angles.shape != (self.head_dim,):
            raise AIEOperatorConstraintError(
                f"Expected q_angles shape ({self.head_dim},), " f"got {q_angles.shape}"
            )

        # Interleave KV cache for DMA layout
        kv_interleaved = interleave_kv_cache(k_cache, v_cache)

        # Pack Q buffer: [Q_group0 | angles | hdr | Q_group1 | angles | hdr ...]
        q_packed = pack_q_with_angles(
            q, q_angles, self.group_size, self.num_kv_heads, seq_len_cur=seq_len_cur
        )

        self.write_buffer("kv_cache", kv_interleaved)
        self.write_buffer("queries", q_packed)
        self.run_runlist()

        result = self.read_buffer_as_torch("output", (self.num_heads, self.head_dim))
        return result
