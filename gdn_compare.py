#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GDN core-operator benchmark: vLLM Triton (BF16) / MegaGDN PTO (FP16) / TileLang (FP16).

Conforms to the canonical gdn_compare contract from gdn.md:
  - inputs staged on device ONCE (make_runner); the timed closure runs only the
    device kernel path (no H2D / cat / dtype cast / layout transpose in the loop);
  - wall time sampled AFTER device sync;
  - canonical input contract (seed 20261027, g=-(0.01+0.09*rand) FP32,
    beta=0.1+0.8*rand, state=0.01*randn, Q/K pre-L2-normalised); vLLM keeps
    g/beta/state in FP32 and passes cu_seqlens as INT32;
  - correctness gate (output+state finite and NRMSE<2%) must pass before timing;
  - canonical JSON schema (cases / batch / length / output_error / state_error /
    correctness_pass / event_span_us_samples / wall_us_samples /
    zero_initial_state / script_sha256 / device + package versions);
  - canonical profiler (Level1 + PipeUtilization, schedule with 5 active steps,
    prof.step() per iter) wrapping only the timed kernel, not the CPU reference.

Target: Ascend 910B4, CANN 9.1.0. One backend per process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401  — must be imported before .npu() use
    _HAS_NPU = True
except Exception:
    _HAS_NPU = False

# ----------------------------------------------------------------------------
# Core shape contract (gdn.md).
# ----------------------------------------------------------------------------
H_DEFAULT = 16
D_DEFAULT = 128
SEED_DEFAULT = 20261027
NRMSE_GATE = 0.02          # exploratory kernel gate (not a production accuracy bar)


# ============================================================================
# Independent CPU FP64 reference recurrence (FLA naive_recurrent, inlined).
# ============================================================================
def gdn_recurrent_ref_fp64(
    q, k, v, beta, g, scale, initial_state=None,
):
    """Token-level gated delta rule, FP64 on CPU. ``g`` is the per-step LOG
    gate (pre-cumsum); the per-step ``h *= exp(g[i])`` accumulates the decay.
    Inputs are the exact (already-quantized) tensors each backend receives,
    upcast to FP64 here — so NRMSE measures pure kernel numerical error, with
    no input-quantization noise mixed in."""
    q = q.to(torch.float64)
    k = k.to(torch.float64)
    v = v.to(torch.float64)
    beta = beta.to(torch.float64)
    g = g.to(torch.float64)
    if initial_state is not None:
        initial_state = initial_state.to(torch.float64)
    B, T, H, D = q.shape
    q = (q * scale).transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    g = g.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous()
    h = torch.zeros(B, H, D, D, dtype=torch.float64)
    if initial_state is not None:
        h = initial_state.to(torch.float64).clone()
    o = torch.zeros(B, H, T, D, dtype=torch.float64)
    for i in range(T):
        bq, bk = q[:, :, i], k[:, :, i]
        bv = v[:, :, i].clone()
        gi = g[:, :, i].exp()
        h = h * gi[..., None, None]
        bbeta = beta[:, :, i]
        bv = bv - (h * bk[..., None]).sum(dim=-2)
        bv = bv * bbeta[..., None]
        h = h + bk.unsqueeze(-1) * bv.unsqueeze(-2)
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", bq, h)
    return o.transpose(1, 2).contiguous(), h


def reference_for_case(seqs, scale):
    """Run FP64 recurrence per sequence; pack outputs to [rows, H, D] and stack
    final states to [N_seq, H, D, D]."""
    os_list, fs_list = [], []
    for s in seqs:
        o_s, h_s = gdn_recurrent_ref_fp64(
            s["q"], s["k"], s["v"], s["beta"], s["g"], scale, s["h0"])
        os_list.append(o_s.squeeze(0))      # [T, H, D]
        fs_list.append(h_s.squeeze(0))      # [H, D, D]
    return torch.cat(os_list, dim=0), torch.stack(fs_list, dim=0)


# ============================================================================
# Canonical input generation (shared source in FP32; reference upcasts to FP64;
# each backend casts Q/K/V — and beta for FP16 backends — to its native dtype).
# ============================================================================
def gen_case_inputs(n_seq, t_len, h, d, seed, zero_state, qkv_dtype, beta_dtype, state_dtype):
    """Canonical inputs from a SINGLE generator (seeded once). Q/K/V/beta are
    quantized to the backend's native dtype here; the FP64 reference upcasts
    these same tensors, so input-quantization error is excluded from kernel
    NRMSE. ``g`` is always FP32 (every backend takes float32 pre-cumsum logits).
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    seqs = []
    for _ in range(n_seq):
        q = F.normalize(torch.randn(1, t_len, h, d, generator=gen, dtype=torch.float32), dim=-1, p=2)
        k = F.normalize(torch.randn(1, t_len, h, d, generator=gen, dtype=torch.float32), dim=-1, p=2)
        v = torch.randn(1, t_len, h, d, generator=gen, dtype=torch.float32)
        beta = 0.1 + 0.8 * torch.rand(1, t_len, h, generator=gen, dtype=torch.float32)
        g = -(0.01 + 0.09 * torch.rand(1, t_len, h, generator=gen, dtype=torch.float32))
        if zero_state:
            h0 = torch.zeros(1, h, d, d, dtype=torch.float32)
        else:
            h0 = 0.01 * torch.randn(1, h, d, d, generator=gen, dtype=torch.float32)
        seqs.append({
            "q": q.to(qkv_dtype), "k": k.to(qkv_dtype), "v": v.to(qkv_dtype),
            "beta": beta.to(beta_dtype), "g": g.contiguous(),            # FP32, never quantized
            "h0": h0.to(state_dtype) if not zero_state else h0.to(state_dtype),
        })
    return seqs


# ============================================================================
# vLLM runtime stubs — standalone single-rank execution.
# ============================================================================
def init_vllm_stubs(device):
    """Patch the chunk module's bound ``get_forward_context`` / ``get_pcp_group``
    to single-rank no-op stubs, so ``chunk_gated_delta_rule_fwd`` runs without a
    vLLM server. We patch the module attributes (not vllm's ContextVar globals)
    — the forward calls ``get_forward_context()`` / ``get_pcp_group()`` resolved
    in the chunk module's own namespace."""
    import types
    import vllm_ascend.ops.triton.fla.chunk as chunk
    chunk.get_forward_context = lambda: types.SimpleNamespace(attn_metadata=None)
    chunk.get_pcp_group = lambda: types.SimpleNamespace(
        world_size=1, rank_in_group=0)


# ============================================================================
# Backend interface: build_runner() stages everything; runner() runs only kernels.
# ============================================================================
@dataclass
class CaseResult:
    batch: int
    length: int
    rows: int
    chunk_size: int
    output_error: dict
    state_error: dict
    correctness_pass: bool
    event_span_us_samples: List[float]
    wall_us_samples: List[float]
    event_span_us_p50: Optional[float]
    event_span_us_p90: Optional[float]
    wall_us_p50: Optional[float]
    wall_us_p90: Optional[float]


class Backend:
    name = "base"
    qkv_dtype = torch.float32
    beta_dtype = torch.float32       # vLLM keeps beta FP32
    state_dtype = torch.float32      # vLLM keeps state FP32
    native_chunk_size = 64
    contract_exception: Optional[str] = None   # deviations from the vLLM-canonical contract

    def build_runner(self, seqs, h, d, scale, device) -> Callable[[], Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        raise NotImplementedError


# ----- vLLM Triton (BF16) ---------------------------------------------------
class VLLMTritonBackend(Backend):
    name = "vllm-triton"
    qkv_dtype = torch.bfloat16
    beta_dtype = torch.float32
    state_dtype = torch.float32
    native_chunk_size = 64

    def __init__(self, prebuilt_meta: bool):
        self.prebuilt_meta = prebuilt_meta

    def _build_meta(self, cu_seqlens_cpu, h, device):
        if not self.prebuilt_meta:
            return None
        from vllm_ascend.ops.gdn_attn_builder import (
            GDNChunkedPrefillMetadata, _compact_empty_segments,
            _GDN_CHUNK_SIZE, _GDN_CUMSUM_WORKING_SET, _GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE,
        )
        from vllm_ascend.ops.triton.fla.utils import (
            prepare_chunk_indices, prepare_chunk_offsets,
            prepare_final_chunk_indices, prepare_update_chunk_offsets,
        )
        cu_host = tuple(cu_seqlens_cpu.to(torch.int64).reshape(-1).tolist())
        ci64 = prepare_chunk_indices(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
        co64 = prepare_chunk_offsets(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
        uco64 = prepare_update_chunk_offsets(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
        fci64 = prepare_final_chunk_indices(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
        cilb = prepare_chunk_indices(cu_seqlens_cpu, _GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE)
        cc = max(1, _GDN_CUMSUM_WORKING_SET // (h * _GDN_CHUNK_SIZE))
        ccs = 1 if cc <= 1 else 1 << (cc - 1).bit_length()
        bic = prepare_chunk_indices(cu_seqlens_cpu, ccs)
        # No zero-length segments in our cases → keep_meta stays None.
        cu_kern, _, keep_meta = _compact_empty_segments(cu_host, None, device=device)
        if keep_meta is None:
            cu_kern = None
        return GDNChunkedPrefillMetadata(
            cu_seqlens_host=cu_host,
            chunk_indices_chunk64_host=tuple(ci64.to(torch.int64).reshape(-1).tolist()),
            chunk_indices_chunk64=ci64.to(device=device, non_blocking=True),
            chunk_offsets_chunk64=co64.to(device=device, non_blocking=True),
            update_chunk_offsets_chunk64=uco64.to(device=device, non_blocking=True),
            final_chunk_indices_chunk64=fci64.to(device=device, non_blocking=True),
            chunk_indices_large_block=cilb.to(device=device, non_blocking=True),
            block_indices_cumsum=bic.to(device=device, non_blocking=True),
            num_decodes=0, cu_seqlens_kern=cu_kern, keep_meta=keep_meta,
        )

    def build_runner(self, seqs, h, d, scale, device):
        from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
        init_vllm_stubs(device)
        n_seq, t_len = len(seqs), seqs[0]["q"].shape[1]
        # Stage device tensors ONCE.
        q = torch.cat([s["q"] for s in seqs], dim=1).to(device=device, dtype=self.qkv_dtype)
        k = torch.cat([s["k"] for s in seqs], dim=1).to(device=device, dtype=self.qkv_dtype)
        v = torch.cat([s["v"] for s in seqs], dim=1).to(device=device, dtype=self.qkv_dtype)
        beta = torch.cat([s["beta"] for s in seqs], dim=1).to(device=device, dtype=self.beta_dtype)
        g = torch.cat([s["g"] for s in seqs], dim=1).to(device=device, dtype=torch.float32)
        h0 = torch.cat([s["h0"] for s in seqs], dim=0).to(device=device, dtype=self.state_dtype)
        cu = torch.tensor([0] + [t_len * (i + 1) for i in range(n_seq)],
                          dtype=torch.int32, device=device)   # INT32 per contract
        meta = self._build_meta(cu.cpu().to(torch.int64), h, device)

        def run():
            with torch.no_grad():
                o, fs = chunk_gated_delta_rule(
                    q, k, v, g, beta,
                    scale=scale, initial_state=h0, output_final_state=True,
                    cu_seqlens=cu, prebuilt_meta=meta, head_first=False,
                    use_qk_l2norm_in_kernel=False,
                )
            return o.squeeze(0), fs
        return run


# ----- MegaGDN PTO (FP16) ---------------------------------------------------
class MegaGdnBackend(Backend):
    name = "megagdn"
    qkv_dtype = torch.float16
    beta_dtype = torch.float16
    state_dtype = torch.float16
    native_chunk_size = 128
    contract_exception = "FP16 Q/K/V/beta/state (vLLM is BF16 Q/K/V, FP32 g/beta/state); g stays FP32"

    def __init__(self, chunk_size: Optional[int]):
        self.chunk_size = chunk_size  # None → run_mega_kernel default (128)

    def build_runner(self, seqs, h, d, scale, device):
        import ctypes, os
        from megagdn_pto.compile import BLOCK_DIM, _KERNELS_PTO
        from megagdn_pto.kernel_libs import (
            _check_supported_heads, chunk_gdn_causal_masks,
            precomputed_minus_identity, total_chunks, _vp,
        )
        from megagdn_pto.mega_kernel import _load_mega_kernel
        n_seq, t_len = len(seqs), seqs[0]["q"].shape[1]
        q = torch.cat([s["q"] for s in seqs], dim=1).to(device=device, dtype=self.qkv_dtype)
        k = torch.cat([s["k"] for s in seqs], dim=1).to(device=device, dtype=self.qkv_dtype)
        v = torch.cat([s["v"] for s in seqs], dim=1).to(device=device, dtype=self.qkv_dtype)
        beta = torch.cat([s["beta"] for s in seqs], dim=1).to(device=device, dtype=self.beta_dtype)
        g_in = torch.cat([s["g"] for s in seqs], dim=1).to(device=device, dtype=torch.float32)
        h0 = torch.cat([s["h0"] for s in seqs], dim=0).to(device=device, dtype=self.state_dtype)
        cu = torch.tensor([0] + [t_len * (i + 1) for i in range(n_seq)],
                          dtype=torch.int32, device=device)

        # Pre-bind everything run_mega_kernel would otherwise re-do per call:
        # masks, workspace tensors, the compiled lib, and the stream handle.
        C = self.chunk_size or 128
        H, D = v.shape[2], q.shape[3]
        kh = q.shape[2]
        _check_supported_heads(H, kh)
        T = q.shape[1]
        N_seq = int(cu.numel()) - 1
        bd = BLOCK_DIM
        dt, di = device.type, device.index if device.index is not None else -1
        msk_lower, msk_full = chunk_gdn_causal_masks(dt, di, C)
        minus_identity = precomputed_minus_identity(dt, di, C)
        tc = total_chunks(N_seq, T, C, cu)
        num_matrices = tc * H

        g_sum      = torch.empty(1, T, H, device=device, dtype=torch.float32)
        g_t        = torch.empty(H, T, device=device, dtype=torch.float32)
        beta_t     = torch.empty(H, T, device=device, dtype=torch.float16)
        A          = torch.zeros(1, T, H, C, device=device, dtype=torch.float16)
        A_inv_f32  = torch.zeros(1, T, H, C, device=device, dtype=torch.float32)
        A_inv      = torch.zeros(1, T, H, C, device=device, dtype=torch.float16)
        w          = torch.empty_like(v)
        u          = torch.empty_like(v)
        s          = torch.zeros(tc * H, D, D, device=device, dtype=torch.float16)
        v_new      = torch.empty_like(v)
        fs         = torch.zeros(N_seq * H, D, D, device=device, dtype=torch.float16)
        h0c        = h0.contiguous()
        kkt_ws     = torch.zeros(bd * 2, C, C, device=device, dtype=torch.float16)
        wy_ws_a1   = torch.zeros(bd, C, C, device=device, dtype=torch.float16)
        wy_ws_a2   = torch.zeros(bd, C, C, device=device, dtype=torch.float16)
        h_ws       = torch.zeros(bd * 4, D, D, device=device, dtype=torch.float16)
        o_ws_qk    = torch.zeros(bd, C, C, device=device, dtype=torch.float16)
        o_ws_qs    = torch.zeros(bd, C, D, device=device, dtype=torch.float16)
        o_ws_gated = torch.zeros(bd, C, C, device=device, dtype=torch.float16)
        o_out      = torch.empty_like(v)

        mtime = os.stat(os.path.join(_KERNELS_PTO, "mega_kernel.cpp")).st_mtime_ns
        lib = _load_mega_kernel(hidden_size=D, chunk_size=C, cpp_mtime_ns=mtime)
        stream_ptr = torch.npu.current_stream()._as_parameter_

        def run():
            with torch.no_grad():
                # run_mega_kernel allocates these as torch.zeros each call
                # (A/A_inv/A_inv_f32/s/fs are read-then-written or carry
                # chunk-0 zero initialisation; workspaces are scratch that
                # the gemms init). Zero our reused buffers in-place instead
                # — same fill launches, no per-call allocation.
                A.zero_()
                A_inv_f32.zero_()
                A_inv.zero_()
                s.zero_()
                fs.zero_()
                kkt_ws.zero_()
                wy_ws_a1.zero_()
                wy_ws_a2.zero_()
                h_ws.zero_()
                o_ws_qk.zero_()
                o_ws_qs.zero_()
                o_ws_gated.zero_()
                lib.call_kernel(
                    bd, stream_ptr,
                    _vp(q), _vp(k), _vp(v), _vp(g_in), _vp(beta),
                    _vp(msk_lower), _vp(msk_full), _vp(minus_identity), _vp(cu),
                    _vp(o_out),
                    _vp(g_sum), _vp(g_t), _vp(beta_t),
                    _vp(A), _vp(A_inv_f32), _vp(A_inv),
                    _vp(w), _vp(u), _vp(s), _vp(v_new), _vp(fs),
                    _vp(h0c), 1,
                    _vp(kkt_ws), _vp(wy_ws_a1), _vp(wy_ws_a2), _vp(h_ws),
                    _vp(o_ws_qk), _vp(o_ws_qs), _vp(o_ws_gated),
                    H, kh,
                    N_seq, T, T, num_matrices,
                )
                o = o_out * scale
            return o.squeeze(0), fs.view(N_seq, H, D, D)
        return run


# ----- TileLang (FP16) ------------------------------------------------------
class TileLangBackend(Backend):
    name = "tilelang"
    qkv_dtype = torch.float16
    beta_dtype = torch.float16
    native_chunk_size = 32          # canonical: C/BK/BV = 32
    contract_exception = ("FP16 Q/K/V/beta; zero-state forced (upstream chunk_h has no "
                          "initial_state param); C=BK=BV=32; T padded to multiple of 32")

    def __init__(self, example_path: str, chunk_size: Optional[int], bk: int, bv: int):
        self.example_path = example_path
        self.chunk_size = chunk_size or 32
        self.bk = bk or 32
        self.bv = bv or 32
        self._loaded = False

    def _load_stages(self):
        if self._loaded:
            return
        if self.example_path not in sys.path:
            sys.path.insert(0, self.example_path)
        # Import the compiled-kernel builders directly so we can pre-bind the
        # kernel objects once (the upstream wrappers re-invoke xxx_ker(...) and
        # re-allocate mask/workspace per call — that dispatch dominates timing
        # for these tiny T=16/32/48 cases).
        from gdn_chunk_cumsum import cumsum_ker
        from gdn_chunk_scaled_dot_kkt import kkt_ker
        from gdn_solve_tril import solve_tril_ker
        from gdn_wy_fast import wy_fast_ker
        from gdn_chunk_h import chunk_h_ker
        from gdn_chunk_o import chunk_o_ker
        self._cumsum_ker = cumsum_ker
        self._kkt_ker = kkt_ker
        self._solve_ker = solve_tril_ker
        self._wy_ker = wy_fast_ker
        self._chunk_h_ker = chunk_h_ker
        self._chunk_o_ker = chunk_o_ker
        self._loaded = True

    def build_runner(self, seqs, h, d, scale, device):
        self._load_stages()
        n_seq, t_len = len(seqs), seqs[0]["q"].shape[1]
        C = self.chunk_size
        L = ((t_len + C - 1) // C) * C          # pad T up to a multiple of C
        BK, BV = self.bk, self.bv
        if L % C != 0:
            raise ValueError(f"TileLang L%C!=0: T={t_len} C={C} L={L}")
        chunk_num = (L + C - 1) // C
        bv_num = (d + BV - 1) // BV

        def stage(src_key, dtype):
            t = torch.cat([s[src_key] for s in seqs], dim=0)        # [B, T, H, (D)]
            if L > t_len:
                # Pad the TIME axis (dim 1). 4D [B,T,H,D] vs 3D [B,T,H].
                if t.ndim == 4:
                    t = F.pad(t, (0, 0, 0, 0, 0, L - t_len))
                else:
                    t = F.pad(t, (0, 0, 0, L - t_len))
            return t.to(device=device, dtype=dtype)

        q = stage("q", self.qkv_dtype).transpose(1, 2).contiguous()   # [B, H, L, D]
        k = stage("k", self.qkv_dtype).transpose(1, 2).contiguous()
        v = stage("v", self.qkv_dtype).transpose(1, 2).contiguous()
        beta = stage("beta", self.beta_dtype).transpose(1, 2).contiguous()  # [B,H,L]
        g = stage("g", torch.float32).transpose(1, 2).contiguous()          # [B,H,L]

        # Pre-bind kernel objects + pre-allocate reusable buffers ONCE.
        k_cum = self._cumsum_ker(n_seq, h, L, C)
        k_kkt = self._kkt_ker(n_seq, h, L, d, C, BK)
        k_slv = self._solve_ker(n_seq, h, L, C)
        k_wy = self._wy_ker(n_seq, h, L, d, d, C, BK, BV)
        k_ch = self._chunk_h_ker(n_seq, h, L, d, d, C, BK, BV)
        k_co = self._chunk_o_ker(n_seq, h, L, d, d, C, BK, BV)
        # Pre-allocate the constant buffers each kernel needs, ONCE (the upstream
        # wrappers re-create these per call). kkt uses a strictly-lower mask
        # (diagonal=-1); chunk_o uses an inclusive-lower mask (diagonal=0);
        # solve_tril needs a CxC identity. All kept in float32 on device.
        kkt_msk = torch.tril(torch.ones((C, C), device=device, dtype=torch.float32), diagonal=-1)
        co_msk = torch.tril(torch.ones((C, C), device=device, dtype=torch.float32), diagonal=0)
        idt = torch.eye(C, device=device, dtype=torch.float32)
        ws3 = torch.zeros((n_seq * h * bv_num, d, BV), device=device, dtype=torch.float16)
        s_buf = torch.zeros((n_seq, h, chunk_num, d, d), device=device, dtype=torch.float16)

        def run():
            with torch.no_grad():
                # chunk_h declares workspace_3 and S as "need to be manually set
                # to 0" (the kernel reads workspace_3 as chunk-0's initial
                # cross-chunk state and never clears it). The wrappers allocate
                # fresh zeros each call; we zero our reused buffers in-place
                # instead — same fill launches, no per-call allocation.
                ws3.zero_()
                s_buf.zero_()
                g_cum = k_cum(g)
                A = k_kkt(k, beta, g_cum, kkt_msk)
                A_inv = k_slv(A, idt)
                w, u = k_wy(k, v, beta, g_cum, A_inv)
                # upstream chunk_h, unmodified (zero-state); reuse ws3 / s_buf.
                new_v, final_s = k_ch(k, w, u, g_cum, ws3, s_buf)
                # chunk_o takes new_v (WY-represented), NOT raw v — FLA dataflow.
                o = k_co(q, k, new_v, s_buf, g_cum, co_msk)
                o = o * scale
            # [B, H, L, D] → first T rows per seq → [B*T, H, D]
            o = o.transpose(1, 2).contiguous()[:, :t_len, :, :].reshape(-1, h, d)
            return o, final_s
        return run


BACKENDS = {
    "vllm-triton": lambda a: VLLMTritonBackend(prebuilt_meta=a.prebuilt_meta),
    "megagdn": lambda a: MegaGdnBackend(chunk_size=a.chunk_size),
    "mega-pto": lambda a: MegaGdnBackend(chunk_size=a.chunk_size),   # alias
    "tilelang": lambda a: TileLangBackend(
        example_path=a.tilelang_path, chunk_size=a.chunk_size,
        bk=a.tilelang_bk, bv=a.tilelang_bv),
}


# ============================================================================
# Timing.
# ============================================================================
def npu_sync():
    if _HAS_NPU:
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def time_runner(run, warmup, repeats):
    npu_sync()
    for _ in range(warmup):
        run()
    npu_sync()
    EventT = torch.npu.Event if _HAS_NPU else (
        torch.cuda.Event if torch.cuda.is_available() else None)
    ev, wall = [], []
    for _ in range(repeats):
        tic = time.perf_counter()
        if EventT is not None:
            start = EventT(enable_timing=True)
            end = EventT(enable_timing=True)
            start.record()
            run()
            end.record()
            npu_sync()
            ev.append(start.elapsed_time(end) * 1000.0)   # ms→us
        else:
            run()
            npu_sync()
            ev.append(0.0)
        wall.append((time.perf_counter() - tic) * 1e6)
    return ev, wall


def pct(samples, q):
    if not samples:
        return None
    s = sorted(samples)
    return s[max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))]


def median(samples):
    return statistics.median(samples) if samples else None


# ============================================================================
# Validation.
# ============================================================================
def nrmse(pred, ref):
    pred = pred.to(torch.float64).flatten()
    ref = ref.to(torch.float64).flatten()
    denom = ref.norm().clamp_min(1e-12)        # scalar clamp, never zero
    return (pred - ref).norm().item() / denom.item()


def max_abs_err(pred, ref):
    return (pred.to(torch.float64) - ref.to(torch.float64)).abs().max().item()


def all_finite(t):
    return True if t is None else bool(torch.isfinite(t.to(torch.float64)).all().item())


def error_dict(pred, ref):
    return {
        "nrmse": (None if pred is None else nrmse(pred, ref)),
        "max_abs": (None if pred is None else max_abs_err(pred, ref)),
        "finite": (None if pred is None else all_finite(pred)),
    }


# ============================================================================
# Cases + main.
# ============================================================================
def parse_cases(raw):
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "x" not in tok:
            raise ValueError(f"Bad case '{tok}', expected BxT")
        b, t = tok.split("x")
        out.append((int(b), int(t)))
    return out


def package_versions():
    pkgs = {}
    for mod_name, dist in [("torch", "torch"), ("torch_npu", "torch-npu"),
                           ("vllm", "vllm"), ("vllm_ascend", "vllm-ascend"),
                           ("tilelang", "tilelang"), ("megagdn_pto", "megagdn-pto")]:
        try:
            m = __import__(mod_name) if mod_name != "vllm_ascend" else None
            v = getattr(m, "__version__", None) if m else None
            if v is None:
                from importlib.metadata import version
                v = version(dist)
            pkgs[mod_name] = str(v)
        except Exception:
            pkgs[mod_name] = None
    return pkgs


def device_name():
    try:
        if _HAS_NPU:
            return f"npu:{torch.npu.current_device()} ({torch.npu.get_device_name()})"
    except Exception:
        pass
    return "cpu"


def script_sha256(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return None


def run_case(backend, n_seq, t_len, h, d, scale, seed, zero_state, warmup, repeats):
    seqs = gen_case_inputs(n_seq, t_len, h, d, seed, zero_state,
                           backend.qkv_dtype, backend.beta_dtype, backend.state_dtype)
    o_ref, fs_ref = reference_for_case(seqs, scale)        # CPU FP64, once

    runner = backend.build_runner(seqs, h, d, scale, torch.device(_device_str()))

    # Validation run (not timed).
    o_npu, fs_npu = runner()
    npu_sync()
    o_cmp = o_npu.detach().to("cpu", torch.float64)
    fs_cmp = fs_npu.detach().to("cpu", torch.float64) if fs_npu is not None else None

    out_err = error_dict(o_cmp, o_ref)
    st_err = error_dict(fs_cmp, fs_ref)
    passed = (bool(out_err["finite"]) and bool(st_err["finite"])
              and out_err["nrmse"] < NRMSE_GATE and st_err["nrmse"] < NRMSE_GATE)

    if not passed:
        print(f"  [FAIL] out_nrmse={out_err['nrmse']:.4%} "
              f"state_nrmse={st_err['nrmse']:.4%} — skipping timing")
        ev, wall = [], []
    else:
        ev, wall = time_runner(runner, warmup, repeats)
        print(f"  [PASS] out_nrmse={out_err['nrmse']:.4%} "
              f"state_nrmse={st_err['nrmse']:.4%} "
              f"event_p50={median(ev):.2f}us wall_p50={median(wall):.2f}us")

    return CaseResult(
        batch=n_seq, length=t_len, rows=n_seq * t_len,
        chunk_size=(getattr(backend, "chunk_size", None) or backend.native_chunk_size),
        output_error=out_err, state_error=st_err, correctness_pass=passed,
        event_span_us_samples=ev, wall_us_samples=wall,
        event_span_us_p50=median(ev), event_span_us_p90=pct(ev, 0.90),
        wall_us_p50=median(wall), wall_us_p90=pct(wall, 0.90),
    )


_device_str = lambda: "cpu"


def profile_case(backend, n_seq, t_len, h, d, scale, seed, zero_state, profile_dir):
    """Canonical profiler: Level1 + PipeUtilization, 5 active steps, prof.step()
    per iter; wraps only the timed kernel, not the CPU reference. Configured via
    ``_ExperimentalConfig`` — no silent fallback (a missing torch_npu profiler
    API is a hard error, not a degraded trace)."""
    if not _HAS_NPU:
        raise RuntimeError("profiling requires torch_npu")
    os.makedirs(profile_dir, exist_ok=True)
    seqs = gen_case_inputs(n_seq, t_len, h, d, seed, zero_state,
                           backend.qkv_dtype, backend.beta_dtype, backend.state_dtype)
    runner = backend.build_runner(seqs, h, d, scale, torch.device(_device_str()))
    npu_sync()
    for _ in range(3):       # warm JIT / PTO compile caches outside the window
        runner()
    npu_sync()

    import torch_npu.profiler as np
    try:
        from torch_npu.profiler import _ExperimentalConfig, ProfilerLevel, PipeUtilization
        exp_cfg = _ExperimentalConfig(
            profiler_level=ProfilerLevel.Level1,
            pipe_utilization=PipeUtilization.Data,
        )
    except Exception as e:
        raise RuntimeError(
            "torch_npu.profiler._ExperimentalConfig(ProfilerLevel.Level1, "
            f"PipeUtilization.Data) unavailable on this CANN build: {e!r}. "
            "Cannot produce a canonical Level1+PipeUtilization trace.")
    prof_ctx = np.profile(
        activities=[np.ProfilerActivity.CPU, np.ProfilerActivity.NPU],
        schedule=np.schedule(wait=1, warmup=1, active=5, repeat=1),
        on_trace_ready=np.tensorboard_trace_handler(profile_dir),
        experimental_config=exp_cfg,
    )
    with prof_ctx as prof:
        for _ in range(7):          # wait(1)+warmup(1)+active(5)
            runner()
            prof.step()
    npu_sync()


def main():
    global _device_str
    ap = argparse.ArgumentParser(description="GDN three-backend benchmark (910B4 / CANN 9.1.0)")
    ap.add_argument("--backend", required=True, choices=list(BACKENDS.keys()))
    ap.add_argument("--cases", required=True, help="Comma-separated BxT, e.g. 5x16,5x32,5x48")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--heads", type=int, default=H_DEFAULT)
    ap.add_argument("--dim", type=int, default=D_DEFAULT)
    ap.add_argument("--device", type=int, default=0, help="NPU device id")
    ap.add_argument("--zero-state", action="store_true",
                    help="Zero initial state (default: nonzero 0.01*randn FP32, the cache-hit contract)")
    ap.add_argument("--prebuilt-meta", action="store_true",
                    help="(vllm-triton) build chunk metadata once, outside the timed region")
    ap.add_argument("--chunk-size", type=int, default=None,
                    help="(megagdn / tilelang) chunk size C. Mega default 128, TileLang default 32")
    ap.add_argument("--tilelang-path", type=str,
                    default="tilelang-ascend/examples/linear_attention_and_rnn/gdn")
    ap.add_argument("--tilelang-bk", type=int, default=32)
    ap.add_argument("--tilelang-bv", type=int, default=32)
    ap.add_argument("--profile-dir", type=str, default=None,
                    help="If set, profile the first case (Level1 + PipeUtilization, 5 active steps)")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    if _HAS_NPU:
        torch.npu.set_device(args.device)
        _device_str = lambda: f"npu:{args.device}"
    else:
        _device_str = lambda: "cpu"
    device = torch.device(_device_str())
    scale = args.dim ** -0.5

    # TileLang: zero-state forced (upstream chunk_h has no initial_state param).
    zero_state = args.zero_state
    if args.backend == "tilelang" and not zero_state:
        print("[gdn_compare] tilelang forces --zero-state (upstream chunk_h starts state at zero)")
        zero_state = True

    backend = BACKENDS[args.backend](args)
    cases = parse_cases(args.cases)
    print(f"[gdn_compare] backend={backend.name} device={device_name()} "
          f"H={args.heads} D={args.dim} scale={scale:.6f} zero_state={zero_state} "
          f"seed={args.seed} warmup={args.warmup} repeats={args.repeats}")

    case_results = []
    for idx, (n_seq, t_len) in enumerate(cases):
        print(f"  -> case {n_seq}x{t_len} ...", flush=True)
        cr = run_case(backend, n_seq, t_len, args.heads, args.dim, scale,
                      args.seed, zero_state, args.warmup, args.repeats)
        case_results.append(cr)
        # Profile only AFTER the correctness gate has passed for this case.
        if args.profile_dir and idx == 0 and cr.correctness_pass:
            print("  [profile] capturing Level1+PipeUtilization trace ...")
            profile_case(backend, n_seq, t_len, args.heads, args.dim, scale,
                         args.seed, zero_state, args.profile_dir)

    out = {
        "backend": backend.name,
        "device": device_name(),
        "versions": package_versions(),
        "contract_exception": backend.contract_exception,
        "heads": args.heads,
        "dim": args.dim,
        "scale": scale,
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "zero_initial_state": zero_state,
        "script_sha256": script_sha256(os.path.abspath(__file__)),
        "cases": [
            {
                "batch": cr.batch, "length": cr.length, "rows": cr.rows,
                "chunk_size": cr.chunk_size,
                "output_error": cr.output_error,
                "state_error": cr.state_error,
                "correctness_pass": cr.correctness_pass,
                "event_span_us_samples": cr.event_span_us_samples,
                "wall_us_samples": cr.wall_us_samples,
                "event_span_us_p50": cr.event_span_us_p50,
                "event_span_us_p90": cr.event_span_us_p90,
                "wall_us_p50": cr.wall_us_p50,
                "wall_us_p90": cr.wall_us_p90,
            } for cr in case_results
        ],
    }
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[gdn_compare] wrote {args.output}")
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
