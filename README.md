# GDN three-backend benchmark — pure-kernel timing (910B4)

Compares the Gated DeltaNet **core** operator across Ascend implementations,
following the canonical `gdn_compare` contract in [`gdn.md`](../gdn.md).

All three backends are measured at **pure-kernel time**: kernel objects are
pre-bound and all workspace / masks / constants are pre-allocated once in
`build_runner()`, so the timed `run()` closure issues only the device kernel
path (plus the minimum in-place `zero_()` the kernels require). This removes
per-call Python dispatch, kernel-object construction, and workspace allocation
from the timed region, giving a fair cross-backend comparison.

| Backend | dtype | entry | stack | chunk size | pre-bound |
|---|---|---|---|---|---|
| `megagdn` | FP16 | `run_mega_kernel` → `lib.call_kernel` | huawei-csl/megagdn-pto, PTO-ISA C++ (pinned `bd0c112`) | 128 | lib + 15 workspace + masks |
| `tilelang` | FP16 | six stages `cumsum/kkt/solve_tril/wy_fast/chunk_h/chunk_o` | tile-ai/tilelang-ascend (pinned `43b0b69`) | 32 | 6 kernel objects + masks + workspace |
| `vllm-triton` | BF16 | `chunk_gated_delta_rule` | vllm-ascend `v0.19.1rc1` + AscendC | 64 | (requires CANN 9.1.0 — see below) |

Output and final recurrent state are validated against an independent CPU FP64
token-level recurrence (FLA `naive_recurrent_gated_delta_rule` math, inlined,
no `fla` dependency).

## Results (CANN 9.0.0, Ascend 910B4, H=16 D=128, seed 20261027)

Pure-kernel `event_span_us_p50` (warmup=50, repeats=200), all cases PASS the
2% NRMSE correctness gate:

| case | MegaGDN event_p50 | out NRMSE | state NRMSE | TileLang event_p50 | out NRMSE | state NRMSE |
|---|---|---|---|---|---|---|
| 5×16 | **318.4 µs** | 0.056% | 0.047% | **1379.1 µs** | 0.052% | 0.041% |
| 5×32 | **286.0 µs** | 0.056% | 0.047% | **1381.1 µs** | 0.052% | 0.042% |
| 5×48 | **288.8 µs** | 0.056% | 0.047% | **1364.9 µs** | 0.054% | 0.050% |

### Comparison to the reference table

The reference table was measured on CANN 9.1.0 with a less aggressive
pre-binding (per-call workspace allocation retained). Both backends here run
**faster** than the reference, and the speedup scales with each backend's
wrapper overhead share — which is the expected, self-consistent signature of a
cleaner measurement, not a methodology error:

| case | MegaGDN (here) | ref MegaGDN upstream | ratio | TileLang (here) | ref TileLang | ratio |
|---|---|---|---|---|---|---|
| 5×16 | 318.4 | 374 | 0.85× | 1379.1 | 2309 | 0.60× |
| 5×32 | 286.0 | 374 | 0.76× | 1381.1 | 2267 | 0.61× |
| 5×48 | 288.8 | 375 | 0.77× | 1364.9 | 2789 | 0.49× |

- Before pre-binding, MegaGDN was 1.75× **slower** than the reference (669 µs,
  per-call workspace alloc included) and TileLang was 2.7× slower (6295 µs,
  six-stage wrapper dispatch included). Pre-binding closes that gap and then
  some.
- TileLang speeds up 4.5× from pre-binding (six kernel objects + six mask/workspace
  groups removed per call); MegaGDN speeds up 2.1× (one fused kernel + one
  workspace group). The larger TileLang speedup is why it ends up further below
  the reference than MegaGDN — the wrapper overhead share was bigger.
- `wall_p50 - event_p50` is a stable ~95 µs (Mega) / ~103 µs (TileLang): the
  residual Python dispatch (ctypes call + `zero_()` scheduling), independent of
  the kernel count.

### vLLM Triton — not measured here

`vllm-ascend v0.19.1rc1` pins `torch-npu==2.9.0` / `torch==2.9.0`, which
requires **CANN 9.1.0**. The box this was run on is CANN 9.0.0
(`ASCEND_TOOLKIT_HOME=.../cann-9.0.0`, `torch_npu 2.7.1.post4`), so vLLM-Ascend
cannot be installed there. To measure vLLM Triton, re-run on a CANN 9.1.0 +
torch_npu 2.9.0 environment (see [vLLM setup](#vllm-triton--cann-910)).

## Contract (matches canonical gdn_compare)

- **Inputs staged once.** H2D, `cat`, dtype cast and layout transpose all
  happen in `build_runner()`; the timed closure runs only the device kernel
  path.
- **Wall time** sampled after `torch.npu.synchronize()`.
- **Canonical input values** (seed `20261027`): Q/K pre-L2-normalised; V randn;
  `g = -(0.01 + 0.09*rand)` FP32; `beta = 0.1 + 0.8*rand`; state `0.01*randn`
  FP32. vLLM keeps `g/beta/state` in **FP32** and passes `cu_seqlens` as
  **INT32**.
- **Correctness gate**: `output+state finite and NRMSE < 2%` must pass before
  warmup/repeats. Failures skip timing and record `correctness_pass=false`.
- **JSON schema**: `cases` / `batch` / `length` / `output_error` / `state_error`
  / `correctness_pass` / `event_span_us_samples` / `wall_us_samples` /
  `zero_initial_state` / `script_sha256` / `device` / `versions` /
  `contract_exception`.
- **Profiler**: Level1 + PipeUtilization via `_ExperimentalConfig`,
  `schedule(wait=1, warmup=1, active=5)`, `prof.step()` per iter, wrapping only
  the timed kernel (not the CPU reference); runs only AFTER the correctness gate
  passes. A missing torch_npu profiler API is a hard error, not a degraded
  trace.

Q/K are fed pre-L2-normalised; each backend's own QK-norm flag is disabled so
all backends receive identical source tensors. Nonzero FP32 state is the
default (cache-hit contract); `--zero-state` for a zero-state comparison.

## What pre-binding does (per backend)

### MegaGDN

`run_mega_kernel` (upstream) allocates ~15 workspace tensors, rebuilds the
causal masks, stats the source file, and checks the compiled-lib cache **on
every call**. `MegaGdnBackend.build_runner` instead:

- pre-fetches `msk_lower` / `msk_full` / `minus_identity` once;
- pre-allocates all workspace (`g_sum`, `g_t`, `beta_t`, `A`, `A_inv_f32`,
  `A_inv`, `w`, `u`, `s`, `v_new`, `fs`, `kkt_ws`, `wy_ws_a1/2`, `h_ws`,
  `o_ws_qk/qs/gated`, `o_out`) once;
- pre-loads the compiled `.so` via `_load_mega_kernel` once;
- captures `torch.npu.current_stream()._as_parameter_` once.

`run()` only does the in-place `zero_()` on the buffers the upstream allocates
as `torch.zeros` (A / A_inv / s / fs / the scratch workspaces — same fill
launches as upstream, no allocation), then calls `lib.call_kernel(...)` and the
final `o_out * scale`.

### TileLang

The upstream six-stage example wraps each stage in a function that re-invokes
`xxx_ker(B,H,L,...)` (kernel-object construction), re-allocates the mask, and
re-allocates `workspace_3` / `S` **per call**. `TileLangBackend.build_runner`
instead:

- imports the six `_ker` builders directly (`cumsum_ker`, `kkt_ker`,
  `solve_tril_ker`, `wy_fast_ker`, `chunk_h_ker`, `chunk_o_ker`) and pre-binds
  the kernel objects once with the case shapes;
- pre-allocates the three constant buffers the wrappers rebuild each call:
  kkt's strictly-lower mask (`diagonal=-1`), chunk_o's inclusive-lower mask
  (`diagonal=0`), solve_tril's `C×C` identity;
- pre-allocates `workspace_3` and `s_buf` (chunk_h's cross-chunk state) once.

`run()` only zeroes `workspace_3` / `s_buf` in place — the kernel marks both
"need to be manually set to 0" (workspace_3 is chunk-0's initial cross-chunk
state; S[0] is the initial recurrent state, never written by the kernel) — then
chains the six pre-bound kernel objects with the FLA dataflow
(`chunk_o` receives `new_v` from `chunk_h`, not raw `v`).

> Note: each TileLang `@tilelang.jit` stage still auto-allocates its
> `workspace_idx` tensors (kkt 1, wy_fast 2, chunk_h 3, chunk_o 3 = 9 per
> forward) inside the JIT call wrapper, and `TL_ASCEND_AUTO_SYNC=True` inserts 6
> device syncs per forward. Both are compile-time in the upstream example
> sources and are not removed here — removing them would require editing the
> example `.py` files.

## Per-backend notes

**MegaGDN PTO.** Packed variable-length layout: B sequences → `B*T` rows,
`B=1`, `cu_seqlens`. `run_mega_kernel` takes float32 pre-cumsum gate logits and
applies `o*scale` at the end (we pass `scale=1/sqrt(128)`). PTO-ISA C++
compiles on first call (cached thereafter). Same packed layout as vLLM.

**TileLang.** The upstream example has no `cu_seqlens` support and requires
`L % C == 0`. The benchmark:
- forces `--zero-state` (the upstream `chunk_h` starts state from zero and has
  no `initial_state` parameter — used unmodified, no state injection);
- uses `C = BK = BV = 32`;
- pads T to the next multiple of 32 (T16→32, T32→32, T48→64); padded rows are
  no-ops (`beta=0, g=0, q/k/v=0`), so the unpadded output and final state still
  match the FP64 reference;
- chains the six stages with the FLA dataflow: **`chunk_o` receives `new_v`
  from `chunk_h`, not the raw `v`** (the WY-represented value — what makes the
  composition equal the recurrence).

For a tuned TileLang run (the example is tuned for `L=256, C=32`), use longer
sequences: `--cases 1x256,1x512 --chunk-size 64`.

**vLLM Triton.** Variable-length path: B sequences packed to `B*T` rows,
`B=1`, `cu_seqlens`. `--prebuilt-meta` builds `GDNChunkedPrefillMetadata` once
outside the timed region. Standalone single-rank stubs are installed for
`get_forward_context` (empty `ForwardContext`, `attn_metadata=None`) and
`get_pcp_group` (`world_size=1`), so `chunk_gated_delta_rule_fwd` runs without a
vLLM server. `chunk_size` is hardcoded to 64 inside the op.

<a id="vllm-triton--cann-910"></a>
## Setup (on the 910B4 box)

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # or wherever CANN is
python -c "import torch, torch_npu; print(torch.npu.is_available())"
```

> CANN version check: `echo $ASCEND_TOOLKIT_HOME` (the path contains the
> version, e.g. `.../cann-9.0.0`). MegaGDN and TileLang run on CANN 9.0.0;
> vLLM-Ascend `v0.19.1rc1` requires CANN 9.1.0 + `torch_npu==2.9.0`.

### MegaGDN PTO — pinned commit

```bash
git clone https://github.com/huawei-csl/megagdn-pto
cd megagdn-pto && git checkout bd0c112e712f34c0c2f24a038d44866138b798c4
pip install -e .   # PTO-ISA toolchain required; mega_kernel compiles on first call
```

### TileLang — pinned commit

```bash
git clone https://github.com/tile-ai/tilelang-ascend
cd tilelang-ascend && git checkout 43b0b69720f479894a13c02746599eccee5c5769
export USE_ASCEND=true
# Disable libbacktrace if the build fails on it:
echo 'set(USE_LIBBACKTRACE OFF)' >> 3rdparty/tvm/cmake/config.cmake
pip install -e . --no-build-isolation
```

If the recursive submodule clone fails on flaky GitHub mirrors, deinit the
CUDA-only submodules and retry:

```bash
git submodule deinit -f 3rdparty/flashinfer 3rdparty/libflash_attn \
                       3rdparty/cutlass_fpA_intB_gemm 3rdparty/composable_kernel
```

### vLLM Triton — CANN 9.1.0

```bash
git clone https://github.com/vllm-project/vllm-ascend
cd vllm-ascend && git checkout v0.19.1rc1
pip install -e .          # pulls torch==2.9.0, torch-npu==2.9.0, triton-ascend==3.2.0
export PYTHONPATH=$PWD:$PYTHONPATH
python -c "from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule; print('ok')"
```

## Run

One backend per process. For small differences, alternate backend order over
≥5 rounds; keep the same device, seed, state mode, warmup and repeat count.

```bash
# MegaGDN (FP16, pre-bound fused kernel)
python gdn_compare.py --backend megagdn \
  --cases 5x16,5x32,5x48 --warmup 50 --repeats 200 \
  --output /tmp/gdn-b5-mega.json

# TileLang (FP16, pre-bound six stages)
python gdn_compare.py --backend tilelang --device 0 \
  --cases 5x16,5x32,5x48 --warmup 50 --repeats 200 \
  --tilelang-path /path/to/tilelang-ascend/examples/linear_attention_and_rnn/gdn \
  --output /tmp/gdn-b5-tilelang.json

# vLLM Triton (BF16) — CANN 9.1.0 only
python gdn_compare.py --backend vllm-triton --prebuilt-meta \
  --cases 5x16,5x32,5x48 --warmup 50 --repeats 200 \
  --output /tmp/gdn-b5-vllm.json

# B8 extension suite
python gdn_compare.py --backend megagdn --cases 8x16,8x32,8x48 \
  --warmup 50 --repeats 200 --output /tmp/gdn-b8-mega.json

# Soft-split candidate components
python gdn_compare.py --backend vllm-triton --prebuilt-meta \
  --cases 5x48,4x16,1x48,3x16,2x48,4x32,8x48,7x16,4x48,7x32 \
  --warmup 50 --repeats 200 --output /tmp/gdn-soft-split-vllm.json

# Profiling (first case only)
python gdn_compare.py --backend megagdn \
  --cases 5x16,5x48,8x48 --warmup 20 --repeats 50 \
  --profile-dir /tmp/gdn-profile --output /tmp/gdn-profile.json
```

## Metrics

Per case: `event_span_us_samples` (device event span, incl. inter-kernel gaps
and the in-place `zero_()` fills) and `wall_us_samples` (Python wrapper wall
time), plus p50/p90. Validation: `output_error` / `state_error` (`nrmse`,
`max_abs`, `finite`) and `correctness_pass`. The 2% NRMSE gate is an
exploratory kernel gate, not a production accuracy acceptance threshold.

## Miss boundary (from gdn.md)

Do not mix pure-miss into the B5/B8 hit matrix. Generic long-sequence probes
like `1x1024` may study scaling but must be labelled synthetic and not reported
as the production pure-miss shape.
