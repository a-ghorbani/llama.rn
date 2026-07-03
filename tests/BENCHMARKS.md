# KV-checkpoint benchmarks (on-device)

Quantifies the KV-checkpoint tradeoff: RAM spent (the serialized state blob) vs
latency saved (prefill / time-to-first-token avoided by restoring instead of
reprocessing chat history). Measured with `reuse_test bench` on real Android
devices; see "Reproduce" below.

## What the bench measures

`./reuse_test <model> bench [n_turns]` runs a growing multi-turn chat with fixed
(canned) replies so the reuse/wipe divergence is identical every run and every
device. Per turn it reports time-to-first-token with the checkpoint **off** (the
cache is wiped and the whole history reprocessed) vs **memory** (the boundary is
restored and only the new tail reprocessed). The checkpoint blob is the marginal
RAM cost. `bench off` / `bench memory` run a single mode and print peak RSS for a
cross-process comparison.

The win only exists when a turn would otherwise WIPE. On a pure-recurrent model
(Mamba) every divergent turn wipes without the checkpoint, so it is the clearest
demonstrator. On a hybrid model with bounded recurrent rollback (LFM2.5) plain
append is absorbed without a wipe, so the checkpoint is unused there — a true
result, not a bench artifact.

## CPU results — Mamba-130M, 8-turn chat, off vs memory

Off-mode TTFT grows linearly with history (full reprocess); memory-mode stays
flat (tail-only). Checkpoint blob: **2.67 MB, constant**. Overall speedup and the
worst single-turn off-mode stall (turn 8, ~614 prompt tokens):

| Device | SoC | overall | turn-8 off → memory |
| --- | --- | --- | --- |
| Pixel 9 | Tensor G4 | 3.5x | 4.6 s → 0.72 s |
| POCO (2511FPC34G) | Dimensity (mt6899) | 4.0x | 4.7 s → 0.68 s |
| Galaxy S23 | Snapdragon 8 Gen 2 | 3.8x | 4.9 s → 0.70 s |
| OnePlus 6 | Snapdragon 845 | 3.9x | 10.2 s → 1.43 s |
| Redmi (Helio) | Helio (mt6768) | 4.2x | 18.7 s → 2.5 s |
| Xiaomi (canoe) | Snapdragon | ~7x | flat ~0.45 s |

Takeaway: memory-mode TTFT is flat regardless of conversation length; off-mode
scales with total context. On low-end silicon the checkpoint converts an 18-second
stall into 2.5 s for a constant 2.67 MB. Raw logs: `results-android/cpu-*.txt`.

## GPU (OpenCL) results — the serialization gate

Built with `-DGGML_OPENCL=ON` (see CMakeLists.txt) and run with layers offloaded
to the GPU. The load-bearing finding is a **correctness** one, not speed:

- On devices where OpenCL actually engaged (Qualcomm Adreno 740 on the S23, and
  the Xiaomi "canoe"), the checkpoint's OpenCL serialization gate fired: memory
  mode reports `action=wipe`, overall ~1.0x. The per-seq state blob can't be
  round-tripped through the GPU backend, so the checkpoint **correctly refuses to
  save/restore and falls back to a full wipe — no benefit, but no corruption.**
  This code path only ever executes on-device; host testing can't reach it.
- MediaTek/Mali and Tensor devices have no shell-accessible OpenCL driver, so they
  fall back to CPU and the checkpoint works normally (`restore`, ~3x).
- The OpenCL backend requires OpenCL 3.0 (`clCreateBufferWithProperties`); the
  2018 Adreno 630 (OpenCL 2.0) on the OnePlus 6 cannot load the binary.

Net: **the KV checkpoint is a CPU-backend optimization**; on the OpenCL GPU
backend it is correctly disabled. Raw logs: `results-android/opencl-*.txt`.

## Hexagon NPU — feasibility assessment (not yet run)

The Hexagon backend exists in-tree (`cpp/ggml-hexagon/`, DSP-side code under
`htp/`) and the pieces are available: the Hexagon SDK 6.4.0.2, and all three
Snapdragon devices (S23, OnePlus 6, Xiaomi canoe) expose FastRPC
(`/vendor/lib64/libcdsprpc.so`). Remaining work and blockers:

1. **DSP library build.** `scripts/build-hexagon-htp.sh` builds `libggml-htp-v*.so`
   for the DSP; it needs the containerized Qualcomm toolchain (Docker) or a native
   Linux build. Available on the DGX (has Docker + the SDK), not on macOS.
2. **CPU-side backend.** Add a `-DGGML_HEXAGON=ON` block to `tests/CMakeLists.txt`
   mirroring `android/src/main/rnllama/CMakeLists.txt` (sources `ggml-hexagon.cpp`
   + `htp-drv.cpp`, FastRPC includes, link `libcdsprpc.so`, `-DLM_GGML_USE_HEXAGON`).
3. **Device blocker.** Running **unsigned** DSP code from an adb-shell binary is
   rejected on retail devices (FastRPC unsigned-PD requires a debuggable/engineering
   build or a Qualcomm test signature). The attached units are retail, so this is
   the likely wall — hence not attempted here.

Hypothesis worth testing on an engineering device: unlike OpenCL, the checkpoint
should **work** on Hexagon, because `llama_state_seq_get/set_data` operates on the
CPU-side KV buffers — the DSP only accelerates compute, it doesn't own the state.
If so, Hexagon would keep the CPU-level checkpoint speedups while offloading matmuls.

## Reproduce

Host (any model in `tests/models/`):

```bash
cd tests && ./build_and_test.sh
./build/reuse_test models/mamba-130m-hf-Q3_K_M.gguf bench 8
```

On-device across every attached phone (build here, adb on `$ADB_HOST`):

```bash
# CPU
ADB_HOST=dgx ./run-android.sh models/mamba-130m-hf-Q3_K_M.gguf 8
# GPU (needs ./scripts/build-opencl.sh once, for bin/<abi>/libOpenCL.so)
BACKEND=opencl ADB_HOST=dgx ./run-android.sh models/mamba-130m-hf-Q3_K_M.gguf 6
```

Leave `ADB_HOST` unset if the devices are attached to the build machine.
