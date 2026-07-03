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

The win only exists when a turn would otherwise WIPE. That depends on the model:

- **Qwen3.5** (hybrid) is the canonical target. Its chat template strips the empty
  `<think></think>` the model emits, so the re-rendered history diverges from the
  cache mid-sequence every turn and the recurrent `seq_rm` fails -> full wipe
  without the checkpoint. This is the case the feature was built for.
  (`RNLLAMA_BENCH_REALCHAT=1` reproduces the think-strip authentically; the hybrid
  `seq_rm` failure also forces the wipe under the default fixed replies.)
- **Mamba** (pure recurrent) wipes on every divergent turn — the clearest small
  demonstrator.
- **Granite-4** (hybrid mamba2+attn) is mixed: its bounded rollback absorbs later
  turns but not the first few, so it restores early then reuses.
- **LFM2.5** (hybrid conv+attn) absorbs plain append entirely via rollback, so the
  checkpoint is unused there — a true result, not a bench artifact.

## CPU results — Qwen3.5-2B (the real-world case), 5-turn chat

Off-mode TTFT grows linearly with history; memory-mode is flat (tail-only).
Checkpoint blob: **22.7 MB** — a real 2B hybrid state, much closer to the design's
54–114 MB envelope than the toy models below. On a 2B model the wipe cost is
severe: by turn 5 (~278 prompt tokens) off-mode already stalls for **~30 seconds**
on flagship silicon, held flat at ~7.5 s by the checkpoint.

| Device | SoC | overall | turn-5 off → memory |
| --- | --- | --- | --- |
| POCO (2511FPC34G) | Dimensity (mt6899) | 2.47x | 29.1 s → 7.4 s |
| Galaxy S23 | Snapdragon 8 Gen 2 | 2.38x | 29.5 s → 7.7 s |
| Xiaomi (canoe) | Snapdragon | 2.49x | 17.8 s → 4.6 s |
| Pixel 9 | Tensor G4 | 1.74x | fastest CPU; ratio still climbing at turn 5 |

The overall multiplier is lower than Mamba's mainly because a 2B model's memory-mode
floor (tail reprocess + generation) is itself ~7 s, and the ratio keeps growing with
turn count. The absolute saving is the story: ~22 s of stall removed per turn by
turn 5, for 22.7 MB. (The two oldest phones, OnePlus 6 / Redmi, weren't run at 2B —
a wipe there would be minutes.) Raw logs: `results-android/cpu-Qwen*.txt`.

## CPU results — Mamba-130M, 8-turn chat (clearest small demonstrator)

Off-mode TTFT grows linearly with history; memory-mode stays flat. Checkpoint blob:
**2.67 MB, constant**. Overall speedup and worst single-turn off stall (turn 8,
~614 prompt tokens):

| Device | SoC | overall | turn-8 off → memory |
| --- | --- | --- | --- |
| Pixel 9 | Tensor G4 | 3.5x | 4.6 s → 0.72 s |
| POCO (2511FPC34G) | Dimensity (mt6899) | 4.0x | 4.7 s → 0.68 s |
| Galaxy S23 | Snapdragon 8 Gen 2 | 3.8x | 4.9 s → 0.70 s |
| OnePlus 6 | Snapdragon 845 | 3.9x | 10.2 s → 1.43 s |
| Redmi (Helio) | Helio (mt6768) | 4.2x | 18.7 s → 2.5 s |
| Xiaomi (canoe) | Snapdragon | ~7x | flat ~0.45 s |

On low-end silicon the checkpoint converts an 18-second stall into 2.5 s for a
constant 2.67 MB. Granite-4 (hybrid) sits between Mamba and LFM2.5 — mixed
restore/reuse, ~1.3x on most devices. Raw logs: `results-android/cpu-mamba-*.txt`.

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

Run the full behavioral + performance suites with real transcripts across every
downloaded model — includes the append/recall, edit-no-leak, and new-session-no-leak
scenarios (plus VLM/tools where applicable) *and* the off-vs-memory perf chat:

```bash
./run-local.sh                 # all models
./run-local.sh qwen mamba      # only matching models
BENCH_TURNS=6 ./run-local.sh   # longer perf chat
SKIP_BENCH=1 ./run-local.sh    # behavioral scenarios only (SKIP_EVAL=1 = perf only)
```

Each model's full transcript is printed to the terminal and saved to
`results-local/<model>.txt` (git-ignored).

Note: the app does NOT need to call `clearCache` between chats. A new
conversation that doesn't extend the cached prefix is rejected by the checkpoint's
token-prefix guard and reprocessed from scratch (`action=wipe`, no leak) — proven
by `reuse_test`'s "new-chat (no clearCache) leak gate". `clearCache` is an optional
explicit reset that also frees the snapshot blob.

For just the perf view of one model: `RNLLAMA_BENCH_VERBOSE=1 ./build/reuse_test
models/<m>.gguf bench 4` (add `RNLLAMA_BENCH_REALCHAT=1` for a genuine multi-turn
chat). For just the correctness gates of one model: `./build/reuse_test models/<m>.gguf`
(runs the clearCache leak gate + checkpoint validation).

On-device across every attached phone (build here, adb on `$ADB_HOST`):

```bash
# CPU
ADB_HOST=dgx ./run-android.sh models/mamba-130m-hf-Q3_K_M.gguf 8
# GPU (needs ./scripts/build-opencl.sh once, for bin/<abi>/libOpenCL.so)
BACKEND=opencl ADB_HOST=dgx ./run-android.sh models/mamba-130m-hf-Q3_K_M.gguf 6
```

Leave `ADB_HOST` unset if the devices are attached to the build machine.
