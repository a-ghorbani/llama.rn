# C++ Tests for llama.rn

Integration and unit tests for the llama.rn C++ core (`cpp/`), including the
KV-cache reuse / KV-checkpoint feature of the legacy completion path.

## Quick start

```bash
cd tests

# Build every test executable (shared core is compiled once)
./build_and_test.sh

# Run the default suite (bundled tiny model, exits non-zero on failure)
./run_tests.sh
```

Or with CMake directly:

```bash
cd tests
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
```

### Requirements

- CMake 3.16+, C++17 compiler
- Bundled test model: `tests/tiny-random-llama.gguf` (pure-attention, random
  weights — fast, deterministic with `temp=0`, but it cannot "answer" anything,
  so capability asserts are only meaningful on real models)

## Model-gated suites

The default `./run_tests.sh` run only uses the bundled tiny model. Suites that
need a real model are gated behind env vars and are reported as SKIP otherwise:

| Env var | Model to provide | What it unlocks |
| --- | --- | --- |
| `RNLLAMA_TEST_HYBRID_MODEL` | hybrid/SSM chat gguf (LFM2/LFM2.5, Granite-4, Mamba, Qwen3.5, Falcon-H) | `reuse_test` KV-checkpoint validation gates (leak, perf, 2nd-reuse chain, determinism, restore-failure, ctx-shift, embedding no-clobber) |
| `RNLLAMA_TEST_MODEL` | any real chat gguf | `eval_scenarios_test` full eval (hard capability asserts) |
| `RNLLAMA_TEST_MMPROJ` / `RNLLAMA_TEST_IMAGE` / `RNLLAMA_TEST_IMAGE2` | matching mmproj + two test images | eval scenario D (VLM grounding, same-image reuse, image-swap answer change) |
| `RNLLAMA_TEST_TOOLS=1` | — | eval scenario E (tool calling consistency) |
| `RNLLAMA_TEST_SWA_MODEL` | SWA gguf (Gemma 3 / Gemma 3n) | `swa_cache_corruption_test` — KNOWN-RED repro, see below |

Example:

```bash
RNLLAMA_TEST_HYBRID_MODEL=~/models/lfm2-1.2b-q4_k_m.gguf \
RNLLAMA_TEST_MODEL=~/models/qwen3-1.7b-q4_k_m.gguf \
./run_tests.sh
```

## Test executables

### `rnllama_tests` (`simple_test.cpp`)

Basic integration: model load, tokenize, a short completion, embeddings plumbing.
Runs on the bundled tiny model. Part of the default suite.

### `parallel_decoding_test` (`parallel_decoding_test.cpp`)

Slot-manager / request-queue coverage for parallel decoding (multiple concurrent
completions, cancellation, slot reuse). Runs on the bundled tiny model. Part of
the default suite.

### `reuse_test` (`reuse_test.cpp`)

The KV-cache reuse + KV-checkpoint suite. Three layers:

1. **Unit section** — always runs (also with no args) and always gates the exit
   code. Asserts the `KvCheckpoint` invariants against `cpp/rn-kv-checkpoint.hpp`
   and `cpp/rn-completion.cpp`:
   - `is_prefix_of` truth table (shorter / exact / extension / mismatches / invalid)
   - `save()` arg-check asymmetry: bad arguments (`ctx==nullptr`, `n_boundary<=0`,
     `n_boundary>tokens.size()`) return `false` **without** invalidating an
     existing snapshot (deliberate: a bad call must not cost the conversation its
     restore point)
   - backend refusal (`backend_serializable=false`) returns `false` **and**
     invalidates a prior snapshot (a stale blob would restore wrong state on
     OpenCL)
   - `invalidate()` really frees (`capacity()==0` via the swap idiom), for both a
     manual multi-MB blob and a real saved blob
   - `cancelKvCheckpointSave()` clears the pending capture but **keeps** the
     snapshot; `dropKvCheckpoint()` clears and frees everything
   - **B3 regression:** after any completion, `endCompletion()` must trim the
     never-decoded tail so `llama_memory_seq_pos_max(mem,0)+1 == embd.size()`
   Uses the bundled tiny model.
2. **Gated validation** — run as `./reuse_test <hybrid-model.gguf>` (or via
   `RNLLAMA_TEST_HYBRID_MODEL`). Needs a recurrent/hybrid model, because only
   there does mid-sequence `seq_rm` fail and the restore path fire; on a
   pure-attention model these gates SKIP loudly instead of fake-passing.
   - clearCache leak gate (stale checkpoint must not leak across chats)
   - PERF gate (memory-mode turn 2 reuses strictly more than the wiped baseline)
   - 2nd-reuse chain (an overshoot turn must not clobber the trimmed boundary)
   - correctness gate (restore output coherent + deterministic mem-vs-mem)
   - restore-failure fallback (corrupted blob ⇒ `WIPE`, coherent output, fresh
     checkpoint re-saved)
   - ctx-shift invalidation (a mid-generation context shift drops the
     checkpoint; the next turn must not `RESTORE`)
   - embedding no-clobber (`embedding()` uses `loadPrompt(..., false)`; the chat
     checkpoint blob must survive, and the next divergent chat turn must still
     `RESTORE`)
3. **Descriptive scenarios** — no-arg runs also print reuse behavior for clean
   extension / mid-divergence / faithful round-trip prompts (observability, no
   asserts).

Extra mode: `./reuse_test <model.gguf> size` prints the serialized per-sequence
state size across context lengths (checkpoint memory footprint per arch).

### `eval_scenarios_test` (`eval_scenarios_test.cpp`)

Model-agnostic evaluation harness printing full transcripts plus machine
asserts. For each KV mode in `{off, memory}` it loads a **fresh context**
(clearCache does not fully reset SWA state, which would contaminate the
cross-mode comparison) and runs:

- **A. APPEND** — multi-turn memory/recall; per-turn reuse action recorded.
- **B. EDIT** — regenerate an edited turn; must reflect the edit and not leak
  the old name/hobby (word-boundary match).
- **C. NEW SESSION** — isolation + coherence after `clearCache`.
- **D. VLM** (`--mmproj`/`--image`, optional `--image2`) — image grounding; the
  same-image follow-up must actually REUSE the image+text prefix
  (`reused > 0`, mtmd bitmap-hash path); with `--image2` the answer to the same
  question must CHANGE when the image is swapped (greedy decoding — a model
  that ignores images fails this).
- **E. TOOLS** (`--tools`) — tool-call behavior must be identical off vs memory.

Cross-mode asserts: off==memory token identity when no RESTORE fired (the
checkpoint must be a byte-level no-op there), coherence when a RESTORE did fire
(SSM restores are not bit-identical to a full reprocess by FP non-associativity),
and per-turn pairing: any turn the off baseline WIPEd, memory mode must RESTORE.

`--smoke` (what the default suite runs with the tiny model): capability asserts
(recall/coherence) become informational; structural asserts (token identity,
isolation, wipe→restore pairing) stay hard.

```bash
./eval_scenarios_test <model.gguf> [--mmproj <p>] [--image <p>] [--image2 <p>] [--tools] [--smoke]
```

### `swa_cache_corruption_test` (`swa_cache_corruption_test.cpp`) — KNOWN-RED

Deterministic repro for the **unfixed** SWA evicted-prefix reuse bug: on an
iSWA model (Gemma 3 / 3n), after a turn longer than `n_swa`, the sliding window
physically evicts the shared prefix, but `seq_rm` still reports success, so
`loadPrompt` "reuses" cells that no longer exist in the SWA layers.

- Needs a real SWA model (`RNLLAMA_TEST_SWA_MODEL`, e.g. a Gemma-3 gguf); on a
  dense model it exits 0 with a SKIP note.
- **Exit 1 is the expected result today** — it proves the bug, it is not a
  regression in your change. The primary assert (`n_past==0` after an evicted
  prefix) flips to PASS once `loadPrompt` gains a `llama_memory_seq_pos_min`
  guard (server parity: `pos_min_thold` in `server-context.cpp`).
- Wired into the build so it cannot rot, but excluded from the default run;
  `run_tests.sh` runs it only when `RNLLAMA_TEST_SWA_MODEL` is set and never
  counts its failure against the suite. If it ever PASSES, the bug got fixed:
  promote it to a regular gated suite and update this section.

## Shared helpers

`test-helpers.hpp` — header-only utilities used by the suites: `Msg`/`to_json`,
`lower`/`contains`/`contains_word` (word-boundary leak checks),
`looks_like_garbage`, `check()` pass/fail counters, `format_chat` (Jinja chat
rendering), and `run_turn` (the one-turn driver mirroring the real JSI flow:
`rewind → initSampling → beginCompletion → loadPrompt → doCompletion loop →
endCompletion`, with `TurnOptions` for checkpoint on/off, media, and tool
wiring).

## Historical notes

- `chat_scenarios_test.cpp` was removed: S1/S3/S4 are covered by eval scenarios
  A and C (with pinned `temp=0`/seed instead of nondeterministic defaults, and
  the direct `kv_checkpoint_enabled` field instead of an env-var toggle), and
  its S5 "SWA garbage" scenario never met the repro preconditions (see the
  header of `swa_cache_corruption_test.cpp`).
- `vlm_test.cpp` was removed: subsumed by eval scenario D, including its unique
  same-image reuse check (now a hard assert).
