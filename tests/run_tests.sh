#!/bin/bash
# Runs the llama.rn C++ test suites.
#
# Default (no env vars): every suite that works with the bundled tiny model
# (tests/tiny-random-llama.gguf) runs, and the script exits non-zero on any
# real failure.
#
# Real models: drop any gguf into tests/models/ (see ./download-models.sh) and
# it is picked up automatically -- reuse_test and eval_scenarios_test detect the
# architecture at runtime and SKIP the suites that don't apply, so a pure-attn,
# hybrid, recurrent, or SWA model each exercises exactly what it should. An mmproj
# sitting next to a model (mmproj-*.gguf) is wired into eval scenario D.
#
# Explicit overrides (take precedence; skipped unless set):
#   RNLLAMA_TEST_HYBRID_MODEL=<path>  hybrid/SSM gguf -> reuse_test validation gates
#   RNLLAMA_TEST_MODEL=<path>         real chat gguf -> eval_scenarios_test full eval
#   RNLLAMA_TEST_MMPROJ=<path>        mmproj gguf   -> adds eval scenario D (VLM)
#   RNLLAMA_TEST_IMAGE=<path>         image for scenario D
#   RNLLAMA_TEST_IMAGE2=<path>        second image for the D image-swap assert
#   RNLLAMA_TEST_TOOLS=1              adds eval scenario E (tool calling)
#   RNLLAMA_TEST_SWA_MODEL=<path>     SWA gguf (Gemma 4 / 3 / 3n) -> runs the
#                                     KNOWN-RED swa_cache_corruption_test repro;
#                                     its failure is EXPECTED (unfixed bug) and
#                                     does not fail the suite. See README.md.
set -u

cd "$(dirname "$0")"

echo "=== Running llama.rn C++ Tests ==="
echo ""

if [ ! -d "build" ]; then
    echo "Error: build directory not found"
    echo "Please run ./build_and_test.sh first"
    exit 1
fi

cd build

TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0
SUMMARY=""

# run_suite <label> <executable> [args...]
run_suite() {
    local label="$1"; shift
    local exe="$1"; shift
    if [ ! -f "$exe" ]; then
        echo "Error: $exe executable not found (run ./build_and_test.sh first)"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        SUMMARY+="  FAIL $label (not built)\n"
        return
    fi
    echo "--- $label ---"
    if "./$exe" "$@"; then
        echo "✓ $label passed"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        SUMMARY+="  PASS $label\n"
    else
        echo "✗ $label failed"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        SUMMARY+="  FAIL $label\n"
    fi
    echo ""
}

skip_suite() {
    TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    SUMMARY+="  SKIP $1 ($2)\n"
    echo "--- $1: SKIPPED ($2) ---"
    echo ""
}

# ---------------- default suites (bundled tiny model) ----------------

run_suite "basic integration (rnllama_tests)" rnllama_tests
run_suite "parallel decoding (parallel_decoding_test)" parallel_decoding_test
# reuse_test with no args: KV-checkpoint unit section (always gates exit code)
# + descriptive reuse scenarios on the tiny model.
run_suite "KV checkpoint units + reuse smoke (reuse_test)" reuse_test
# eval scenarios in smoke mode: structural asserts (cross-mode token identity,
# isolation) stay hard; capability asserts (recall/coherence) are informational.
run_suite "eval scenarios smoke (eval_scenarios_test --smoke)" \
    eval_scenarios_test ../tiny-random-llama.gguf --smoke

# ---------------- KNOWN-RED SWA repro helper ----------------

# Documents the unfixed SWA evicted-prefix reuse bug. Never fails the suite; a
# PASS means the bug got fixed (then promote it and update README.md).
# $1 model. If quiet=1 (2nd arg), a non-SWA skip (exit 3) prints and counts
# nothing -- used by the auto-loop, which attempts the repro for every model.
run_swa_repro() {
    local model="$1" quiet="${2:-0}"
    ./swa_cache_corruption_test "$model" > /tmp/rnllama_swa.out 2>&1
    local rc=$?
    if [ "$rc" -eq 3 ]; then
        [ "$quiet" = "1" ] && return
        cat /tmp/rnllama_swa.out
        skip_suite "SWA evicted-prefix repro (swa_cache_corruption_test)" "model is not SWA"
        return
    fi
    echo "--- SWA evicted-prefix repro (swa_cache_corruption_test, KNOWN-RED): $(basename "$model") ---"
    cat /tmp/rnllama_swa.out
    if [ "$rc" -eq 0 ]; then
        echo "✓ SWA repro PASSED — the seq_pos_min guard appears to be fixed; update README.md"
        SUMMARY+="  PASS swa_cache_corruption_test (bug appears FIXED - promote this suite)\n"
    else
        echo "✗ SWA repro failed as EXPECTED (known unfixed bug; not counted)"
        SUMMARY+="  KNOWN-RED swa_cache_corruption_test (expected failure, not counted)\n"
    fi
    echo ""
}

# ---------------- auto-discovered models (tests/models/*.gguf) ----------------
# Each model runs through both reuse_test and eval_scenarios_test; the binaries
# self-detect arch and SKIP the gates that don't apply, so one loop covers
# pure-attn / hybrid / recurrent / SWA. mmproj-*.gguf files are not models --
# they attach to the model with the matching stem for eval scenario D.

shopt -s nullglob
MODEL_FILES=(../models/*.gguf)
shopt -u nullglob

for model in "${MODEL_FILES[@]}"; do
    base="$(basename "$model")"
    case "$base" in mmproj-*) continue ;; esac  # projector, not a standalone model

    run_suite "reuse_test :: $base" reuse_test "$model"

    # --smoke: feature/structural asserts stay hard (token identity, isolation,
    # wipe->restore pairing) but capability asserts (recall/coherence) are
    # informational -- the auto-loop validates the FEATURE across architectures,
    # not model quality, so a weak 130-350M model must not red-flag it. Use the
    # RNLLAMA_TEST_MODEL override for a full-capability eval on a chosen model.
    EVAL_ARGS=("$model" --smoke)
    # Attach a projector for scenario D. The mmproj quant usually differs from the
    # model's (e.g. model -Q4_K_M, mmproj -f16), so match on the family stem (base
    # minus the trailing quant token) rather than an exact name.
    stem="${base%.gguf}"; stem="${stem%-Q[0-9]*}"; stem="${stem%-IQ[0-9]*}"
    shopt -s nullglob
    mmcands=(../models/mmproj-*"${stem}"*.gguf)
    shopt -u nullglob
    if [ "${#mmcands[@]}" -gt 0 ]; then
        EVAL_ARGS+=(--mmproj "${mmcands[0]}")
        # Scenario D needs images. Default to two visually distinct images bundled
        # with llama.cpp (the image-swap assert needs the answers to differ) unless
        # the caller supplied their own.
        DEF_IMG1="../../third_party/llama.cpp/tools/mtmd/test-1.jpeg"
        DEF_IMG2="../../third_party/llama.cpp/media/matmul.png"
        EVAL_ARGS+=(--image "${RNLLAMA_TEST_IMAGE:-$DEF_IMG1}")
        EVAL_ARGS+=(--image2 "${RNLLAMA_TEST_IMAGE2:-$DEF_IMG2}")
    else
        [ -n "${RNLLAMA_TEST_IMAGE:-}" ]  && EVAL_ARGS+=(--image "$RNLLAMA_TEST_IMAGE")
        [ -n "${RNLLAMA_TEST_IMAGE2:-}" ] && EVAL_ARGS+=(--image2 "$RNLLAMA_TEST_IMAGE2")
    fi
    [ "${RNLLAMA_TEST_TOOLS:-0}" = "1" ] && EVAL_ARGS+=(--tools)
    run_suite "eval_scenarios :: $base" eval_scenarios_test "${EVAL_ARGS[@]}"

    # SWA models additionally drive the KNOWN-RED repro; quiet=1 so a dense model
    # (exit 3) is silently ignored rather than reported.
    run_swa_repro "$model" 1
done

# ---------------- explicit env-var overrides ----------------

if [ -n "${RNLLAMA_TEST_HYBRID_MODEL:-}" ]; then
    run_suite "KV checkpoint validation on hybrid model (reuse_test)" \
        reuse_test "$RNLLAMA_TEST_HYBRID_MODEL"
fi

if [ -n "${RNLLAMA_TEST_MODEL:-}" ]; then
    EVAL_ARGS=("$RNLLAMA_TEST_MODEL")
    [ -n "${RNLLAMA_TEST_MMPROJ:-}" ] && EVAL_ARGS+=(--mmproj "$RNLLAMA_TEST_MMPROJ")
    [ -n "${RNLLAMA_TEST_IMAGE:-}" ]  && EVAL_ARGS+=(--image "$RNLLAMA_TEST_IMAGE")
    [ -n "${RNLLAMA_TEST_IMAGE2:-}" ] && EVAL_ARGS+=(--image2 "$RNLLAMA_TEST_IMAGE2")
    [ "${RNLLAMA_TEST_TOOLS:-0}" = "1" ] && EVAL_ARGS+=(--tools)
    run_suite "eval scenarios full (eval_scenarios_test)" eval_scenarios_test "${EVAL_ARGS[@]}"
fi

[ -n "${RNLLAMA_TEST_SWA_MODEL:-}" ] && run_swa_repro "$RNLLAMA_TEST_SWA_MODEL"

if [ "${#MODEL_FILES[@]}" -eq 0 ] && [ -z "${RNLLAMA_TEST_HYBRID_MODEL:-}${RNLLAMA_TEST_MODEL:-}${RNLLAMA_TEST_SWA_MODEL:-}" ]; then
    skip_suite "real-model suites (reuse_test / eval_scenarios / swa repro)" \
        "no models in tests/models/ -- run ./download-models.sh"
fi

# ---------------- summary ----------------

echo "=== Test Summary ==="
printf "%b" "$SUMMARY"
echo "Passed: $TESTS_PASSED  Failed: $TESTS_FAILED  Skipped: $TESTS_SKIPPED"

if [ "$TESTS_FAILED" -eq 0 ]; then
    echo "✓ All executed test suites passed"
    exit 0
else
    echo "✗ Some tests failed"
    exit 1
fi
