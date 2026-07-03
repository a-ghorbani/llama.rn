#!/bin/bash
# Runs the llama.rn C++ test suites.
#
# Default (no env vars): every suite that works with the bundled tiny model
# (tests/tiny-random-llama.gguf) runs, and the script exits non-zero on any
# real failure.
#
# Model-gated suites (skipped unless the env var is set):
#   RNLLAMA_TEST_HYBRID_MODEL=<path>  hybrid/SSM gguf (LFM2, Granite, Mamba,
#                                     Qwen3.5, ...) -> reuse_test KV-checkpoint
#                                     validation gates
#   RNLLAMA_TEST_MODEL=<path>         real chat gguf -> eval_scenarios_test full
#                                     eval (hard capability asserts)
#   RNLLAMA_TEST_MMPROJ=<path>        mmproj gguf   -> adds eval scenario D (VLM)
#   RNLLAMA_TEST_IMAGE=<path>         image for scenario D
#   RNLLAMA_TEST_IMAGE2=<path>        second image for the D image-swap assert
#   RNLLAMA_TEST_TOOLS=1              adds eval scenario E (tool calling)
#   RNLLAMA_TEST_SWA_MODEL=<path>     SWA gguf (Gemma 3 / 3n) -> runs the
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

# ---------------- model-gated suites ----------------

if [ -n "${RNLLAMA_TEST_HYBRID_MODEL:-}" ]; then
    run_suite "KV checkpoint validation on hybrid model (reuse_test)" \
        reuse_test "$RNLLAMA_TEST_HYBRID_MODEL"
else
    skip_suite "KV checkpoint validation (reuse_test <hybrid>)" "set RNLLAMA_TEST_HYBRID_MODEL"
fi

if [ -n "${RNLLAMA_TEST_MODEL:-}" ]; then
    EVAL_ARGS=("$RNLLAMA_TEST_MODEL")
    [ -n "${RNLLAMA_TEST_MMPROJ:-}" ] && EVAL_ARGS+=(--mmproj "$RNLLAMA_TEST_MMPROJ")
    [ -n "${RNLLAMA_TEST_IMAGE:-}" ]  && EVAL_ARGS+=(--image "$RNLLAMA_TEST_IMAGE")
    [ -n "${RNLLAMA_TEST_IMAGE2:-}" ] && EVAL_ARGS+=(--image2 "$RNLLAMA_TEST_IMAGE2")
    [ "${RNLLAMA_TEST_TOOLS:-0}" = "1" ] && EVAL_ARGS+=(--tools)
    run_suite "eval scenarios full (eval_scenarios_test)" eval_scenarios_test "${EVAL_ARGS[@]}"
else
    skip_suite "eval scenarios full (eval_scenarios_test <model>)" "set RNLLAMA_TEST_MODEL"
fi

# KNOWN-RED repro: documents the unfixed SWA evicted-prefix reuse bug. Runs only
# on request and never fails the suite; a PASS means the bug got fixed (then
# promote it to a regular gated suite and update README.md).
if [ -n "${RNLLAMA_TEST_SWA_MODEL:-}" ]; then
    echo "--- SWA evicted-prefix repro (swa_cache_corruption_test, KNOWN-RED) ---"
    if ./swa_cache_corruption_test "$RNLLAMA_TEST_SWA_MODEL"; then
        echo "✓ SWA repro PASSED — the seq_pos_min guard appears to be fixed; update README.md"
        SUMMARY+="  PASS swa_cache_corruption_test (bug appears FIXED - promote this suite)\n"
    else
        echo "✗ SWA repro failed as EXPECTED (known unfixed bug; not counted)"
        SUMMARY+="  KNOWN-RED swa_cache_corruption_test (expected failure, not counted)\n"
    fi
    echo ""
else
    skip_suite "SWA evicted-prefix repro (swa_cache_corruption_test)" "set RNLLAMA_TEST_SWA_MODEL; KNOWN-RED"
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
