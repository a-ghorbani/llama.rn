#!/bin/bash
# Run the KV-checkpoint suites on every model in tests/models/ ON THIS MACHINE,
# printing real conversations so you can eyeball behavior. Two parts per model:
#
#  1. BEHAVIORAL (eval_scenarios_test --smoke): full transcripts + hard asserts for
#       A  append / multi-turn recall
#       B  edit a turn  -> must reflect the edit and NOT leak the old name/hobby
#       C  new session after clearCache -> must NOT leak the prior chat, stay coherent
#       D  VLM image grounding (only if an mmproj-<model>.gguf sits next to the model)
#       E  tool calling  (only with --tools)
#     --smoke keeps the structural/leak asserts HARD; recall/coherence become
#     informational so a weak small model doesn't spuriously fail.
#
#  2. PERFORMANCE (reuse_test bench, verbose): the checkpoint off-vs-memory growing
#     chat, printing each turn's reply + reuse action (cold/reuse/restore/wipe).
#
#   ./run-local.sh                 # all models
#   ./run-local.sh qwen mamba      # only models whose filename matches
#   BENCH_TURNS=6 ./run-local.sh   # longer perf chat
#   WITH_TOOLS=1 ./run-local.sh    # add scenario E
#   SKIP_BENCH=1 ./run-local.sh    # behavioral only    (SKIP_EVAL=1 for perf only)
set -u
cd "$(dirname "$0")"
for exe in eval_scenarios_test reuse_test; do
    [ -x "build/$exe" ] || { echo "build first: ./build_and_test.sh"; exit 1; }
done

FILTERS=("$@")
BENCH_TURNS="${BENCH_TURNS:-4}"
DEF_IMG1="../third_party/llama.cpp/tools/mtmd/test-1.jpeg"
DEF_IMG2="../third_party/llama.cpp/media/matmul.png"
RESULTS="results-local"       # each model's full transcript is also saved here
mkdir -p "$RESULTS"

shopt -s nullglob
for m in models/*.gguf; do
    base="$(basename "$m")"
    case "$base" in mmproj-*) continue ;; esac
    if [ "${#FILTERS[@]}" -gt 0 ]; then
        match=0; lc_base="$(echo "$base" | tr '[:upper:]' '[:lower:]')"
        for f in "${FILTERS[@]}"; do
            lc_f="$(echo "$f" | tr '[:upper:]' '[:lower:]')"
            [[ "$lc_base" == *"$lc_f"* ]] && match=1
        done
        [ "$match" = 1 ] || continue
    fi
    # Run all of this model's output through one group piped to tee, so it prints
    # to the terminal AND is saved under results-local/<model>.txt.
    OUT="$RESULTS/${base%.gguf}.txt"
    {
        echo
        echo "############################## $base ##############################"

        if [ "${SKIP_EVAL:-0}" != 1 ]; then
            echo "---- BEHAVIORAL (append / edit-no-leak / new-session-no-leak) ----"
            EVAL_ARGS=("$m" --smoke)
            stem="${base%.gguf}"; stem="${stem%-Q[0-9]*}"; stem="${stem%-IQ[0-9]*}"
            mmcands=(models/mmproj-*"${stem}"*.gguf)
            if [ "${#mmcands[@]}" -gt 0 ]; then
                EVAL_ARGS+=(--mmproj "${mmcands[0]}" --image "$DEF_IMG1" --image2 "$DEF_IMG2")
            fi
            [ "${WITH_TOOLS:-0}" = 1 ] && EVAL_ARGS+=(--tools)
            ./build/eval_scenarios_test "${EVAL_ARGS[@]}"
        fi

        if [ "${SKIP_BENCH:-0}" != 1 ]; then
            echo "---- PERFORMANCE (checkpoint off vs memory, growing chat) ----"
            RNLLAMA_BENCH_VERBOSE=1 RNLLAMA_BENCH_REALCHAT="${RNLLAMA_BENCH_REALCHAT:-1}" \
                ./build/reuse_test "$m" bench "$BENCH_TURNS"
        fi
    } 2>&1 | tee "$OUT"
    echo "   saved -> $OUT"
done
