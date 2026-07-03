#!/bin/bash
# Run the KV-checkpoint bench on every model in tests/models/ ON THIS MACHINE,
# printing the actual conversation (user prompt + the model's reply, with the
# per-turn reuse action) for both checkpoint off and memory, plus the timing table.
#
# Eyeball coherence -- especially that RESTORE turns produce sane output, and that
# a model without a checkpoint WIPEs (Mamba, Qwen3.5) while one with rollback
# reuses (LFM2.5).
#
#   ./run-local.sh              # 4 turns, real replies (model's own output)
#   ./run-local.sh 6            # 6 turns
#   RNLLAMA_BENCH_REALCHAT=0 ./run-local.sh   # fixed canned replies (deterministic)
#   ./run-local.sh 4 mamba qwen # only models whose filename matches these
set -u
cd "$(dirname "$0")"
[ -x build/reuse_test ] || { echo "build first: ./build_and_test.sh"; exit 1; }

TURNS="${1:-4}"; shift || true
FILTERS=("$@")
export RNLLAMA_BENCH_VERBOSE=1
export RNLLAMA_BENCH_REALCHAT="${RNLLAMA_BENCH_REALCHAT:-1}"

shopt -s nullglob
for m in models/*.gguf; do
    base="$(basename "$m")"
    case "$base" in mmproj-*) continue ;; esac
    if [ "${#FILTERS[@]}" -gt 0 ]; then
        match=0
        for f in "${FILTERS[@]}"; do [[ "$base" == *"$f"* ]] && match=1; done
        [ "$match" = 1 ] || continue
    fi
    echo
    echo "################################ $base ################################"
    ./build/reuse_test "$m" bench "$TURNS"
done
