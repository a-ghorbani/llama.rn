#!/bin/bash
# Downloads one real model per memory architecture the KV-checkpoint feature
# branches on, into tests/models/ (git-ignored). run_tests.sh then auto-routes
# each *.gguf to the suite its architecture applies to; the test binaries detect
# arch at runtime (llama_model_is_recurrent / is_hybrid / n_swa) and SKIP the
# suites that don't apply, so nothing needs per-model wiring here.
#
# Usage:
#   ./download-models.sh                 # core set (skips the ~4 GB Gemma-4)
#   ./download-models.sh --all           # everything, including Gemma-4 + mmproj
#   ./download-models.sh smollm2 lfm2    # only the named models
#   QUANT=Q4_K_M ./download-models.sh    # smaller quant (default Q8_0; Gemma-4 is always Q4_K_M)
#
# Model coverage:
#   smollm2   pure-attention        checkpoint is a no-op; off==memory bit-identical
#   lfm2      hybrid (conv+attn)    main restore path (LFM2.5, arch reports "lfm2")
#   granite4  hybrid (mamba2+attn)  different hybrid state layout
#   mamba     pure recurrent (SSM)  seq_rm always fails, no attention state
#   gemma4    SWA + vision          KNOWN-RED swa test, the n_swa gate, and VLM scenario D
set -eu

cd "$(dirname "$0")/models"

QUANT="${QUANT:-Q8_0}"
HF="https://huggingface.co"

# name  repo  file-template ($QUANT expands)   -- one line per model
# Gemma-4 is pinned to Q4_K_M (Q8 is ~4.7 GB) and pulls its mmproj too.
manifest() {
    case "$1" in
        smollm2)  echo "prithivMLmods/SmolLM2-135M-Instruct-GGUF|SmolLM2-135M-Instruct.${QUANT}.gguf" ;;
        # LFM2.5 declares general.architecture=lfm2 (same code path), newer + smaller
        # than LFM2-350M. Pinned to Q4_K_M (this repo has no Q8_0 for the 230M).
        lfm2)     echo "LiquidAI/LFM2.5-230M-GGUF|LFM2.5-230M-Q4_K_M.gguf" ;;
        granite4) echo "unsloth/granite-4.0-h-350m-GGUF|granite-4.0-h-350m-${QUANT}.gguf" ;;
        # Pinned quant: the QuantFactory Q8_0 fails to load on the current
        # llama.cpp sync (unreadable tensor info); tensorblock's Q3_K_M loads.
        mamba)    echo "tensorblock/mamba-130m-hf-GGUF|mamba-130m-hf-Q3_K_M.gguf" ;;
        gemma4)   echo "bartowski/google_gemma-4-E2B-it-GGUF|google_gemma-4-E2B-it-Q4_K_M.gguf"
                  echo "bartowski/google_gemma-4-E2B-it-GGUF|mmproj-google_gemma-4-E2B-it-f16.gguf" ;;
        *) echo "unknown model: $1" >&2; return 1 ;;
    esac
}

CORE="smollm2 lfm2 granite4 mamba"
ALL="$CORE gemma4"

if [ "$#" -eq 0 ]; then
    SELECTED="$CORE"
    echo "Downloading core set (quant=$QUANT). Add gemma4 or --all for the ~4 GB SWA/VLM model."
elif [ "$1" = "--all" ]; then
    SELECTED="$ALL"
else
    SELECTED="$*"
fi

fetch() {
    local repo="$1" file="$2"
    if [ -f "$file" ]; then
        echo "  ✓ $file (already present)"
        return
    fi
    echo "  ↓ $file"
    # -C - resumes a partial download; write to a .part then rename so an
    # interrupted run never leaves a truncated .gguf that looks complete.
    curl -fL -C - -o "$file.part" "$HF/$repo/resolve/main/$file"
    mv "$file.part" "$file"
}

for name in $SELECTED; do
    echo "== $name =="
    manifest "$name" | while IFS='|' read -r repo file; do
        fetch "$repo" "$file"
    done
done

echo ""
echo "Done. Run the full matrix with:  ./run_tests.sh"
echo "Models in tests/models/ are picked up automatically."
