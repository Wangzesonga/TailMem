#!/usr/bin/env bash
set -euo pipefail

# Run the exact paper method inside the accompanying modified EpiCache tree.
# Usage:
#   EPICACHE_ROOT=/path/to/ml-epicache GPU=0 \
#   ./scripts/run_frozen_eval.sh llama locomo '/path/to/manifests/history_*.json' output.json

LABEL=${1:?model label: qwen|qwen25_3b|qwen3|llama}
DATASET=${2:?dataset: locomo|realtalk|longmemeval}
MANIFEST_PATTERN=${3:?quoted manifest glob}
OUTPUT=${4:?output JSON}
EPICACHE_ROOT=${EPICACHE_ROOT:?set EPICACHE_ROOT to the modified EpiCache repository}
GPU=${GPU:-0}
METHOD=${METHOD:-v12_gemini_bipartite_syncfree_exact_final_native}

mapfile -t MANIFESTS < <(compgen -G "$MANIFEST_PATTERN" | sort)
if [[ ${#MANIFESTS[@]} -eq 0 ]]; then
  echo "No manifests matched: $MANIFEST_PATTERN" >&2
  exit 2
fi

cd "$EPICACHE_ROOT"
CUDA_VISIBLE_DEVICES="$GPU" \
HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} \
TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1} \
PYTHONPATH="$EPICACHE_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
python -u -m experiments_new.graph_suite.runner \
  --manifest "${MANIFESTS[@]}" \
  --label "$LABEL" \
  --dataset "$DATASET" \
  --methods "$METHOD" \
  --output "$OUTPUT"
