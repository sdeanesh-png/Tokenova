#!/usr/bin/env bash
#
# Fetches the MiniLM-L6-v2 ONNX export from HuggingFace into ./models/.
# Required only when the `onnx` Cargo feature is enabled on the classifier
# crate; the default build uses the zero-dependency HeuristicEmbedder.
#
# Usage:
#   scripts/fetch-model.sh [MODEL_VARIANT]
#
# MODEL_VARIANT defaults to `all-MiniLM-L6-v2` (22M params, 384 dim).
# Alternatives:
#   all-MiniLM-L12-v2  (33M params, 384 dim — closer to PRD §6.2's 80M spec)

set -euo pipefail

MODEL="${1:-all-MiniLM-L6-v2}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/models"
mkdir -p "$DEST"

# Xenova maintains ONNX exports of the sentence-transformers models.
BASE="https://huggingface.co/Xenova/${MODEL}/resolve/main"

echo "fetching ${MODEL} ONNX model + tokenizer → ${DEST}"
curl -fL -o "${DEST}/${MODEL}.onnx"          "${BASE}/onnx/model.onnx"
curl -fL -o "${DEST}/${MODEL}.tokenizer.json" "${BASE}/tokenizer.json"

cat <<EOF

done.

To build with ONNX classification enabled:
  # macOS (Homebrew):
  brew install onnxruntime
  export ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib

  cargo build --release --features "tokenova-classifier/onnx" -p tokenova-proxy

EOF
