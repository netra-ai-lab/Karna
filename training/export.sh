#!/usr/bin/env bash
# Export a model as the ONNX model folder the karna package runs (no PyTorch needed there).
#
#   training/export.sh SOURCE [OUT] [-- extra export_onnx flags]
#
# SOURCE is a PyTorch release folder (scripts/06_export_pruned.py), a training checkpoint,
# a short name (karna-base) or a Hugging Face repo id. OUT defaults to onnx/<name> next to a
# release folder. The folder is then checked against PyTorch through the package's own code;
# CHECK=<folder of audio, or a list of paths> adds real audio to that check.
#
# Use it:
#   KARNA_BUNDLES=$(dirname OUT) karna transcribe speech.wav --model <name>
set -euo pipefail
cd "$(dirname "$0")"
source scripts/env.sh
[[ $# -eq 0 || "$1" == -h || "$1" == --help ]] && { sed -n '2,/^$/{s/^# \{0,1\}//;p}' "$0"; exit 0; }

src="$1"; shift
if [[ $# -gt 0 && "$1" != --* ]]; then out="$1"; shift; else
  name="$(basename "$src")"; name="${name%.pt}"
  if [[ -d "$src" ]]; then out="$(dirname "$src")/onnx/$name"; else out="runs/onnx/$name"; fi
fi
[[ "${1:-}" == -- ]] && shift
args=("$src" "$out")
[[ -n "${CHECK:-}" ]] && args+=(--check "$CHECK")
"$PY" -m karna_train.export_onnx "${args[@]}" "$@"
