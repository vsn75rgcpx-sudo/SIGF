#!/usr/bin/env bash
# Path-parameterized wrapper for the reviewed lossless data utility.
# It is inspect-only unless --execute is supplied.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="${PYTHON:-python3}"
MEMBERSHIP=""
BRATS_STAGE=""
GOAT_STAGE=""
DATA_ROOT=""
EVAL_ROOT=""
EXECUTE=0

usage() {
    printf '%s\n' \
        'Usage: prepare_data.sh --membership MANIFEST --brats-stage DIR --goat-stage DIR --data-root DIR --eval-root DIR [--python PYTHON] [--execute]' \
        'The membership manifest and stages must come from an authorized data owner.' \
        'Use runtime/baselines/prepare_data_lossless.py directly for membership recovery and lossless staging.'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) PYTHON=${2:?missing value}; shift 2 ;;
        --membership) MEMBERSHIP=${2:?missing value}; shift 2 ;;
        --brats-stage) BRATS_STAGE=${2:?missing value}; shift 2 ;;
        --goat-stage) GOAT_STAGE=${2:?missing value}; shift 2 ;;
        --data-root) DATA_ROOT=${2:?missing value}; shift 2 ;;
        --eval-root) EVAL_ROOT=${2:?missing value}; shift 2 ;;
        --execute) EXECUTE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 64 ;;
    esac
done
[[ -n "$MEMBERSHIP" && -n "$BRATS_STAGE" && -n "$GOAT_STAGE" && -n "$DATA_ROOT" && -n "$EVAL_ROOT" ]] || { usage >&2; exit 64; }
if [[ "$EXECUTE" -eq 0 ]]; then
    printf '%s\n' 'INSPECT_ONLY: no data is read or written.'
    printf '%s\n' "python=$PYTHON" "membership=$MEMBERSHIP" "brats_stage=$BRATS_STAGE" "goat_stage=$GOAT_STAGE" "data_root=$DATA_ROOT" "eval_root=$EVAL_ROOT"
    exit 0
fi
[[ "$(uname -s)" == "Linux" ]] || { printf '%s\n' 'Data preparation is intended for Linux.' >&2; exit 70; }
DATA_ROOT="$(mkdir -p "$DATA_ROOT" && cd "$DATA_ROOT" && pwd -P)"
EVAL_ROOT="$(mkdir -p "$EVAL_ROOT" && cd "$EVAL_ROOT" && pwd -P)"
if [[ "$DATA_ROOT" == "$EVAL_ROOT" || "$DATA_ROOT" == "$EVAL_ROOT"/* || "$EVAL_ROOT" == "$DATA_ROOT"/* ]]; then
    printf '%s\n' 'data-root and eval-root must be separate.' >&2
    exit 72
fi
mkdir -p "$DATA_ROOT/nnUNet_raw" "$DATA_ROOT/nnUNet_preprocessed"
"$PYTHON" "$ROOT/runtime/baselines/prepare_data_lossless.py" build-raw \
    --membership "$MEMBERSHIP" --brats-stage "$BRATS_STAGE" --goat-stage "$GOAT_STAGE" \
    --raw-root "$DATA_ROOT/nnUNet_raw" --preprocessed-root "$DATA_ROOT/nnUNet_preprocessed" \
    --eval-root "$EVAL_ROOT" --dataset-json "$ROOT/frozen/dataset.json" \
    --plans-json "$ROOT/frozen/nnUNetPlans.json"
