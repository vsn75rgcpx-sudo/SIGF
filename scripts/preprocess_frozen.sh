#!/usr/bin/env bash
# Guarded Linux-only wrapper for frozen-plan preprocessing. It does not call
# a planner and does not submit a scheduler job.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="${PYTHON:-python3}"
DATA_ROOT=""
PERSISTENT_ROOT=""
PROCESSES=2
EXECUTE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root) DATA_ROOT=${2:?missing value for --data-root}; shift 2 ;;
        --persistent-root) PERSISTENT_ROOT=${2:?missing value for --persistent-root}; shift 2 ;;
        --python) PYTHON=${2:?missing value for --python}; shift 2 ;;
        --processes) PROCESSES=${2:?missing value for --processes}; shift 2 ;;
        --execute) EXECUTE=1; shift ;;
        -h|--help) sed -n '1,24p' "$0"; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 64 ;;
    esac
done
[[ -n "$DATA_ROOT" ]] || { printf '%s\n' 'Specify --data-root.' >&2; exit 64; }
[[ "$PROCESSES" =~ ^[12]$ ]] || { printf '%s\n' '--processes must be 1 or 2; do not exceed the bounded preparation budget.' >&2; exit 64; }
if [[ "$EXECUTE" -eq 0 ]]; then
    INSPECT_ARGS=(--package-root "$ROOT" --data-root "$DATA_ROOT" --processes "$PROCESSES")
    [[ -n "$PERSISTENT_ROOT" ]] && INSPECT_ARGS+=(--persistent-root "$PERSISTENT_ROOT")
    exec "$PYTHON" "$ROOT/scripts/preprocess_frozen.py" "${INSPECT_ARGS[@]}"
fi

[[ -n "$PERSISTENT_ROOT" ]] || { printf '%s\n' '--persistent-root is required with --execute.' >&2; exit 64; }

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
exec "$PYTHON" "$ROOT/scripts/preprocess_frozen.py" --package-root "$ROOT" --data-root "$DATA_ROOT" --persistent-root "$PERSISTENT_ROOT" --processes "$PROCESSES" --execute
