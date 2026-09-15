#!/usr/bin/env bash
# Install the reviewed Linux runtime into a fresh prefix.
# Default mode is inspect-only. This script does not rent, upload, or contact
# a project server.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="${PYTHON:-python3}"
PROFILE=""
PREFIX=""
APPLY=0

usage() {
    printf '%s\n' \
        'Usage: install_environment.sh --profile blackwell5090-cu128|ada4090-cu118 --prefix DIR [--python PYTHON] [--apply]' \
        'Without --apply the command only prints the selected recipe and local interpreter information.' \
        'Use --apply only on a compatible Linux host with a fresh prefix on persistent storage.'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) PROFILE=${2:?missing value for --profile}; shift 2 ;;
        --prefix) PREFIX=${2:?missing value for --prefix}; shift 2 ;;
        --python) PYTHON=${2:?missing value for --python}; shift 2 ;;
        --apply) APPLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 64 ;;
    esac
done

[[ -n "$PROFILE" && -n "$PREFIX" ]] || { usage >&2; exit 64; }
case "$PROFILE" in
    blackwell5090-cu128)
        CHANNEL=cu128
        GPU_HINT=5090
        PROFILE_REQUIREMENTS="$ROOT/requirements/constraints-blackwell5090-cu128.txt"
        ;;
    ada4090-cu118)
        CHANNEL=cu118
        GPU_HINT=4090
        PROFILE_REQUIREMENTS="$ROOT/requirements/constraints-ada4090-cu118.txt"
        ;;
    *) printf 'Unsupported profile: %s\n' "$PROFILE" >&2; exit 64 ;;
esac
COMMON_REQUIREMENTS="$ROOT/requirements/constraints-common-runtime.txt"
[[ -f "$PROFILE_REQUIREMENTS" && -f "$COMMON_REQUIREMENTS" ]] || {
    printf '%s\n' 'Installation recipe is incomplete.' >&2
    exit 66
}

printf '%s\n' \
    "profile=$PROFILE" \
    "prefix=$PREFIX" \
    "python=$PYTHON" \
    "profile_requirements=$PROFILE_REQUIREMENTS" \
    "common_requirements=$COMMON_REQUIREMENTS" \
    "pytorch_index=https://download.pytorch.org/whl/$CHANNEL" \
    "gpu_hint=RTX $GPU_HINT"

if [[ "$APPLY" -eq 0 ]]; then
    printf '%s\n' 'INSPECT_ONLY: no environment, download, driver or GPU change.'
    "$PYTHON" -c 'import platform,sys; print({"executable":sys.executable,"version":platform.python_version(),"system":platform.system()})' || true
    exit 0
fi

[[ "$(uname -s)" == "Linux" ]] || {
    printf '%s\n' 'Refusing installation outside Linux.' >&2
    exit 70
}
command -v "$PYTHON" >/dev/null 2>&1 || {
    printf 'Python executable not found: %s\n' "$PYTHON" >&2
    exit 69
}
"$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)' || {
    printf '%s\n' 'Python 3.10.x is required.' 'Select a real Python 3.10 interpreter and rerun with --python /absolute/path/bin/python.' >&2
    exit 70
}
[[ "$PREFIX" == /* ]] || {
    printf '%s\n' '--prefix must be an absolute path on persistent storage.' >&2
    exit 72
}
command -v nvidia-smi >/dev/null 2>&1 || {
    printf '%s\n' 'nvidia-smi is required for profile validation.' >&2
    exit 70
}
GPU_INFO="$(nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader)"
printf '%s\n' "$GPU_INFO"
printf '%s\n' "$GPU_INFO" | tr '[:upper:]' '[:lower:]' | grep -q "$GPU_HINT" || {
    printf 'Detected GPU does not match profile %s: %s\n' "$PROFILE" "$GPU_INFO" >&2
    exit 71
}

PREFIX="$( "$PYTHON" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$PREFIX" )"
mkdir -p "$(dirname "$PREFIX")"
if [[ -e "$PREFIX" || -L "$PREFIX" ]]; then
    printf 'Refusing to overwrite an existing prefix: %s\n' "$PREFIX" >&2
    exit 73
fi
LOCK="$PREFIX.installing.lock"
mkdir "$LOCK"
cleanup() {
    if [[ -d "$LOCK" ]]; then
        rmdir "$LOCK" 2>/dev/null || true
    fi
}
trap cleanup EXIT

"$PYTHON" -m venv "$PREFIX"
ENV_PYTHON="$PREFIX/bin/python"
export PYTHONNOUSERSITE=1
"$ENV_PYTHON" -m pip install --no-user --requirement "$PROFILE_REQUIREMENTS" --index-url "https://download.pytorch.org/whl/$CHANNEL" --extra-index-url "https://pypi.org/simple"
"$ENV_PYTHON" -m pip install --no-user --requirement "$COMMON_REQUIREMENTS"
"$ENV_PYTHON" "$ROOT/scripts/install_custom_trainer.py" --apply
"$ENV_PYTHON" "$ROOT/scripts/verify_environment.py" --profile "$PROFILE" --out "$PREFIX/runtime_verification.json"
printf '%s\n' "ENVIRONMENT_INSTALLED: $PREFIX"
