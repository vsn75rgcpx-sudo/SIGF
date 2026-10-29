#!/usr/bin/env bash
# Public path-parameterized nnU-Net training launcher.
# Default is inspect-only. A real run needs an explicit preflight and a fresh
# run root; resume is allowed only within that run root.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
EXPERIMENT=""
PROFILE=""
ENV_PREFIX=""
DATA_ROOT=""
PERSISTENT_ROOT=""
RUN_ROOT=""
PREFLIGHT=""
AUG_WORKERS=4
RESUME=0
EXECUTE=0
RUN_SEED=42
SEED_POLICY=fixed_by_seeded_trainer

usage() {
    printf '%s\n' \
        'Usage: train_one_experiment.sh --experiment full_molg|no_maf|no_swin|concat_fusion|proposed_loss_ds_matched --profile PROFILE --env-prefix DIR --data-root DIR --persistent-root DIR --run-root DIR [--preflight FILE] [--augmentation-workers N] [--resume] [--execute-training]' \
        'Without --execute-training the command only prints the binding.' \
        'A new run must use a previously absent run root; --resume requires its own checkpoint_latest.pth.'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment) EXPERIMENT=${2:?missing value}; shift 2 ;;
        --profile) PROFILE=${2:?missing value}; shift 2 ;;
        --env-prefix) ENV_PREFIX=${2:?missing value}; shift 2 ;;
        --data-root) DATA_ROOT=${2:?missing value}; shift 2 ;;
        --persistent-root) PERSISTENT_ROOT=${2:?missing value}; shift 2 ;;
        --run-root) RUN_ROOT=${2:?missing value}; shift 2 ;;
        --preflight) PREFLIGHT=${2:?missing value}; shift 2 ;;
        --augmentation-workers) AUG_WORKERS=${2:?missing value}; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --execute-training) EXECUTE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 64 ;;
    esac
done

case "$EXPERIMENT" in
    full_molg) TRAINER=nnUNetTrainerParallelModalSwinUNetB24; RUN_SEED=unfixed; SEED_POLICY=not_fixed_by_entrypoint ;;
    no_maf) TRAINER=nnUNetTrainerParallelModalSwinUNetB24NoMAF ;;
    no_swin) TRAINER=nnUNetTrainerParallelModalSwinUNetB24NoSwin ;;
    concat_fusion) TRAINER=nnUNetTrainerParallelModalSwinUNetB24ConcatFusion ;;
    proposed_loss_ds_matched) TRAINER=nnUNetTrainerParallelModalSwinUNetB24ProposedLossDSMatchedSeed42 ;;
    *) printf 'Unsupported experiment: %s\n' "$EXPERIMENT" >&2; exit 64 ;;
esac
[[ -n "$PROFILE" && -n "$ENV_PREFIX" && -n "$DATA_ROOT" && -n "$PERSISTENT_ROOT" && -n "$RUN_ROOT" ]] || {
    usage >&2
    exit 64
}
[[ "$AUG_WORKERS" =~ ^[1-9][0-9]*$ ]] || { printf '%s\n' 'augmentation-workers must be positive.' >&2; exit 64; }

if [[ "$EXECUTE" -eq 0 ]]; then
    printf '%s\n' 'INSPECT_ONLY: no run root, GPU, environment or training process will be created.'
    printf '%s\n' \
        "experiment=$EXPERIMENT" "trainer=$TRAINER" "profile=$PROFILE" \
        'dataset=901 configuration=3d_fullres plans=nnUNetPlans fold=all' \
        "seed=$RUN_SEED seed_policy=$SEED_POLICY epochs=1000 train_iterations=250 validation_iterations=50 patch=128,160,112 batch=2" \
        "augmentation_workers=$AUG_WORKERS validation_workers=$(( AUG_WORKERS / 2 > 0 ? AUG_WORKERS / 2 : 1 )) library_threads=1 amp=true compile=true resume=$RESUME"
    exit 0
fi

[[ "$(uname -s)" == "Linux" ]] || { printf '%s\n' 'Training requires Linux.' >&2; exit 70; }
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || {
    printf '%s\n' 'Set exactly one CUDA_VISIBLE_DEVICES value.' >&2
    exit 70
}
PYTHON="$ENV_PREFIX/bin/python"
TRAIN="$ENV_PREFIX/bin/nnUNetv2_train"
[[ -x "$PYTHON" && -x "$TRAIN" ]] || { printf '%s\n' 'Environment is incomplete.' >&2; exit 69; }
[[ -n "$PREFLIGHT" && -f "$PREFLIGHT" ]] || { printf '%s\n' 'An explicit preflight report is required.' >&2; exit 70; }
[[ -d "$DATA_ROOT" && -d "$PERSISTENT_ROOT" ]] || { printf '%s\n' 'Both roots must exist.' >&2; exit 70; }
DATA_ROOT="$(cd "$DATA_ROOT" && pwd -P)"
PERSISTENT_ROOT="$(cd "$PERSISTENT_ROOT" && pwd -P)"
RUN_ROOT="$( "$PYTHON" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$RUN_ROOT" )"
if [[ "$DATA_ROOT" == "$PERSISTENT_ROOT" || "$DATA_ROOT" == "$PERSISTENT_ROOT"/* || "$PERSISTENT_ROOT" == "$DATA_ROOT"/* ]]; then
    printf '%s\n' 'data-root and persistent-root must be separate and non-nested.'
    exit 72
fi
[[ "$RUN_ROOT" == "$PERSISTENT_ROOT"/* ]] || { printf '%s\n' 'run-root must be under persistent-root.' >&2; exit 72; }
if [[ "$RESUME" -eq 0 && ( -e "$RUN_ROOT" || -L "$RUN_ROOT" ) ]]; then
    printf 'Refusing to reuse a run root for a new run: %s\n' "$RUN_ROOT" >&2
    exit 73
fi
if [[ "$RESUME" -eq 1 && ! -d "$RUN_ROOT" ]]; then
    printf 'Resume run root does not exist: %s\n' "$RUN_ROOT" >&2
    exit 70
fi

"$PYTHON" - "$PREFLIGHT" "$EXPERIMENT" "$PROFILE" <<'PY'
import json, sys
payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
if payload.get("status") not in ("PASS_BINDING_GPU_VISIBLE", "PASS"):
    raise SystemExit("preflight is not passing")
if payload.get("experiment") != sys.argv[2] or payload.get("profile") != sys.argv[3]:
    raise SystemExit("preflight experiment/profile mismatch")
PY
mkdir -p "$RUN_ROOT"
RECHECK="$RUN_ROOT/preflight_recheck.json"
"$PYTHON" "$ROOT/scripts/preflight.py" \
    --experiment "$EXPERIMENT" --profile "$PROFILE" --data-root "$DATA_ROOT" \
    --persistent-root "$PERSISTENT_ROOT" --augmentation-workers "$AUG_WORKERS" \
    --out "$RECHECK" --execute >/dev/null

if [[ "$RESUME" -eq 1 ]]; then
    MODEL_DIR="$RUN_ROOT/nnUNet_results/$EXPERIMENT/Dataset901_BraTS2023_TCRegion/${TRAINER}__nnUNetPlans__3d_fullres/fold_all"
    [[ -f "$MODEL_DIR/checkpoint_latest.pth" ]] || { printf '%s\n' 'Same-run checkpoint_latest.pth is missing.' >&2; exit 70; }
    "$PYTHON" "$ROOT/scripts/checkpoint_contract.py" \
        --checkpoint "$MODEL_DIR/checkpoint_latest.pth" --out "$RUN_ROOT/resume_checkpoint_contract.json"
    "$PYTHON" - "$RUN_ROOT/run_identity.json" "$EXPERIMENT" "$TRAINER" "$RUN_SEED" "$SEED_POLICY" "$RUN_ROOT/resume_identity_check.json" <<'PY'
import json, sys
payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
for key, value in (("experiment", sys.argv[2]), ("trainer", sys.argv[3]), ("total_epochs", 1000)):
    if payload.get(key) != value:
        raise SystemExit(f"same-run identity mismatch for {key}")
expected_seed = None if sys.argv[4] == "unfixed" else int(sys.argv[4])
legacy_unverified_seed = (
    sys.argv[2] == "full_molg" and "seed_policy" not in payload and payload.get("seed") == 42
)
if not legacy_unverified_seed and payload.get("seed") != expected_seed:
    raise SystemExit("same-run identity mismatch for seed")
if "seed_policy" in payload and payload["seed_policy"] != sys.argv[5]:
    raise SystemExit("same-run identity mismatch for seed_policy")
record = {
    "experiment": sys.argv[2], "trainer": sys.argv[3],
    "entrypoint_seed": expected_seed, "seed_policy": sys.argv[5],
    "recorded_identity_seed": payload.get("seed"),
    "legacy_seed_metadata_unverified": legacy_unverified_seed,
    "note": "The existing run identity is preserved. Legacy full_molg seed=42 metadata does not prove that the trainer set a seed. Checkpoint field checks do not prove restoration; nnU-Net --c must restore this run's complete checkpoint state.",
}
open(sys.argv[6], "w", encoding="utf-8").write(json.dumps(record, indent=2, sort_keys=True) + "\n")
if legacy_unverified_seed:
    print("Legacy full_molg seed=42 metadata is unverified; preserving run_identity.json and recording the correction separately.", file=sys.stderr)
PY
else
    "$PYTHON" - "$RUN_ROOT/run_identity.json" "$EXPERIMENT" "$TRAINER" "$PROFILE" "$RUN_ROOT" "$AUG_WORKERS" "$RUN_SEED" "$SEED_POLICY" <<'PY'
import json, sys, time
payload = {
    "status": "TRAINING_STARTED", "experiment": sys.argv[2], "trainer": sys.argv[3],
    "profile": sys.argv[4], "run_root": sys.argv[5],
    "seed": None if sys.argv[7] == "unfixed" else int(sys.argv[7]),
    "seed_policy": sys.argv[8], "total_epochs": 1000,
    "train_iterations_per_epoch": 250, "validation_iterations_per_epoch": 50,
    "patch_size": [128, 160, 112], "batch_size": 2,
    "augmentation_workers": int(sys.argv[6]), "validation_workers_derived": max(1, int(sys.argv[6]) // 2),
    "library_threads": 1, "amp": True, "torch_compile": True, "resume": False,
    "started_unix": time.time(),
}
open(sys.argv[1], "x", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
fi

if [[ "$RUN_SEED" == unfixed ]]; then
    # The historical B24 trainer does not consume this variable.
    unset NNUNET_RUN_SEED
else
    export NNUNET_RUN_SEED="$RUN_SEED"
fi
export NNUNET_NUM_EPOCHS=1000
export nnUNet_compile=true
export nnUNet_n_proc_DA="$AUG_WORKERS"
export nnUNET_RAW="$DATA_ROOT/nnUNet_raw"
export nnUNet_raw="$DATA_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$DATA_ROOT/nnUNet_preprocessed"
export nnUNet_results="$RUN_ROOT/nnUNet_results/$EXPERIMENT"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
export MPLCONFIGDIR="$RUN_ROOT/cache/mpl"
export TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/cache/torchinductor"
export TRITON_CACHE_DIR="$RUN_ROOT/cache/triton"
RUN_TMPDIR="$(mktemp -d -p /tmp "molg-${EXPERIMENT}.XXXXXX")"
export TMPDIR="$RUN_TMPDIR"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

LOCK="$RUN_ROOT/.training.lock"
if [[ -e "$LOCK" || -L "$LOCK" ]]; then
    printf 'A run lock already exists; inspect it before proceeding: %s\n' "$LOCK" >&2
    exit 74
fi
mkdir "$LOCK"
printf '%s\n' "$$" > "$LOCK/pid"
cleanup() {
    rm -f "$LOCK/pid" 2>/dev/null || true
    rmdir "$LOCK" 2>/dev/null || true
    rm -rf "$RUN_TMPDIR" 2>/dev/null || true
}
trap cleanup EXIT

TRAIN_ARGS=(901 3d_fullres all -tr "$TRAINER" -p nnUNetPlans)
if [[ "$RESUME" -eq 1 ]]; then TRAIN_ARGS+=(--c); fi
printf '%q ' "$TRAIN" "${TRAIN_ARGS[@]}" > "$RUN_ROOT/train_command.txt"
printf '\n' >> "$RUN_ROOT/train_command.txt"
mkdir -p "$RUN_ROOT/logs"
set +e
"$TRAIN" "${TRAIN_ARGS[@]}" 2>&1 | tee "$RUN_ROOT/logs/launcher.log"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e
"$PYTHON" - "$RUN_ROOT/exit_status.json" "$TRAIN_STATUS" "$RUN_TMPDIR" "$RESUME" <<'PY'
import json, sys, time
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps({
    "exit_code": int(sys.argv[2]), "temporary_directory": sys.argv[3],
    "resume": bool(int(sys.argv[4])), "finished_unix": time.time(),
}, indent=2, sort_keys=True) + "\n")
PY
if [[ "$TRAIN_STATUS" -ne 0 ]]; then
    printf 'TRAINING_FAILED exit_code=%s\n' "$TRAIN_STATUS" >&2
    exit "$TRAIN_STATUS"
fi
printf '%s\n' "TRAINING_PROCESS_EXITED_0: $RUN_ROOT"
