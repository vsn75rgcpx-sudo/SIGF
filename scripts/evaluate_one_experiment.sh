#!/usr/bin/env bash
# Evaluate one final checkpoint on one authorized case group.
# Default is inspect-only; this launcher never trains or changes the checkpoint.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
EXPERIMENT=""
GROUP=""
ENV_PREFIX=""
DATA_ROOT=""
PERSISTENT_ROOT=""
RUN_ROOT=""
IMAGES_DIR=""
LABEL_DIR=""
OUTPUT_DIR=""
EXECUTE=0

usage() {
    printf '%s\n' \
        'Usage: evaluate_one_experiment.sh --experiment full_molg|no_maf|no_swin|concat_fusion|proposed_loss_ds_matched --group internal375|external_novel100 --env-prefix DIR --data-root DIR --persistent-root DIR --run-root DIR --images-dir DIR --label-dir DIR --output-dir DIR [--execute-evaluation]' \
        'The labels directory defines the exact authorized case set; the evaluator rejects missing predictions.'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment) EXPERIMENT=${2:?missing value}; shift 2 ;;
        --group) GROUP=${2:?missing value}; shift 2 ;;
        --env-prefix) ENV_PREFIX=${2:?missing value}; shift 2 ;;
        --data-root) DATA_ROOT=${2:?missing value}; shift 2 ;;
        --persistent-root) PERSISTENT_ROOT=${2:?missing value}; shift 2 ;;
        --run-root) RUN_ROOT=${2:?missing value}; shift 2 ;;
        --images-dir) IMAGES_DIR=${2:?missing value}; shift 2 ;;
        --label-dir) LABEL_DIR=${2:?missing value}; shift 2 ;;
        --output-dir) OUTPUT_DIR=${2:?missing value}; shift 2 ;;
        --execute-evaluation) EXECUTE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 64 ;;
    esac
done
case "$EXPERIMENT" in
    full_molg) TRAINER=nnUNetTrainerParallelModalSwinUNetB24 ;;
    no_maf) TRAINER=nnUNetTrainerParallelModalSwinUNetB24NoMAF ;;
    no_swin) TRAINER=nnUNetTrainerParallelModalSwinUNetB24NoSwin ;;
    concat_fusion) TRAINER=nnUNetTrainerParallelModalSwinUNetB24ConcatFusion ;;
    proposed_loss_ds_matched) TRAINER=nnUNetTrainerParallelModalSwinUNetB24ProposedLossDSMatchedSeed42 ;;
    *) printf 'Unsupported experiment: %s\n' "$EXPERIMENT" >&2; exit 64 ;;
esac
case "$GROUP" in internal375|external_novel100) ;; *) printf 'Unsupported group: %s\n' "$GROUP" >&2; exit 64 ;; esac
[[ -n "$ENV_PREFIX" && -n "$DATA_ROOT" && -n "$PERSISTENT_ROOT" && -n "$RUN_ROOT" && -n "$IMAGES_DIR" && -n "$LABEL_DIR" && -n "$OUTPUT_DIR" ]] || { usage >&2; exit 64; }

if [[ "$EXECUTE" -eq 0 ]]; then
    printf '%s\n' 'INSPECT_ONLY: no inference, metric computation or output is created.'
    printf '%s\n' "experiment=$EXPERIMENT" "trainer=$TRAINER" "group=$GROUP" "checkpoint=checkpoint_final.pth" "step_size=0.5 gaussian=true mirroring=true npp=1 nps=1 postprocess=WT300,TC50,ET30 hd95_empty=374"
    exit 0
fi

PYTHON="$ENV_PREFIX/bin/python"
PREDICT="$ENV_PREFIX/bin/nnUNetv2_predict"
[[ -x "$PYTHON" && -x "$PREDICT" ]] || { printf '%s\n' 'Environment or nnUNetv2_predict is missing.' >&2; exit 69; }
[[ -d "$IMAGES_DIR" && -d "$LABEL_DIR" && -d "$RUN_ROOT" && -d "$DATA_ROOT" && -d "$PERSISTENT_ROOT" ]] || { printf '%s\n' 'Input roots are missing.' >&2; exit 70; }
if [[ "$DATA_ROOT" == "$PERSISTENT_ROOT" || "$DATA_ROOT" == "$PERSISTENT_ROOT"/* || "$PERSISTENT_ROOT" == "$DATA_ROOT"/* ]]; then
    printf '%s\n' 'data-root and persistent-root must be separate and non-nested.' >&2
    exit 72
fi
MODEL_DIR="$RUN_ROOT/nnUNet_results/$EXPERIMENT/Dataset901_BraTS2023_TCRegion/${TRAINER}__nnUNetPlans__3d_fullres/fold_all"
CHECKPOINT="$MODEL_DIR/checkpoint_final.pth"
[[ -s "$CHECKPOINT" ]] || { printf 'Missing final checkpoint: %s\n' "$CHECKPOINT" >&2; exit 70; }
mkdir -p "$OUTPUT_DIR/predictions" "$OUTPUT_DIR/metrics"
export nnUNet_raw="$DATA_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$DATA_ROOT/nnUNet_preprocessed"
export nnUNet_results="$RUN_ROOT/nnUNet_results/$EXPERIMENT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONNOUSERSITE=1
"$PREDICT" -i "$IMAGES_DIR" -o "$OUTPUT_DIR/predictions" -d 901 -c 3d_fullres -tr "$TRAINER" -p nnUNetPlans -f all -chk checkpoint_final.pth -step_size 0.5 -npp 1 -nps 1
"$PYTHON" "$ROOT/scripts/verify_case_coverage.py" --prediction-dir "$OUTPUT_DIR/predictions" --label-dir "$LABEL_DIR" --expected-count "$( [[ "$GROUP" == internal375 ]] && printf 375 || printf 100 )"
"$PYTHON" "$ROOT/runtime/baselines/evaluate_prediction_folder.py" \
    --pred_dir "$OUTPUT_DIR/predictions" --label_dir "$LABEL_DIR" \
    --out "$OUTPUT_DIR/metrics/final_metrics.json" --csv_out "$OUTPUT_DIR/metrics/case_metrics.csv"
"$PYTHON" - "$OUTPUT_DIR/evaluation_binding.json" "$EXPERIMENT" "$GROUP" "$CHECKPOINT" <<'PY'
import hashlib, json, pathlib, sys, time
p = pathlib.Path(sys.argv[4])
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "experiment": sys.argv[2], "group": sys.argv[3],
    "checkpoint": str(p), "checkpoint_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    "completed_unix": time.time(),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
