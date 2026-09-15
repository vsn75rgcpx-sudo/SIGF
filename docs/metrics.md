# Evaluation and metrics

The independent evaluation entry point is
runtime/baselines/evaluate_prediction_folder.py. It reads one prediction
NIfTI per label NIfTI, checks that the case sets match, maps label 4 to 3,
and computes the WT, TC and ET regions.

Dice is computed with the fixed 1e-5 smoothing term. Hierarchical connected
components are applied to predictions with minimum sizes of 300 voxels for
WT, 50 for TC and 30 for ET; nesting is then enforced. HD95 uses voxel-grid
coordinates from the loaded arrays. Both masks empty gives 0.0; exactly one
mask empty, or a missing surface, gives the fixed 374 sentinel. Summary
statistics are the arithmetic mean and sample standard deviation (ddof=1)
over case-level values. These definitions are not tunable command-line
choices for the formal protocol.

Training-time pseudo Dice is a validation signal, not an internal375 or
external novel100 test result. Formal scoring must use checkpoint_final.pth
from the corresponding experiment and the authorized evaluation group.
Do not substitute checkpoint_best, delete failed cases, or replace HD95
sentinels with a finite value.

Example for an authorized prediction folder:

    python runtime/baselines/evaluate_prediction_folder.py --pred_dir /private/predictions --label_dir /private/labels --out /private/metrics/final_metrics.json --csv_out /private/metrics/case_metrics.csv
