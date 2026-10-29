# Evaluation and metrics

The independent evaluation entry point is
runtime/baselines/evaluate_prediction_folder.py. It reads one prediction
NIfTI per label NIfTI, checks that the case sets match, maps label 4 to 3,
and computes the WT, TC and ET regions for SIGF and its structural variants.

Dice is computed with the fixed 1e-5 smoothing term. Hierarchical connected
components are applied to predictions with minimum sizes of 300 voxels for
WT, 50 for TC and 30 for ET; nesting is then enforced. HD95 uses voxel-grid
coordinates from the loaded arrays. NSD uses equally weighted surface voxels
and a distance tolerance of 1.0 voxel. At the frozen 1 mm isotropic spacing,
this threshold corresponds to 1 mm. For HD95, both masks empty gives 0.0;
exactly one mask empty, or a missing surface, gives the fixed 374 sentinel.
For Dice and NSD, both masks empty gives 1.0; when only one is empty, Dice is
approximately zero because of smoothing and NSD is zero. Summary
statistics are the arithmetic mean and sample standard deviation (ddof=1)
over case-level values. Within each case, the reported Mean is the arithmetic
mean across WT, TC and ET. These definitions are not tunable command-line
choices for the formal protocol.

The fold=all training-time pseudo Dice monitors the training process using
the training case pool. It is not a held-out validation result or a test result
for internal375 or the supplemental novel100 cohort. The group identifier
external_novel100 is retained for compatibility; the paper describes it as a
supplemental test cohort. Formal scoring must use checkpoint_final.pth
from the corresponding experiment and the authorized evaluation group.
Do not substitute checkpoint_best, delete failed cases, or replace HD95
sentinels with a finite value.

Example for an authorized prediction folder:

    python runtime/baselines/evaluate_prediction_folder.py --pred_dir /private/predictions --label_dir /private/labels --out /private/metrics/final_metrics.json --csv_out /private/metrics/case_metrics.csv
