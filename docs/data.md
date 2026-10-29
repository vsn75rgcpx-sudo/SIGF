# Data, membership and preparation

The implementation expects nnU-Net v2 data for
Dataset901_BraTS2023_TCRegion, configuration 3d_fullres, with the channel order
T1, T1ce, T2, FLAIR. Labels use WT = labels greater than zero, TC = labels 1
or 3, and ET = label 3; source label 4 is normalized to 3.

The frozen protocol uses 876 training cases, 375 internal cases and 100
supplemental cases. The internal and supplemental case lists are evaluation-only. The
876-case training membership was recovered as the complement of the frozen
internal375 set within the 2023 archive, with consistency checks against four
historical case-metrics CSV sources. No original splits_final.json was
available, so the recovered list must not be presented as one.

Case identifiers and source CSVs are withheld until their distribution
permission is confirmed. Authorized users should create the membership
manifest with the lossless utility and the four matching internal and four
matching external case-metrics CSVs:

    python runtime/baselines/prepare_data_lossless.py membership --internal-csv /authorized/internal-source-1.csv /authorized/internal-source-2.csv /authorized/internal-source-3.csv /authorized/internal-source-4.csv --external-csv /authorized/external-source-1.csv /authorized/external-source-2.csv /authorized/external-source-3.csv /authorized/external-source-4.csv --brats-archive /authorized/BraTS2023.zip --goat-archive /authorized/GoAT2024.zip --out /path/to/private/membership

Then materialize the two lossless staging areas and build the nnU-Net raw
layout. The archive names, case lists and output roots below are placeholders;
the release does not provide or download the source data:

    python runtime/baselines/prepare_data_lossless.py convert-brats2023 --archive /authorized/BraTS2023.zip --case-list /path/to/private/membership/training876.txt --expected-cases 876 --out /data/molg/staged/brats2023
    python runtime/baselines/prepare_data_lossless.py extract-goat --archive /authorized/GoAT2024.zip --case-list /path/to/private/membership/external_novel100.txt --expected-cases 100 --out /data/molg/staged/goat2024
    bash scripts/prepare_data.sh --python /persistent/molg/envs/molg-cu128/bin/python --membership /path/to/private/membership --brats-stage /data/molg/staged/brats2023 --goat-stage /data/molg/staged/goat2024 --data-root /data/molg --eval-root /persistent/molg/eval --execute

Audit the two stages before building raw data if they were not produced by
the commands above:

    python runtime/baselines/prepare_data_lossless.py audit-stage --stage /data/molg/staged/brats2023 --case-list /path/to/private/membership/training876.txt --expected-cases 876 --kind brats2023
    python runtime/baselines/prepare_data_lossless.py audit-stage --stage /data/molg/staged/goat2024 --case-list /path/to/private/membership/external_novel100.txt --expected-cases 100 --kind goat2024

After raw data is built, run only the frozen-plan preprocessor. It calls
nnU-Net's preprocessing API with the supplied plans identifier and never calls
the planner:

    bash scripts/preprocess_frozen.sh --python /persistent/molg/envs/molg-cu128/bin/python --data-root /data/molg --persistent-root /persistent/molg --processes 2 --execute

The utility streams nested 2023 ZIP NIfTI members without loading and
re-serializing voxel arrays. Existing GoAT .nii.gz payloads are copied
byte-for-byte. It rejects unsafe ZIP members, duplicate modalities,
unmarked collisions and incomplete markers. The planner is not run
automatically.

The caller must keep raw/staged/preprocessed data on a data volume and keep
code, logs, checkpoints and recovery records on a persistent volume. Do not
place either volume inside the other. The source archives remain under the
data owner's control.

## Relationship to the paper cohorts

The paper uses the names BraTS 2023 test list (375 cases) and image-deduplicated
BraTS-GoAT 2024 supplemental cohort (100 cases). The existing directory and
command identifier external_novel100 refers to the latter cohort; it is kept
for compatibility and is not a claim of independent clinical-site validation.

The membership command above recovers the already selected case lists from
historical case-metrics CSVs and checks their counts and archive availability.
It does not perform the cross-dataset image comparison used to construct the
paper's supplemental cohort. The public package does not include the original
image-deduplication workflow or its audit records. Those materials and the
cohort lists are needed to independently verify the same cohort selection.

Public access to the BraTS source images alone does not supply the private
CSVs required by this membership procedure. If the cohort files cannot be
obtained from the authors under the applicable data terms, readers can run
the method on their own documented split but cannot claim an exact reproduction
of the paper's specified cohorts or table values. No script in this package
fetches private files or reconstructs missing case membership by assumption.
