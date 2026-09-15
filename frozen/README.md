# Frozen data contract

This directory contains non-case-specific metadata used by the current MoLG
implementation:

- dataset.json describes the four input channels, region labels, and the
  876-case training count.
- nnUNetPlans.json is the supplied Dataset901_BraTS2023_TCRegion,
  3d_fullres, nnUNetPlans contract.
- dataset_fingerprint.json records the corresponding preprocessing statistics.

The 375 internal cases, 100 external novel cases, and the recovered 876-case
training membership are intentionally not shipped in this public repository.
The 876-case set was reconstructed from matching historical case-metrics CSV
files and archive complements; it must not be described as an original
splits_final.json. A data owner may provide authorized case lists in the format
accepted by runtime/baselines/prepare_data_lossless.py.

The metadata hashes are recorded in provenance/frozen_metadata_sha256.txt.
Raw MRI, labels, ZIP archives, preprocessed arrays, predictions and checkpoints
are not part of this repository.
