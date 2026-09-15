# Source and packaging differences

The files under runtime/ were copied from the reviewed recipe-v2 runtime
snapshot. Three legacy model-helper files had trailing whitespace removed as
a non-semantic packaging cleanup; their published SHA-256 values are listed in
provenance/source_code_sha256.txt. The executable source, model, trainer,
loss, data utility and evaluator behavior was not changed.

The following changes are public packaging changes, not scientific changes:

1. The original FedMoLG-TCFP prototype was moved under
   legacy/fedmolg_tcfp so it remains available but is not confused with the
   current nnU-Net v2 entry point.
2. The public config/experiments.json describes the current variants and keeps
   the proposed matched deep-supervision weights explicit.
3. New path-parameterized entry points replace private deployment paths:
   install_environment.sh, install_custom_trainer.py, verify_environment.py,
   preflight.py, train_one_experiment.sh, prepare_data.sh,
   preprocess_frozen.py, preprocess_frozen.sh and evaluate_one_experiment.sh.
   They refuse implicit installation, preprocessing, training or evaluation.
4. Private membership manifests, source paths, run identities, wheel URLs,
   weights, predictions and logs were not copied.

No model layer, loss equation, label mapping, post-processing threshold,
HD95 rule, optimizer, learning-rate schedule, patch, batch or epoch setting
was changed in this release preparation.
