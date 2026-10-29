# Source and packaging changes

## Runtime source

Files under `runtime/` came from the reviewed recipe-v2 snapshot. Three model
helper files had trailing whitespace removed during the original packaging;
the published source hashes are recorded in
`provenance/source_code_sha256.txt`.

The 2026-10-29 SIGF alignment preserves the runtime model, trainer, loss,
data preparation, and evaluator implementations. It does not change layers,
loss equations, label mapping, component thresholds, HD95 handling, optimizer,
learning-rate schedule, patch size, batch size, or epoch count.

## SIGF alignment on 2026-10-29

The release presentation now matches the current manuscript:

1. SIGF is the paper-facing name. Existing MoLG Python identifiers and paths
   remain available for checkpoint and command compatibility.
2. `config/experiments.json` identifies `full_molg` and the three structural
   ablations as paper experiments. `proposed_loss_ds_matched` is an additional
   experiment; its implementation is retained.
3. Main reproduction examples select the standard-loss `full_molg` entry.
4. Preflight and launcher records distinguish per-experiment seed behavior.
   The main B24 trainer does not apply the custom seed variable, while seeded
   ablation trainers do. Historical metadata is not proof of RNG state.
5. Data and evaluation documentation identifies the material available in the
   public package and the extra membership information needed for the exact
   study cohorts.

These changes correct public configuration and execution records. They do not
retroactively change the protocol or results of an earlier experiment.

## Original release packaging

The earlier FedMoLG-TCFP prototype was moved to `legacy/fedmolg_tcfp`.
Path-parameterized install, environment, preflight, preparation, preprocessing,
training, and evaluation entry points replaced private deployment paths.
Private membership manifests, source paths, run identities, wheel URLs,
weights, predictions, and logs were omitted from the public tree.
