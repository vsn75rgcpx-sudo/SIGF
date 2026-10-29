# Release notes

## SIGF manuscript alignment on 2026-10-29

The repository now presents **SIGF: Sequence Integration and Gated Fusion for
Brain Tumor Segmentation** as its paper-facing name. The GitHub repository
was renamed from MoLG-UNet to SIGF on 2026-10-29; its current URL is
https://github.com/vsn75rgcpx-sudo/SIGF. Model classes, the experiment key
`full_molg`, and checkpoint layouts retain their compatible identifiers.

The paper experiments are the standard Dice–BCE main model (`full_molg`) and
three structural ablations (`no_maf`, `no_swin`, `concat_fusion`). The
composite-loss experiment `proposed_loss_ds_matched` is documented separately
as an additional experiment. Main training and evaluation examples now select
`full_molg`.

The update also corrects seed reporting: the main B24 trainer does not apply
the custom seed variable, while the three ablation trainers apply seed 42.
Launcher metadata must describe the selected trainer's behavior rather than
imply that exporting a variable fixes all models' RNGs. This documentation
update does not establish historical run seeds.

The model architecture, training loss implementations, frozen plans,
postprocessing, and metric implementations are preserved. Documentation
clarifies the data/materials available and distinguishes static verification
from a full GPU reproduction. No new training results are introduced by this
update.

## Public MoLG-UNet release on 2026-09-15

The initial FedMoLG-TCFP tree was retained under `legacy/fedmolg_tcfp`. Its
federated, 100-round, and Dice–HD95 model-selection workflow remains available
as historical code.

The runtime sources were taken from the reviewed recipe-v2 snapshot. Public
packaging added the current trainer/model modules, frozen metadata, version
recipes, and path-parameterized environment, preprocessing, training, and
evaluation entry points. Private run identities, paths, membership files,
weights, predictions, and full logs were not distributed.

The original release covered three structural ablations and a composite-loss
experiment, with the full standard-loss model listed as a reference. The
2026-10-29 update aligns that presentation with the current SIGF manuscript.
