# Provenance and environment

The runtime sources under runtime/ were copied from the reviewed recipe-v2
snapshot used for the Featurize deployment preparation. Three legacy
model-helper files had trailing whitespace removed as a non-semantic
packaging cleanup; the source file hashes below list the exact bytes shipped
here. The release wrappers and documentation are path-parameterized public
packaging around those sources; they do not alter model, loss, labels,
post-processing thresholds, or metric definitions.

The actual package freeze in environment_freeze_public.txt is a package-name
and version snapshot from the verified Linux Blackwell runtime. It contains no
local wheel URLs or private filesystem paths. The pinned installation inputs
are requirements/constraints-common-runtime.txt plus exactly one GPU profile
file. The historical timm 1.0.27 constraint is not an installation input.

The original source snapshot had no Git commit identifier. Its identity is
therefore represented by SHA-256 file hashes. Private campus paths, Featurize
run identities, checkpoint hashes, case-level CSVs and credentials are
deliberately omitted.

The release has not performed a fresh Linux cold-start install, CUDA driver
check, GPU preflight, full data preparation, training, or evaluation from this
repository on the authoring Mac.

## SIGF paper alignment

The paper and repository are named SIGF. Trainer class names and checkpoint
directories retain earlier MoLG identifiers for the same implementation.
The 2026-10-29 alignment updates public documentation, experiment metadata and
run seed records. Runtime model, trainer, loss, preprocessing and evaluation
sources under runtime/ and the frozen scientific metadata are unchanged.

The source hash manifest covers only files actually shipped under runtime/.
An obsolete entry for compare_case_metrics.py was removed because that file
was not included in the public package. This manifest does not establish the
identity of private checkpoints or reproduce the paper's case-level statistics.
The package does not include the complete statistical analysis workflow.

The full_molg entry retains its original B24 trainer, which does not read the
custom NNUNET_RUN_SEED variable. Its seed is therefore recorded as not fixed
by this entry point. The three structural ablations use seeded trainers with
seed 42. Metadata from older wrappers may say seed 42 for full_molg; that field
alone does not show that the seed was applied. Existing historical records must
be retained rather than rewritten as proof of a different training protocol.
