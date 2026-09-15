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
