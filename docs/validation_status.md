# Validation status

Checked locally on 2026-09-15:

- all Python files parse with the standard-library AST parser;
- all public shell entry points pass bash -n;
- the JSON experiment contract loads and contains all four formal variants;
- current runtime source hashes match the reviewed recipe-v2 source snapshot;
- frozen dataset, plans and fingerprint hashes match the approved metadata;
- public-tree checks reject case lists, weights, NIfTI, ZIP archives and other
  scientific artifacts;
- credential/private-path pattern scans pass for the release tree and the
  preserved initial Git history;
- synthetic nested-BraTS ZIP streaming preserves the payload and rejects path
  traversal;
- inspect-only entry points run without creating an environment, data tree,
  GPU context, training process or evaluation output.

Not run here:

- Linux cold-start installation and resolver/wheel availability;
- NVIDIA driver, CUDA, Blackwell/5090 or Ada/4090 compatibility;
- actual trainer import from an installed nnU-Net package;
- real archive staging, preprocessing, training, resume loading or evaluation;
- internal375/external novel100 reproduction from this public clone.

The authoring Mac lacks the runtime scientific Python packages needed for
NumPy/SciPy metric execution, Torch model construction and GPU checks. This
is recorded as an environment limitation, not a test pass.
