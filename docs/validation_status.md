# Validation status

## SIGF alignment on 2026-10-29

The current update aligns the public experiment definitions and reproduction
instructions with the SIGF manuscript. The full standard-loss model and its
three structural ablations are the paper experiments; the composite-loss
variant is additional. Existing scientific runtime sources and frozen
metadata are preserved.

Static review confirms that the main B24 trainer does not inherit the seed
mixin, whereas the three structural ablations do. Public execution metadata
must distinguish this behavior. A seed recorded by a launcher is not evidence
of the RNG state of a historical training run.

Checks completed for this update:

- all 14 dependency-free tests passed, including the real launcher shell with
  mocked GPU/preflight/checkpoint/training programs; these verify experiment
  bindings, per-experiment seed records, environment variables, command
  arguments, preservation of legacy run identities, and rejection of
  incompatible resume identities;
- all 40 Python files parsed and all eight shell entry points passed bash -n;
- 19 local documentation links resolved and four Bash example blocks parsed;
- all 13 published runtime source hashes and all three frozen metadata hashes
  matched; runtime/ and frozen/ have no changes from the audited snapshot;
- git diff --check passed, and independent review found no remaining release
  blockers.

Mocked entry-point tests do not execute a scientific trainer or load a real
checkpoint. This update does not include training or evaluation on a GPU.

## Source audit on 2026-10-28

The local source audit performed:

- Python AST parsing of 39 files and shell parsing of eight entry points;
- all seven existing dependency-free contract and synthetic-data checks;
- verification of the frozen dataset, plans, and fingerprint hashes;
- comparison of the paper's architecture, three ablation definitions,
  training/inference settings, and model parameter count with the code.

The source-hash list originally included an unshipped comparison utility.
That packaging inconsistency was identified separately from the 13 existing
runtime files whose recorded hashes matched.

The model has 19,144,082 registered parameters by static layer counting.
This count was not produced by constructing a Torch model in the authoring
environment.

## Historical release checks on 2026-09-15

The original release records Python and shell parsing, JSON contract checks,
public-material and credential/private-path scans, frozen metadata checks,
synthetic nested-ZIP payload preservation and path-traversal rejection, and
inspect-only entry-point checks. Those records describe that release's scope;
they are not evidence that a later public clone reproduced the paper tables.

## Execution not performed in the authoring environment

- Linux cold-start installation and package resolution;
- NVIDIA driver, CUDA, Blackwell/5090, or Ada/4090 compatibility testing;
- importing the trainer from an installed nnU-Net scientific runtime;
- GPU model construction, forward/backward execution, or checkpoint loading;
- real archive staging, preprocessing, training, resume, or evaluation;
- reproduction of the internal375 or supplemental novel100 results from the
  public clone.

These tasks require the appropriate runtime, hardware, data, and membership
information. Passing static checks is not a substitute for those executions.
