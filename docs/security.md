# Public-release security boundary

Before publication, the release tree and the preserved Git history were
scanned for credentials, private filesystem paths, campus and rental-server
identifiers, private run identities and large scientific artifacts. No
credential or private path is intended to be present in the release.

The repository excludes raw MRI and labels, case-level lists and CSVs,
prediction NIfTI, checkpoints, full logs, virtual environments, wheelhouses,
ZIP archives and preprocessed arrays. The public data section gives the
format and the generation/checking method instead.

The existing LICENSE is retained as MIT for this repository's authored
material. Dependencies and any upstream components remain under their own
licenses; see THIRD_PARTY_NOTICES.md. This repository does not relicense
third-party packages as MIT.
