# Release notes

## Public MoLG-UNet release

The initial FedMoLG-TCFP tree is retained under
legacy/fedmolg_tcfp. Its federated, 100-round and Dice-HD95 model-selection
workflow remains available as historical code, but it is not the default
reproduction route.

The current runtime sources were taken from the reviewed recipe-v2 snapshot.
The scientific files are copied unchanged. Public changes are limited to:

- relocating the earlier public prototype under a clearly labelled legacy
  path;
- adding the current trainer and model modules, frozen metadata and exact
  version recipes;
- adding generic path-parameterized install, environment, preflight, train
  and evaluation entry points;
- documenting the four formal ablations, the full reference trainer, data
  boundaries, checkpoint policy and metrics;
- removing private run identities, private paths and case-level membership
  files from the release surface.

The release does not claim that every package can be cold-start installed on
every Linux distribution, or that one model is universally superior. Reported
experimental results are kept with the private evaluation backup rather than
used to alter the implementation.
