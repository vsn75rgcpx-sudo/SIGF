#!/usr/bin/env python3
"""CPU-only checkpoint field check; it does not initialize or restore a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED = ("trainer_name", "current_epoch", "network_weights", "optimizer_state", "grad_scaler_state")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import torch

    path = Path(args.checkpoint)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("checkpoint top level is not a mapping")
    fields = {name: name in payload for name in REQUIRED}
    result = {
        "checkpoint": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "top_level_fields": sorted(payload),
        "field_presence": fields,
        "grad_scaler_state_is_none": payload.get("grad_scaler_state") is None,
        "note": "field completeness only; actual trainer restore is proven only by nnU-Net --c execution",
    }
    if not all(fields.values()):
        raise RuntimeError(json.dumps(result, sort_keys=True))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
