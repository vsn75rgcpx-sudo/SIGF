from __future__ import annotations

import gzip
import importlib.util
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "runtime/baselines/prepare_data_lossless.py"
SPEC = importlib.util.spec_from_file_location("molg_lossless", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_nested_brats_member_is_streamed_without_payload_change():
    case = "BraTS-GLI-00001-000"
    payload = (348).to_bytes(4, "little", signed=True) + bytes(range(4)) * 100
    with tempfile.TemporaryDirectory() as directory:
        archive_path = Path(directory) / "synthetic.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for role in ("t1n", "t1c", "t2w", "t2f", "seg"):
                archive.writestr(f"nested/{case}/{case}-{role}.nii", payload)
        indexed = MODULE.index_brats_archive(archive_path)
        assert set(indexed) == {case}
        with zipfile.ZipFile(archive_path) as archive:
            info = archive.getinfo(f"nested/{case}/{case}-t1n.nii")
            destination = Path(directory) / "out.nii.gz"
            record = MODULE.stream_brats_member(archive, info, destination)
        with gzip.open(destination, "rb") as handle:
            assert handle.read() == payload
        assert record["source_bytes"] == len(payload)
        assert record["output_bytes"] == destination.stat().st_size


def test_zip_path_traversal_is_rejected():
    try:
        MODULE.safe_zip_name("../outside")
    except RuntimeError:
        pass
    else:
        raise AssertionError("unsafe ZIP path was accepted")
