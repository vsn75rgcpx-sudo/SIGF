#!/usr/bin/env python3
"""Prepare the closeout data without changing source NIfTI voxel streams.

This utility has three deliberately separate operations:

* ``membership`` recovers the fixed train/internal/external case sets from the
  four historical CSVs for each evaluation group.
* ``convert-brats2023`` streams nested ``.nii`` ZIP members into deterministic
  ``.nii.gz`` files. The NIfTI payload is never loaded or re-serialized.
* ``extract-goat`` copies the existing ``.nii.gz`` member payloads from the
  GoAT ZIP without a second compression pass.

The commands are intended to run in a versioned closeout directory. They are
resumable only for complete files carrying a matching marker; an unmarked
different file is treated as a collision and stops the job.
"""

import argparse
import csv
import errno
import gzip
import hashlib
import json
import os
import posixpath
import re
import shutil
import stat
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


CHUNK_SIZE = 1024 * 1024
BRATS_MODALITY_DIRS = {
    "t1n": "t1n",
    "t1c": "t1c",
    "t2w": "t2w",
    "t2f": "t2f",
}
MODALITY_ORDER = ("t1n", "t1c", "t2w", "t2f")
BRATS_CASE_RE = re.compile(r"BraTS-GLI-\d{5}-\d{3}")
GOAT_CASE_RE = re.compile(r"BraTS-GoAT-\d{5}")


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def safe_zip_name(name: str) -> str:
    if "\\" in name or name.startswith("/"):
        fail(f"Unsafe ZIP member name: {name!r}")
    normalized = posixpath.normpath(name)
    if normalized in ("", ".") or normalized == ".." or normalized.startswith("../"):
        fail(f"Unsafe ZIP member traversal: {name!r}")
    return normalized


def ensure_regular_zip_member(info: zipfile.ZipInfo) -> str:
    if info.filename.endswith("/") or info.is_dir():
        return "directory"
    safe_zip_name(info.filename)
    mode = (info.external_attr >> 16) & 0o170000
    if mode == stat.S_IFLNK or mode in (stat.S_IFCHR, stat.S_IFBLK, stat.S_IFIFO):
        fail(f"Refusing non-regular ZIP member: {info.filename!r}")
    return "file"


def read_case_csv(path: Path) -> List[str]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "case" not in reader.fieldnames:
            fail(f"{path} does not have the required actual 'case' column: {reader.fieldnames}")
        cases: List[str] = []
        seen = set()
        for row_number, row in enumerate(reader, start=2):
            case = (row.get("case") or "").strip()
            if not case:
                fail(f"Empty case at {path}:{row_number}")
            if case in seen:
                fail(f"Duplicate case {case} at {path}:{row_number}")
            seen.add(case)
            cases.append(case)
    return sorted(cases)


def csv_record(path: Path, cases: Sequence[str]) -> Dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "count": len(cases),
        "cases_sha256": hashlib.sha256(("\n".join(cases) + "\n").encode()).hexdigest(),
    }


def archive_record(path: Path, kind: str, provided_sha256: str) -> Dict[str, object]:
    archive_sha256 = provided_sha256
    if not archive_sha256:
        archive_sha256 = sha256_file(path) if os.environ.get("CLOSEOUT_HASH_ARCHIVES") == "1" else "not-rehashed-in-membership-pass"
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": archive_sha256,
        "kind": kind,
    }


def index_brats_archive(path: Path) -> Dict[str, Dict[str, object]]:
    result: Dict[str, Dict[str, object]] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if ensure_regular_zip_member(info) == "directory":
                continue
            name = safe_zip_name(info.filename)
            matches = BRATS_CASE_RE.findall(name)
            if not matches:
                continue
            if len(set(matches)) != 1:
                fail(f"Ambiguous 2023 case in ZIP member: {name}")
            case = matches[0]
            parts = name.split("/")
            case_positions = [i for i, part in enumerate(parts) if part == case]
            if len(case_positions) != 1:
                fail(f"Unexpected 2023 case path: {name}")
            role: Optional[str] = None
            if parts[-1] == case + "-seg.nii":
                role = "seg"
            elif len(parts) >= 2 and parts[-2].startswith(case + "-") and parts[-2].endswith(".nii"):
                token = parts[-2][len(case) + 1:-4]
                role = BRATS_MODALITY_DIRS.get(token)
            elif parts[-1].startswith(case + "-") and parts[-1].endswith(".nii"):
                # Most 2023 cases use a nested component directory. A small
                # subset marked with the -001 suffix stores the same payload
                # as a flat member; both are source-equivalent layouts.
                token = parts[-1][len(case) + 1:-4]
                role = BRATS_MODALITY_DIRS.get(token)
            if role is None:
                fail(f"Unexpected 2023 member under known case: {name}")
            entry = result.setdefault(case, {"members": {}})
            members = entry["members"]
            assert isinstance(members, dict)
            if role in members:
                fail(f"Duplicate 2023 {role} member for {case}: {members[role]} and {name}")
            members[role] = {
                "name": name,
                "file_size": info.file_size,
                "compress_size": info.compress_size,
                "crc": info.CRC,
                "compress_type": info.compress_type,
            }
    for case, entry in result.items():
        members = entry["members"]
        assert isinstance(members, dict)
        expected = set(MODALITY_ORDER) | {"seg"}
        if set(members) != expected:
            fail(f"2023 case {case} has roles {sorted(members)}, expected {sorted(expected)}")
    return result


def index_goat_archive(path: Path) -> Dict[str, Dict[str, object]]:
    result: Dict[str, Dict[str, object]] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if ensure_regular_zip_member(info) == "directory":
                continue
            name = safe_zip_name(info.filename)
            matches = GOAT_CASE_RE.findall(name)
            if not matches:
                continue
            if len(set(matches)) != 1:
                fail(f"Ambiguous GoAT case in ZIP member: {name}")
            case = matches[0]
            basename = name.rsplit("/", 1)[-1]
            expected_prefix = case + "-"
            if not basename.startswith(expected_prefix) or not basename.endswith(".nii.gz"):
                fail(f"Unexpected GoAT member under known case: {name}")
            token = basename[len(expected_prefix):-7]
            role = "seg" if token == "seg" else BRATS_MODALITY_DIRS.get(token)
            if role is None:
                fail(f"Unexpected GoAT modality in member: {name}")
            entry = result.setdefault(case, {"members": {}})
            members = entry["members"]
            assert isinstance(members, dict)
            if role in members:
                fail(f"Duplicate GoAT {role} member for {case}: {members[role]} and {name}")
            members[role] = {
                "name": name,
                "file_size": info.file_size,
                "compress_size": info.compress_size,
                "crc": info.CRC,
                "compress_type": info.compress_type,
            }
    for case, entry in result.items():
        members = entry["members"]
        assert isinstance(members, dict)
        expected = set(MODALITY_ORDER) | {"seg"}
        if set(members) != expected:
            fail(f"GoAT case {case} has roles {sorted(members)}, expected {sorted(expected)}")
    return result


def require_same_sets(records: Sequence[Tuple[Path, List[str]]], label: str) -> List[str]:
    if not records:
        fail(f"No {label} CSVs supplied")
    reference = set(records[0][1])
    for path, cases in records[1:]:
        if set(cases) != reference:
            fail(f"{label} membership mismatch in {path}: symmetric difference {sorted(reference ^ set(cases))[:20]}")
    return sorted(reference)


def membership_command(args: argparse.Namespace) -> None:
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        fail(f"Membership output is not empty; refusing overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)

    internal_records = [(Path(p), read_case_csv(Path(p))) for p in args.internal_csv]
    external_records = [(Path(p), read_case_csv(Path(p))) for p in args.external_csv]
    internal = require_same_sets(internal_records, "internal")
    external = require_same_sets(external_records, "external")
    if len(internal) != 375:
        fail(f"Expected 375 recovered internal cases, got {len(internal)}")
    if len(external) != 100:
        fail(f"Expected 100 recovered external cases, got {len(external)}")
    if "BraTS-GoAT-00598" in external:
        fail("Known reprocessed overlap BraTS-GoAT-00598 is present in external novel100")

    brats_archive = Path(args.brats_archive)
    goat_archive = Path(args.goat_archive)
    brats = index_brats_archive(brats_archive)
    goat = index_goat_archive(goat_archive)
    brats_cases = set(brats)
    goat_cases = set(goat)
    if len(brats_cases) != 1251:
        fail(f"Expected 1251 2023 cases, got {len(brats_cases)}")
    if len(goat_cases) != 1351:
        fail(f"Expected 1351 GoAT cases, got {len(goat_cases)}")
    if not set(internal).issubset(brats_cases):
        fail(f"Internal cases missing from 2023 archive: {sorted(set(internal) - brats_cases)}")
    if not set(external).issubset(goat_cases):
        fail(f"External cases missing from GoAT archive: {sorted(set(external) - goat_cases)}")
    train = sorted(brats_cases - set(internal))
    if len(train) != 876 or set(train) & set(internal):
        fail(f"Recovered training complement is not exactly 876 disjoint cases: {len(train)}")

    records = {
        "schema_version": 1,
        "created_unix": time.time(),
        "case_column": "case",
        "provenance": "recovered from four matching historical case_metrics.csv files per evaluation group; not an original split manifest",
        "internal375": {
            "count": len(internal),
            "cases": internal,
            "csvs": [csv_record(path, cases) for path, cases in internal_records],
        },
        "external_novel100": {
            "count": len(external),
            "cases": external,
            "csvs": [csv_record(path, cases) for path, cases in external_records],
            "known_excluded_reprocessed_overlap": "BraTS-GoAT-00598",
        },
        "training876": {
            "count": len(train),
            "cases": train,
            "construction": "set(2023 archive cases) - internal375",
        },
        "archives": {
            "brats2023": dict(archive_record(brats_archive, "nested_uncompressed_nifti", args.brats_sha256), case_count=len(brats_cases), member_count=sum(len(v["members"]) for v in brats.values())),
            "goat2024": dict(archive_record(goat_archive, "compressed_nifti", args.goat_sha256), case_count=len(goat_cases), member_count=sum(len(v["members"]) for v in goat.values())),
        },
        "archive_crosscheck": {
            "internal_all_present": True,
            "external_all_present": True,
            "training_complement_exact": True,
            "external_excludes_known_reprocessed_overlap": True,
        },
    }
    atomic_json(out / "membership_manifest.json", records)
    (out / "internal375.txt").write_text("\n".join(internal) + "\n", encoding="utf-8")
    (out / "external_novel100.txt").write_text("\n".join(external) + "\n", encoding="utf-8")
    (out / "training876.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    print(json.dumps({"internal375": len(internal), "training876": len(train), "external_novel100": len(external)}, sort_keys=True))


def load_case_list(path: Path, expected_count: int) -> List[str]:
    cases = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(cases) != expected_count or len(set(cases)) != expected_count:
        fail(f"Case list {path} expected {expected_count} unique cases, got {len(cases)}")
    return sorted(cases)


def ensure_new_directory(path: Path) -> None:
    if path.exists() and not path.is_dir():
        fail(f"Expected directory, found non-directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def check_nifti_header(header: bytes, member_name: str) -> None:
    if len(header) < 352:
        fail(f"NIfTI member is shorter than the required header: {member_name}")
    sizeof_little = int.from_bytes(header[0:4], "little", signed=True)
    sizeof_big = int.from_bytes(header[0:4], "big", signed=True)
    if sizeof_little != 348 and sizeof_big != 348:
        fail(f"Member does not have a NIfTI-1 header: {member_name}")


def marker_valid(marker: Path, expected: Mapping[str, object], output: Path) -> bool:
    if not marker.exists() or not output.exists() or output.is_symlink():
        return False
    try:
        existing = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if existing != dict(expected):
        return False
    return output.stat().st_size == int(expected["output_bytes"]) and sha256_file(output) == expected["output_sha256"]


def stream_brats_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, destination: Path) -> Dict[str, object]:
    partial = destination.with_name(destination.name + ".partial")
    header = bytearray()
    source_digest = hashlib.sha256()
    source_bytes = 0
    try:
        with archive.open(info, "r") as source, partial.open("wb") as raw_output:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed:
                for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
                    if len(header) < 352:
                        header.extend(chunk[:352 - len(header)])
                    source_digest.update(chunk)
                    source_bytes += len(chunk)
                    compressed.write(chunk)
        if source_bytes != info.file_size:
            fail(f"ZIP member size changed while reading {info.filename}: {source_bytes} != {info.file_size}")
        check_nifti_header(bytes(header), info.filename)
        os.replace(str(partial), str(destination))
    except OSError as exc:
        if exc.errno in (errno.ENOSPC, errno.EDQUOT):
            print(f"STORAGE_ERROR errno={exc.errno} destination={destination}", file=sys.stderr, flush=True)
        raise
    return {
        "source_bytes": source_bytes,
        "source_sha256": source_digest.hexdigest(),
        "output_bytes": destination.stat().st_size,
        "output_sha256": sha256_file(destination),
    }


def stream_goat_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, destination: Path) -> Dict[str, object]:
    partial = destination.with_name(destination.name + ".partial")
    digest = hashlib.sha256()
    copied = 0
    try:
        with archive.open(info, "r") as source, partial.open("wb") as target:
            for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
                digest.update(chunk)
                copied += len(chunk)
                target.write(chunk)
        if copied != info.file_size:
            fail(f"ZIP member size changed while reading {info.filename}: {copied} != {info.file_size}")
        os.replace(str(partial), str(destination))
    except OSError as exc:
        if exc.errno in (errno.ENOSPC, errno.EDQUOT):
            print(f"STORAGE_ERROR errno={exc.errno} destination={destination}", file=sys.stderr, flush=True)
        raise
    return {
        "payload_bytes": copied,
        "payload_sha256": digest.hexdigest(),
        "output_bytes": destination.stat().st_size,
        "output_sha256": sha256_file(destination),
    }


def convert_brats_command(args: argparse.Namespace) -> None:
    cases = load_case_list(Path(args.case_list), args.expected_cases)
    out = Path(args.out)
    ensure_new_directory(out)
    source = Path(args.archive)
    source_stat = source.stat()
    all_records = []
    with zipfile.ZipFile(source) as archive:
        indexed = index_brats_archive(source)
        for index, case in enumerate(cases, start=1):
            if case not in indexed:
                fail(f"Requested 2023 case is not in archive: {case}")
            case_dir = out / case
            ensure_new_directory(case_dir)
            members = indexed[case]["members"]
            assert isinstance(members, dict)
            case_record = {"case": case, "files": {}}
            for role in MODALITY_ORDER + ("seg",):
                member = members[role]
                assert isinstance(member, dict)
                info = archive.getinfo(member["name"])
                destination = case_dir / f"{case}_{role}.nii.gz"
                marker = destination.with_name(destination.name + ".json")
                expected_prefix = {
                    "source_member": member["name"],
                    "source_file_size": member["file_size"],
                    "source_crc": member["crc"],
                    "packaging": "gzip-mtime0-no-filename",
                }
                if marker.exists():
                    existing = json.loads(marker.read_text(encoding="utf-8"))
                    if not all(existing.get(key) == value for key, value in expected_prefix.items()):
                        fail(f"Existing marker differs for {destination}")
                    if not marker_valid(marker, existing, destination):
                        fail(f"Existing 2023 output failed marker/hash validation: {destination}")
                    record = existing
                else:
                    if destination.exists() or destination.is_symlink():
                        fail(f"Unmarked existing output is a collision: {destination}")
                    record = dict(expected_prefix)
                    record.update(stream_brats_member(archive, info, destination))
                    atomic_json(marker, record)
                case_record["files"][role] = record
            atomic_json(case_dir / "case_complete.json", {"case": case, "files": case_record["files"]})
            all_records.append(case_record)
            if index == 1 or index % 25 == 0 or index == len(cases):
                print(f"[2023] {index}/{len(cases)} {case}", flush=True)
    atomic_json(out / "stage_manifest.json", {
        "schema_version": 1,
        "kind": "brats2023_nested_stream_gzip",
        "source_archive": str(source),
        "source_archive_bytes": source_stat.st_size,
        "source_archive_sha256": os.environ.get("CLOSEOUT_BRATS_ARCHIVE_SHA256", "provided-by-migration"),
        "case_count": len(all_records),
        "cases": all_records,
    })


def extract_goat_command(args: argparse.Namespace) -> None:
    cases = load_case_list(Path(args.case_list), args.expected_cases)
    out = Path(args.out)
    ensure_new_directory(out)
    source = Path(args.archive)
    source_stat = source.stat()
    all_records = []
    with zipfile.ZipFile(source) as archive:
        indexed = index_goat_archive(source)
        for index, case in enumerate(cases, start=1):
            if case not in indexed:
                fail(f"Requested GoAT case is not in archive: {case}")
            case_dir = out / case
            ensure_new_directory(case_dir)
            members = indexed[case]["members"]
            assert isinstance(members, dict)
            case_record = {"case": case, "files": {}}
            for role in MODALITY_ORDER + ("seg",):
                member = members[role]
                assert isinstance(member, dict)
                info = archive.getinfo(member["name"])
                destination = case_dir / f"{case}_{role}.nii.gz"
                marker = destination.with_name(destination.name + ".json")
                expected_prefix = {
                    "source_member": member["name"],
                    "source_file_size": member["file_size"],
                    "source_crc": member["crc"],
                    "packaging": "copied-inner-gzip-payload",
                }
                if marker.exists():
                    existing = json.loads(marker.read_text(encoding="utf-8"))
                    if not all(existing.get(key) == value for key, value in expected_prefix.items()):
                        fail(f"Existing marker differs for {destination}")
                    if not marker_valid(marker, existing, destination):
                        fail(f"Existing GoAT output failed marker/hash validation: {destination}")
                    record = existing
                else:
                    if destination.exists() or destination.is_symlink():
                        fail(f"Unmarked existing output is a collision: {destination}")
                    record = dict(expected_prefix)
                    record.update(stream_goat_member(archive, info, destination))
                    atomic_json(marker, record)
                case_record["files"][role] = record
            atomic_json(case_dir / "case_complete.json", {"case": case, "files": case_record["files"]})
            all_records.append(case_record)
            if index == 1 or index % 25 == 0 or index == len(cases):
                print(f"[GoAT] {index}/{len(cases)} {case}", flush=True)
    atomic_json(out / "stage_manifest.json", {
        "schema_version": 1,
        "kind": "goat2024_selected_inner_gzip_payloads",
        "source_archive": str(source),
        "source_archive_bytes": source_stat.st_size,
        "source_archive_sha256": os.environ.get("CLOSEOUT_GOAT_ARCHIVE_SHA256", "provided-by-migration"),
        "case_count": len(all_records),
        "cases": all_records,
    })


def audit_stage_command(args: argparse.Namespace) -> None:
    """Re-read every staged gzip stream and verify its marker provenance.

    The conversion/extraction pass already checks the ZIP member CRC while it
    streams. This second pass is intentionally independent: it validates the
    completed gzip CRC/EOF, NIfTI header, output hash, and (for the 2023
    conversion) the uncompressed source digest recorded in the marker.
    """
    cases = load_case_list(Path(args.case_list), args.expected_cases)
    stage = Path(args.stage)
    manifest_path = stage / "stage_manifest.json"
    if not manifest_path.is_file():
        fail(f"Missing stage manifest: {manifest_path}")
    stage_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_kind = {
        "brats2023": "brats2023_nested_stream_gzip",
        "goat2024": "goat2024_selected_inner_gzip_payloads",
    }[args.kind]
    if stage_manifest.get("kind") != expected_kind:
        fail(f"Stage kind mismatch at {manifest_path}: {stage_manifest.get('kind')!r}")
    manifest_cases = [record.get("case") for record in stage_manifest.get("cases", [])]
    if sorted(manifest_cases) != cases or len(manifest_cases) != args.expected_cases:
        fail(f"Stage manifest case set/count mismatch: {manifest_path}")

    total_files = 0
    total_output_bytes = 0
    total_uncompressed_bytes = 0
    case_summaries = []
    for index, case in enumerate(cases, start=1):
        case_dir = stage / case
        case_file = case_dir / "case_complete.json"
        if not case_file.is_file():
            fail(f"Missing complete-case marker: {case_file}")
        case_complete = json.loads(case_file.read_text(encoding="utf-8"))
        files = case_complete.get("files")
        if not isinstance(files, dict) or set(files) != set(MODALITY_ORDER) | {"seg"}:
            fail(f"Incomplete case marker: {case_file}")
        case_uncompressed = 0
        for role in MODALITY_ORDER + ("seg",):
            output = case_dir / f"{case}_{role}.nii.gz"
            marker = output.with_name(output.name + ".json")
            if not marker.is_file() or not output.is_file() or output.is_symlink():
                fail(f"Missing staged output or marker: {output}")
            marker_data = json.loads(marker.read_text(encoding="utf-8"))
            if not marker_valid(marker, marker_data, output):
                fail(f"Output hash/size does not match marker: {output}")
            digest = hashlib.sha256()
            uncompressed_bytes = 0
            header = bytearray()
            with gzip.open(output, "rb") as source:
                for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
                    if len(header) < 352:
                        header.extend(chunk[:352 - len(header)])
                    digest.update(chunk)
                    uncompressed_bytes += len(chunk)
            check_nifti_header(bytes(header), str(output))
            if args.kind == "brats2023":
                if uncompressed_bytes != int(marker_data["source_bytes"]):
                    fail(f"Uncompressed size mismatch: {output}")
                if digest.hexdigest() != marker_data["source_sha256"]:
                    fail(f"Uncompressed source digest mismatch: {output}")
            case_uncompressed += uncompressed_bytes
            total_files += 1
            total_output_bytes += output.stat().st_size
            total_uncompressed_bytes += uncompressed_bytes
        case_summaries.append({"case": case, "files": 5, "uncompressed_bytes": case_uncompressed})
        if index == 1 or index % 25 == 0 or index == len(cases):
            print(f"[audit {args.kind}] {index}/{len(cases)} {case}", flush=True)

    atomic_json(stage / "audit_manifest.json", {
        "schema_version": 1,
        "kind": args.kind,
        "stage": str(stage),
        "stage_manifest_sha256": sha256_file(manifest_path),
        "case_count": len(cases),
        "file_count": total_files,
        "output_bytes": total_output_bytes,
        "uncompressed_bytes": total_uncompressed_bytes,
        "gzip_crc_and_nifti_header_verified": True,
        "case_summaries": case_summaries,
        "created_unix": time.time(),
    })
    print(json.dumps({
        "stage": str(stage),
        "kind": args.kind,
        "cases": len(cases),
        "files": total_files,
        "output_bytes": total_output_bytes,
        "uncompressed_bytes": total_uncompressed_bytes,
    }, sort_keys=True))


def install_link(destination: Path, source: Path) -> None:
    if not source.exists() or source.is_symlink() and not source.exists():
        fail(f"Missing staged source for link: {source}")
    if destination.is_symlink():
        if os.path.realpath(destination) == str(source.resolve()):
            return
        fail(f"Existing link points elsewhere: {destination} -> {os.readlink(destination)}")
    if destination.exists():
        fail(f"Existing non-link file would be overwritten: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(source), str(destination))


def copy_if_identical_or_absent(source: Path, destination: Path) -> None:
    if not source.is_file():
        fail(f"Missing frozen metadata source: {source}")
    if destination.exists() or destination.is_symlink():
        if not destination.is_file() or sha256_file(source) != sha256_file(destination):
            fail(f"Frozen metadata collision at {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(destination))


def build_raw_command(args: argparse.Namespace) -> None:
    membership = Path(args.membership)
    train = load_case_list(membership / "training876.txt", 876)
    internal = load_case_list(membership / "internal375.txt", 375)
    external = load_case_list(membership / "external_novel100.txt", 100)
    raw_root = Path(args.raw_root)
    raw_dataset = raw_root / "Dataset901_BraTS2023_TCRegion"
    preprocessed_root = Path(args.preprocessed_root)
    preprocessed_dataset = preprocessed_root / "Dataset901_BraTS2023_TCRegion"
    for path in (raw_dataset / "imagesTr", raw_dataset / "labelsTr", raw_dataset / "imagesTs", raw_dataset / "labelsTs", preprocessed_dataset):
        ensure_new_directory(path)
    copy_if_identical_or_absent(Path(args.dataset_json), raw_dataset / "dataset.json")
    copy_if_identical_or_absent(Path(args.dataset_json), preprocessed_dataset / "dataset.json")
    copy_if_identical_or_absent(Path(args.plans_json), preprocessed_dataset / "nnUNetPlans.json")

    brats_stage = Path(args.brats_stage)
    goat_stage = Path(args.goat_stage)
    eval_root = Path(args.eval_root)
    all_layout = []

    def add_brats_case(case: str, images_dir: Path, labels_dir: Path) -> None:
        case_dir = brats_stage / case
        if not case_dir.is_dir():
            fail(f"Missing staged 2023 case: {case}")
        for channel, role in enumerate(MODALITY_ORDER):
            source = case_dir / f"{case}_{role}.nii.gz"
            destination = images_dir / f"{case}_{channel:04d}.nii.gz"
            install_link(destination, source)
        source_label = case_dir / f"{case}_seg.nii.gz"
        install_link(labels_dir / f"{case}.nii.gz", source_label)

    for case in train:
        add_brats_case(case, raw_dataset / "imagesTr", raw_dataset / "labelsTr")
    for case in internal:
        add_brats_case(case, raw_dataset / "imagesTs", raw_dataset / "labelsTs")

    def add_goat_case(case: str, images_dir: Path, labels_dir: Path) -> None:
        case_dir = goat_stage / case
        if not case_dir.is_dir():
            fail(f"Missing staged GoAT case: {case}")
        for channel, role in enumerate(MODALITY_ORDER):
            install_link(images_dir / f"{case}_{channel:04d}.nii.gz", case_dir / f"{case}_{role}.nii.gz")
        install_link(labels_dir / f"{case}.nii.gz", case_dir / f"{case}_seg.nii.gz")

    internal_eval = eval_root / "internal375"
    external_eval = eval_root / "external_novel100"
    for directory in (internal_eval / "imagesTs", internal_eval / "labelsTs", external_eval / "imagesTs", external_eval / "labelsTs"):
        ensure_new_directory(directory)
    for case in internal:
        add_brats_case(case, internal_eval / "imagesTs", internal_eval / "labelsTs")
    for case in external:
        add_goat_case(case, external_eval / "imagesTs", external_eval / "labelsTs")

    for case in train:
        all_layout.append({"case": case, "group": "training876", "source": "brats2023"})
    for case in internal:
        all_layout.append({"case": case, "group": "internal375", "source": "brats2023"})
    for case in external:
        all_layout.append({"case": case, "group": "external_novel100", "source": "goat2024"})
    atomic_json(raw_root / "closeout_layout_manifest.json", {
        "schema_version": 1,
        "dataset": "Dataset901_BraTS2023_TCRegion",
        "training_count": len(train),
        "internal_count": len(internal),
        "external_count": len(external),
        "link_policy": "absolute symlinks to versioned lossless staged payloads",
        "cases": all_layout,
    })
    print(json.dumps({"raw_dataset": str(raw_dataset), "training876": len(train), "internal375": len(internal), "external_novel100": len(external)}, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    membership = subparsers.add_parser("membership")
    membership.add_argument("--internal-csv", nargs=4, required=True)
    membership.add_argument("--external-csv", nargs=4, required=True)
    membership.add_argument("--brats-archive", required=True)
    membership.add_argument("--goat-archive", required=True)
    membership.add_argument("--brats-sha256", default="")
    membership.add_argument("--goat-sha256", default="")
    membership.add_argument("--out", required=True)
    membership.set_defaults(func=membership_command)

    convert = subparsers.add_parser("convert-brats2023")
    convert.add_argument("--archive", required=True)
    convert.add_argument("--case-list", required=True)
    convert.add_argument("--expected-cases", type=int, required=True)
    convert.add_argument("--out", required=True)
    convert.set_defaults(func=convert_brats_command)

    goat = subparsers.add_parser("extract-goat")
    goat.add_argument("--archive", required=True)
    goat.add_argument("--case-list", required=True)
    goat.add_argument("--expected-cases", type=int, required=True)
    goat.add_argument("--out", required=True)
    goat.set_defaults(func=extract_goat_command)

    audit = subparsers.add_parser("audit-stage")
    audit.add_argument("--stage", required=True)
    audit.add_argument("--case-list", required=True)
    audit.add_argument("--expected-cases", type=int, required=True)
    audit.add_argument("--kind", choices=("brats2023", "goat2024"), required=True)
    audit.set_defaults(func=audit_stage_command)

    raw = subparsers.add_parser("build-raw")
    raw.add_argument("--membership", required=True)
    raw.add_argument("--brats-stage", required=True)
    raw.add_argument("--goat-stage", required=True)
    raw.add_argument("--raw-root", required=True)
    raw.add_argument("--preprocessed-root", required=True)
    raw.add_argument("--eval-root", required=True)
    raw.add_argument("--dataset-json", required=True)
    raw.add_argument("--plans-json", required=True)
    raw.set_defaults(func=build_raw_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except OSError as exc:
        if exc.errno in (errno.ENOSPC, errno.EDQUOT):
            print(f"STORAGE_ERROR errno={exc.errno}: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
