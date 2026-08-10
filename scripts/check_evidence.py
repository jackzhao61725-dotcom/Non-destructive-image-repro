"""Verify the bundled retained synthetic evidence without rerunning analysis."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import struct
from typing import Any, Mapping, Sequence
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "evidence" / "retained_v1" / "manifest.json"
EXPECTED_STATUS = "curated_public_derivative_of_admitted_synthetic_evidence"


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _safe_relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError("evidence path must be non-empty text")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"unsafe evidence path: {value!r}")
    return path


def _csv_data_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.reader(handle)
        try:
            next(rows)
        except StopIteration:
            return 0
        return sum(1 for _ in rows)


def _npy_header(handle: Any) -> Mapping[str, Any]:
    if handle.read(6) != b"\x93NUMPY":
        raise ValueError("missing NPY magic")
    version = handle.read(2)
    if len(version) != 2:
        raise ValueError("truncated NPY version")
    major = version[0]
    if major == 1:
        size_bytes = handle.read(2)
        if len(size_bytes) != 2:
            raise ValueError("truncated NPY header size")
        header_size = struct.unpack("<H", size_bytes)[0]
    elif major in {2, 3}:
        size_bytes = handle.read(4)
        if len(size_bytes) != 4:
            raise ValueError("truncated NPY header size")
        header_size = struct.unpack("<I", size_bytes)[0]
    else:
        raise ValueError(f"unsupported NPY major version {major}")
    encoding = "utf-8" if major == 3 else "latin1"
    header = handle.read(header_size).decode(encoding).strip()
    payload = ast.literal_eval(header)
    if not isinstance(payload, Mapping):
        raise ValueError("NPY header is not a mapping")
    return payload


def _npz_schema_failures(path: Path, expected: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    with zipfile.ZipFile(path) as archive:
        members = {
            PurePosixPath(name).stem: name
            for name in archive.namelist()
            if name.endswith(".npy")
        }
        if set(members) != set(expected):
            failures.append(
                f"{path.name}: NPZ arrays expected {sorted(expected)}, "
                f"observed {sorted(members)}"
            )
            return failures
        for name, contract in expected.items():
            if not isinstance(contract, Mapping):
                failures.append(f"{path.name}:{name}: schema must be an object")
                continue
            with archive.open(members[name]) as handle:
                header = _npy_header(handle)
            observed_shape = list(header.get("shape", ()))
            expected_shape = contract.get("shape")
            if observed_shape != expected_shape:
                failures.append(
                    f"{path.name}:{name}: expected shape {expected_shape}, "
                    f"observed {observed_shape}"
                )
            descriptor = header.get("descr")
            exact_descriptor = contract.get("npy_descr")
            if exact_descriptor is not None and descriptor != exact_descriptor:
                failures.append(
                    f"{path.name}:{name}: expected dtype descriptor "
                    f"{exact_descriptor!r}, observed {descriptor!r}"
                )
            expected_kind = contract.get("dtype_kind")
            if expected_kind is not None:
                observed_kind = (
                    descriptor.lstrip("<>=|")[:1]
                    if isinstance(descriptor, str)
                    else ""
                )
                if observed_kind != expected_kind:
                    failures.append(
                        f"{path.name}:{name}: expected dtype kind "
                        f"{expected_kind!r}, observed {observed_kind!r}"
                    )
    return failures


def check_bundle(manifest_path: Path) -> list[str]:
    manifest = _read_json(manifest_path)
    root = manifest_path.parent
    failures: list[str] = []
    if manifest.get("status") != EXPECTED_STATUS:
        failures.append("manifest has the wrong evidence classification")
    if manifest.get("scientific_compute_executed") is not False:
        failures.append("manifest must state that no scientific compute was executed")

    entries = manifest.get("files")
    if not isinstance(entries, list):
        return failures + ["manifest files must be a list"]
    declared_paths: set[str] = set()
    declared_bytes = 0
    raw_hashes: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            failures.append("manifest file entry must be an object")
            continue
        try:
            relative = _safe_relative_path(entry.get("path"))
        except ValueError as error:
            failures.append(str(error))
            continue
        relative_text = relative.as_posix()
        if relative_text in declared_paths:
            failures.append(f"duplicate manifest path: {relative_text}")
            continue
        declared_paths.add(relative_text)
        path = root.joinpath(*relative.parts)
        if not path.is_file():
            failures.append(f"{relative_text}: missing")
            continue
        expected_bytes = entry.get("bytes")
        if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool):
            failures.append(f"{relative_text}: bytes must be an integer")
        else:
            declared_bytes += expected_bytes
            if path.stat().st_size != expected_bytes:
                failures.append(
                    f"{relative_text}: expected {expected_bytes} bytes, "
                    f"observed {path.stat().st_size}"
                )
        expected_hash = entry.get("sha256")
        observed_hash = _sha256_file(path)
        if expected_hash != observed_hash:
            failures.append(
                f"{relative_text}: expected SHA-256 {expected_hash}, "
                f"observed {observed_hash}"
            )
        raw_hashes[relative_text] = observed_hash
        expected_rows = entry.get("rows")
        if expected_rows is not None:
            if path.suffix.lower() != ".csv":
                failures.append(f"{relative_text}: rows declared for a non-CSV file")
            else:
                observed_rows = _csv_data_rows(path)
                if observed_rows != expected_rows:
                    failures.append(
                        f"{relative_text}: expected {expected_rows} data rows, "
                        f"observed {observed_rows}"
                    )
        expected_schema = entry.get("npz_schema")
        if expected_schema is not None:
            if not isinstance(expected_schema, Mapping):
                failures.append(f"{relative_text}: npz_schema must be an object")
            else:
                failures.extend(_npz_schema_failures(path, expected_schema))

    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual_paths != declared_paths:
        failures.append(
            "bundle file inventory differs from manifest: "
            f"missing={sorted(declared_paths - actual_paths)}, "
            f"extra={sorted(actual_paths - declared_paths)}"
        )
    if manifest.get("bundle_file_count_excluding_manifest") != len(declared_paths):
        failures.append("declared bundle file count does not match the manifest entries")
    if manifest.get("bundle_bytes_excluding_manifest") != declared_bytes:
        failures.append("declared bundle byte count does not match the manifest entries")

    fixed_manifest = _read_json(root / "fixed_field" / "target_manifest.json")
    if fixed_manifest.get("raw_target_sha256") != raw_hashes.get(
        "fixed_field/target_raw_roles.npz"
    ):
        failures.append("fixed-field target manifest does not identify its raw NPZ")
    orientation_manifest = _read_json(
        root / "orientation" / "redacted_target_manifest.json"
    )
    observed_raw = orientation_manifest.get("observed_raw")
    if not isinstance(observed_raw, Mapping) or observed_raw.get(
        "sha256"
    ) != raw_hashes.get("orientation/observed_raw.npz"):
        failures.append("orientation target manifest does not identify its raw NPZ")
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        failures = check_bundle(args.manifest.resolve())
    except (OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        print(f"FAIL evidence manifest: {error}")
        return 2
    if failures:
        print(f"FAIL evidence ({len(failures)} mismatches)")
        for failure in failures:
            print(f"  {failure}")
        return 1
    manifest = _read_json(args.manifest.resolve())
    print(
        "PASS evidence "
        f"({manifest['bundle_file_count_excluding_manifest']} files, "
        f"{manifest['bundle_bytes_excluding_manifest']} bytes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
