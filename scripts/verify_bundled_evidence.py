"""Verify the six admitted result families bundled with the public release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "release" / "public_release_allowlist.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _safe_relative(value: object, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} is not a portable relative path: {value}")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise ValueError(f"{label} is not a safe relative path: {value}")
    return path


def verify_evidence(root: Path = ROOT, allowlist: Path = ALLOWLIST) -> dict[str, int | str]:
    root = root.resolve()
    release = _read_json(allowlist.resolve())
    trees = release.get("evidence_trees")
    if not isinstance(trees, list) or not trees:
        raise ValueError("release inventory has no evidence trees")

    tree_count = 0
    artifact_count = 0
    total_bytes = 0
    seen_trees: set[str] = set()
    for tree_index, record in enumerate(trees):
        if not isinstance(record, dict):
            raise ValueError(f"evidence tree {tree_index} is not an object")
        relative = _safe_relative(record.get("path"), label=f"evidence tree {tree_index}")
        relative_text = relative.as_posix()
        if relative_text in seen_trees:
            raise ValueError(f"duplicate evidence tree: {relative_text}")
        seen_trees.add(relative_text)

        directory = root.joinpath(*relative.parts)
        manifest = directory / "artifact_manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        if _sha256(manifest) != record.get("artifact_manifest_sha256"):
            raise ValueError(f"evidence manifest identity changed: {manifest}")

        payload = _read_json(manifest)
        if payload.get("status") != "admitted_immutable" or payload.get("admitted") is not True:
            raise ValueError(f"evidence family is not admitted and immutable: {manifest}")
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, list) or payload.get("artifact_count") != len(artifacts):
            raise ValueError(f"invalid artifact inventory: {manifest}")

        recorded_paths: set[str] = set()
        tree_bytes = 0
        for artifact_index, artifact_record in enumerate(artifacts):
            if not isinstance(artifact_record, dict):
                raise ValueError(f"artifact {artifact_index} is not an object: {manifest}")
            artifact_relative = _safe_relative(
                artifact_record.get("path"), label=f"artifact {artifact_index}"
            )
            artifact_text = artifact_relative.as_posix()
            if artifact_text in recorded_paths:
                raise ValueError(f"duplicate artifact path in {manifest}: {artifact_text}")
            recorded_paths.add(artifact_text)
            artifact = directory.joinpath(*artifact_relative.parts)
            if not artifact.is_file():
                raise FileNotFoundError(artifact)
            size = artifact.stat().st_size
            if size != artifact_record.get("bytes") or _sha256(artifact) != artifact_record.get("sha256"):
                raise ValueError(f"artifact identity changed: {artifact}")
            tree_bytes += size

        actual_paths = {
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*")
            if path.is_file() and path.name != "artifact_manifest.json"
        }
        if actual_paths != recorded_paths:
            raise ValueError(f"evidence files differ from the manifest inventory: {manifest}")
        if tree_bytes != payload.get("total_bytes"):
            raise ValueError(f"evidence byte total changed: {manifest}")

        tree_count += 1
        artifact_count += len(artifacts)
        total_bytes += tree_bytes

    return {
        "status": "pass",
        "evidence_tree_count": tree_count,
        "artifact_count": artifact_count,
        "artifact_bytes": total_bytes,
    }


def main() -> int:
    print(json.dumps(verify_evidence(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
