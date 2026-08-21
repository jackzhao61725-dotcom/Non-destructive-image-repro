"""Verify and collect the exact figures used by the current dissertation.

The public gallery is deliberately separate from scientific recomputation.
Each bundled PDF is checked against the release inventory before it is copied;
the public example and the evidence-specific renderers cover the calculations
that are intended to be rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "release" / "public_release_allowlist.json"
DEFAULT_OUTPUT = ROOT / "output" / "public" / "manuscript_figures"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_inventory(path: Path = ALLOWLIST) -> list[dict[str, str]]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    figures = payload.get("manuscript_figures")
    if not isinstance(figures, list) or not figures:
        raise ValueError("public release inventory has no manuscript figures")

    records: list[dict[str, str]] = []
    names: set[str] = set()
    for index, record in enumerate(figures):
        if not isinstance(record, dict):
            raise ValueError(f"manuscript figure {index} is not an object")
        relative = record.get("path")
        expected = record.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError(f"manuscript figure {index} lacks path or hash")
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        observed = _sha256(source)
        if observed != expected:
            raise ValueError(f"manuscript figure identity changed: {source}")
        name = source.name
        if name in names:
            raise ValueError(f"duplicate public figure name: {name}")
        names.add(name)
        records.append(
            {
                "name": name,
                "sha256": expected,
                "kind": str(record.get("kind", "unspecified")),
                "source": relative,
            }
        )
    return records


def collect_figures(output: Path, *, check_only: bool = False) -> dict[str, Any]:
    records = _load_inventory()
    if check_only:
        return {"status": "pass", "figure_count": len(records), "written": False}

    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True)
    for record in records:
        source = ROOT / record["source"]
        destination = output / record["name"]
        shutil.copyfile(source, destination)
        if _sha256(destination) != record["sha256"]:
            raise ValueError(f"copied figure identity changed: {destination}")

    index = {
        "schema_version": 1,
        "status": "verified_exact_manuscript_figures",
        "figure_count": len(records),
        "figures": records,
        "interpretation": (
            "These PDFs are the exact presentation files used by the dissertation. "
            "Their hashes establish identity; they are not a claim that every long-run "
            "calculation was repeated by this collection step."
        ),
    }
    (output / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"status": "pass", "figure_count": len(records), "written": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    arguments = parser.parse_args()
    summary = collect_figures(arguments.output, check_only=arguments.check_only)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
