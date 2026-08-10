"""Release-surface checks for the independent public repository."""

from __future__ import annotations

import json
from pathlib import Path
import re
import tomllib

import non_destructive_image


ROOT = Path(__file__).resolve().parents[1]


def test_release_version_is_synchronised() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project_version = tomllib.load(handle)["project"]["version"]
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r"^version:\s*([^\s]+)\s*$", citation, flags=re.MULTILINE)

    assert match is not None
    assert project_version == "1.0.0"
    assert non_destructive_image.__version__ == project_version
    assert match.group(1) == project_version


def test_public_configs_do_not_reference_the_internal_repository() -> None:
    old_repository = "github.com/jackzhao61725-dotcom/Non-destructive-image\""
    for path in sorted((ROOT / "configs").glob("*.json")):
        text = path.read_text(encoding="utf-8")
        json.loads(text)
        assert old_repository not in text
        assert "runtime_transform" not in text
        assert '"measured_scale"' not in text
        assert '"source_fit"' not in text


def test_public_top_level_is_allowlisted() -> None:
    allowed_files = {
        ".gitattributes",
        ".gitignore",
        "CITATION.cff",
        "LICENSE",
        "README.md",
        "THIRD_PARTY.md",
        "pyproject.toml",
    }
    allowed_directories = {
        ".git",
        ".github",
        ".pytest_cache",
        "build",
        "configs",
        "docs",
        "outputs",
        "reference",
        "scripts",
        "src",
        "tests",
    }
    for path in ROOT.iterdir():
        allowed = allowed_directories if path.is_dir() else allowed_files
        assert path.name in allowed, path


def test_public_runtime_surface_uses_repository_independent_language() -> None:
    roots = (ROOT / "src", ROOT / "scripts", ROOT / "configs")
    forbidden = (
        "chapter_5",
        "Chapter 5",
        "C:\\Users\\",
        "dissertation convention",
        "agent instructions",
        "handoff record",
    )
    for root in roots:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path.suffix not in {".json", ".py"}:
                continue
            text = path.read_text(encoding="utf-8")
            assert all(term not in text for term in forbidden), path
