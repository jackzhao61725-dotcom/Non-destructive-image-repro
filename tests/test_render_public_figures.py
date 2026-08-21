import hashlib
import json
from pathlib import Path

from scripts.render_public_figures import ALLOWLIST, collect_figures


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_figure_collection_is_exact(tmp_path: Path) -> None:
    output = tmp_path / "figures"
    summary = collect_figures(output)

    assert summary == {"status": "pass", "figure_count": 17, "written": True}
    index = json.loads((output / "index.json").read_text(encoding="utf-8"))
    assert index["status"] == "verified_exact_manuscript_figures"
    assert index["figure_count"] == 17
    for record in index["figures"]:
        assert _sha256(output / record["name"]) == record["sha256"]

    assert collect_figures(tmp_path / "unused", check_only=True) == {
        "status": "pass",
        "figure_count": 17,
        "written": False,
    }
    assert ALLOWLIST.is_file()
