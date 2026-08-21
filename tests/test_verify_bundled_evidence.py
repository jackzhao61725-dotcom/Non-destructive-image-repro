from scripts.verify_bundled_evidence import verify_evidence


def test_bundled_evidence_matches_all_manifests() -> None:
    assert verify_evidence() == {
        "status": "pass",
        "evidence_tree_count": 6,
        "artifact_count": 183,
        "artifact_bytes": 67080996,
    }
