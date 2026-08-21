"""Prepare or validate the non-admitted static Chapter 4 staging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from non_destructive_image.chapter4_static_retained_staging import (  # noqa: E402
    prepare_staging,
    validate_sources,
    validate_staging,
)


DEFAULT_CONFIG = ROOT / "configs/chapter_4_three_state_four_method_static_retained_v3.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-check", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--validate-only", type=Path)
    args = parser.parse_args()
    selected = sum((args.source_check, args.prepare, args.validate_only is not None))
    if selected != 1:
        parser.error("choose exactly one action")
    if args.source_check:
        config = validate_sources(ROOT, args.config.resolve())
        result = {
            "status": "PASS",
            "family": config["family"],
            "writes": 0,
            "staging_exists": (ROOT / config["staging"]).exists(),
            "target_exists": (ROOT / config["target"]).exists(),
        }
    elif args.validate_only is not None:
        result = validate_staging(args.validate_only.resolve())
    else:
        output = prepare_staging(ROOT, args.config.resolve())
        result = validate_staging(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
