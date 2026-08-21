"""Build, admit and validate the Section 5.3 density-recovery result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from non_destructive_image.chapter5_three_state_density_recovery import (
    admit,
    build_candidate,
    validate,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/chapter_5_three_state_density_recovery_v1.json",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--candidate", action="store_true")
    action.add_argument("--admit", action="store_true")
    action.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.candidate:
        result = build_candidate(ROOT, config_path)
    elif args.admit:
        result = admit(ROOT, config_path)
    else:
        result = validate(ROOT / config["target"])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
