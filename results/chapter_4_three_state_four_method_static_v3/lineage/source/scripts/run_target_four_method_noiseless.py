"""Run or validate the target-scale noiseless four-method comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from isolated_non_destructive_image import (  # noqa: E402
    load_isolated_non_destructive_image_module,
)


MODULE = load_isolated_non_destructive_image_module(
    "target_four_method_noiseless",
    namespace="_target_four_method_noiseless",
)
DEFAULT_CONFIG = ROOT / "configs" / "target_three_state_four_method_noiseless_v4.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-check", action="store_true")
    parser.add_argument("--validate-only", type=Path)
    arguments = parser.parse_args()
    config_path = arguments.config.resolve()
    config = MODULE.validate_config(ROOT, config_path)
    if arguments.source_check:
        print(json.dumps({"status": "source_check_pass", "label": config["label"]}))
        return 0
    if arguments.validate_only is not None:
        summary = MODULE.validate_output(arguments.validate_only.resolve(), config)
        print(json.dumps({"status": "validation_pass", "label": summary["label"]}))
        return 0
    output = MODULE.run_diagnostic(ROOT, config_path)
    print(json.dumps({"status": "generation_pass", "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
