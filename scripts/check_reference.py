"""Check public reproductions against software-verification reference values."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # Package import in tests; direct import when executed as a file.
    from scripts import reproduce_forward_model, reproduce_inference
    from scripts._common import (
        CONFIG_FILENAMES,
        REPOSITORY_ROOT,
        read_json,
        sha256_file,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by CLI acceptance
    import reproduce_forward_model
    import reproduce_inference
    from _common import CONFIG_FILENAMES, REPOSITORY_ROOT, read_json, sha256_file


REFERENCE_CLASSIFICATION = (
    "software_verification_reference_not_experimental_data"
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=REPOSITORY_ROOT / "configs",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=REPOSITORY_ROOT / "reference" / "expected_results.json",
    )
    parser.add_argument("--include-inference", action="store_true")
    return parser.parse_args(argv)


def _path_value(payload: Any, path: Sequence[str | int]) -> Any:
    value = payload
    for part in path:
        if isinstance(part, int):
            if not isinstance(value, list):
                raise TypeError(f"expected a list before index {part}")
            value = value[part]
        else:
            if not isinstance(value, Mapping):
                raise TypeError(f"expected an object before key {part!r}")
            value = value[part]
    return value


def check_section(
    payload: Mapping[str, Any],
    section: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Return mismatches and observed summary values for one reference section."""

    checks = section.get("checks")
    if not isinstance(checks, list):
        raise ValueError("reference section checks must be a list")
    failures: list[str] = []
    observed_by_key: dict[str, Any] = {}
    for check in checks:
        if not isinstance(check, Mapping):
            raise ValueError("each reference check must be an object")
        key = str(check["key"])
        path = check["path"]
        if not isinstance(path, list) or not all(
            isinstance(part, (str, int)) and not isinstance(part, bool)
            for part in path
        ):
            raise ValueError(f"{key}: path must contain only string keys or indexes")
        try:
            observed = _path_value(payload, path)
        except (IndexError, KeyError, TypeError) as error:
            failures.append(f"{key}: missing ({error})")
            continue
        observed_by_key[key] = observed
        expected = check["expected"]
        if "absolute_tolerance" in check or "relative_tolerance" in check:
            if (
                isinstance(observed, bool)
                or isinstance(expected, bool)
                or not isinstance(observed, (int, float))
                or not isinstance(expected, (int, float))
            ):
                failures.append(f"{key}: numeric tolerance applied to non-number")
                continue
            absolute = float(check.get("absolute_tolerance", 0.0))
            relative = float(check.get("relative_tolerance", 0.0))
            if not math.isfinite(float(observed)) or not math.isclose(
                float(observed),
                float(expected),
                abs_tol=absolute,
                rel_tol=relative,
            ):
                failures.append(
                    f"{key}: expected {expected!r}, observed {observed!r} "
                    f"(abs_tol={absolute:g}, rel_tol={relative:g})"
                )
        elif observed != expected:
            failures.append(f"{key}: expected {expected!r}, observed {observed!r}")
    return failures, observed_by_key


def _print_result(
    name: str,
    section: Mapping[str, Any],
    failures: Sequence[str],
    observed_by_key: Mapping[str, Any],
) -> None:
    if failures:
        print(f"FAIL {name} ({len(failures)} mismatches)")
        for failure in failures:
            print(f"  {failure}")
        return
    checks = section["checks"]
    summary_keys = section.get("summary_keys", [])
    summary = ", ".join(
        f"{key}={observed_by_key[str(key)]:.12g}"
        for key in summary_keys
    )
    suffix = "" if not summary else f": {summary}"
    print(f"PASS {name} ({len(checks)} keys){suffix}")


def _config_identity_failures(
    reference: Mapping[str, Any],
    config_dir: Path,
) -> list[str]:
    expected = reference.get("config_sha256")
    if not isinstance(expected, Mapping):
        return ["reference config_sha256 must be an object"]
    if set(expected) != set(CONFIG_FILENAMES):
        return ["reference config_sha256 keys do not match the public configs"]
    failures: list[str] = []
    for name in CONFIG_FILENAMES:
        expected_hash = expected[name]
        if not isinstance(expected_hash, str):
            failures.append(f"{name}: reference hash must be text")
            continue
        observed_hash = sha256_file(config_dir / name)
        if observed_hash != expected_hash:
            failures.append(
                f"{name}: expected config SHA-256 {expected_hash}, "
                f"observed {observed_hash}"
            )
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    reference = read_json(args.reference)
    if reference.get("classification") != REFERENCE_CLASSIFICATION:
        print("FAIL reference: missing software-verification classification")
        return 2
    identity_failures = _config_identity_failures(reference, args.config_dir)
    if identity_failures:
        print(f"FAIL reference identity ({len(identity_failures)} mismatches)")
        for failure in identity_failures:
            print(f"  {failure}")
        return 2
    sections = reference.get("sections")
    if not isinstance(sections, Mapping):
        print("FAIL reference: sections must be an object")
        return 2

    selected = ["forward"]
    if args.include_inference:
        selected.append("inference")
    payloads: dict[str, Mapping[str, Any]] = {
        "forward": reproduce_forward_model.reproduce(
            args.config_dir,
            invocation_arguments={
                "caller": "check_reference",
                "config_dir": args.config_dir,
                "include_inference": args.include_inference,
                "reference": args.reference,
            },
        )
    }
    if args.include_inference:
        payloads["inference"] = reproduce_inference.reproduce(
            args.config_dir,
            draws=1,
            invocation_arguments={
                "caller": "check_reference",
                "config_dir": args.config_dir,
                "draws": 1,
                "include_inference": True,
                "reference": args.reference,
            },
        )

    mismatch_count = 0
    for name in selected:
        section = sections.get(name)
        if not isinstance(section, Mapping):
            print(f"FAIL reference: section {name!r} is missing")
            mismatch_count += 1
            continue
        failures, observed = check_section(payloads[name], section)
        _print_result(name, section, failures, observed)
        mismatch_count += len(failures)
    return 1 if mismatch_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
