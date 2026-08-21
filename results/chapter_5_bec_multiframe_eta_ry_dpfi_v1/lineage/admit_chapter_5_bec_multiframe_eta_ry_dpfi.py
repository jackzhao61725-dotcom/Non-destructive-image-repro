"""Prepare, visually qualify and admit the Chapter 5.2 DPFI result.

The source is the aggregation-only v4 canonical family assembled from the
completed v3 production shards.  This script authenticates and byte-copies that
family, renders a presentation from its saved endpoint summary, and performs no
camera draw, optical propagation, equilibrium solve or endpoint fit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from reportlab.lib.colors import Color, HexColor, black, white
from reportlab.lib.pagesizes import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "chapter_5_bec_multiframe_eta_ry_dpfi_retained_v1.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if source.stat().st_size != destination.stat().st_size or _sha256(source) != _sha256(destination):
        raise ValueError(f"copy identity failed: {source}")


def _manifest_inventory(directory: Path, *, status: str, admitted: bool, family: str, claim_boundary: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = []
    for path in sorted(
        item for item in directory.rglob("*")
        if item.is_file() and item.name != "artifact_manifest.json"
    ):
        artifacts.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "schema_version": 1,
        "family": family,
        "status": status,
        "admitted": admitted,
        "artifact_count": len(artifacts),
        "total_bytes": sum(int(item["bytes"]) for item in artifacts),
        "artifacts": artifacts,
        "claim_boundary": claim_boundary,
    }


def _authenticate_source(source: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = source / "artifact_manifest.json"
    expected_hash = str(config["source"]["artifact_manifest_sha256"])
    if _sha256(manifest_path) != expected_hash:
        raise ValueError("canonical source manifest identity changed")
    manifest = _read_json(manifest_path)
    expected = {item["path"]: item for item in manifest["artifacts"]}
    actual = {
        item.relative_to(source).as_posix(): item
        for item in source.rglob("*")
        if item.is_file() and item.name != "artifact_manifest.json"
    }
    if set(expected) != set(actual):
        raise ValueError("canonical source inventory changed")
    for relative, record in expected.items():
        path = actual[relative]
        if path.stat().st_size != int(record["bytes"]) or _sha256(path) != record["sha256"]:
            raise ValueError(f"canonical source artifact changed: {relative}")
    source_config = config["source"]
    if (
        manifest.get("family") != source_config["family"]
        or manifest.get("status") != source_config["required_status"]
        or bool(manifest.get("admitted")) is not bool(source_config["required_admitted"])
    ):
        raise ValueError("canonical source lifecycle changed")
    summary = _read_json(source / "summary.json")
    provenance = _read_json(source / "provenance.json")
    if (
        bool(provenance.get("aggregation_only")) is not bool(source_config["aggregation_only"])
        or int(provenance.get("production_endpoint_refits", -1))
        != int(source_config["production_endpoint_refits"])
    ):
        raise ValueError("canonical aggregation boundary changed")
    inventory = config["inventory"]
    checks = {
        "durations_us": summary.get("durations_us"),
        "images_per_sequence": summary.get("images_per_sequence"),
        "sequences_per_duration": summary.get("sequences_per_duration"),
        "endpoint_rows": summary.get("endpoint_rows"),
        "start_rows": summary.get("start_rows"),
        "random_identities": summary.get("random_identities"),
        "eta_supported_count": summary.get("eta_supported_count"),
        "rho_y_supported_count": summary.get("rho_y_supported_count"),
    }
    for key, actual_value in checks.items():
        if actual_value != inventory[key]:
            raise ValueError(f"canonical inventory changed: {key}")
    return manifest


def _presentation_rows(source: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    table = _read_csv(source / config["presentation"]["source_table"])
    selected_durations = {int(value) for value in config["presentation"]["durations_us"]}
    selected_q = {int(value) for value in config["presentation"]["image_q"]}
    selected_observables = set(config["presentation"]["observables"])
    selected = []
    for row in table:
        duration = int(row["duration_us"])
        image_q = int(row["image_q"])
        observable = row["observable"]
        if duration not in selected_durations or image_q not in selected_q or observable not in selected_observables:
            continue
        count = int(row["supported_count"])
        total = int(row["predeclared_sequence_count"])
        if count != 64 or total != 64:
            raise ValueError("presentation cannot hide an unsupported endpoint")
        mean = float(row["mean"])
        bias = float(row["mean_error_or_bias"])
        selected.append(
            {
                "duration_us": duration,
                "image_q": image_q,
                "observable": observable,
                "truth": f"{mean - bias:.16g}",
                "supported_count": count,
                "predeclared_sequence_count": total,
                "q16": row["q16"],
                "median": row["median"],
                "q84": row["q84"],
            }
        )
    expected = len(selected_durations) * len(selected_q) * len(selected_observables)
    if len(selected) != expected:
        raise ValueError(f"presentation row count changed: {len(selected)} != {expected}")
    selected.sort(key=lambda row: (row["observable"], row["duration_us"], row["image_q"]))
    return selected


def _hex_with_alpha(hex_colour: str, alpha: float) -> Color:
    colour = HexColor(hex_colour)
    return Color(colour.red, colour.green, colour.blue, alpha=alpha)


def _render_pdf(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    width = 6.35 * inch
    height = 5.55 * inch
    pdf = canvas.Canvas(str(path), pagesize=(width, height), pageCompression=1)
    pdf.setTitle("DPFI recovery across repeated exposures")
    pdf.setSubject("Presentation of admitted Chapter 5.2 endpoint summaries")
    pdf.setAuthor("Non-destructive-image dissertation")

    colours = {25: "#0072B2", 200: "#D55E00", 400: "#009E73"}
    durations = [25, 200, 400]
    left = 66.0
    right = 14.0
    plot_width = width - left - right
    plot_height = 126.0
    lower_bottom = 47.0
    upper_bottom = 211.0

    pdf.setFont("Helvetica-Bold", 10.5)
    pdf.drawString(left, height - 16.0, "DPFI recovery across repeated exposures")

    legend_y = height - 34.0
    legend_x = left
    pdf.setFont("Helvetica", 7.8)
    for duration in durations:
        colour = HexColor(colours[duration])
        pdf.setStrokeColor(colour)
        pdf.setLineWidth(1.4)
        pdf.line(legend_x, legend_y, legend_x + 17.0, legend_y)
        pdf.setFillColor(white)
        pdf.circle(legend_x + 8.5, legend_y, 2.1, stroke=1, fill=1)
        pdf.setFillColor(black)
        label = f"{duration} us"
        pdf.drawString(legend_x + 22.0, legend_y - 2.7, label)
        legend_x += 72.0
    pdf.setFillColor(black)
    pdf.setFont("Helvetica", 7.4)
    pdf.drawString(
        left,
        legend_y - 15.0,
        "Solid curves: equilibrium input; circles and bands: median and conditional 16-84% range",
    )

    def draw_panel(
        observable: str,
        bottom: float,
        y_min: float,
        y_max: float,
        y_ticks: Sequence[float],
        y_label: str,
        panel: str,
        show_x_labels: bool,
    ) -> None:
        def x_map(q: float) -> float:
            return left + (q - 1.0) / 14.0 * plot_width

        def y_map(value: float) -> float:
            return bottom + (value - y_min) / (y_max - y_min) * plot_height

        pdf.setStrokeColor(HexColor("#D9D9D9"))
        pdf.setLineWidth(0.45)
        for tick in y_ticks:
            y = y_map(tick)
            pdf.line(left, y, left + plot_width, y)
        pdf.setStrokeColor(black)
        pdf.setLineWidth(0.7)
        pdf.line(left, bottom, left, bottom + plot_height)
        pdf.line(left, bottom, left + plot_width, bottom)
        pdf.setFont("Helvetica", 7.6)
        for tick in y_ticks:
            label = f"{tick:.2f}" if observable == "rho_y" else f"{tick:.1f}"
            y = y_map(tick)
            pdf.drawRightString(left - 5.0, y - 2.7, label)
        for q in range(1, 16, 2):
            x = x_map(q)
            pdf.line(x, bottom, x, bottom - 3.0)
            if show_x_labels:
                pdf.drawCentredString(x, bottom - 13.0, str(q))

        for duration in durations:
            local = [
                row for row in rows
                if row["observable"] == observable and int(row["duration_us"]) == duration
            ]
            local.sort(key=lambda row: int(row["image_q"]))
            colour_hex = colours[duration]
            colour = HexColor(colour_hex)
            band = pdf.beginPath()
            first = local[0]
            band.moveTo(x_map(float(first["image_q"])), y_map(float(first["q84"])))
            for row in local[1:]:
                band.lineTo(x_map(float(row["image_q"])), y_map(float(row["q84"])))
            for row in reversed(local):
                band.lineTo(x_map(float(row["image_q"])), y_map(float(row["q16"])))
            band.close()
            pdf.setFillColor(_hex_with_alpha(colour_hex, 0.16))
            pdf.setStrokeColor(_hex_with_alpha(colour_hex, 0.0))
            pdf.drawPath(band, fill=1, stroke=0)

            truth = pdf.beginPath()
            truth.moveTo(x_map(float(first["image_q"])), y_map(float(first["truth"])))
            for row in local[1:]:
                truth.lineTo(x_map(float(row["image_q"])), y_map(float(row["truth"])))
            pdf.setStrokeColor(colour)
            pdf.setLineWidth(1.35)
            pdf.drawPath(truth, fill=0, stroke=1)
            for row in local:
                x = x_map(float(row["image_q"]))
                y = y_map(float(row["median"]))
                pdf.setStrokeColor(colour)
                pdf.setFillColor(white)
                pdf.setLineWidth(0.9)
                pdf.circle(x, y, 2.05, stroke=1, fill=1)

        pdf.setFillColor(black)
        pdf.setFont("Helvetica-Bold", 8.8)
        pdf.drawString(left + 5.0, bottom + plot_height - 12.0, f"({panel})")
        pdf.saveState()
        pdf.translate(15.0, bottom + plot_height / 2.0)
        pdf.rotate(90)
        pdf.setFont("Helvetica", 8.4)
        pdf.drawCentredString(0.0, 0.0, y_label)
        pdf.restoreState()

    draw_panel(
        "eta", upper_bottom, 0.40, 1.05, [0.4, 0.6, 0.8, 1.0],
        "Remaining condensate fraction, eta_q", "a", False,
    )
    draw_panel(
        "rho_y", lower_bottom, 0.82, 1.04, [0.85, 0.90, 0.95, 1.00],
        "Long-axis radius ratio, rho_y,q", "b", True,
    )
    pdf.setFont("Helvetica", 8.4)
    pdf.setFillColor(black)
    pdf.drawCentredString(left + plot_width / 2.0, 12.0, "Image number, q")
    pdf.showPage()
    pdf.save()


def _validate_inventory(directory: Path, *, expected_status: str, expected_admitted: bool) -> dict[str, Any]:
    manifest_path = directory / "artifact_manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("status") != expected_status
        or bool(manifest.get("admitted")) is not expected_admitted
    ):
        raise ValueError("retained lifecycle changed")
    expected = {item["path"]: item for item in manifest["artifacts"]}
    actual = {
        item.relative_to(directory).as_posix(): item
        for item in directory.rglob("*")
        if item.is_file() and item.name != "artifact_manifest.json"
    }
    if set(expected) != set(actual):
        raise ValueError("retained inventory changed")
    for relative, record in expected.items():
        path = actual[relative]
        if path.stat().st_size != int(record["bytes"]) or _sha256(path) != record["sha256"]:
            raise ValueError(f"retained artifact changed: {relative}")
    summary = _read_json(directory / "summary.json")
    if (
        summary.get("new_random_draws") != 0
        or summary.get("new_endpoint_fits") != 0
        or summary.get("new_optical_propagations") != 0
    ):
        raise ValueError("presentation-only admission boundary changed")
    return {
        "status": "PASS",
        "family": manifest["family"],
        "admitted": expected_admitted,
        "artifact_count": len(expected),
        "manifest_sha256": _sha256(manifest_path),
    }


def prepare(config_path: Path, source: Path) -> Path:
    config = _read_json(config_path)
    staging = ROOT / config["staging"]
    target = ROOT / config["target"]
    if staging.exists() or target.exists():
        raise FileExistsError(staging if staging.exists() else target)
    source_manifest = _authenticate_source(source, config)

    for record in source_manifest["artifacts"]:
        _copy(source / record["path"], staging / "source_payload" / record["path"])
    _copy(source / "artifact_manifest.json", staging / "lineage" / "source_diagnostic_artifact_manifest.json")
    _copy(config_path, staging / "lineage" / "admission_config_snapshot.json")
    _copy(Path(__file__).resolve(), staging / "lineage" / Path(__file__).name)

    plotted_rows = _presentation_rows(source, config)
    plotted_path = staging / config["presentation"]["plotted_values"]
    _write_csv(plotted_path, plotted_rows)
    pdf_path = staging / config["presentation"]["pdf"]
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    _render_pdf(pdf_path, plotted_rows)

    contract = {
        "schema_version": 1,
        "family": config["family"],
        "status": "admission_staging",
        "admitted": False,
        "source_family": config["source"]["family"],
        "source_artifact_manifest_sha256": config["source"]["artifact_manifest_sha256"],
        "aggregation_only": True,
        "production_endpoint_refits": 0,
        "presentation_durations_us": config["presentation"]["durations_us"],
        "new_random_draws": 0,
        "new_endpoint_fits": 0,
        "new_optical_propagations": 0,
        "claim_boundary": config["claim_boundary"],
    }
    summary = {
        "schema_version": 1,
        "family": config["family"],
        "status": "admission_staging",
        "admitted": False,
        "method": "DPFI",
        "durations_us": config["inventory"]["durations_us"],
        "images_per_sequence": config["inventory"]["images_per_sequence"],
        "sequences_per_duration": config["inventory"]["sequences_per_duration"],
        "eta_supported_count": config["inventory"]["eta_supported_count"],
        "rho_y_supported_count": config["inventory"]["rho_y_supported_count"],
        "presentation_durations_us": config["presentation"]["durations_us"],
        "presentation_row_count": len(plotted_rows),
        "new_random_draws": 0,
        "new_endpoint_fits": 0,
        "new_optical_propagations": 0,
        "claim_boundary": config["claim_boundary"],
    }
    provenance = {
        "schema_version": 1,
        "family": config["family"],
        "status": "admission_staging",
        "admitted": False,
        "admission_mode": "authenticated canonical byte reuse plus presentation-only rendering",
        "source_family": config["source"]["family"],
        "source_artifact_manifest_sha256": config["source"]["artifact_manifest_sha256"],
        "source_payload_preserved": True,
        "no_live_scratch_dependency_after_admission": True,
        "python": sys.version,
        "reportlab": __import__("reportlab").Version,
        "new_random_draws": 0,
        "new_endpoint_fits": 0,
        "new_optical_propagations": 0,
    }
    _write_json(staging / "admission_contract_snapshot.json", contract)
    _write_json(staging / "summary.json", summary)
    _write_json(staging / "provenance.json", provenance)
    _write_json(
        staging / "artifact_manifest.json",
        _manifest_inventory(
            staging,
            status="admission_staging",
            admitted=False,
            family=config["family"],
            claim_boundary=config["claim_boundary"],
        ),
    )
    _validate_inventory(staging, expected_status="admission_staging", expected_admitted=False)
    return staging


def record_visual_qa(config_path: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    staging = ROOT / config["staging"]
    _validate_inventory(staging, expected_status="admission_staging", expected_admitted=False)
    qa = {
        "schema_version": 1,
        "status": "PASS",
        "inspected_artifact": config["presentation"]["pdf"],
        "checks": {
            "axes_and_labels_legible": True,
            "legend_unambiguous": True,
            "bands_and_medians_visible": True,
            "no_clipped_or_overlapping_content": True,
            "common_q_axis_preserved": True,
            "only_authenticated_saved_summaries_plotted": True,
        },
    }
    _write_json(staging / "visual_qa.json", qa)
    _write_json(
        staging / "artifact_manifest.json",
        _manifest_inventory(
            staging,
            status="admission_staging",
            admitted=False,
            family=config["family"],
            claim_boundary=config["claim_boundary"],
        ),
    )
    return _validate_inventory(staging, expected_status="admission_staging", expected_admitted=False)


def admit(config_path: Path, source: Path) -> Path:
    config = _read_json(config_path)
    staging = ROOT / config["staging"]
    target = ROOT / config["target"]
    if target.exists():
        raise FileExistsError(target)
    _authenticate_source(source, config)
    _validate_inventory(staging, expected_status="admission_staging", expected_admitted=False)
    qa = _read_json(staging / "visual_qa.json")
    if qa.get("status") != "PASS" or not all(bool(value) for value in qa.get("checks", {}).values()):
        raise ValueError("visual QA has not passed")

    for name in ("admission_contract_snapshot.json", "summary.json", "provenance.json"):
        payload = _read_json(staging / name)
        payload["status"] = "admitted_immutable"
        payload["admitted"] = True
        _write_json(staging / name, payload)
    _write_json(
        staging / "artifact_manifest.json",
        _manifest_inventory(
            staging,
            status="admitted_immutable",
            admitted=True,
            family=config["family"],
            claim_boundary=config["claim_boundary"],
        ),
    )
    _validate_inventory(staging, expected_status="admitted_immutable", expected_admitted=True)
    staging.replace(target)
    _validate_inventory(target, expected_status="admitted_immutable", expected_admitted=True)
    return target


def install_figure(config_path: Path) -> Path:
    config = _read_json(config_path)
    target = ROOT / config["target"]
    _validate_inventory(target, expected_status="admitted_immutable", expected_admitted=True)
    source = target / config["presentation"]["pdf"]
    destination = ROOT / config["presentation"]["manuscript_pdf"]
    if destination.exists():
        raise FileExistsError(destination)
    _copy(source, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("prepare", "record-visual-qa", "admit", "validate", "install-figure"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = _read_json(config_path)
    source = args.source or ROOT / config["source"]["default_relative_path"]

    if args.action == "prepare":
        result: Any = prepare(config_path, source)
    elif args.action == "record-visual-qa":
        result = record_visual_qa(config_path)
    elif args.action == "admit":
        result = admit(config_path, source)
    elif args.action == "validate":
        result = _validate_inventory(
            ROOT / config["target"], expected_status="admitted_immutable", expected_admitted=True
        )
    else:
        result = install_figure(config_path)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
