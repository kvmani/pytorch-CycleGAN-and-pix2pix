"""Legacy inference command construction and output packaging."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from microi2i.core.contracts import ScriptWorkflowConfig


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def build_inference_command(config: ScriptWorkflowConfig, *, repo_root: Path) -> list[str]:
    """Build the command that runs the current legacy inference script."""

    script = Path(config.command.legacy_script)
    if not script.is_absolute():
        script = repo_root / script
    return [sys.executable, str(script), *config.command.legacy_args]


def materialize_inference_inputs(
    inputs: dict[str, Any],
    *,
    repo_root: Path,
    run_dir: Path,
) -> dict[str, Any]:
    """Normalize configured inference inputs into a manifest and optional staging folder."""

    mode = str(inputs.get("mode", "legacy")).lower()
    if mode == "legacy":
        report = {
            "schema_version": "microi2i.inference_inputs.v1",
            "mode": mode,
            "status": "skipped",
            "reason": "legacy dataroot is used directly",
            "samples": [],
        }
        (run_dir / "inference_inputs.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return report

    recursive = bool(inputs.get("recursive", False))
    copy_to_run = bool(inputs.get("copy_to_run", True))
    source_paths = _select_input_paths(inputs, repo_root=repo_root, recursive=recursive)
    staging_dir = run_dir / "inputs"
    samples: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    if copy_to_run:
        staging_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(source_paths):
        try:
            with Image.open(source) as img:
                width, height = img.size
                mode_name = img.mode
                channels = len(img.getbands())
            target_relative = ""
            if copy_to_run:
                target = staging_dir / f"{index:05d}_{source.name}"
                shutil.copy2(source, target)
                target_relative = str(target.relative_to(run_dir))
            samples.append(
                {
                    "index": index,
                    "source_path": str(source),
                    "run_relative_path": target_relative,
                    "filename": source.name,
                    "width": width,
                    "height": height,
                    "mode": mode_name,
                    "channels": channels,
                    "size_bytes": source.stat().st_size,
                }
            )
        except Exception as exc:
            failures.append({"source_path": str(source), "error": str(exc)})

    report = {
        "schema_version": "microi2i.inference_inputs.v1",
        "mode": mode,
        "status": "ready" if samples and not failures else ("failed" if failures and not samples else "partial"),
        "copy_to_run": copy_to_run,
        "recursive": recursive,
        "sample_count": len(samples),
        "failure_count": len(failures),
        "staging_dir": str(staging_dir) if copy_to_run else "",
        "samples": samples,
        "failures": failures,
    }
    (run_dir / "inference_inputs.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_inputs_csv(samples, run_dir / "inference_inputs.csv")
    return report


def _repo_path(value: str | Path, *, repo_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def _select_input_paths(inputs: dict[str, Any], *, repo_root: Path, recursive: bool) -> list[Path]:
    mode = str(inputs.get("mode", "")).lower()
    if mode == "single":
        path = _repo_path(str(inputs.get("path", "")), repo_root=repo_root)
        return [path] if path.exists() else []
    if mode == "folder":
        root = _repo_path(str(inputs.get("path", "")), repo_root=repo_root)
        iterator = root.rglob("*") if recursive else root.glob("*")
        return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if mode == "manifest":
        manifest = _repo_path(str(inputs.get("manifest_path", "")), repo_root=repo_root)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        items = payload.get("samples", payload if isinstance(payload, list) else [])
        paths: list[Path] = []
        if isinstance(items, list):
            for item in items:
                if isinstance(item, str):
                    paths.append(_repo_path(item, repo_root=repo_root))
                elif isinstance(item, dict):
                    raw = item.get("path", item.get("source_path", ""))
                    if raw:
                        paths.append(_repo_path(str(raw), repo_root=repo_root))
        return [path for path in paths if path.exists() and path.suffix.lower() in IMAGE_EXTENSIONS]
    raise ValueError("inference.inputs.mode must be one of: legacy, single, folder, manifest")


def _write_inputs_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["index", "filename", "source_path", "run_relative_path", "width", "height", "mode", "channels", "size_bytes"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def package_prediction_images(
    source_dir: str | Path,
    run_dir: Path,
    *,
    postprocess: dict[str, Any] | None = None,
    references_dir: str | Path | None = None,
) -> dict[str, object]:
    """Copy and summarize prediction images into a normalized run folder."""

    source = Path(source_dir)
    references = Path(references_dir) if references_dir else None
    predictions_dir = run_dir / "predictions"
    postprocess = postprocess or {}
    rows: list[dict[str, Any]] = []
    if not source.exists() or not source.is_dir():
        return {
            "status": "skipped",
            "reason": "expected_output_dir does not exist",
            "source_dir": str(source),
            "prediction_count": 0,
            "predictions_dir": str(predictions_dir),
        }

    predictions_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            relative = path.relative_to(source)
            target = predictions_dir / _target_name(relative, len(rows), postprocess)
            target.parent.mkdir(parents=True, exist_ok=True)
            row = _copy_or_postprocess_image(path, target, run_dir, postprocess)
            reference = references / relative if references is not None else None
            if reference is not None and reference.exists():
                row["reference_path"] = str(reference)
                row["has_reference"] = True
            else:
                row["reference_path"] = ""
                row["has_reference"] = False
            rows.append(row)
    batch_summary = {
        "schema_version": "microi2i.inference_batch_summary.v1",
        "source_dir": str(source),
        "predictions_dir": str(predictions_dir),
        "prediction_count": len(rows),
        "postprocess": postprocess,
        "references_dir": str(references) if references is not None else "",
        "predictions": rows,
    }
    (run_dir / "batch_summary.json").write_text(json.dumps(batch_summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_batch_csv(rows, run_dir / "batch_summary.csv")
    review = _write_review_html(rows, run_dir / "review.html")
    comparison = _write_comparison_html(rows, run_dir / "comparison_review.html")
    return {
        "status": "copied",
        "source_dir": str(source),
        "prediction_count": len(rows),
        "reference_count": len([row for row in rows if row.get("has_reference")]),
        "predictions_dir": str(predictions_dir),
        "files": [row["run_relative_path"] for row in rows],
        "batch_summary_json": "batch_summary.json",
        "batch_summary_csv": "batch_summary.csv",
        "review_html": str(review.name),
        "comparison_review_html": str(comparison.name),
    }


def _target_name(relative: Path, index: int, postprocess: dict[str, Any]) -> Path:
    prefix = str(postprocess.get("rename_prefix", "")).strip()
    if not prefix:
        return relative
    return relative.with_name(f"{prefix}_{index:05d}{relative.suffix.lower()}")


def _copy_or_postprocess_image(source: Path, target: Path, run_dir: Path, postprocess: dict[str, Any]) -> dict[str, Any]:
    grayscale = bool(postprocess.get("grayscale", False))
    resize = postprocess.get("resize", None)
    threshold = postprocess.get("threshold", None)
    auto_contrast = bool(postprocess.get("auto_contrast", False))

    if grayscale or resize or threshold is not None or auto_contrast:
        with Image.open(source) as img:
            out = img.convert("L") if grayscale or threshold is not None else img.convert("RGB")
            if auto_contrast:
                from PIL import ImageOps

                out = ImageOps.autocontrast(out)
            if resize:
                if not isinstance(resize, list | tuple) or len(resize) != 2:
                    raise ValueError("inference.postprocess.resize must be [width, height]")
                out = out.resize((int(resize[0]), int(resize[1])))
            if threshold is not None:
                cutoff = int(threshold)
                out = out.point(lambda value: 255 if value >= cutoff else 0)
            out.save(target)
    else:
        shutil.copy2(source, target)

    with Image.open(target) as img:
        width, height = img.size
        mode = img.mode
        channels = len(img.getbands())
    return {
        "source_path": str(source),
        "run_relative_path": str(target.relative_to(run_dir)),
        "filename": target.name,
        "width": width,
        "height": height,
        "mode": mode,
        "channels": channels,
        "size_bytes": target.stat().st_size,
    }


def _write_batch_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["filename", "run_relative_path", "source_path", "width", "height", "mode", "channels", "size_bytes"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_review_html(rows: list[dict[str, Any]], path: Path) -> Path:
    cards = []
    for row in rows[:200]:
        image_path = row["run_relative_path"].replace("\\", "/")
        cards.append(
            "<figure>"
            f"<img src='{image_path}' alt='{row['filename']}'>"
            f"<figcaption>{row['filename']}<br>{row['width']}x{row['height']} {row['mode']}</figcaption>"
            "</figure>"
        )
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>MicroI2I Inference Review</title>"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:2rem;background:#f7fbfd;color:#102a43}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:1rem}"
        "figure{margin:0;background:white;border:1px solid #d9e5ec;border-radius:10px;padding:.75rem;box-shadow:0 6px 18px rgba(16,42,67,.10)}"
        "img{width:100%;height:150px;object-fit:contain;background:#eef4f8;border-radius:8px}"
        "figcaption{font-size:.85rem;margin-top:.5rem}</style></head><body>"
        "<h1>MicroI2I Inference Review</h1>"
        f"<p>Images shown: {min(len(rows), 200)} of {len(rows)}</p>"
        f"<div class='grid'>{''.join(cards)}</div></body></html>",
        encoding="utf-8",
    )
    return path


def _write_comparison_html(rows: list[dict[str, Any]], path: Path) -> Path:
    comparison_rows = [row for row in rows if row.get("has_reference")]
    cards = []
    for row in comparison_rows[:100]:
        prediction_path = row["run_relative_path"].replace("\\", "/")
        reference_path = Path(str(row["reference_path"])).as_posix()
        cards.append(
            "<section class='pair'>"
            f"<h2>{row['filename']}</h2>"
            "<div class='images'>"
            f"<figure><img src='{prediction_path}' alt='prediction'><figcaption>Prediction</figcaption></figure>"
            f"<figure><img src='file:///{reference_path}' alt='reference'><figcaption>Reference</figcaption></figure>"
            "</div></section>"
        )
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>MicroI2I Comparison Review</title>"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:2rem;background:#f8fafc;color:#0f172a}"
        ".pair{background:white;border:1px solid #cbd5e1;border-radius:12px;padding:1rem;margin-bottom:1rem}"
        ".images{display:grid;grid-template-columns:1fr 1fr;gap:1rem}"
        "img{width:100%;height:240px;object-fit:contain;background:#f1f5f9;border-radius:8px}"
        "figcaption{text-align:center;font-weight:600}</style></head><body>"
        "<h1>MicroI2I Prediction/Reference Review</h1>"
        f"<p>Pairs shown: {min(len(comparison_rows), 100)} of {len(comparison_rows)}</p>"
        f"{''.join(cards)}</body></html>",
        encoding="utf-8",
    )
    return path
