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


def package_prediction_images(
    source_dir: str | Path,
    run_dir: Path,
    *,
    postprocess: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Copy and summarize prediction images into a normalized run folder."""

    source = Path(source_dir)
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
            rows.append(row)
    batch_summary = {
        "schema_version": "microi2i.inference_batch_summary.v1",
        "source_dir": str(source),
        "predictions_dir": str(predictions_dir),
        "prediction_count": len(rows),
        "postprocess": postprocess,
        "predictions": rows,
    }
    (run_dir / "batch_summary.json").write_text(json.dumps(batch_summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_batch_csv(rows, run_dir / "batch_summary.csv")
    review = _write_review_html(rows, run_dir / "review.html")
    return {
        "status": "copied",
        "source_dir": str(source),
        "prediction_count": len(rows),
        "predictions_dir": str(predictions_dir),
        "files": [row["run_relative_path"] for row in rows],
        "batch_summary_json": "batch_summary.json",
        "batch_summary_csv": "batch_summary.csv",
        "review_html": str(review.name),
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
