"""Image translation metric helpers."""

from __future__ import annotations

from math import log10, sqrt
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def _psnr(rmse: float) -> float:
    if rmse <= 0:
        return float("inf")
    return 20.0 * log10(255.0 / rmse)


def evaluate_paired_directories(predictions_dir: str | Path, targets_dir: str | Path) -> dict[str, Any]:
    """Evaluate same-named prediction and target images with basic fidelity metrics."""

    pred_root = Path(predictions_dir)
    target_root = Path(targets_dir)
    if not pred_root.exists() or not target_root.exists():
        return {
            "status": "skipped",
            "reason": "predictions_dir or targets_dir does not exist",
            "predictions_dir": str(pred_root),
            "targets_dir": str(target_root),
            "samples": [],
            "aggregate": {},
        }

    target_by_name = {
        path.name: path
        for path in target_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    rows: list[dict[str, Any]] = []
    for pred_path in sorted(pred_root.iterdir()):
        if not pred_path.is_file() or pred_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        target_path = target_by_name.get(pred_path.name)
        if target_path is None:
            continue
        pred = _load_image(pred_path)
        target = _load_image(target_path)
        if pred.shape != target.shape:
            target = np.asarray(Image.open(target_path).convert("RGB").resize((pred.shape[1], pred.shape[0])), dtype=np.float32)
        diff = pred - target
        mae = float(np.mean(np.abs(diff)))
        rmse = float(sqrt(float(np.mean(diff * diff))))
        rows.append(
            {
                "sample": pred_path.name,
                "mae": mae,
                "rmse": rmse,
                "psnr": _psnr(rmse),
            }
        )

    if not rows:
        return {
            "status": "skipped",
            "reason": "no same-named image pairs found",
            "predictions_dir": str(pred_root),
            "targets_dir": str(target_root),
            "samples": [],
            "aggregate": {},
        }

    return {
        "status": "computed",
        "predictions_dir": str(pred_root),
        "targets_dir": str(target_root),
        "samples": rows,
        "aggregate": {
            "sample_count": len(rows),
            "mae_mean": float(np.mean([row["mae"] for row in rows])),
            "rmse_mean": float(np.mean([row["rmse"] for row in rows])),
            "psnr_mean": float(np.mean([row["psnr"] for row in rows if np.isfinite(row["psnr"])])),
        },
    }
