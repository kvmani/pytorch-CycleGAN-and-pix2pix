from __future__ import annotations

import numpy as np
from PIL import Image

from microi2i.evaluation.image_translation import evaluate_paired_directories


def test_evaluate_paired_directories_computes_basic_metrics(tmp_path) -> None:
    pred_dir = tmp_path / "pred"
    target_dir = tmp_path / "target"
    pred_dir.mkdir()
    target_dir.mkdir()
    Image.new("RGB", (4, 4), (10, 10, 10)).save(pred_dir / "sample.png")
    Image.new("RGB", (4, 4), (12, 12, 12)).save(target_dir / "sample.png")

    report = evaluate_paired_directories(pred_dir, target_dir)

    assert report["status"] == "computed"
    assert report["aggregate"]["sample_count"] == 1
    assert report["samples"][0]["mae"] == 2.0
    assert "histogram_l1" in report["samples"][0]
    assert "edge_mae_mean" in report["aggregate"]


def test_evaluate_identical_gradient_image_has_perfect_fidelity_metrics(tmp_path) -> None:
    pred_dir = tmp_path / "pred"
    target_dir = tmp_path / "target"
    pred_dir.mkdir()
    target_dir.mkdir()
    arr = np.tile(np.arange(8, dtype=np.uint8) * 20, (8, 1))
    rgb = np.stack([arr, arr, arr], axis=2)
    Image.fromarray(rgb).save(pred_dir / "sample.png")
    Image.fromarray(rgb).save(target_dir / "sample.png")

    report = evaluate_paired_directories(pred_dir, target_dir)
    sample = report["samples"][0]

    assert sample["mae"] == 0.0
    assert sample["rmse"] == 0.0
    assert sample["histogram_l1"] == 0.0
    assert sample["gradient_correlation"] == 1.0
    assert report["aggregate"]["psnr_mean"] == float("inf")
