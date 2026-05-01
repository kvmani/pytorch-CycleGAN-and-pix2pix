from __future__ import annotations

import numpy as np
from PIL import Image

from microi2i.evaluation.image_translation import evaluate_paired_directories, write_evaluation_review_artifacts


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
    assert report["metric_families"]["fidelity"]["metric_count"] == 4
    assert "ebsd_kikuchi" in report["metric_families"]


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
    assert sample["ebsd_band_contrast_delta"] == 0.0
    assert sample["ebsd_band_sharpness_ratio"] == 1.0
    assert sample["orientation_coherence_delta"] == 0.0
    assert report["aggregate"]["psnr_mean"] == float("inf")


def test_ebsd_proxy_metrics_detect_blurred_band_structure(tmp_path) -> None:
    pred_dir = tmp_path / "pred"
    target_dir = tmp_path / "target"
    pred_dir.mkdir()
    target_dir.mkdir()
    target = np.zeros((16, 16), dtype=np.uint8)
    target[:, 7:9] = 255
    pred = np.zeros((16, 16), dtype=np.uint8)
    pred[:, 6:10] = 128
    Image.fromarray(np.stack([pred, pred, pred], axis=2)).save(pred_dir / "band.png")
    Image.fromarray(np.stack([target, target, target], axis=2)).save(target_dir / "band.png")

    report = evaluate_paired_directories(pred_dir, target_dir)
    sample = report["samples"][0]

    assert sample["ebsd_band_contrast_delta"] > 0.0
    assert sample["ebsd_band_sharpness_ratio"] < 1.0
    assert "ebsd_band_contrast_delta_mean" in report["aggregate"]


def test_evaluation_review_artifacts_include_best_and_worst_samples(tmp_path) -> None:
    pred_dir = tmp_path / "pred"
    target_dir = tmp_path / "target"
    pred_dir.mkdir()
    target_dir.mkdir()
    Image.new("RGB", (8, 8), (10, 10, 10)).save(pred_dir / "good.png")
    Image.new("RGB", (8, 8), (10, 10, 10)).save(target_dir / "good.png")
    Image.new("RGB", (8, 8), (0, 0, 0)).save(pred_dir / "bad.png")
    Image.new("RGB", (8, 8), (100, 100, 100)).save(target_dir / "bad.png")
    report = evaluate_paired_directories(pred_dir, target_dir)

    review = write_evaluation_review_artifacts(report, tmp_path / "run", metric="mae", limit=1)

    assert review["sample_count"] == 2
    assert (tmp_path / "run" / "evaluation_outliers.csv").exists()
    assert (tmp_path / "run" / "evaluation_review.html").exists()
    csv_text = (tmp_path / "run" / "evaluation_outliers.csv").read_text(encoding="utf-8")
    assert "good.png" in csv_text
    assert "bad.png" in csv_text
