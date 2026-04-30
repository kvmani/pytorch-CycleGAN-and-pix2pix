from __future__ import annotations

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
