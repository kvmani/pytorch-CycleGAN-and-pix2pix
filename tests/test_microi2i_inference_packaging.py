from __future__ import annotations

import json

from PIL import Image

from microi2i.inference.legacy_runner import package_prediction_images


def test_package_prediction_images_writes_batch_outputs(tmp_path) -> None:
    source = tmp_path / "legacy_results"
    source.mkdir()
    Image.new("RGB", (10, 12), (10, 20, 30)).save(source / "a_fake_B.png")
    Image.new("RGB", (8, 8), (30, 20, 10)).save(source / "nested_b_fake_B.jpg")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    result = package_prediction_images(source, run_dir)

    assert result["status"] == "copied"
    assert result["prediction_count"] == 2
    assert (run_dir / "predictions" / "a_fake_B.png").exists()
    assert (run_dir / "batch_summary.json").exists()
    assert (run_dir / "batch_summary.csv").exists()
    assert (run_dir / "review.html").exists()
    summary = json.loads((run_dir / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["prediction_count"] == 2


def test_package_prediction_images_applies_postprocessing(tmp_path) -> None:
    source = tmp_path / "legacy_results"
    source.mkdir()
    Image.new("RGB", (10, 12), (10, 20, 30)).save(source / "sample.png")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    result = package_prediction_images(
        source,
        run_dir,
        postprocess={"grayscale": True, "resize": [6, 5], "rename_prefix": "pred"},
    )

    output = run_dir / result["files"][0]
    with Image.open(output) as img:
        assert img.mode == "L"
        assert img.size == (6, 5)
    assert output.name == "pred_00000.png"
