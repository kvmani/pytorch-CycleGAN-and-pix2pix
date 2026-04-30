from __future__ import annotations

from pathlib import Path

from PIL import Image

from microi2i.app.cli import main
from microi2i.core.contracts import DatasetQAConfig
from microi2i.dataops.dataset_qa import run_dataset_qa


def _write_image(path: Path, color: tuple[int, int, int] = (100, 90, 80), size: tuple[int, int] = (8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_dataset_qa_detects_duplicates_and_writes_contact_sheet(tmp_path) -> None:
    source = tmp_path / "source"
    _write_image(source / "specimen_a" / "one.png", color=(10, 20, 30))
    _write_image(source / "specimen_b" / "duplicate.png", color=(10, 20, 30))
    (source / "specimen_c").mkdir(parents=True)
    (source / "specimen_c" / "bad.png").write_text("not an image", encoding="utf-8")
    output = tmp_path / "qa"
    cfg = DatasetQAConfig.from_mapping(
        {
            "schema_version": "microi2i.dataset_qa_config.v1",
            "dataset_id": "qa_test",
            "task_type": "paired_translation",
            "source_roots": [str(source)],
            "output_dir": str(output),
            "leakage_group_policy": {"mode": "parent"},
            "contact_sheet": {"max_images": 10, "thumb_size": 32},
        }
    )

    report = run_dataset_qa(cfg, repo_root=tmp_path)

    assert report["status"] == "failed"
    assert report["summary"]["total_images"] == 3
    assert report["summary"]["unreadable_images"] == 1
    assert report["summary"]["duplicate_groups"] == 1
    assert (output / "dataset_qa_report.json").exists()
    assert (output / "dataset_qa_report.html").exists()
    assert (output / "contact_sheet.jpg").exists()


def test_cli_data_qa_writes_run_artifacts(tmp_path) -> None:
    source = tmp_path / "source"
    _write_image(source / "specimen_a" / "one.png")
    qa_output = tmp_path / "qa_output"
    runs = tmp_path / "runs"

    exit_code = main(
        [
            "data-qa",
            "--config",
            "configs/dataset_qa.default.yml",
            "--set",
            f"source_roots=[{source.as_posix()}]",
            "--set",
            f"output_dir={qa_output.as_posix()}",
            "--set",
            f"output_root={runs.as_posix()}",
        ]
    )

    run_dir = next(runs.iterdir())
    assert exit_code == 0
    assert (run_dir / "dataset_qa_report.json").exists()
    assert (run_dir / "dataset_qa_report.html").exists()
    assert (run_dir / "contact_sheet.jpg").exists()
