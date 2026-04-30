from __future__ import annotations

from microi2i.app.cli import main


def test_cli_validate_registry() -> None:
    assert main(["validate-registry"]) == 0


def test_cli_train_dry_run_with_override(tmp_path) -> None:
    exit_code = main(
        [
            "train",
            "--config",
            "configs/train/pix2pix.default.yml",
            "--dry-run",
            "--set",
            f"output_root={tmp_path.as_posix()}",
        ]
    )

    assert exit_code == 0
    assert any(tmp_path.iterdir())


def test_cli_infer_dry_run_writes_inference_report(tmp_path) -> None:
    exit_code = main(
        [
            "infer",
            "--config",
            "configs/inference/folder.default.yml",
            "--dry-run",
            "--set",
            f"output_root={tmp_path.as_posix()}",
        ]
    )

    run_dir = next(tmp_path.iterdir())
    assert exit_code == 0
    assert (run_dir / "report.json").exists()
    assert (run_dir / "run_report.json").exists()
    assert (run_dir / "artifact_manifest.json").exists()
