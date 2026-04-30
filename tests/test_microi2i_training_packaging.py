from __future__ import annotations

import json

from microi2i.app.cli import main
from microi2i.core.contracts import ScriptWorkflowConfig
from microi2i.training.legacy_runner import (
    apply_smoke_training_overrides,
    build_training_command,
    build_training_preflight,
    package_training_outputs,
    parse_legacy_loss_log,
    parse_legacy_args,
)


def test_parse_legacy_args_extracts_training_metadata() -> None:
    parsed = parse_legacy_args(["--dataroot", "data", "--model", "pix2pix", "--continue_train"])

    assert parsed["dataroot"] == "data"
    assert parsed["model"] == "pix2pix"
    assert parsed["continue_train"] is True


def test_training_preflight_reports_missing_dataroot_as_dry_run_warning(tmp_path) -> None:
    cfg = {
        "schema_version": "microi2i.train_config.v1",
        "output_root": str(tmp_path),
        "training": {
            "legacy_script": "train.py",
            "legacy_args": ["--dataroot", "missing", "--name", "demo", "--model", "pix2pix"],
        },
    }
    config = ScriptWorkflowConfig.from_mapping(cfg, section="training")
    command = build_training_command(config, repo_root=tmp_path)

    report = build_training_preflight(
        config,
        repo_root=tmp_path,
        resolved_config=cfg,
        command=command,
        dry_run=True,
    )

    assert report["schema_version"] == "microi2i.training_preflight.v1"
    assert report["checks"]["dataroot_exists"] is False
    assert report["warnings"]


def test_cli_train_dry_run_writes_training_package(tmp_path) -> None:
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

    run_dir = next(tmp_path.iterdir())
    report = json.loads((run_dir / "training_report.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["schema_version"] == "microi2i.training_report.v1"
    assert (run_dir / "metrics_log.csv").exists()
    assert (run_dir / "metrics_log.jsonl").exists()
    assert (run_dir / "training_summary.html").exists()


def test_smoke_training_overrides_force_cpu_and_tiny_dataset() -> None:
    args = ["--dataroot", "data", "--gpu_ids", "0", "--n_epochs", "100"]

    effective = apply_smoke_training_overrides(
        args,
        {"enabled": True, "max_epochs": 1, "max_dataset_size": 2, "image_size": 32},
    )
    parsed = parse_legacy_args(effective)

    assert parsed["gpu_ids"] == "-1"
    assert parsed["n_epochs"] == "1"
    assert parsed["n_epochs_decay"] == "0"
    assert parsed["max_dataset_size"] == "2"
    assert parsed["load_size"] == "32"
    assert parsed["crop_size"] == "32"
    assert parsed["no_html"] is True


def test_parse_legacy_loss_log_and_package_training_outputs(tmp_path) -> None:
    experiment = tmp_path / "checkpoints" / "demo"
    images = experiment / "web" / "images"
    images.mkdir(parents=True)
    (experiment / "loss_log.txt").write_text(
        "================ Training Loss ================\n"
        "(epoch: 1, iters: 2, time: 0.001, data: 0.002) G_GAN: 1.250 D_real: 0.500\n",
        encoding="utf-8",
    )
    from PIL import Image

    Image.new("RGB", (8, 8), (10, 20, 30)).save(images / "epoch001_fake_B.png")
    rows = parse_legacy_loss_log(experiment / "loss_log.txt")

    outputs = package_training_outputs(
        tmp_path / "run",
        {
            "paths": {"checkpoints_dir": str(tmp_path / "checkpoints")},
            "experiment_name": "demo",
        },
    )

    assert rows[0]["loss_G_GAN"] == 1.25
    assert outputs["loss_rows"] == 1
    assert outputs["validation_sample_count"] == 1
    assert (tmp_path / "run" / "metrics_log.csv").exists()
    assert (tmp_path / "run" / "validation_samples.html").exists()
