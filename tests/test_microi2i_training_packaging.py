from __future__ import annotations

import json

from microi2i.app.cli import main
from microi2i.core.contracts import ScriptWorkflowConfig
from microi2i.training.legacy_runner import build_training_command, build_training_preflight, parse_legacy_args


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
