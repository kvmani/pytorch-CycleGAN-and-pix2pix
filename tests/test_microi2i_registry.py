from __future__ import annotations

import json

from microi2i.app.cli import main
from microi2i.plugins.registry import (
    compare_run_reports,
    load_model_registry,
    merge_local_registry_overlay,
    update_model_status,
    validate_model_registry,
    write_run_comparison_html,
)


def test_default_model_registry_is_valid() -> None:
    registry = load_model_registry("frozen_checkpoints/model_registry.json")

    assert validate_model_registry(registry) == []


def test_registry_validation_reports_missing_fields() -> None:
    registry = {
        "schema_version": "microi2i.model_registry.v1",
        "models": [{"model_id": "bad"}],
    }

    errors = validate_model_registry(registry)

    assert errors
    assert "missing fields" in errors[0]


def test_update_model_status_records_lifecycle_history() -> None:
    registry = load_model_registry("frozen_checkpoints/model_registry.json")

    updated = update_model_status(
        registry,
        model_id="smoke_pix2pix_unet256",
        status="candidate",
        note="unit test promotion",
        reviewer="unit-test-reviewer",
        metrics={"mae_mean": 1.25},
    )

    record = next(item for item in updated["models"] if item["model_id"] == "smoke_pix2pix_unet256")
    assert record["status"] == "candidate"
    assert record["metrics"]["mae_mean"] == 1.25
    assert record["lifecycle_history"]
    assert record["lifecycle_history"][-1]["reviewer"] == "unit-test-reviewer"
    assert registry["models"][0]["status"] == "smoke"


def test_compare_run_reports_ranks_by_metric(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps({"schema_version": "microi2i.evaluation_report.v1", "metrics": {"aggregate": {"mae_mean": 2.0}}}),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps({"schema_version": "microi2i.evaluation_report.v1", "metrics": {"aggregate": {"mae_mean": 1.0}}}),
        encoding="utf-8",
    )

    report = compare_run_reports([first, second], metric="mae_mean")

    assert report["ranked"][0]["path"] == str(second)


def test_write_run_comparison_html_creates_manual_review_dashboard(tmp_path) -> None:
    report = {
        "schema_version": "microi2i.run_comparison.v1",
        "metric": "mae_mean",
        "lower_is_better": True,
        "ranked": [{"path": "run_a/report.json", "value": 1.0, "sample_count": 2, "status": "computed"}],
    }

    path = write_run_comparison_html(report, tmp_path / "comparison.html")

    text = path.read_text(encoding="utf-8")
    assert "Run Comparison" in text
    assert "does not promote models automatically" in text


def test_cli_promote_model_dry_run_does_not_modify_registry() -> None:
    exit_code = main(
        [
            "promote-model",
            "--model-id",
            "smoke_pix2pix_unet256",
            "--status",
            "candidate",
            "--note",
            "dry run only",
            "--dry-run",
        ]
    )

    assert exit_code == 0


def test_local_registry_overlay_adds_machine_specific_metadata() -> None:
    registry = load_model_registry("frozen_checkpoints/model_registry.json")

    merged = merge_local_registry_overlay(
        registry,
        {
            "schema_version": "microi2i.model_registry.v1",
            "models": [
                {
                    "model_id": "smoke_pix2pix_unet256",
                    "checkpoint_path": "D:/models/latest_net_G.pth",
                }
            ],
        },
    )

    record = next(item for item in merged["models"] if item["model_id"] == "smoke_pix2pix_unet256")
    assert record["local_overlay"]["checkpoint_path"] == "D:/models/latest_net_G.pth"
    assert "local_overlay" not in registry["models"][0]
