"""Legacy training command construction and run preflight helpers."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

from microi2i.core.contracts import ScriptWorkflowConfig


def parse_legacy_args(args: list[str]) -> dict[str, str | bool]:
    """Parse simple ``--key value`` legacy options into a metadata mapping."""

    parsed: dict[str, str | bool] = {}
    index = 0
    while index < len(args):
        token = args[index]
        if not token.startswith("--"):
            index += 1
            continue
        key = token[2:].replace("-", "_")
        next_index = index + 1
        if next_index >= len(args) or args[next_index].startswith("--"):
            parsed[key] = True
            index += 1
        else:
            parsed[key] = args[next_index]
            index += 2
    return parsed


def build_training_command(config: ScriptWorkflowConfig, *, repo_root: Path) -> list[str]:
    """Build the command that runs the current legacy training script."""

    script = Path(config.command.legacy_script)
    if not script.is_absolute():
        script = repo_root / script
    return [sys.executable, str(script), *config.command.legacy_args]


def build_training_preflight(
    config: ScriptWorkflowConfig,
    *,
    repo_root: Path,
    resolved_config: dict[str, Any],
    command: list[str],
    dry_run: bool,
) -> dict[str, Any]:
    """Create a structured preflight report before launching training."""

    script = Path(config.command.legacy_script)
    if not script.is_absolute():
        script = repo_root / script
    options = parse_legacy_args(config.command.legacy_args)
    dataroot_value = str(options.get("dataroot", ""))
    dataroot = Path(dataroot_value)
    if dataroot_value and not dataroot.is_absolute():
        dataroot = repo_root / dataroot
    checkpoints_value = str(options.get("checkpoints_dir", "checkpoints"))
    checkpoints_dir = Path(checkpoints_value)
    if not checkpoints_dir.is_absolute():
        checkpoints_dir = repo_root / checkpoints_dir

    training_section = resolved_config.get("training", {})
    if not isinstance(training_section, dict):
        training_section = {}
    dataset_manifest = str(training_section.get("dataset_manifest_path", "")).strip()
    dataset_manifest_path = Path(dataset_manifest) if dataset_manifest else None
    if dataset_manifest_path is not None and not dataset_manifest_path.is_absolute():
        dataset_manifest_path = repo_root / dataset_manifest_path

    checks = {
        "legacy_script_exists": script.exists(),
        "dataroot_configured": bool(dataroot_value),
        "dataroot_exists": dataroot.exists() if dataroot_value else False,
        "checkpoints_parent_exists": checkpoints_dir.parent.exists(),
        "dataset_manifest_configured": dataset_manifest_path is not None,
        "dataset_manifest_exists": dataset_manifest_path.exists() if dataset_manifest_path is not None else False,
    }
    errors: list[str] = []
    warnings: list[str] = []
    if not checks["legacy_script_exists"]:
        errors.append(f"legacy training script does not exist: {script}")
    if not checks["dataroot_configured"]:
        errors.append("training legacy_args must include --dataroot")
    elif not checks["dataroot_exists"]:
        message = f"training dataroot does not exist: {dataroot}"
        if dry_run:
            warnings.append(message)
        else:
            errors.append(message)
    if dataset_manifest_path is None:
        warnings.append("training.dataset_manifest_path is not configured; provenance is weaker")
    elif not checks["dataset_manifest_exists"]:
        warnings.append(f"configured dataset manifest does not exist: {dataset_manifest_path}")

    return {
        "schema_version": "microi2i.training_preflight.v1",
        "dry_run": dry_run,
        "command": command,
        "legacy_options": options,
        "model": options.get("model", ""),
        "dataset_mode": options.get("dataset_mode", ""),
        "experiment_name": options.get("name", ""),
        "paths": {
            "legacy_script": str(script),
            "dataroot": str(dataroot) if dataroot_value else "",
            "checkpoints_dir": str(checkpoints_dir),
            "dataset_manifest": str(dataset_manifest_path) if dataset_manifest_path is not None else "",
        },
        "runtime": {
            "gpu_ids": config.runtime.gpu_ids,
            "seed": config.runtime.seed,
        },
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }


def write_training_metric_placeholders(run_dir: Path) -> list[Path]:
    """Create structured metric logs that legacy training can later populate."""

    csv_path = run_dir / "metrics_log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "iteration", "loss_g", "loss_d", "learning_rate", "phase"])
    jsonl_path = run_dir / "metrics_log.jsonl"
    jsonl_path.write_text("", encoding="utf-8")
    return [csv_path, jsonl_path]


def write_training_summary_html(run_dir: Path, report: dict[str, Any]) -> Path:
    """Write a human-readable training preflight and summary document."""

    body = json.dumps(report, indent=2, sort_keys=True)
    path = run_dir / "training_summary.html"
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>microi2i training summary</title>"
        "<style>body{font-family:Arial,sans-serif;margin:2rem;line-height:1.45}"
        "pre{background:#f6f8fa;border-radius:10px;padding:1rem;overflow:auto}"
        ".ok{color:#0f766e}.warn{color:#b45309}.fail{color:#b91c1c}</style></head>"
        "<body><h1>microi2i training summary</h1>"
        f"<p>Status: <strong>{report.get('status', 'unknown')}</strong></p>"
        f"<pre>{body}</pre></body></html>",
        encoding="utf-8",
    )
    return path
