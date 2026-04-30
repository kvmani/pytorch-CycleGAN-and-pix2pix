"""Canonical command-line interface for microi2i workflows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from microi2i.core.contracts import DatasetPrepareConfig, ScriptWorkflowConfig
from microi2i.dataops.dataset_prepare import prepare_dataset
from microi2i.io.configuration import apply_overrides, load_config
from microi2i.inference.legacy_runner import build_inference_command, package_prediction_images
from microi2i.manifests.reporting import finalize_run, start_run
from microi2i.evaluation.image_translation import evaluate_paired_directories
from microi2i.plugins.registry import load_model_registry, validate_model_registry
from microi2i.training.legacy_runner import build_training_command


ROOT = Path(__file__).resolve().parents[3]


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def _run_subprocess(command: list[str], *, cwd: Path) -> int:
    process = subprocess.run(command, cwd=str(cwd), check=False)
    return int(process.returncode)


def _write_html_report(run_dir: Path, title: str, payload: dict[str, Any]) -> Path:
    path = run_dir / "report.html"
    body = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title></head>"
        f"<body><h1>{title}</h1><pre>{body}</pre></body></html>",
        encoding="utf-8",
    )
    return path


def _run_wrapped_workflow(args: argparse.Namespace, workflow: str) -> int:
    cfg = apply_overrides(load_config(args.config), args.set_values or [])
    section = "training" if workflow == "training" else "inference"
    config = ScriptWorkflowConfig.from_mapping(cfg, section=section)
    run = start_run(
        workflow=workflow,
        config_path=args.config,
        resolved_config=cfg,
        output_root=_repo_path(config.base.output_root),
        command=sys.argv,
    )
    dry_run = bool(config.base.dry_run or args.dry_run)
    status = "success"
    exit_code = 0
    try:
        if workflow == "training":
            command = build_training_command(config, repo_root=ROOT)
        else:
            command = build_inference_command(config, repo_root=ROOT)
        run.add_artifact("command.json", "command", "Resolved legacy command", {"command": command})
        if dry_run:
            run.add_artifact("dry_run.json", "dry_run", "Dry-run marker", {"skipped_command": command})
        else:
            exit_code = _run_subprocess(command, cwd=ROOT)
            if exit_code != 0:
                status = "failed"
        if workflow == "inference":
            packaged = package_prediction_images(config.command.expected_output_dir, run.run_dir)
            report = {
                "schema_version": "microi2i.inference_report.v1",
                "status": "dry_run" if dry_run else status,
                "command": command,
                "packaged_predictions": packaged,
            }
            run.add_artifact("report.json", "report", "Inference report", report)
            html_path = _write_html_report(run.run_dir, "microi2i inference report", report)
            run.artifacts.append(
                {
                    "path": html_path.name,
                    "kind": "html_report",
                    "description": "Human-readable inference report",
                    "exists": html_path.exists(),
                    "size_bytes": html_path.stat().st_size if html_path.exists() else 0,
                    "sha256": "",
                }
            )
    except Exception as exc:  # manifest failure context is more useful than a bare traceback
        status = "failed"
        exit_code = 1
        run.add_artifact("error_report.json", "error", "Workflow error report", {"error": str(exc)})
    finalize_run(run, status=status, exit_code=exit_code)
    print(str(run.run_dir))
    return exit_code


def cmd_train(args: argparse.Namespace) -> int:
    return _run_wrapped_workflow(args, "training")


def cmd_infer(args: argparse.Namespace) -> int:
    return _run_wrapped_workflow(args, "inference")


def cmd_prepare_dataset(args: argparse.Namespace) -> int:
    cfg = apply_overrides(load_config(args.config), args.set_values or [])
    config = DatasetPrepareConfig.from_mapping(cfg)
    run = start_run(
        workflow="prepare_dataset",
        config_path=args.config,
        resolved_config=cfg,
        output_root=_repo_path(config.base.output_root),
        command=sys.argv,
    )
    status = "success"
    exit_code = 0
    try:
        if bool(config.base.dry_run or args.dry_run):
            dataset_payload = {
                "schema_version": "microi2i.dataset_manifest.v1",
                "dataset_id": config.dataset_id,
                "task_type": config.task_type,
                "source_roots": config.source_roots,
                "output_dataset_dir": config.output_dataset_dir,
                "split_policy": cfg.get("split_policy", {}),
                "preprocessing": config.preprocessing,
                "leakage_group_policy": cfg.get("leakage_group_policy", {}),
                "dry_run": True,
            }
        else:
            dataset_payload = prepare_dataset(config, repo_root=ROOT)
        run.add_artifact("dataset_manifest.json", "dataset_manifest", "Dataset preparation manifest", dataset_payload)
    except Exception as exc:
        status = "failed"
        exit_code = 1
        run.add_artifact("error_report.json", "error", "Dataset preparation error", {"error": str(exc)})
    finalize_run(run, status=status, exit_code=exit_code)
    print(str(run.run_dir))
    return exit_code


def cmd_evaluate(args: argparse.Namespace) -> int:
    cfg = apply_overrides(load_config(args.config), args.set_values or [])
    run = start_run(
        workflow="evaluate",
        config_path=args.config,
        resolved_config=cfg,
        output_root=_repo_path(str(cfg.get("output_root", "artifacts/runs"))),
        command=sys.argv,
    )
    inputs = cfg.get("inputs", {})
    if not isinstance(inputs, dict):
        inputs = {}
    metric_payload = evaluate_paired_directories(
        inputs.get("predictions_dir", ""),
        inputs.get("targets_dir", ""),
    )
    report = {
        "schema_version": "microi2i.evaluation_report.v1",
        "status": metric_payload["status"],
        "configured_metrics": cfg.get("metrics", []),
        "metrics": metric_payload,
    }
    run.add_artifact("report.json", "report", "Evaluation report", report)
    html_path = _write_html_report(run.run_dir, "microi2i evaluation report", report)
    run.artifacts.append(
        {
            "path": html_path.name,
            "kind": "html_report",
            "description": "Human-readable evaluation report",
            "exists": html_path.exists(),
            "size_bytes": html_path.stat().st_size if html_path.exists() else 0,
            "sha256": "",
        }
    )
    finalize_run(run, status="success", exit_code=0)
    print(str(run.run_dir))
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    registry_path = _repo_path(args.registry)
    registry = load_model_registry(registry_path)
    records = registry.get("models", [])
    if args.details:
        print(json.dumps(registry, indent=2, sort_keys=True))
    else:
        for record in records:
            print(f"{record.get('model_id', '<missing>')}: {record.get('display_name', '<unnamed>')}")
    return 0


def cmd_validate_registry(args: argparse.Namespace) -> int:
    cfg = apply_overrides(load_config(args.config), args.set_values or []) if args.config else {}
    registry_path = _repo_path(str(cfg.get("registry_path", args.registry)))
    registry = load_model_registry(registry_path)
    errors = validate_model_registry(registry)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Registry valid: {registry_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="microi2i")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_config_flags(child: argparse.ArgumentParser) -> None:
        child.add_argument("--config", required=True)
        child.add_argument("--set", dest="set_values", action="append", default=[])
        child.add_argument("--dry-run", action="store_true")

    train = sub.add_parser("train", help="Run a configured training workflow")
    add_config_flags(train)
    train.set_defaults(func=cmd_train)

    infer = sub.add_parser("infer", help="Run a configured inference workflow")
    add_config_flags(infer)
    infer.set_defaults(func=cmd_infer)

    prep = sub.add_parser("prepare-dataset", help="Create or validate dataset preparation metadata")
    add_config_flags(prep)
    prep.set_defaults(func=cmd_prepare_dataset)

    evaluate = sub.add_parser("evaluate", help="Run evaluation or emit evaluation report metadata")
    add_config_flags(evaluate)
    evaluate.set_defaults(func=cmd_evaluate)

    models = sub.add_parser("models", help="List registered models")
    models.add_argument("--registry", default="frozen_checkpoints/model_registry.json")
    models.add_argument("--details", action="store_true")
    models.set_defaults(func=cmd_models)

    validate = sub.add_parser("validate-registry", help="Validate model registry metadata")
    validate.add_argument("--config", default="configs/registry_validation.default.yml")
    validate.add_argument("--registry", default="frozen_checkpoints/model_registry.json")
    validate.add_argument("--set", dest="set_values", action="append", default=[])
    validate.set_defaults(func=cmd_validate_registry)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
