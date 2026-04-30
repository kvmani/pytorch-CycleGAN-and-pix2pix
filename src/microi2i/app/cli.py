"""Canonical command-line interface for microi2i workflows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from microi2i.io.configuration import apply_overrides, load_config
from microi2i.manifests.reporting import RunContext, finalize_run, start_run
from microi2i.evaluation.image_translation import evaluate_paired_directories
from microi2i.plugins.registry import load_model_registry, validate_model_registry


ROOT = Path(__file__).resolve().parents[3]


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def _run_subprocess(command: list[str], *, cwd: Path) -> int:
    process = subprocess.run(command, cwd=str(cwd), check=False)
    return int(process.returncode)


def _legacy_command(cfg: dict[str, Any], workflow: str) -> list[str]:
    workflow_cfg = cfg.get(workflow, {})
    if not isinstance(workflow_cfg, dict):
        raise ValueError(f"{workflow} config must be a mapping")
    script = str(workflow_cfg.get("legacy_script", ""))
    if not script:
        raise ValueError(f"{workflow}.legacy_script is required for this migration-layer command")
    args = workflow_cfg.get("legacy_args", [])
    if not isinstance(args, list):
        raise ValueError(f"{workflow}.legacy_args must be a list")
    return [sys.executable, str(_repo_path(script)), *[str(item) for item in args]]


def _run_wrapped_workflow(args: argparse.Namespace, workflow: str) -> int:
    cfg = apply_overrides(load_config(args.config), args.set_values or [])
    run = start_run(
        workflow=workflow,
        config_path=args.config,
        resolved_config=cfg,
        output_root=_repo_path(str(cfg.get("output_root", "artifacts/runs"))),
        command=sys.argv,
    )
    dry_run = bool(cfg.get("dry_run", False) or args.dry_run)
    status = "success"
    exit_code = 0
    try:
        command = _legacy_command(cfg, workflow)
        run.add_artifact("command.json", "command", "Resolved legacy command", {"command": command})
        if dry_run:
            run.add_artifact("dry_run.json", "dry_run", "Dry-run marker", {"skipped_command": command})
        else:
            exit_code = _run_subprocess(command, cwd=ROOT)
            if exit_code != 0:
                status = "failed"
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
    run = start_run(
        workflow="prepare_dataset",
        config_path=args.config,
        resolved_config=cfg,
        output_root=_repo_path(str(cfg.get("output_root", "artifacts/runs"))),
        command=sys.argv,
    )
    dataset_payload = {
        "schema_version": "microi2i.dataset_manifest.v1",
        "dataset_id": cfg.get("dataset_id", run.run_id),
        "task_type": cfg.get("task_type", "paired_translation"),
        "source_roots": cfg.get("source_roots", []),
        "split_policy": cfg.get("split_policy", {}),
        "preprocessing": cfg.get("preprocessing", {}),
        "leakage_group_policy": cfg.get("leakage_group_policy", {}),
    }
    run.add_artifact("dataset_manifest.json", "dataset_manifest", "Dataset preparation manifest", dataset_payload)
    finalize_run(run, status="success", exit_code=0)
    print(str(run.run_dir))
    return 0


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
    run.add_artifact("report.json", "report", "Evaluation report placeholder", report)
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
