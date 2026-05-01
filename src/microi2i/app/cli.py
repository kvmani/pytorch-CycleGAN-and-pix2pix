"""Canonical command-line interface for microi2i workflows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from microi2i.core.contracts import DatasetPrepareConfig, DatasetQAConfig, ScriptWorkflowConfig, SmokeDatasetConfig
from microi2i.dataops.dataset_prepare import prepare_dataset
from microi2i.dataops.dataset_qa import run_dataset_qa
from microi2i.dataops.smoke_data import create_smoke_datasets
from microi2i.io.configuration import apply_overrides, load_config
from microi2i.inference.legacy_runner import materialize_inference_inputs, package_prediction_images
from microi2i.manifests.reporting import finalize_run, start_run
from microi2i.evaluation.image_translation import evaluate_paired_directories, write_evaluation_review_artifacts
from microi2i.models.backends import get_model_backend, infer_backend_id
from microi2i.plugins.registry import (
    compare_run_reports,
    load_model_registry,
    load_model_registry_with_overlay,
    save_model_registry,
    update_model_status,
    validate_model_registry,
    write_run_comparison_html,
)
from microi2i.training.legacy_runner import (
    build_training_preflight,
    package_training_outputs,
    write_training_metric_placeholders,
    write_training_summary_html,
)


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


def _record_existing_artifact(run: Any, path: Path, kind: str, description: str) -> None:
    run.artifacts.append(
        {
            "path": path.relative_to(run.run_dir).as_posix() if path.is_relative_to(run.run_dir) else path.name,
            "kind": kind,
            "description": description,
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "sha256": "",
        }
    )


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
        backend = get_model_backend(infer_backend_id(config, workflow=workflow))
        backend_metadata = backend.metadata()
        if workflow == "training":
            command = backend.train_command(config, repo_root=ROOT, resolved_config=cfg)
            preflight = build_training_preflight(
                config,
                repo_root=ROOT,
                resolved_config=cfg,
                command=command,
                dry_run=dry_run,
            )
            training_status = "ready" if not preflight["errors"] else "blocked"
            report = {
                "schema_version": "microi2i.training_report.v1",
                "status": "dry_run" if dry_run else training_status,
                "model_backend": backend_metadata,
                "preflight": preflight,
            }
            run.add_artifact("training_report.json", "training_report", "Training preflight and run report", report)
            for path in write_training_metric_placeholders(run.run_dir):
                _record_existing_artifact(
                    run,
                    path,
                    "metrics_log",
                    "Structured training metrics log placeholder",
                )
            html_path = write_training_summary_html(run.run_dir, report)
            _record_existing_artifact(run, html_path, "html_report", "Human-readable training summary")
            if preflight["errors"] and not dry_run:
                status = "failed"
                exit_code = 1
                finalize_run(run, status=status, exit_code=exit_code)
                print(str(run.run_dir))
                return exit_code
        else:
            command = backend.infer_command(config, repo_root=ROOT)
        run.add_artifact(
            "command.json",
            "command",
            "Resolved model backend command",
            {"command": command, "model_backend": backend_metadata},
        )
        if dry_run:
            run.add_artifact("dry_run.json", "dry_run", "Dry-run marker", {"skipped_command": command})
        else:
            exit_code = _run_subprocess(command, cwd=ROOT)
            if exit_code != 0:
                status = "failed"
        if workflow == "training":
            outputs = package_training_outputs(run.run_dir, preflight)
            run.add_artifact("training_outputs.json", "training_outputs", "Packaged training logs and panels", outputs)
            for name, kind, description in (
                ("loss_curves.csv", "loss_curves", "Tabular training loss curve data"),
                ("loss_curves.svg", "loss_curves", "SVG training loss curve plot"),
            ):
                curve_path = run.run_dir / name
                if curve_path.exists():
                    _record_existing_artifact(run, curve_path, kind, description)
            panel_path = run.run_dir / "validation_samples.html"
            if panel_path.exists():
                _record_existing_artifact(run, panel_path, "html_review", "Training validation sample panel")
        if workflow == "inference":
            inference_cfg = cfg.get("inference", {})
            if not isinstance(inference_cfg, dict):
                inference_cfg = {}
            input_report = materialize_inference_inputs(
                inference_cfg.get("inputs", {"mode": "legacy"}),
                repo_root=ROOT,
                run_dir=run.run_dir,
            )
            for name, kind, description in (
                ("inference_inputs.json", "inference_inputs", "Normalized inference input manifest"),
                ("inference_inputs.csv", "inference_inputs", "Normalized inference input table"),
            ):
                artifact_path = run.run_dir / name
                if artifact_path.exists():
                    _record_existing_artifact(run, artifact_path, kind, description)
            packaged = package_prediction_images(
                config.command.expected_output_dir,
                run.run_dir,
                postprocess=inference_cfg.get("postprocess", {}),
                references_dir=inference_cfg.get("references_dir", ""),
            )
            report = {
                "schema_version": "microi2i.inference_report.v1",
                "status": "dry_run" if dry_run else status,
                "model_backend": backend_metadata,
                "command": command,
                "inputs": input_report,
                "packaged_predictions": packaged,
            }
            run.add_artifact("report.json", "report", "Inference report", report)
            for name, kind, description in (
                ("batch_summary.json", "batch_summary", "Per-image inference batch summary"),
                ("batch_summary.csv", "batch_summary", "Per-image inference batch summary table"),
                ("review.html", "html_review", "Human-readable inference image review"),
                ("comparison_review.html", "html_review", "Prediction/reference comparison review"),
            ):
                artifact_path = run.run_dir / name
                if artifact_path.exists():
                    run.artifacts.append(
                        {
                            "path": name,
                            "kind": kind,
                            "description": description,
                            "exists": True,
                            "size_bytes": artifact_path.stat().st_size,
                            "sha256": "",
                        }
                    )
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


def cmd_data_qa(args: argparse.Namespace) -> int:
    cfg = apply_overrides(load_config(args.config), args.set_values or [])
    config = DatasetQAConfig.from_mapping(cfg)
    run = start_run(
        workflow="data_qa",
        config_path=args.config,
        resolved_config=cfg,
        output_root=_repo_path(config.base.output_root),
        command=sys.argv,
    )
    status = "success"
    exit_code = 0
    try:
        report = run_dataset_qa(config, repo_root=ROOT)
        run.add_artifact("dataset_qa_report.json", "dataset_qa_report", "Dataset QA report", report)
        qa_dir = _repo_path(config.output_dir)
        for name, kind, description in (
            ("dataset_qa_report.html", "html_report", "Human-readable dataset QA report"),
            ("contact_sheet.jpg", "contact_sheet", "Visual sample contact sheet"),
        ):
            path = qa_dir / name
            if path.exists():
                target = run.run_dir / name
                target.write_bytes(path.read_bytes())
                run.artifacts.append(
                    {
                        "path": target.name,
                        "kind": kind,
                        "description": description,
                        "exists": target.exists(),
                        "size_bytes": target.stat().st_size if target.exists() else 0,
                        "sha256": "",
                    }
                )
        if report["status"] == "failed":
            status = "failed"
            exit_code = 1
    except Exception as exc:
        status = "failed"
        exit_code = 1
        run.add_artifact("error_report.json", "error", "Dataset QA error", {"error": str(exc)})
    finalize_run(run, status=status, exit_code=exit_code)
    print(str(run.run_dir))
    return exit_code


def cmd_create_smoke_data(args: argparse.Namespace) -> int:
    cfg = apply_overrides(load_config(args.config), args.set_values or [])
    config = SmokeDatasetConfig.from_mapping(cfg)
    run = start_run(
        workflow="create_smoke_data",
        config_path=args.config,
        resolved_config=cfg,
        output_root=_repo_path(config.base.output_root),
        command=sys.argv,
    )
    status = "success"
    exit_code = 0
    try:
        if bool(config.base.dry_run or args.dry_run):
            report = {
                "schema_version": "microi2i.smoke_dataset_manifest.v1",
                "dry_run": True,
                "output_dir": config.output_dir,
                "image_size": config.image_size,
                "sample_count": config.sample_count,
            }
        else:
            report = create_smoke_datasets(config, repo_root=ROOT)
        run.add_artifact("smoke_dataset_manifest.json", "smoke_dataset_manifest", "Tiny smoke dataset manifest", report)
    except Exception as exc:
        status = "failed"
        exit_code = 1
        run.add_artifact("error_report.json", "error", "Smoke data generation error", {"error": str(exc)})
    finalize_run(run, status=status, exit_code=exit_code)
    print(str(run.run_dir))
    return exit_code


def cmd_run_domain(args: argparse.Namespace) -> int:
    cfg = apply_overrides(load_config(args.config), args.set_values or [])
    config = ScriptWorkflowConfig.from_mapping(cfg, section="domain")
    run = start_run(
        workflow="domain",
        config_path=args.config,
        resolved_config=cfg,
        output_root=_repo_path(config.base.output_root),
        command=sys.argv,
    )
    dry_run = bool(config.base.dry_run or args.dry_run)
    status = "success"
    exit_code = 0
    try:
        script = _repo_path(config.command.legacy_script)
        command = [sys.executable, str(script), *config.command.legacy_args]
        report = {
            "schema_version": "microi2i.domain_workflow_report.v1",
            "status": "dry_run" if dry_run else "launched",
            "domain": cfg.get("domain_name", ""),
            "task": cfg.get("task", ""),
            "command": command,
            "parameters": cfg.get("parameters", {}),
            "legacy_script": str(script),
        }
        run.add_artifact("command.json", "command", "Resolved domain legacy command", {"command": command})
        if dry_run:
            run.add_artifact("dry_run.json", "dry_run", "Dry-run marker", {"skipped_command": command})
        else:
            if not script.exists():
                raise FileNotFoundError(f"domain script does not exist: {script}")
            exit_code = _run_subprocess(command, cwd=ROOT)
            if exit_code != 0:
                status = "failed"
                report["status"] = "failed"
        run.add_artifact("report.json", "report", "Domain workflow report", report)
        html_path = _write_html_report(run.run_dir, "microi2i domain workflow report", report)
        _record_existing_artifact(run, html_path, "html_report", "Human-readable domain workflow report")
    except Exception as exc:
        status = "failed"
        exit_code = 1
        run.add_artifact("error_report.json", "error", "Domain workflow error", {"error": str(exc)})
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
    if metric_payload["status"] == "computed":
        review_cfg = cfg.get("review", {})
        if not isinstance(review_cfg, dict):
            review_cfg = {}
        review = write_evaluation_review_artifacts(
            metric_payload,
            run.run_dir,
            metric=str(review_cfg.get("ranking_metric", "mae")),
            lower_is_better=bool(review_cfg.get("lower_is_better", True)),
            limit=int(review_cfg.get("limit", 5)),
        )
        report["review"] = review
    run.add_artifact("report.json", "report", "Evaluation report", report)
    for name, kind, description in (
        ("evaluation_outliers.csv", "evaluation_review", "Best/worst evaluation sample table"),
        ("evaluation_review.html", "html_review", "Best/worst evaluation image review"),
    ):
        path = run.run_dir / name
        if path.exists():
            _record_existing_artifact(run, path, kind, description)
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
    registry = load_model_registry_with_overlay(registry_path, _repo_path(args.overlay) if args.overlay else "")
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
    overlay_path = str(cfg.get("overlay_path", args.overlay or ""))
    registry = load_model_registry_with_overlay(registry_path, _repo_path(overlay_path) if overlay_path else "")
    errors = validate_model_registry(registry)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Registry valid: {registry_path}")
    return 0


def _parse_metric_overrides(values: list[str]) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"metric override must use key=value: {value}")
        key, raw = value.split("=", 1)
        try:
            metrics[key] = float(raw)
        except ValueError:
            metrics[key] = raw
    return metrics


def cmd_promote_model(args: argparse.Namespace) -> int:
    registry_path = _repo_path(args.registry)
    registry = load_model_registry(registry_path)
    metrics = _parse_metric_overrides(args.metric or [])
    updated = update_model_status(
        registry,
        model_id=args.model_id,
        status=args.status,
        note=args.note,
        reviewer=args.reviewer,
        metrics=metrics,
    )
    errors = validate_model_registry(updated)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if args.dry_run:
        print(json.dumps(updated, indent=2, sort_keys=True))
    else:
        save_model_registry(updated, registry_path)
        print(f"Updated {args.model_id} -> {args.status} in {registry_path}")
    return 0


def cmd_compare_runs(args: argparse.Namespace) -> int:
    lower_is_better = bool(args.lower_is_better)
    if args.higher_is_better:
        lower_is_better = False
    report = compare_run_reports(args.reports, metric=args.metric, lower_is_better=lower_is_better)
    if args.output:
        output_path = _repo_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(str(output_path))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    if args.html_output:
        html_path = write_run_comparison_html(report, _repo_path(args.html_output))
        print(str(html_path))
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

    data_qa = sub.add_parser("data-qa", help="Run dataset quality assurance")
    add_config_flags(data_qa)
    data_qa.set_defaults(func=cmd_data_qa)

    smoke_data = sub.add_parser("create-smoke-data", help="Create tiny deterministic pix2pix/CycleGAN smoke datasets")
    add_config_flags(smoke_data)
    smoke_data.set_defaults(func=cmd_create_smoke_data)

    domain = sub.add_parser("run-domain", help="Run a configured EBSD/Kikuchi domain workflow")
    add_config_flags(domain)
    domain.set_defaults(func=cmd_run_domain)

    evaluate = sub.add_parser("evaluate", help="Run evaluation or emit evaluation report metadata")
    add_config_flags(evaluate)
    evaluate.set_defaults(func=cmd_evaluate)

    models = sub.add_parser("models", help="List registered models")
    models.add_argument("--registry", default="frozen_checkpoints/model_registry.json")
    models.add_argument("--overlay", default="")
    models.add_argument("--details", action="store_true")
    models.set_defaults(func=cmd_models)

    validate = sub.add_parser("validate-registry", help="Validate model registry metadata")
    validate.add_argument("--config", default="configs/registry_validation.default.yml")
    validate.add_argument("--registry", default="frozen_checkpoints/model_registry.json")
    validate.add_argument("--overlay", default="")
    validate.add_argument("--set", dest="set_values", action="append", default=[])
    validate.set_defaults(func=cmd_validate_registry)

    promote = sub.add_parser("promote-model", help="Update model lifecycle status in the registry")
    promote.add_argument("--registry", default="frozen_checkpoints/model_registry.json")
    promote.add_argument("--model-id", required=True)
    promote.add_argument("--status", required=True, choices=["smoke", "candidate", "promoted", "deprecated"])
    promote.add_argument("--note", default="")
    promote.add_argument("--reviewer", default="")
    promote.add_argument("--metric", action="append", default=[])
    promote.add_argument("--dry-run", action="store_true")
    promote.set_defaults(func=cmd_promote_model)

    compare = sub.add_parser("compare-runs", help="Compare evaluation reports by an aggregate metric")
    compare.add_argument("--reports", nargs="+", required=True)
    compare.add_argument("--metric", default="mae_mean")
    compare.add_argument("--lower-is-better", action="store_true", default=True)
    compare.add_argument("--higher-is-better", action="store_true")
    compare.add_argument("--output", default="")
    compare.add_argument("--html-output", default="")
    compare.set_defaults(func=cmd_compare_runs)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
