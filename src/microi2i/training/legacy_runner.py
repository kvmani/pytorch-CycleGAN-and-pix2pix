"""Legacy training command construction and run preflight helpers."""

from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from microi2i.core.contracts import ScriptWorkflowConfig


SMOKE_OPTION_DEFAULTS = {
    "gpu_ids": "-1",
    "n_epochs": "1",
    "n_epochs_decay": "0",
    "max_dataset_size": "2",
    "batch_size": "1",
    "num_threads": "0",
    "load_size": "32",
    "crop_size": "32",
    "display_id": "-1",
    "print_freq": "1",
    "save_latest_freq": "2",
    "save_epoch_freq": "1",
}


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


def _replace_option(args: list[str], key: str, value: str | bool) -> list[str]:
    flag = f"--{key}"
    output: list[str] = []
    index = 0
    replaced = False
    while index < len(args):
        token = args[index]
        if token == flag:
            replaced = True
            if value is True:
                output.append(flag)
                index += 1
                if index < len(args) and not args[index].startswith("--"):
                    index += 1
            else:
                output.extend([flag, str(value)])
                index += 1
                if index < len(args) and not args[index].startswith("--"):
                    index += 1
        else:
            output.append(token)
            index += 1
    if not replaced:
        output.append(flag)
        if value is not True:
            output.append(str(value))
    return output


def apply_smoke_training_overrides(args: list[str], smoke: dict[str, Any]) -> list[str]:
    """Return legacy args with CPU-safe smoke settings applied."""

    if not smoke.get("enabled", False):
        return list(args)
    effective = list(args)
    defaults = dict(SMOKE_OPTION_DEFAULTS)
    if "max_epochs" in smoke:
        defaults["n_epochs"] = str(smoke["max_epochs"])
    if "max_dataset_size" in smoke:
        defaults["max_dataset_size"] = str(smoke["max_dataset_size"])
    if "image_size" in smoke:
        defaults["load_size"] = str(smoke["image_size"])
        defaults["crop_size"] = str(smoke["image_size"])
    if "checkpoints_dir" in smoke:
        defaults["checkpoints_dir"] = str(smoke["checkpoints_dir"])
    if bool(smoke.get("cpu_only", True)):
        defaults["gpu_ids"] = "-1"
    for key, value in defaults.items():
        effective = _replace_option(effective, key, value)
    if bool(smoke.get("disable_html", True)):
        effective = _replace_option(effective, "no_html", True)
    return effective


def _training_section(resolved_config: dict[str, Any]) -> dict[str, Any]:
    section = resolved_config.get("training", {})
    return section if isinstance(section, dict) else {}


def build_training_command(
    config: ScriptWorkflowConfig,
    *,
    repo_root: Path,
    resolved_config: dict[str, Any] | None = None,
) -> list[str]:
    """Build the command that runs the current legacy training script."""

    script = Path(config.command.legacy_script)
    if not script.is_absolute():
        script = repo_root / script
    args = config.command.legacy_args
    if resolved_config is not None:
        smoke = _training_section(resolved_config).get("smoke", {})
        if isinstance(smoke, dict):
            args = apply_smoke_training_overrides(args, smoke)
    return [sys.executable, str(script), *args]


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
    options = parse_legacy_args(command[2:] if len(command) >= 2 else config.command.legacy_args)
    dataroot_value = str(options.get("dataroot", ""))
    dataroot = Path(dataroot_value)
    if dataroot_value and not dataroot.is_absolute():
        dataroot = repo_root / dataroot
    checkpoints_value = str(options.get("checkpoints_dir", "checkpoints"))
    checkpoints_dir = Path(checkpoints_value)
    if not checkpoints_dir.is_absolute():
        checkpoints_dir = repo_root / checkpoints_dir

    training_section = _training_section(resolved_config)
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
    curve_link = ""
    if (run_dir / "loss_curves.svg").exists():
        curve_link = "<h2>Loss Curves</h2><img src='loss_curves.svg' alt='Training loss curves'>"
    monitor_link = ""
    if (run_dir / "validation_monitor" / "index.html").exists():
        monitor_link = "<h2>Validation Monitor</h2><p><a href='validation_monitor/index.html'>Open epoch validation dashboard</a></p>"
    elif (run_dir / "validation_monitor" / "report.json").exists():
        monitor_link = "<h2>Validation Monitor</h2><p><a href='validation_monitor/report.json'>Open epoch validation report</a></p>"
    elif (run_dir / "validation_monitor_manifest.json").exists():
        monitor_link = "<h2>Validation Monitor</h2><p><a href='validation_monitor_manifest.json'>Open validation monitor manifest</a></p>"
    path = run_dir / "training_summary.html"
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>microi2i training summary</title>"
        "<style>body{font-family:Arial,sans-serif;margin:2rem;line-height:1.45}"
        "pre{background:#f6f8fa;border-radius:10px;padding:1rem;overflow:auto}img{max-width:100%}"
        ".ok{color:#0f766e}.warn{color:#b45309}.fail{color:#b91c1c}</style></head>"
        "<body><h1>microi2i training summary</h1>"
        f"<p>Status: <strong>{report.get('status', 'unknown')}</strong></p>"
        f"{curve_link}"
        f"{monitor_link}"
        f"<pre>{body}</pre></body></html>",
        encoding="utf-8",
    )
    return path


LOSS_LINE_RE = re.compile(
    r"\(epoch:\s*(?P<epoch>\d+),\s*iters:\s*(?P<iters>\d+),\s*time:\s*(?P<time>[0-9.eE+-]+),\s*data:\s*(?P<data>[0-9.eE+-]+)\)\s*(?P<losses>.*)"
)
LOSS_VALUE_RE = re.compile(r"(?P<name>[A-Za-z0-9_/-]+):\s*(?P<value>[-+0-9.eE]+)")


def parse_legacy_loss_log(path: str | Path) -> list[dict[str, Any]]:
    """Parse the upstream ``loss_log.txt`` format into structured rows."""

    log_path = Path(path)
    if not log_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LOSS_LINE_RE.search(line)
        if not match:
            continue
        row: dict[str, Any] = {
            "epoch": int(match.group("epoch")),
            "iteration": int(match.group("iters")),
            "time": float(match.group("time")),
            "data_time": float(match.group("data")),
        }
        for loss_match in LOSS_VALUE_RE.finditer(match.group("losses")):
            row[f"loss_{loss_match.group('name')}"] = float(loss_match.group("value"))
        rows.append(row)
    return rows


def write_structured_loss_logs(run_dir: Path, rows: list[dict[str, Any]]) -> list[Path]:
    """Write parsed loss rows as JSONL and CSV."""

    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "metrics_log.jsonl"
    csv_path = run_dir / "metrics_log.csv"
    if not rows:
        return [csv_path, jsonl_path]
    jsonl_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return [csv_path, jsonl_path]


def write_loss_curve_artifacts(run_dir: Path, rows: list[dict[str, Any]]) -> list[Path]:
    """Write simple CSV/SVG loss-curve artifacts from structured loss rows."""

    csv_path = run_dir / "loss_curves.csv"
    svg_path = run_dir / "loss_curves.svg"
    loss_keys = sorted(key for key in {key for row in rows for key in row} if key.startswith("loss_"))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", *loss_keys])
        for index, row in enumerate(rows):
            writer.writerow([index, *[row.get(key, "") for key in loss_keys]])

    if not rows or not loss_keys:
        svg_path.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='640' height='180'>"
            "<text x='20' y='40' font-family='Arial' font-size='18'>No loss rows available</text></svg>",
            encoding="utf-8",
        )
        return [csv_path, svg_path]

    width, height = 720, 360
    left, right, top, bottom = 60, 20, 30, 50
    values = [float(row[key]) for row in rows for key in loss_keys if isinstance(row.get(key), int | float)]
    min_value, max_value = min(values), max(values)
    span = max(max_value - min_value, 1e-9)
    max_step = max(len(rows) - 1, 1)
    colors = ["#2563eb", "#dc2626", "#059669", "#9333ea", "#ea580c", "#0891b2"]

    def point(step: int, value: float) -> tuple[float, float]:
        x = left + ((width - left - right) * step / max_step)
        y = top + ((height - top - bottom) * (max_value - value) / span)
        return x, y

    paths: list[str] = []
    legend: list[str] = []
    for key_index, key in enumerate(loss_keys):
        coords = [
            point(step, float(row[key]))
            for step, row in enumerate(rows)
            if isinstance(row.get(key), int | float)
        ]
        if not coords:
            continue
        d = " ".join(("M" if index == 0 else "L") + f"{x:.2f},{y:.2f}" for index, (x, y) in enumerate(coords))
        color = colors[key_index % len(colors)]
        paths.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2'/>")
        legend.append(
            f"<text x='{left + 10}' y='{height - 25 - (key_index * 18)}' font-family='Arial' font-size='12' fill='{color}'>{key}</text>"
        )

    svg_path.write_text(
        "<svg xmlns='http://www.w3.org/2000/svg' width='720' height='360' viewBox='0 0 720 360'>"
        "<rect width='720' height='360' fill='white'/>"
        "<text x='60' y='22' font-family='Arial' font-size='18' fill='#111827'>Training Loss Curves</text>"
        f"<line x1='{left}' y1='{height-bottom}' x2='{width-right}' y2='{height-bottom}' stroke='#374151'/>"
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{height-bottom}' stroke='#374151'/>"
        f"<text x='{left}' y='{height-12}' font-family='Arial' font-size='12'>step</text>"
        f"<text x='10' y='{top+10}' font-family='Arial' font-size='12'>loss</text>"
        + "".join(paths)
        + "".join(legend)
        + "</svg>",
        encoding="utf-8",
    )
    return [csv_path, svg_path]


def package_training_outputs(run_dir: Path, preflight: dict[str, Any]) -> dict[str, Any]:
    """Collect legacy training logs and visual samples into a normalized package."""

    paths = preflight.get("paths", {})
    checkpoints_dir = Path(str(paths.get("checkpoints_dir", "")))
    experiment_name = str(preflight.get("experiment_name", ""))
    experiment_dir = checkpoints_dir / experiment_name if experiment_name else checkpoints_dir
    loss_log = experiment_dir / "loss_log.txt"
    rows = parse_legacy_loss_log(loss_log)
    write_structured_loss_logs(run_dir, rows)
    curve_paths = write_loss_curve_artifacts(run_dir, rows)

    copied_images: list[dict[str, Any]] = []
    image_dir = experiment_dir / "web" / "images"
    target_dir = run_dir / "validation_samples"
    if image_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(image_dir.glob("*.png"))[:24]:
            target = target_dir / image_path.name
            shutil.copy2(image_path, target)
            copied_images.append({"source": str(image_path), "target": str(target.relative_to(run_dir))})

    panel_path = run_dir / "validation_samples.html"
    image_tags = "\n".join(
        f"<figure><img src='{item['target']}' alt='{Path(item['target']).name}'><figcaption>{Path(item['target']).name}</figcaption></figure>"
        for item in copied_images
    )
    panel_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>microi2i validation samples</title>"
        "<style>body{font-family:Arial,sans-serif;margin:2rem}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:1rem}"
        "img{max-width:100%;border:1px solid #d1d5db;border-radius:8px}figcaption{font-size:.85rem;color:#374151}</style></head>"
        f"<body><h1>Validation Samples</h1><p>Copied images: {len(copied_images)}</p><div class='grid'>{image_tags}</div></body></html>",
        encoding="utf-8",
    )
    return {
        "schema_version": "microi2i.training_outputs.v1",
        "experiment_dir": str(experiment_dir),
        "loss_log": str(loss_log),
        "loss_rows": len(rows),
        "loss_curve_artifacts": [str(path) for path in curve_paths],
        "validation_sample_count": len(copied_images),
        "validation_samples": copied_images,
        "validation_samples_html": str(panel_path),
    }
