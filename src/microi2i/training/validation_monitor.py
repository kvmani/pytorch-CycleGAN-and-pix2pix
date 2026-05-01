"""Epoch-level validation monitoring for legacy GAN training."""

from __future__ import annotations

import copy
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from microi2i.evaluation.image_translation import evaluate_paired_directories
from microi2i.training.legacy_runner import parse_legacy_args


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MONITOR_ENV_VAR = "MICROI2I_VALIDATION_MONITOR_CONFIG"


def _as_dict(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _image_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _validation_roots(dataroot: Path, dataset_mode: str) -> tuple[str, list[Path], list[str]]:
    warnings: list[str] = []
    if dataset_mode == "unaligned":
        for phase in ("val", "test"):
            root_a = dataroot / f"{phase}A"
            root_b = dataroot / f"{phase}B"
            if root_a.exists() or root_b.exists():
                return phase, [root_a, root_b], warnings
        warnings.append(f"no CycleGAN validation roots found under {dataroot}; expected valA/valB or testA/testB")
        return "val", [dataroot / "valA", dataroot / "valB"], warnings
    for phase in ("val", "test"):
        root = dataroot / phase
        if root.exists():
            return phase, [root], warnings
    warnings.append(f"no pix2pix validation root found under {dataroot}; expected val or test")
    return "val", [dataroot / "val"], warnings


def discover_validation_pool(
    *,
    repo_root: str | Path,
    legacy_options: dict[str, Any],
    monitor_config: dict[str, Any],
) -> dict[str, Any]:
    """Discover validation samples for aligned and unaligned legacy datasets."""

    root = Path(repo_root)
    dataroot = _repo_path(root, str(legacy_options.get("dataroot", "")))
    dataset_mode = str(legacy_options.get("dataset_mode", "aligned"))
    phase, roots, warnings = _validation_roots(dataroot, dataset_mode)
    explicit = [str(item) for item in monitor_config.get("fixed_images", []) if str(item).strip()]
    samples: list[dict[str, Any]] = []

    if dataset_mode == "unaligned":
        paths_a = _image_paths(roots[0])
        paths_b = _image_paths(roots[1]) if len(roots) > 1 else []
        for index, path in enumerate(paths_a):
            target = paths_b[index] if index < len(paths_b) else None
            samples.append(
                {
                    "sample_id": path.stem,
                    "index": index,
                    "role": "A",
                    "input_path": str(path),
                    "target_path": str(target) if target else "",
                    "fixed": False,
                }
            )
    else:
        for index, path in enumerate(_image_paths(roots[0])):
            samples.append(
                {
                    "sample_id": path.stem,
                    "index": index,
                    "role": "AB",
                    "input_path": str(path),
                    "target_path": str(path),
                    "fixed": False,
                }
            )

    if explicit:
        by_abs = {str(Path(row["input_path"]).resolve()).lower(): row for row in samples}
        by_name = {Path(row["input_path"]).name.lower(): row for row in samples}
        by_id = {str(row["sample_id"]).lower(): row for row in samples}
        fixed: list[dict[str, Any]] = []
        for item in explicit:
            candidate = _repo_path(root, item)
            row = (
                by_abs.get(str(candidate.resolve()).lower())
                or by_name.get(Path(item).name.lower())
                or by_id.get(str(item).lower())
            )
            if row is None:
                warnings.append(f"configured fixed validation image was not found in validation pool: {item}")
                continue
            fixed.append(dict(row, fixed=True))
    else:
        fixed_count = max(_as_int(monitor_config.get("fixed_count"), 5), 0)
        fixed = [dict(row, fixed=True) for row in samples[:fixed_count]]

    fixed_ids = {row["sample_id"] for row in fixed}
    for row in samples:
        row["fixed"] = row["sample_id"] in fixed_ids
    return {
        "schema_version": "microi2i.validation_pool.v1",
        "dataset_mode": dataset_mode,
        "phase": phase,
        "roots": [str(path) for path in roots],
        "sample_count": len(samples),
        "samples": samples,
        "fixed_samples": fixed,
        "warnings": warnings,
    }


def select_epoch_samples(
    samples: list[dict[str, Any]],
    fixed_samples: list[dict[str, Any]],
    *,
    total_count: int,
    seed: int,
    epoch: int,
) -> list[dict[str, Any]]:
    """Select fixed samples plus deterministic per-epoch random samples."""

    total = max(int(total_count), 0)
    fixed = [dict(row, fixed=True, selection_role="fixed") for row in fixed_samples[:total]]
    remaining_slots = max(total - len(fixed), 0)
    fixed_ids = {row["sample_id"] for row in fixed}
    random_pool = [row for row in samples if row.get("sample_id") not in fixed_ids]
    rng = random.Random(int(seed) + int(epoch))
    extra = rng.sample(random_pool, min(remaining_slots, len(random_pool))) if remaining_slots else []
    return fixed + [dict(row, fixed=False, selection_role="random") for row in extra]


def build_validation_monitor_manifest(
    *,
    repo_root: str | Path,
    run_dir: str | Path,
    resolved_config: dict[str, Any],
    command: list[str],
    dry_run: bool,
) -> dict[str, Any]:
    """Build the run-level validation monitor manifest."""

    training = _as_dict(resolved_config.get("training"))
    monitor = _as_dict(training.get("validation_monitor"))
    enabled = bool(monitor.get("enabled", True))
    legacy_options = parse_legacy_args(command[2:] if len(command) >= 2 else training.get("legacy_args", []))
    seed = _as_int(monitor.get("seed"), _as_int(_as_dict(resolved_config.get("runtime")).get("seed"), 42))
    fixed_count = _as_int(monitor.get("fixed_count"), 5)
    total_count = _as_int(monitor.get("total_count"), max(fixed_count, 5))
    frequency = max(_as_int(monitor.get("eval_frequency_epochs"), 1), 1)
    pool = discover_validation_pool(repo_root=repo_root, legacy_options=legacy_options, monitor_config=monitor)
    preview = select_epoch_samples(
        pool["samples"],
        pool["fixed_samples"],
        total_count=total_count,
        seed=seed,
        epoch=1,
    )
    warnings = list(pool.get("warnings", []))
    if enabled and not pool["samples"]:
        warnings.append("validation monitor is enabled but no validation samples were discovered")
    return {
        "schema_version": "microi2i.validation_monitor_manifest.v1",
        "enabled": enabled,
        "dry_run": dry_run,
        "repo_root": str(Path(repo_root)),
        "run_dir": str(Path(run_dir)),
        "policy": {
            "fixed_images": [str(item) for item in monitor.get("fixed_images", [])],
            "fixed_count": fixed_count,
            "total_count": total_count,
            "seed": seed,
            "eval_frequency_epochs": frequency,
            "metrics": monitor.get("metrics", []),
            "export_html": bool(monitor.get("export_html", True)),
        },
        "legacy_options": legacy_options,
        "pool": pool,
        "fixed_samples": pool["fixed_samples"],
        "epoch_selection_preview": {"epoch": 1, "samples": preview},
        "epoch_selections": [],
        "warnings": warnings,
    }


def write_validation_monitor_manifest(run_dir: str | Path, manifest: dict[str, Any]) -> Path:
    path = Path(run_dir) / "validation_monitor_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _batchify(data: dict[str, Any]) -> dict[str, Any]:
    batched: dict[str, Any] = {}
    for key, value in data.items():
        if hasattr(value, "unsqueeze"):
            batched[key] = value.unsqueeze(0)
        elif key.endswith("_paths"):
            batched[key] = [value]
        else:
            batched[key] = value
    return batched


def _safe_sample_name(sample: dict[str, Any]) -> str:
    raw = str(sample.get("sample_id") or Path(str(sample.get("input_path", "sample"))).stem)
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw) or "sample"


def _save_visuals(visuals: dict[str, Any], sample: dict[str, Any], epoch_dir: Path) -> dict[str, str]:
    from util import util as legacy_util

    sample_name = _safe_sample_name(sample)
    saved: dict[str, str] = {}
    for label, tensor in visuals.items():
        image = legacy_util.tensor2im(tensor)
        out_dir = epoch_dir / label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{sample_name}.png"
        legacy_util.save_image(image, str(out_path))
        saved[label] = out_path.relative_to(epoch_dir.parent).as_posix()
    return saved


def _write_epoch_html(epoch_dir: Path, epoch_report: dict[str, Any]) -> Path:
    cards = []
    for row in epoch_report.get("samples", []):
        visuals = row.get("visuals", {})
        figures = []
        for label in ("real_A", "fake_B", "real_B", "fake_A", "rec_A", "rec_B"):
            rel = visuals.get(label)
            if rel:
                src = rel.replace("\\", "/")
                if src.startswith(epoch_dir.name + "/"):
                    src = src[len(epoch_dir.name) + 1 :]
                figures.append(f"<figure><img src='{src}' alt='{label}'><figcaption>{label}</figcaption></figure>")
        cards.append(
            "<section class='card'>"
            f"<h2>{row.get('selection_role', '')}: {row.get('sample_id', '')}</h2>"
            f"<p>{row.get('input_path', '')}</p><div class='grid'>{''.join(figures)}</div></section>"
        )
    path = epoch_dir / "index.html"
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Validation Monitor Epoch</title>"
        "<style>body{font-family:Arial,sans-serif;margin:2rem;background:#f8fafc;color:#0f172a}"
        ".card{background:white;border:1px solid #cbd5e1;border-radius:14px;padding:1rem;margin-bottom:1rem}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:1rem}"
        "img{width:100%;height:220px;object-fit:contain;background:#eef2ff;border-radius:10px}"
        "figcaption{text-align:center;font-weight:700}</style></head><body>"
        f"<h1>Validation Monitor - Epoch {epoch_report.get('epoch')}</h1>{''.join(cards)}</body></html>",
        encoding="utf-8",
    )
    return path


def _write_index_html(monitor_dir: Path, report: dict[str, Any]) -> Path:
    epochs = report.get("epochs", [])
    rows = []
    fixed_by_sample: dict[str, list[dict[str, str]]] = {}
    for item in epochs:
        metrics = item.get("metrics", {})
        aggregate = metrics.get("aggregate", {}) if isinstance(metrics, dict) else {}
        rows.append(
            "<tr>"
            f"<td><a href='epoch_{int(item.get('epoch', 0)):03d}/index.html'>{item.get('epoch')}</a></td>"
            f"<td>{item.get('sample_count', 0)}</td>"
            f"<td>{aggregate.get('mae_mean', '')}</td>"
            f"<td>{aggregate.get('psnr_mean', '')}</td>"
            f"<td>{aggregate.get('ssim_mean', '')}</td>"
            "</tr>"
        )
        for sample in item.get("samples", []):
            if sample.get("selection_role") != "fixed":
                continue
            rel = sample.get("visuals", {}).get("fake_B", "")
            if rel:
                fixed_by_sample.setdefault(str(sample.get("sample_id", "")), []).append(
                    {"epoch": str(item.get("epoch", "")), "image": rel}
                )
    progression_cards = []
    for sample_id, images in sorted(fixed_by_sample.items()):
        figures = []
        for image in images:
            src = str(image["image"]).replace("\\", "/")
            figures.append(
                f"<figure><img src='{src}' alt='epoch {image['epoch']} {sample_id}'><figcaption>epoch {image['epoch']}</figcaption></figure>"
            )
        progression_cards.append(
            f"<section class='card'><h2>Fixed sample: {sample_id}</h2><div class='grid'>{''.join(figures)}</div></section>"
        )
    path = monitor_dir / "index.html"
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>MicroI2I Validation Monitor</title>"
        "<style>body{font-family:Arial,sans-serif;margin:2rem;background:#f8fafc;color:#0f172a}"
        "h1{font-size:2.2rem}.card{background:white;border:1px solid #cbd5e1;border-radius:14px;padding:1rem;margin:1rem 0}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem}"
        "img{width:100%;height:220px;object-fit:contain;background:#eef2ff;border-radius:10px}"
        "figcaption{text-align:center;font-weight:700;color:#334155}"
        "table{border-collapse:collapse;width:100%;background:white;border-radius:12px;overflow:hidden}"
        "th,td{border:1px solid #cbd5e1;padding:.65rem;text-align:left}th{background:#dbeafe}"
        ".note{color:#475569}</style></head><body>"
        "<h1>Training Validation Monitor</h1>"
        "<p class='note'>Fixed samples are repeated across epochs for longitudinal review; random samples add coverage.</p>"
        "<h2>Epoch Metrics</h2>"
        "<table><thead><tr><th>Epoch</th><th>Samples</th><th>MAE</th><th>PSNR</th><th>SSIM</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table><h2>Fixed Sample Progression</h2>{''.join(progression_cards)}</body></html>",
        encoding="utf-8",
    )
    return path


def _copy_manifest_targets(manifest: dict[str, Any], selected: list[dict[str, Any]], epoch_dir: Path) -> None:
    selected_dir = epoch_dir / "selected_inputs"
    selected_dir.mkdir(parents=True, exist_ok=True)
    for sample in selected:
        source = Path(str(sample.get("input_path", "")))
        if source.exists():
            shutil.copy2(source, selected_dir / f"{_safe_sample_name(sample)}{source.suffix}")


def run_epoch_validation_monitor(model: Any, opt: Any, epoch: int) -> None:
    """Run configured validation monitor at epoch end inside legacy training."""

    config_path = os.environ.get(MONITOR_ENV_VAR, "")
    if not config_path:
        return
    manifest_path = Path(config_path)
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("enabled", True):
        return
    frequency = max(int(manifest.get("policy", {}).get("eval_frequency_epochs", 1)), 1)
    if int(epoch) % frequency != 0:
        return

    run_dir = Path(str(manifest["run_dir"]))
    monitor_dir = run_dir / "validation_monitor"
    epoch_dir = monitor_dir / f"epoch_{int(epoch):03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    selected = select_epoch_samples(
        manifest.get("pool", {}).get("samples", []),
        manifest.get("fixed_samples", []),
        total_count=int(manifest.get("policy", {}).get("total_count", 5)),
        seed=int(manifest.get("policy", {}).get("seed", 42)),
        epoch=int(epoch),
    )
    _copy_manifest_targets(manifest, selected, epoch_dir)

    from data import create_dataset

    val_opt = copy.copy(opt)
    val_opt.phase = manifest.get("pool", {}).get("phase", "val")
    val_opt.serial_batches = True
    val_opt.no_flip = True
    val_opt.batch_size = 1
    val_opt.num_threads = 0
    val_opt.max_dataset_size = max(len(manifest.get("pool", {}).get("samples", [])), 1)
    dataset_wrapper = create_dataset(val_opt)
    dataset = dataset_wrapper.dataset
    path_attr = "A_paths" if hasattr(dataset, "A_paths") else "AB_paths"
    dataset_paths = [str(Path(path).resolve()) for path in getattr(dataset, path_attr, [])]
    path_to_index = {path.lower(): index for index, path in enumerate(dataset_paths)}

    if hasattr(model, "eval"):
        model.eval()
    sample_rows: list[dict[str, Any]] = []
    for sample in selected:
        index = path_to_index.get(str(Path(str(sample.get("input_path", ""))).resolve()).lower())
        if index is None:
            continue
        model.set_input(_batchify(dataset[index]))
        model.test()
        visuals = _save_visuals(model.get_current_visuals(), sample, epoch_dir)
        sample_rows.append({**sample, "visuals": visuals})

    for name in getattr(model, "model_names", []):
        net = getattr(model, "net" + name, None)
        if net is not None and hasattr(net, "train"):
            net.train()

    pred_dir = epoch_dir / "fake_B"
    target_dir = epoch_dir / "real_B"
    dataset_mode = str(manifest.get("pool", {}).get("dataset_mode", ""))
    if dataset_mode == "unaligned":
        metrics = {
            "status": "skipped",
            "reason": "CycleGAN validation samples are unpaired; paired metrics require explicit references",
            "samples": [],
            "aggregate": {},
        }
    else:
        metrics = evaluate_paired_directories(pred_dir, target_dir) if pred_dir.exists() and target_dir.exists() else {
        "status": "skipped",
        "reason": "paired fake_B/real_B outputs were not available",
        "samples": [],
        "aggregate": {},
        }
    epoch_report = {
        "schema_version": "microi2i.validation_monitor_epoch.v1",
        "epoch": int(epoch),
        "sample_count": len(sample_rows),
        "samples": sample_rows,
        "metrics": metrics,
    }
    (epoch_dir / "report.json").write_text(json.dumps(epoch_report, indent=2, sort_keys=True), encoding="utf-8")
    if bool(manifest.get("policy", {}).get("export_html", True)):
        _write_epoch_html(epoch_dir, epoch_report)

    report_path = monitor_dir / "report.json"
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {"schema_version": "microi2i.validation_monitor_report.v1", "epochs": []}
    )
    report["epochs"] = [row for row in report.get("epochs", []) if row.get("epoch") != int(epoch)] + [epoch_report]
    report["epochs"].sort(key=lambda row: int(row.get("epoch", 0)))
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if bool(manifest.get("policy", {}).get("export_html", True)):
        _write_index_html(monitor_dir, report)

    manifest.setdefault("epoch_selections", []).append({"epoch": int(epoch), "samples": selected})
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
