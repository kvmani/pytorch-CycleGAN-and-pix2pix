"""Model registry loading, validation, comparison, and lifecycle helpers."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


ALLOWED_MODEL_STATUSES = {"smoke", "candidate", "promoted", "deprecated"}
KNOWN_MODEL_BACKENDS = {"legacy_pix2pix", "legacy_cyclegan"}
REQUIRED_MODEL_FIELDS = {
    "model_id",
    "display_name",
    "model_family",
    "task_type",
    "framework",
    "checkpoint_path_hint",
    "input_assumptions",
    "training_dataset",
    "metrics",
    "scientific_use",
    "limitations",
    "status",
    "model_backend",
}


def load_model_registry(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Model registry root must be an object")
    return payload


def merge_local_registry_overlay(registry: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge machine-local registry metadata by model_id without mutating the base registry."""

    merged = deepcopy(registry)
    overlay_models = overlay.get("models", [])
    if not isinstance(overlay_models, list):
        raise ValueError("overlay.models must be a list")
    base_models = merged.get("models", [])
    if not isinstance(base_models, list):
        raise ValueError("registry.models must be a list")
    by_id = {record.get("model_id"): record for record in base_models if isinstance(record, dict)}
    for overlay_record in overlay_models:
        if not isinstance(overlay_record, dict):
            raise ValueError("overlay model records must be objects")
        model_id = overlay_record.get("model_id")
        if model_id not in by_id:
            raise KeyError(f"overlay references unknown model_id: {model_id}")
        target = by_id[model_id]
        local = target.setdefault("local_overlay", {})
        if not isinstance(local, dict):
            local = {}
            target["local_overlay"] = local
        local.update({key: value for key, value in overlay_record.items() if key != "model_id"})
    return merged


def load_model_registry_with_overlay(path: str | Path, overlay_path: str | Path = "") -> dict[str, Any]:
    """Load base registry and optionally merge a git-ignored local overlay."""

    registry = load_model_registry(path)
    if not overlay_path:
        return registry
    overlay_file = Path(overlay_path)
    if not overlay_file.exists():
        return registry
    overlay = load_model_registry(overlay_file)
    return merge_local_registry_overlay(registry, overlay)


def save_model_registry(registry: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_model_registry(registry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if registry.get("schema_version") != "microi2i.model_registry.v1":
        errors.append("registry.schema_version must be microi2i.model_registry.v1")
    models = registry.get("models")
    if not isinstance(models, list):
        errors.append("registry.models must be a list")
        return errors
    seen: set[str] = set()
    for index, record in enumerate(models):
        if not isinstance(record, dict):
            errors.append(f"models[{index}] must be an object")
            continue
        missing = sorted(REQUIRED_MODEL_FIELDS - set(record))
        if missing:
            errors.append(f"models[{index}] missing fields: {', '.join(missing)}")
        model_id = str(record.get("model_id", "")).strip()
        if not model_id:
            errors.append(f"models[{index}].model_id is required")
        elif model_id in seen:
            errors.append(f"duplicate model_id: {model_id}")
        seen.add(model_id)
        status = str(record.get("status", "")).strip()
        if status and status not in ALLOWED_MODEL_STATUSES:
            errors.append(f"models[{index}].status must be one of {sorted(ALLOWED_MODEL_STATUSES)}")
        backend = str(record.get("model_backend", "")).strip()
        if backend and backend not in KNOWN_MODEL_BACKENDS:
            errors.append(f"models[{index}].model_backend must be one of {sorted(KNOWN_MODEL_BACKENDS)}")
    return errors


def update_model_status(
    registry: dict[str, Any],
    *,
    model_id: str,
    status: str,
    note: str = "",
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a registry copy with one model lifecycle status updated."""

    if status not in ALLOWED_MODEL_STATUSES:
        raise ValueError(f"status must be one of {sorted(ALLOWED_MODEL_STATUSES)}")
    updated = deepcopy(registry)
    models = updated.get("models", [])
    if not isinstance(models, list):
        raise ValueError("registry.models must be a list")
    for record in models:
        if isinstance(record, dict) and record.get("model_id") == model_id:
            history = record.setdefault("lifecycle_history", [])
            if not isinstance(history, list):
                history = []
                record["lifecycle_history"] = history
            history.append(
                {
                    "changed_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                    "from_status": record.get("status", ""),
                    "to_status": status,
                    "note": note,
                    "metrics": metrics or {},
                }
            )
            record["status"] = status
            if metrics:
                existing_metrics = record.setdefault("metrics", {})
                if isinstance(existing_metrics, dict):
                    existing_metrics.update(metrics)
            return updated
    raise KeyError(f"model_id not found: {model_id}")


def _extract_report_metrics(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics", {})
    if isinstance(metrics, dict):
        aggregate = metrics.get("aggregate", {})
        if isinstance(aggregate, dict):
            return dict(aggregate)
    aggregate = report.get("aggregate", {})
    return dict(aggregate) if isinstance(aggregate, dict) else {}


def compare_run_reports(paths: list[str | Path], *, metric: str, lower_is_better: bool = True) -> dict[str, Any]:
    """Compare evaluation reports by a selected aggregate metric."""

    rows: list[dict[str, Any]] = []
    for path_value in paths:
        path = Path(path_value)
        report = json.loads(path.read_text(encoding="utf-8"))
        metrics = _extract_report_metrics(report)
        value = metrics.get(metric)
        rows.append(
            {
                "path": str(path),
                "schema_version": report.get("schema_version", ""),
                "status": report.get("status", ""),
                "metric": metric,
                "value": value,
                "sample_count": metrics.get("sample_count"),
            }
        )
    sortable = [row for row in rows if isinstance(row.get("value"), int | float)]
    sortable.sort(key=lambda item: float(item["value"]), reverse=not lower_is_better)
    missing = [row for row in rows if not isinstance(row.get("value"), int | float)]
    return {
        "schema_version": "microi2i.run_comparison.v1",
        "metric": metric,
        "lower_is_better": lower_is_better,
        "ranked": sortable + missing,
    }
