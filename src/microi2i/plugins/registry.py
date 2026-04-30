"""Model registry loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
}


def load_model_registry(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Model registry root must be an object")
    return payload


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
    return errors
