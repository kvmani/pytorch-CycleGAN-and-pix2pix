from __future__ import annotations

from microi2i.plugins.registry import load_model_registry, validate_model_registry


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
