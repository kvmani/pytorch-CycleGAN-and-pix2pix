"""YAML configuration loading and dotted-key overrides."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def _coerce_value(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply dotted-key overrides such as ``training.n_epochs=10``."""

    result = deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be KEY=VALUE, got {override!r}")
        key, raw_value = override.split("=", 1)
        parts = [part for part in key.split(".") if part]
        if not parts:
            raise ValueError(f"Override key cannot be empty: {override!r}")
        cursor: Any = result
        for part in parts[:-1]:
            if isinstance(cursor, list):
                if not part.isdigit():
                    raise ValueError(f"List override segment must be numeric: {part}")
                index = int(part)
                if index >= len(cursor):
                    raise ValueError(f"List override index out of range: {part}")
                next_value = cursor[index]
            elif isinstance(cursor, dict):
                next_value = cursor.setdefault(part, {})
            else:
                raise ValueError(f"Cannot set nested override under scalar key: {part}")
            if not isinstance(next_value, dict | list):
                raise ValueError(f"Cannot set nested override under non-container key: {part}")
            cursor = next_value
        final = parts[-1]
        if isinstance(cursor, list):
            if not final.isdigit():
                raise ValueError(f"List override segment must be numeric: {final}")
            index = int(final)
            if index >= len(cursor):
                raise ValueError(f"List override index out of range: {final}")
            cursor[index] = _coerce_value(raw_value)
        elif isinstance(cursor, dict):
            cursor[final] = _coerce_value(raw_value)
        else:
            raise ValueError(f"Cannot set override under scalar key: {final}")
    return result
