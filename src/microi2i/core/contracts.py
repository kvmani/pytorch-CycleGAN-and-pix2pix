"""Typed workflow configuration contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _as_list(value: object, *, field_name: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return list(value)


def _as_dict(value: object, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return dict(value)


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime settings shared by train and inference workflows."""

    gpu_ids: str = "-1"
    seed: int = 42

    @classmethod
    def from_mapping(cls, payload: object) -> "RuntimeConfig":
        data = _as_dict(payload, field_name="runtime")
        return cls(gpu_ids=str(data.get("gpu_ids", "-1")), seed=int(data.get("seed", 42)))


@dataclass(frozen=True)
class LegacyCommandConfig:
    """Configuration for running an existing repository script."""

    legacy_script: str
    legacy_args: list[str] = field(default_factory=list)
    expected_output_dir: str = ""

    @classmethod
    def from_mapping(cls, payload: object, *, field_name: str) -> "LegacyCommandConfig":
        data = _as_dict(payload, field_name=field_name)
        script = str(data.get("legacy_script", "")).strip()
        if not script:
            raise ValueError(f"{field_name}.legacy_script is required")
        args = [str(item) for item in _as_list(data.get("legacy_args", []), field_name=f"{field_name}.legacy_args")]
        return cls(
            legacy_script=script,
            legacy_args=args,
            expected_output_dir=str(data.get("expected_output_dir", "")),
        )


@dataclass(frozen=True)
class WorkflowConfig:
    """Common top-level workflow fields."""

    schema_version: str
    output_root: str = "artifacts/runs"
    dry_run: bool = False

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "WorkflowConfig":
        return cls(
            schema_version=str(payload.get("schema_version", "")),
            output_root=str(payload.get("output_root", "artifacts/runs")),
            dry_run=bool(payload.get("dry_run", False)),
        )


@dataclass(frozen=True)
class ScriptWorkflowConfig:
    """Validated config for legacy-backed train/inference workflows."""

    base: WorkflowConfig
    runtime: RuntimeConfig
    command: LegacyCommandConfig

    @classmethod
    def from_mapping(cls, payload: dict[str, Any], *, section: str) -> "ScriptWorkflowConfig":
        return cls(
            base=WorkflowConfig.from_mapping(payload),
            runtime=RuntimeConfig.from_mapping(payload.get("runtime", {})),
            command=LegacyCommandConfig.from_mapping(payload.get(section, {}), field_name=section),
        )


@dataclass(frozen=True)
class SplitPolicy:
    """Deterministic split ratios and seed."""

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42

    @classmethod
    def from_mapping(cls, payload: object) -> "SplitPolicy":
        data = _as_dict(payload, field_name="split_policy")
        policy = cls(
            train_ratio=float(data.get("train_ratio", 0.8)),
            val_ratio=float(data.get("val_ratio", 0.1)),
            test_ratio=float(data.get("test_ratio", 0.1)),
            seed=int(data.get("seed", 42)),
        )
        total = policy.train_ratio + policy.val_ratio + policy.test_ratio
        if total <= 0:
            raise ValueError("split ratios must sum to a positive value")
        return policy


@dataclass(frozen=True)
class LeakageGroupPolicy:
    """Policy for assigning related samples to the same split."""

    mode: str = "parent"
    regex: str = ""
    required: bool = False

    @classmethod
    def from_mapping(cls, payload: object) -> "LeakageGroupPolicy":
        data = _as_dict(payload, field_name="leakage_group_policy")
        return cls(
            mode=str(data.get("mode", "parent")),
            regex=str(data.get("regex", "")),
            required=bool(data.get("required", False)),
        )


@dataclass(frozen=True)
class DatasetPrepareConfig:
    """Dataset preparation workflow config."""

    base: WorkflowConfig
    dataset_id: str
    task_type: str
    source_roots: list[str]
    output_dataset_dir: str
    split_policy: SplitPolicy
    preprocessing: dict[str, Any]
    leakage_group_policy: LeakageGroupPolicy

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "DatasetPrepareConfig":
        dataset_id = str(payload.get("dataset_id", "")).strip()
        if not dataset_id:
            raise ValueError("dataset_id is required")
        source_roots = [str(item) for item in _as_list(payload.get("source_roots", []), field_name="source_roots")]
        if not source_roots:
            raise ValueError("source_roots must contain at least one path")
        output_dataset_dir = str(payload.get("output_dataset_dir", Path("artifacts") / "prepared_datasets" / dataset_id))
        return cls(
            base=WorkflowConfig.from_mapping(payload),
            dataset_id=dataset_id,
            task_type=str(payload.get("task_type", "paired_translation")),
            source_roots=source_roots,
            output_dataset_dir=output_dataset_dir,
            split_policy=SplitPolicy.from_mapping(payload.get("split_policy", {})),
            preprocessing=_as_dict(payload.get("preprocessing", {}), field_name="preprocessing"),
            leakage_group_policy=LeakageGroupPolicy.from_mapping(payload.get("leakage_group_policy", {})),
        )


@dataclass(frozen=True)
class DatasetQAConfig:
    """Dataset quality-assurance workflow config."""

    base: WorkflowConfig
    dataset_id: str
    task_type: str
    source_roots: list[str]
    output_dir: str
    leakage_group_policy: LeakageGroupPolicy
    metadata: dict[str, Any]
    contact_sheet: dict[str, Any]

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "DatasetQAConfig":
        dataset_id = str(payload.get("dataset_id", "")).strip()
        if not dataset_id:
            raise ValueError("dataset_id is required")
        source_roots = [str(item) for item in _as_list(payload.get("source_roots", []), field_name="source_roots")]
        if not source_roots:
            raise ValueError("source_roots must contain at least one path")
        return cls(
            base=WorkflowConfig.from_mapping(payload),
            dataset_id=dataset_id,
            task_type=str(payload.get("task_type", "paired_translation")),
            source_roots=source_roots,
            output_dir=str(payload.get("output_dir", Path("artifacts") / "dataset_qa" / dataset_id)),
            leakage_group_policy=LeakageGroupPolicy.from_mapping(payload.get("leakage_group_policy", {})),
            metadata=_as_dict(payload.get("metadata", {}), field_name="metadata"),
            contact_sheet=_as_dict(payload.get("contact_sheet", {}), field_name="contact_sheet"),
        )
