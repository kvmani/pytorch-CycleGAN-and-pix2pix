"""Model execution adapters for CLI-only MicroI2I workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from microi2i.core.contracts import ScriptWorkflowConfig
from microi2i.core.device import resolve_device_policy
from microi2i.inference.legacy_runner import build_inference_command
from microi2i.training.legacy_runner import build_training_command


class ModelBackend(Protocol):
    """Common contract for model execution adapters."""

    backend_id: str
    display_name: str
    model_family: str

    def train_command(
        self,
        config: ScriptWorkflowConfig,
        *,
        repo_root: Path,
        resolved_config: dict[str, Any],
    ) -> list[str]:
        """Build a training command for this backend."""

    def infer_command(self, config: ScriptWorkflowConfig, *, repo_root: Path) -> list[str]:
        """Build an inference command for this backend."""

    def metadata(self) -> dict[str, Any]:
        """Return backend metadata for manifests and reports."""


def _replace_option(args: list[str], key: str, value: str) -> list[str]:
    flag = f"--{key}"
    output: list[str] = []
    index = 0
    replaced = False
    while index < len(args):
        token = args[index]
        if token == flag:
            output.extend([flag, value])
            replaced = True
            index += 1
            if index < len(args) and not args[index].startswith("--"):
                index += 1
        else:
            output.append(token)
            index += 1
    if not replaced:
        output.extend([flag, value])
    return output


def _apply_device_to_command(command: list[str], device_policy: dict[str, Any]) -> list[str]:
    if len(command) <= 2:
        return list(command)
    return [*command[:2], *_replace_option(command[2:], "gpu_ids", str(device_policy["effective_gpu_ids"]))]


@dataclass(frozen=True)
class LegacyScriptBackend:
    """Adapter around the existing upstream train.py/test.py scripts."""

    backend_id: str
    display_name: str
    model_family: str

    def train_command(
        self,
        config: ScriptWorkflowConfig,
        *,
        repo_root: Path,
        resolved_config: dict[str, Any],
    ) -> list[str]:
        command = build_training_command(config, repo_root=repo_root, resolved_config=resolved_config)
        return _apply_device_to_command(command, resolve_device_policy(config.runtime))

    def infer_command(self, config: ScriptWorkflowConfig, *, repo_root: Path) -> list[str]:
        command = build_inference_command(config, repo_root=repo_root)
        return _apply_device_to_command(command, resolve_device_policy(config.runtime))

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": "microi2i.model_backend.v1",
            "backend_id": self.backend_id,
            "display_name": self.display_name,
            "model_family": self.model_family,
            "execution": "legacy_script",
        }


BACKENDS: dict[str, ModelBackend] = {
    "legacy_pix2pix": LegacyScriptBackend("legacy_pix2pix", "Legacy pix2pix adapter", "pix2pix"),
    "legacy_cyclegan": LegacyScriptBackend("legacy_cyclegan", "Legacy CycleGAN adapter", "cycle_gan"),
}


def infer_backend_id(config: ScriptWorkflowConfig, *, workflow: str) -> str:
    """Infer a legacy backend when older configs do not yet name one."""

    if config.model_backend:
        return config.model_backend
    args = config.command.legacy_args
    if workflow == "training":
        for index, token in enumerate(args[:-1]):
            if token == "--model":
                model = args[index + 1]
                if model == "pix2pix":
                    return "legacy_pix2pix"
                if model == "cycle_gan":
                    return "legacy_cyclegan"
    return "legacy_pix2pix"


def get_model_backend(backend_id: str) -> ModelBackend:
    """Return a registered model backend or fail with a clear message."""

    try:
        return BACKENDS[backend_id]
    except KeyError as exc:
        known = ", ".join(sorted(BACKENDS))
        raise ValueError(f"unknown model_backend {backend_id!r}; expected one of: {known}") from exc


def list_model_backends() -> list[dict[str, Any]]:
    """Return registered backend metadata."""

    return [backend.metadata() for backend in BACKENDS.values()]
