from __future__ import annotations

import pytest

from microi2i.core.contracts import RuntimeConfig, ScriptWorkflowConfig
from microi2i.core.device import resolve_device_policy
from microi2i.models.backends import get_model_backend, infer_backend_id, list_model_backends


def test_model_backend_registry_resolves_legacy_adapters() -> None:
    ids = {item["backend_id"] for item in list_model_backends()}

    assert {"legacy_pix2pix", "legacy_cyclegan"} <= ids
    assert get_model_backend("legacy_pix2pix").metadata()["model_family"] == "pix2pix"


def test_unknown_model_backend_fails_clearly() -> None:
    with pytest.raises(ValueError, match="unknown model_backend"):
        get_model_backend("not_real")


def test_runtime_device_policy_cpu_forces_minus_one_gpu_ids() -> None:
    policy = resolve_device_policy(RuntimeConfig(device="cpu", gpu_ids="0"))

    assert policy["effective_device"] == "cpu"
    assert policy["effective_gpu_ids"] == "-1"


def test_runtime_device_policy_cuda_unavailable_fails() -> None:
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device_policy(RuntimeConfig(device="cuda", gpu_ids="0"))


def test_backend_train_command_applies_device_policy(tmp_path) -> None:
    cfg = {
        "schema_version": "microi2i.train_config.v1",
        "model_backend": "legacy_pix2pix",
        "runtime": {"device": "cpu", "gpu_ids": "0"},
        "training": {
            "legacy_script": "train.py",
            "legacy_args": ["--dataroot", "data", "--model", "pix2pix", "--gpu_ids", "0"],
        },
    }
    config = ScriptWorkflowConfig.from_mapping(cfg, section="training")
    backend = get_model_backend(infer_backend_id(config, workflow="training"))

    command = backend.train_command(config, repo_root=tmp_path, resolved_config=cfg)

    assert command[-2:] == ["--gpu_ids", "-1"]
