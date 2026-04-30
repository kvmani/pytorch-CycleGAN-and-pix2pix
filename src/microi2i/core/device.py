"""Runtime device policy for CLI-only CPU/GPU execution."""

from __future__ import annotations

import importlib.util
from typing import Any

from microi2i.core.contracts import RuntimeConfig


def _cuda_available() -> bool:
    """Return whether PyTorch reports an available CUDA device."""

    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device_policy(runtime: RuntimeConfig) -> dict[str, Any]:
    """Resolve runtime device settings into effective legacy GPU arguments."""

    cuda_available = _cuda_available()
    if runtime.device == "cpu":
        effective_device = "cpu"
        effective_gpu_ids = "-1"
    elif runtime.device == "cuda":
        if not cuda_available:
            raise RuntimeError("runtime.device=cuda was requested, but CUDA is not available")
        effective_device = "cuda"
        effective_gpu_ids = runtime.gpu_ids if runtime.gpu_ids != "-1" else "0"
    else:
        if runtime.require_cuda and not cuda_available:
            raise RuntimeError("runtime.require_cuda=true, but CUDA is not available")
        if cuda_available and runtime.gpu_ids != "-1":
            effective_device = "cuda"
            effective_gpu_ids = runtime.gpu_ids
        else:
            effective_device = "cpu"
            effective_gpu_ids = "-1"

    return {
        "schema_version": "microi2i.device_policy.v1",
        "requested_device": runtime.device,
        "requested_gpu_ids": runtime.gpu_ids,
        "require_cuda": runtime.require_cuda,
        "cuda_available": cuda_available,
        "effective_device": effective_device,
        "effective_gpu_ids": effective_gpu_ids,
    }
