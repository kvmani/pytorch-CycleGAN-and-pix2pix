"""Legacy inference command construction and output packaging."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from microi2i.core.contracts import ScriptWorkflowConfig


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def build_inference_command(config: ScriptWorkflowConfig, *, repo_root: Path) -> list[str]:
    """Build the command that runs the current legacy inference script."""

    script = Path(config.command.legacy_script)
    if not script.is_absolute():
        script = repo_root / script
    return [sys.executable, str(script), *config.command.legacy_args]


def package_prediction_images(source_dir: str | Path, run_dir: Path) -> dict[str, object]:
    """Copy prediction images into a normalized run folder when a source exists."""

    source = Path(source_dir)
    predictions_dir = run_dir / "predictions"
    copied: list[str] = []
    if not source.exists() or not source.is_dir():
        return {
            "status": "skipped",
            "reason": "expected_output_dir does not exist",
            "source_dir": str(source),
            "prediction_count": 0,
            "predictions_dir": str(predictions_dir),
        }

    predictions_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            relative = path.relative_to(source)
            target = predictions_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied.append(str(target.relative_to(run_dir)))
    return {
        "status": "copied",
        "source_dir": str(source),
        "prediction_count": len(copied),
        "predictions_dir": str(predictions_dir),
        "files": copied,
    }
