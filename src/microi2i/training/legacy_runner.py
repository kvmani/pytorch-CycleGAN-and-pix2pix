"""Legacy training command construction."""

from __future__ import annotations

import sys
from pathlib import Path

from microi2i.core.contracts import ScriptWorkflowConfig


def build_training_command(config: ScriptWorkflowConfig, *, repo_root: Path) -> list[str]:
    """Build the command that runs the current legacy training script."""

    script = Path(config.command.legacy_script)
    if not script.is_absolute():
        script = repo_root / script
    return [sys.executable, str(script), *config.command.legacy_args]
