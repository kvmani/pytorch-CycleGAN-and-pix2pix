from __future__ import annotations

from pathlib import Path

from microi2i.core.contracts import ScriptWorkflowConfig
from microi2i.inference.legacy_runner import build_inference_command
from microi2i.training.legacy_runner import build_training_command


def test_training_command_is_built_from_structured_config(tmp_path) -> None:
    config = ScriptWorkflowConfig.from_mapping(
        {
            "schema_version": "microi2i.train_config.v1",
            "training": {"legacy_script": "train.py", "legacy_args": ["--model", "pix2pix"]},
        },
        section="training",
    )

    command = build_training_command(config, repo_root=tmp_path)

    assert command[1] == str(Path(tmp_path) / "train.py")
    assert command[-2:] == ["--model", "pix2pix"]


def test_inference_command_is_built_from_structured_config(tmp_path) -> None:
    config = ScriptWorkflowConfig.from_mapping(
        {
            "schema_version": "microi2i.inference_config.v1",
            "inference": {"legacy_script": "test.py", "legacy_args": ["--model", "test"]},
        },
        section="inference",
    )

    command = build_inference_command(config, repo_root=tmp_path)

    assert command[1] == str(Path(tmp_path) / "test.py")
    assert command[-2:] == ["--model", "test"]
