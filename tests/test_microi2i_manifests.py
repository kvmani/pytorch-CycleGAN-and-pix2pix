from __future__ import annotations

import json

from microi2i.manifests.reporting import finalize_run, start_run


def test_run_manifest_and_artifact_manifest_are_written(tmp_path) -> None:
    run = start_run(
        workflow="test_workflow",
        config_path="config.yml",
        resolved_config={"a": 1},
        output_root=tmp_path,
        command=["microi2i", "test"],
    )
    run.add_artifact("custom.json", "custom", "Custom payload", {"ok": True})

    finalize_run(run, status="success", exit_code=0)

    manifest = json.loads((run.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    artifacts = json.loads((run.run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    report = json.loads((run.run_dir / "report.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "microi2i.run_manifest.v1"
    assert manifest["status"] == "success"
    assert artifacts["schema_version"] == "microi2i.artifact_manifest.v1"
    assert any(item["path"] == "custom.json" for item in artifacts["files"])
    assert report["workflow"] == "test_workflow"
