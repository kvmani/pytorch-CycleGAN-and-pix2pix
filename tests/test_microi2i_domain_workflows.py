from __future__ import annotations

import json

from microi2i.app.cli import main


def test_domain_workflow_dry_run_writes_report(tmp_path) -> None:
    exit_code = main(
        [
            "run-domain",
            "--config",
            "configs/domain/ebsd_make_cyclegan.default.yml",
            "--set",
            f"output_root={tmp_path.as_posix()}",
        ]
    )

    run_dir = next(tmp_path.iterdir())
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["schema_version"] == "microi2i.domain_workflow_report.v1"
    assert report["domain"] == "ebsd"
    assert report["status"] == "dry_run"
    assert (run_dir / "command.json").exists()
