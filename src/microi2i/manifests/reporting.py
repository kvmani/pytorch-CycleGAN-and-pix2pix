"""Run and artifact manifest helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _git_state(root: Path) -> dict[str, Any]:
    def run_git(args: list[str]) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=str(root), text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""

    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "dirty": bool(run_git(["status", "--short"])),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class RunContext:
    run_id: str
    workflow: str
    run_dir: Path
    config_path: str
    resolved_config: dict[str, Any]
    command: list[str]
    started_utc: str = field(default_factory=utc_now)
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    def add_artifact(self, relative_path: str, kind: str, description: str, payload: dict[str, Any]) -> Path:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        item = {
            "path": relative_path,
            "kind": kind,
            "description": description,
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "sha256": _sha256(path) if path.exists() else "",
        }
        self.artifacts.append(item)
        return path


def start_run(
    *,
    workflow: str,
    config_path: str,
    resolved_config: dict[str, Any],
    output_root: Path,
    command: list[str],
) -> RunContext:
    run_id = f"{workflow}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(
        run_id=run_id,
        workflow=workflow,
        run_dir=run_dir,
        config_path=config_path,
        resolved_config=resolved_config,
        command=list(command),
    )


def finalize_run(run: RunContext, *, status: str, exit_code: int) -> None:
    root = Path(__file__).resolve().parents[3]
    finished = utc_now()
    report = {
        "schema_version": "microi2i.run_report.v1",
        "run_id": run.run_id,
        "workflow": run.workflow,
        "status": status,
        "exit_code": exit_code,
        "started_utc": run.started_utc,
        "finished_utc": finished,
    }
    run.add_artifact("report.json", "report", "Workflow report", report)
    artifact_manifest = {
        "schema_version": "microi2i.artifact_manifest.v1",
        "run_id": run.run_id,
        "created_utc": finished,
        "files": run.artifacts,
    }
    (run.run_dir / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "microi2i.run_manifest.v1",
        "run_id": run.run_id,
        "workflow": run.workflow,
        "started_utc": run.started_utc,
        "finished_utc": finished,
        "status": status,
        "exit_code": exit_code,
        "command": run.command,
        "config_path": run.config_path,
        "resolved_config": run.resolved_config,
        "code_state": _git_state(root),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.node(),
            "pid": os.getpid(),
        },
        "artifacts": run.artifacts,
    }
    (run.run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
