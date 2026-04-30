from __future__ import annotations

import scripts.smoke_gate as smoke_gate


def test_smoke_gate_dry_run_command_plan(monkeypatch) -> None:
    calls: list[tuple[str, list[str]]] = []

    def fake_run(label: str, command: list[str]) -> int:
        calls.append((label, command))
        return 0

    monkeypatch.setattr(smoke_gate, "_run", fake_run)

    assert smoke_gate.main(["--skip-data"]) == 0
    assert len(calls) == 3
    assert all("--dry-run" in command for label, command in calls if "train" in label)
    assert any("infer" in command for _, command in calls)


def test_smoke_gate_real_training_reports_missing_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(smoke_gate, "_missing_real_training_dependencies", lambda: ["dominate"])

    assert smoke_gate.main(["--run-training"]) == 2
