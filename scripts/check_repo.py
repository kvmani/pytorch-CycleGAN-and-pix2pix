"""Run the local MicroI2I quality gate."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(label: str, command: list[str]) -> int:
    print(f"\n== {label} ==")
    print(" ".join(command))
    proc = subprocess.run(command, cwd=str(ROOT), check=False)
    if proc.returncode != 0:
        print(f"{label} failed with exit code {proc.returncode}", file=sys.stderr)
    return int(proc.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run tests, registry validation, and docs build")
    parser.add_argument("--skip-docs", action="store_true", help="Skip Sphinx docs build")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest")
    parser.add_argument("--include-smoke", action="store_true", help="Run dry-run smoke workflow checks")
    args = parser.parse_args(argv)

    checks: list[tuple[str, list[str]]] = []
    if not args.skip_tests:
        checks.append(("tests", [sys.executable, "-m", "pytest", "tests"]))
    checks.append(("registry validation", [sys.executable, "scripts/microi2i_cli.py", "validate-registry"]))
    if not args.skip_docs:
        checks.append(("docs html build", [sys.executable, "scripts/build_docs.py", "--html-only"]))
    if args.include_smoke:
        checks.append(("smoke gate", [sys.executable, "scripts/smoke_gate.py"]))

    for label, command in checks:
        exit_code = _run(label, command)
        if exit_code != 0:
            return exit_code

    print("\nRepository quality gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
