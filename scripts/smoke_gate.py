"""Run CPU-safe MicroI2I smoke workflow checks."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REAL_TRAINING_DEPENDENCIES = ["torch", "torchvision", "dominate"]


def _missing_real_training_dependencies() -> list[str]:
    return [name for name in REAL_TRAINING_DEPENDENCIES if importlib.util.find_spec(name) is None]


def _run(label: str, command: list[str]) -> int:
    print(f"\n== {label} ==")
    print(" ".join(command))
    process = subprocess.run(command, cwd=str(ROOT), check=False)
    if process.returncode != 0:
        print(f"{label} failed with exit code {process.returncode}", file=sys.stderr)
    return int(process.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run MicroI2I smoke workflow checks")
    parser.add_argument("--run-training", action="store_true", help="Run real tiny CPU training instead of dry-runs")
    parser.add_argument("--skip-data", action="store_true", help="Skip smoke dataset creation")
    args = parser.parse_args(argv)
    if args.run_training:
        missing = _missing_real_training_dependencies()
        if missing:
            print(
                "Real smoke training requires missing dependencies: " + ", ".join(missing),
                file=sys.stderr,
            )
            return 2

    checks: list[tuple[str, list[str]]] = []
    if not args.skip_data:
        checks.append(
            (
                "create smoke data",
                [sys.executable, "scripts/microi2i_cli.py", "create-smoke-data", "--config", "configs/smoke/default.yml"],
            )
        )
    train_suffix = [] if args.run_training else ["--dry-run"]
    checks.extend(
        [
            (
                "pix2pix smoke train",
                [
                    sys.executable,
                    "scripts/microi2i_cli.py",
                    "train",
                    "--config",
                    "configs/train/pix2pix.smoke.yml",
                    *train_suffix,
                ],
            ),
            (
                "cyclegan smoke train",
                [
                    sys.executable,
                    "scripts/microi2i_cli.py",
                    "train",
                    "--config",
                    "configs/train/cyclegan.smoke.yml",
                    *train_suffix,
                ],
            ),
            (
                "folder inference smoke dry-run",
                [
                    sys.executable,
                    "scripts/microi2i_cli.py",
                    "infer",
                    "--config",
                    "configs/inference/folder.default.yml",
                    "--dry-run",
                ],
            ),
        ]
    )

    for label, command in checks:
        exit_code = _run(label, command)
        if exit_code != 0:
            return exit_code
    print("\nSmoke gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
