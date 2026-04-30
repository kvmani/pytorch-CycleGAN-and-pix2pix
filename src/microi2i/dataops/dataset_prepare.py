"""Deterministic microscopy dataset materialization."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import random
import re
import shutil
from typing import Iterable

from microi2i.core.contracts import DatasetPrepareConfig, LeakageGroupPolicy, SplitPolicy


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _repo_path(value: str | Path, *, repo_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def _image_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"source root does not exist: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _leakage_group(path: Path, root: Path, policy: LeakageGroupPolicy) -> str:
    relative = path.relative_to(root)
    text = str(relative).replace("\\", "/")
    if policy.regex:
        match = re.search(policy.regex, text)
        if match:
            if match.groupdict():
                return next(iter(match.groupdict().values()))
            return match.group(1) if match.groups() else match.group(0)
        if policy.required:
            raise ValueError(f"no leakage regex match for {path}")
    if policy.mode in {"parent", "folder_or_specimen_id"}:
        return relative.parts[0] if len(relative.parts) > 1 else path.stem
    if policy.mode == "stem":
        return path.stem
    return path.stem


def _assign_splits(files: list[Path], root: Path, policy: SplitPolicy, leakage: LeakageGroupPolicy) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        grouped[_leakage_group(path, root, leakage)].append(path)
    groups = sorted(grouped)
    rng = random.Random(policy.seed)
    rng.shuffle(groups)

    total = policy.train_ratio + policy.val_ratio + policy.test_ratio
    train_cut = policy.train_ratio / total
    val_cut = (policy.train_ratio + policy.val_ratio) / total

    splits: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    n = len(groups)
    for index, group in enumerate(groups):
        fraction = (index + 1) / max(n, 1)
        if fraction <= train_cut:
            split = "train"
        elif fraction <= val_cut:
            split = "val"
        else:
            split = "test"
        splits[split].extend(grouped[group])

    if n and not splits["train"]:
        first_nonempty = next(name for name in ("val", "test") if splits[name])
        splits["train"].append(splits[first_nonempty].pop(0))
    return {name: sorted(paths) for name, paths in splits.items()}


def _copy_files(files: Iterable[Path], root: Path, target_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in files:
        relative = path.relative_to(root)
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        rows.append({"source": str(path), "target": str(target)})
    return rows


def prepare_dataset(config: DatasetPrepareConfig, *, repo_root: Path) -> dict[str, object]:
    """Materialize a paired pix2pix or unpaired CycleGAN dataset."""

    output_dir = _repo_path(config.output_dataset_dir, repo_root=repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.task_type == "paired_translation":
        source_root = _repo_path(config.source_roots[0], repo_root=repo_root)
        files = _image_files(source_root)
        splits = _assign_splits(files, source_root, config.split_policy, config.leakage_group_policy)
        copies = {
            split: _copy_files(paths, source_root, output_dir / split)
            for split, paths in splits.items()
        }
        layout = "pix2pix_aligned"
    elif config.task_type == "unpaired_translation":
        if len(config.source_roots) < 2:
            raise ValueError("unpaired_translation requires two source_roots")
        roots = [_repo_path(item, repo_root=repo_root) for item in config.source_roots[:2]]
        domain_names = ["A", "B"]
        copies = {}
        for root, domain in zip(roots, domain_names):
            splits = _assign_splits(_image_files(root), root, config.split_policy, config.leakage_group_policy)
            for split, paths in splits.items():
                copies[f"{split}{domain}"] = _copy_files(paths, root, output_dir / f"{split}{domain}")
        layout = "cyclegan_unaligned"
    else:
        raise ValueError(f"unsupported task_type: {config.task_type}")

    sample_counts = {name: len(rows) for name, rows in copies.items()}
    return {
        "schema_version": "microi2i.dataset_manifest.v1",
        "dataset_id": config.dataset_id,
        "task_type": config.task_type,
        "layout": layout,
        "source_roots": config.source_roots,
        "output_dataset_dir": str(output_dir),
        "split_policy": asdict(config.split_policy),
        "preprocessing": config.preprocessing,
        "leakage_group_policy": asdict(config.leakage_group_policy),
        "sample_counts": sample_counts,
        "copied_files": copies,
    }
