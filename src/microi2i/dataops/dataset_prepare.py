"""Deterministic microscopy dataset materialization."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import hashlib
from pathlib import Path
import random
import re
import shutil
from typing import Any, Iterable

from PIL import Image

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


def _size_pair(value: Any, *, field_name: str) -> tuple[int, int] | None:
    if value in (None, "", []):
        return None
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"preprocessing.{field_name} must be [width, height]")
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"preprocessing.{field_name} must contain positive dimensions")
    return width, height


def _crop_box(image: Image.Image, size: tuple[int, int], *, rng: random.Random | None = None) -> tuple[int, int, int, int]:
    width, height = image.size
    crop_width, crop_height = min(size[0], width), min(size[1], height)
    if rng is None:
        left = max((width - crop_width) // 2, 0)
        top = max((height - crop_height) // 2, 0)
    else:
        left = rng.randint(0, max(width - crop_width, 0))
        top = rng.randint(0, max(height - crop_height, 0))
    return left, top, left + crop_width, top + crop_height


def _letterbox(image: Image.Image, size: tuple[int, int], *, fill: int) -> Image.Image:
    target_width, target_height = size
    scale = min(target_width / image.size[0], target_height / image.size[1])
    new_size = (max(1, int(round(image.size[0] * scale))), max(1, int(round(image.size[1] * scale))))
    resized = image.resize(new_size)
    canvas = Image.new(image.mode, size, color=fill)
    offset = ((target_width - new_size[0]) // 2, (target_height - new_size[1]) // 2)
    canvas.paste(resized, offset)
    return canvas


def _pad_to(image: Image.Image, size: tuple[int, int], *, fill: int) -> Image.Image:
    target_width, target_height = size
    if image.size[0] > target_width or image.size[1] > target_height:
        return image
    canvas = Image.new(image.mode, size, color=fill)
    offset = ((target_width - image.size[0]) // 2, (target_height - image.size[1]) // 2)
    canvas.paste(image, offset)
    return canvas


def _preprocess_image(source: Path, target: Path, policy: dict[str, Any], *, seed: int, relative_key: str) -> None:
    if not policy:
        shutil.copy2(source, target)
        return

    with Image.open(source) as opened:
        image = opened.copy()
    color_mode = str(policy.get("color_mode", "preserve")).lower()
    if color_mode in {"rgb", "3ch", "three_channel"}:
        image = image.convert("RGB")
    elif color_mode in {"grayscale", "gray", "l"}:
        image = image.convert("L")
    elif color_mode != "preserve":
        raise ValueError("preprocessing.color_mode must be preserve, rgb, or grayscale")

    center_crop = _size_pair(policy.get("center_crop"), field_name="center_crop")
    if center_crop is not None:
        image = image.crop(_crop_box(image, center_crop))
    random_crop = _size_pair(policy.get("random_crop"), field_name="random_crop")
    if random_crop is not None:
        digest = hashlib.sha256(f"{seed}:{relative_key}".encode("utf-8")).hexdigest()
        image = image.crop(_crop_box(image, random_crop, rng=random.Random(int(digest[:12], 16))))
    fill = int(policy.get("fill", 0))
    letterbox = _size_pair(policy.get("letterbox"), field_name="letterbox")
    if letterbox is not None:
        image = _letterbox(image, letterbox, fill=fill)
    pad_to = _size_pair(policy.get("pad_to"), field_name="pad_to")
    if pad_to is not None:
        image = _pad_to(image, pad_to, fill=fill)
    resize = _size_pair(policy.get("resize"), field_name="resize")
    if resize is not None:
        image = image.resize(resize)
    image.save(target)


def _copy_files(
    files: Iterable[Path],
    root: Path,
    target_root: Path,
    *,
    preprocessing: dict[str, Any],
    seed: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in files:
        relative = path.relative_to(root)
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        _preprocess_image(path, target, preprocessing, seed=seed, relative_key=relative.as_posix())
        rows.append({"source": str(path), "target": str(target), "global_id": hashlib.sha256(str(path).encode("utf-8")).hexdigest()})
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
            split: _copy_files(
                paths,
                source_root,
                output_dir / split,
                preprocessing=config.preprocessing,
                seed=config.split_policy.seed,
            )
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
                copies[f"{split}{domain}"] = _copy_files(
                    paths,
                    root,
                    output_dir / f"{split}{domain}",
                    preprocessing=config.preprocessing,
                    seed=config.split_policy.seed,
                )
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
