"""Tiny deterministic datasets for CPU-safe workflow smoke tests."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from microi2i.core.contracts import SmokeDatasetConfig


def _repo_path(value: str | Path, *, repo_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def _pattern(size: int, *, seed: int, domain: str) -> Image.Image:
    rng = random.Random(seed)
    grid_y, grid_x = np.mgrid[0:size, 0:size]
    base = ((grid_x * (seed % 7 + 1)) + (grid_y * (seed % 5 + 1))) % 255
    if domain == "B":
        base = 255 - base
    arr = np.stack(
        [
            base,
            np.roll(base, rng.randint(1, max(size - 1, 1)), axis=0),
            np.roll(base, rng.randint(1, max(size - 1, 1)), axis=1),
        ],
        axis=2,
    ).astype(np.uint8)
    image = Image.fromarray(arr)
    draw = ImageDraw.Draw(image)
    draw.rectangle([size // 4, size // 4, size // 2, size // 2], outline=(255, 255, 255), width=1)
    return image


def _write_pix2pix(root: Path, *, image_size: int, sample_count: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        count = sample_count if split == "train" else max(1, sample_count // 2)
        for index in range(count):
            left = _pattern(image_size, seed=seed + index, domain="A")
            right = _pattern(image_size, seed=seed + index, domain="B")
            paired = Image.new("RGB", (image_size * 2, image_size))
            paired.paste(left, (0, 0))
            paired.paste(right, (image_size, 0))
            path = split_dir / f"smoke_{index:03d}.png"
            paired.save(path)
            rows.append({"split": split, "path": str(path), "layout": "concatenated_ab"})
    return rows


def _write_cyclegan(root: Path, *, image_size: int, sample_count: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        count = sample_count if split == "train" else max(1, sample_count // 2)
        for domain in ("A", "B"):
            split_dir = root / f"{split}{domain}"
            split_dir.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                path = split_dir / f"smoke_{domain.lower()}_{index:03d}.png"
                _pattern(image_size, seed=seed + index + (1000 if domain == "B" else 0), domain=domain).save(path)
                rows.append({"split": f"{split}{domain}", "path": str(path), "domain": domain})
    return rows


def create_smoke_datasets(config: SmokeDatasetConfig, *, repo_root: Path) -> dict[str, Any]:
    """Create deterministic pix2pix and/or CycleGAN smoke datasets."""

    output_dir = _repo_path(config.output_dir, repo_root=repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, Any] = {}
    if config.include_pix2pix:
        pix2pix_root = output_dir / "pix2pix"
        rows = _write_pix2pix(
            pix2pix_root,
            image_size=config.image_size,
            sample_count=config.sample_count,
            seed=config.seed,
        )
        datasets["pix2pix"] = {
            "layout": "pix2pix_aligned",
            "dataroot": str(pix2pix_root),
            "sample_count": len(rows),
            "samples": rows,
        }
    if config.include_cyclegan:
        cyclegan_root = output_dir / "cyclegan"
        rows = _write_cyclegan(
            cyclegan_root,
            image_size=config.image_size,
            sample_count=config.sample_count,
            seed=config.seed,
        )
        datasets["cyclegan"] = {
            "layout": "cyclegan_unaligned",
            "dataroot": str(cyclegan_root),
            "sample_count": len(rows),
            "samples": rows,
        }
    manifest = {
        "schema_version": "microi2i.smoke_dataset_manifest.v1",
        "config": asdict(config),
        "output_dir": str(output_dir),
        "datasets": datasets,
    }
    (output_dir / "smoke_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest
