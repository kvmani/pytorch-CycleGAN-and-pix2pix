from __future__ import annotations

from PIL import Image

from microi2i.core.contracts import DatasetPrepareConfig
from microi2i.dataops.dataset_prepare import prepare_dataset


def _write_image(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (120, 80, 40)).save(path)


def test_prepare_paired_dataset_materializes_pix2pix_layout(tmp_path) -> None:
    source = tmp_path / "source"
    for index in range(6):
        _write_image(source / f"specimen_{index}" / f"pair_{index}.png")
    output = tmp_path / "prepared"
    cfg = DatasetPrepareConfig.from_mapping(
        {
            "schema_version": "microi2i.dataset_prepare_config.v1",
            "dataset_id": "paired_test",
            "task_type": "paired_translation",
            "source_roots": [str(source)],
            "output_dataset_dir": str(output),
            "split_policy": {"train_ratio": 0.5, "val_ratio": 0.25, "test_ratio": 0.25, "seed": 7},
            "leakage_group_policy": {"mode": "parent", "required": True},
        }
    )

    manifest = prepare_dataset(cfg, repo_root=tmp_path)

    assert manifest["layout"] == "pix2pix_aligned"
    assert sum(manifest["sample_counts"].values()) == 6
    assert any((output / split).exists() for split in ("train", "val", "test"))


def test_prepare_unpaired_dataset_materializes_cyclegan_layout(tmp_path) -> None:
    domain_a = tmp_path / "domain_a"
    domain_b = tmp_path / "domain_b"
    for index in range(4):
        _write_image(domain_a / f"group_{index}" / f"a_{index}.png")
        _write_image(domain_b / f"group_{index}" / f"b_{index}.png")
    output = tmp_path / "prepared"
    cfg = DatasetPrepareConfig.from_mapping(
        {
            "schema_version": "microi2i.dataset_prepare_config.v1",
            "dataset_id": "unpaired_test",
            "task_type": "unpaired_translation",
            "source_roots": [str(domain_a), str(domain_b)],
            "output_dataset_dir": str(output),
            "split_policy": {"train_ratio": 0.5, "val_ratio": 0.25, "test_ratio": 0.25, "seed": 3},
            "leakage_group_policy": {"mode": "parent", "required": True},
        }
    )

    manifest = prepare_dataset(cfg, repo_root=tmp_path)

    assert manifest["layout"] == "cyclegan_unaligned"
    assert sum(count for name, count in manifest["sample_counts"].items() if name.endswith("A")) == 4
    assert sum(count for name, count in manifest["sample_counts"].items() if name.endswith("B")) == 4
    assert (output / "trainA").exists()
    assert (output / "trainB").exists()


def test_prepare_dataset_applies_preprocessing_policy(tmp_path) -> None:
    source = tmp_path / "source"
    _write_image(source / "specimen_1" / "pair.png")
    output = tmp_path / "prepared"
    cfg = DatasetPrepareConfig.from_mapping(
        {
            "schema_version": "microi2i.dataset_prepare_config.v1",
            "dataset_id": "preprocess_test",
            "task_type": "paired_translation",
            "source_roots": [str(source)],
            "output_dataset_dir": str(output),
            "split_policy": {"train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0, "seed": 7},
            "preprocessing": {"center_crop": [6, 6], "resize": [4, 4], "color_mode": "grayscale"},
            "leakage_group_policy": {"mode": "parent", "required": True},
        }
    )

    manifest = prepare_dataset(cfg, repo_root=tmp_path)
    prepared = Image.open(output / "train" / "specimen_1" / "pair.png")

    assert prepared.size == (4, 4)
    assert prepared.mode == "L"
    assert manifest["copied_files"]["train"][0]["global_id"]
