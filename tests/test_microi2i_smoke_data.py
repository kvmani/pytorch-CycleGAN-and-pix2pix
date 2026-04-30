from __future__ import annotations

import json

from PIL import Image

from microi2i.app.cli import main
from microi2i.core.contracts import SmokeDatasetConfig
from microi2i.dataops.smoke_data import create_smoke_datasets


def test_create_smoke_datasets_materializes_pix2pix_and_cyclegan(tmp_path) -> None:
    cfg = SmokeDatasetConfig.from_mapping(
        {
            "schema_version": "microi2i.smoke_data_config.v1",
            "smoke_dataset": {
                "output_dir": str(tmp_path / "smoke"),
                "image_size": 16,
                "sample_count": 2,
                "seed": 11,
                "include_pix2pix": True,
                "include_cyclegan": True,
            },
        }
    )

    manifest = create_smoke_datasets(cfg, repo_root=tmp_path)

    assert manifest["schema_version"] == "microi2i.smoke_dataset_manifest.v1"
    assert (tmp_path / "smoke" / "pix2pix" / "train" / "smoke_000.png").exists()
    assert (tmp_path / "smoke" / "cyclegan" / "trainA" / "smoke_a_000.png").exists()
    paired = Image.open(tmp_path / "smoke" / "pix2pix" / "train" / "smoke_000.png")
    assert paired.size == (32, 16)


def test_cli_create_smoke_data_writes_run_manifest(tmp_path) -> None:
    exit_code = main(
        [
            "create-smoke-data",
            "--config",
            "configs/smoke/default.yml",
            "--set",
            f"output_root={tmp_path.as_posix()}/runs",
            "--set",
            f"smoke_dataset.output_dir={tmp_path.as_posix()}/data",
        ]
    )

    run_dir = next((tmp_path / "runs").iterdir())
    report = json.loads((run_dir / "smoke_dataset_manifest.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["datasets"]["pix2pix"]["layout"] == "pix2pix_aligned"
    assert (tmp_path / "data" / "smoke_dataset_manifest.json").exists()
