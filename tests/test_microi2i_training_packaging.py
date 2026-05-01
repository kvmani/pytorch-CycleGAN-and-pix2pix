from __future__ import annotations

import json

from microi2i.app.cli import main
from microi2i.core.contracts import ScriptWorkflowConfig
from microi2i.training.legacy_runner import (
    apply_smoke_training_overrides,
    build_training_command,
    build_training_preflight,
    package_training_outputs,
    parse_legacy_loss_log,
    parse_legacy_args,
    write_loss_curve_artifacts,
)
from microi2i.training.validation_monitor import (
    MONITOR_ENV_VAR,
    build_validation_monitor_manifest,
    discover_validation_pool,
    run_epoch_validation_monitor,
    select_epoch_samples,
    write_validation_monitor_manifest,
)


def test_parse_legacy_args_extracts_training_metadata() -> None:
    parsed = parse_legacy_args(["--dataroot", "data", "--model", "pix2pix", "--continue_train"])

    assert parsed["dataroot"] == "data"
    assert parsed["model"] == "pix2pix"
    assert parsed["continue_train"] is True


def test_training_preflight_reports_missing_dataroot_as_dry_run_warning(tmp_path) -> None:
    cfg = {
        "schema_version": "microi2i.train_config.v1",
        "output_root": str(tmp_path),
        "training": {
            "legacy_script": "train.py",
            "legacy_args": ["--dataroot", "missing", "--name", "demo", "--model", "pix2pix"],
        },
    }
    config = ScriptWorkflowConfig.from_mapping(cfg, section="training")
    command = build_training_command(config, repo_root=tmp_path)

    report = build_training_preflight(
        config,
        repo_root=tmp_path,
        resolved_config=cfg,
        command=command,
        dry_run=True,
    )

    assert report["schema_version"] == "microi2i.training_preflight.v1"
    assert report["checks"]["dataroot_exists"] is False
    assert report["warnings"]


def test_cli_train_dry_run_writes_training_package(tmp_path) -> None:
    exit_code = main(
        [
            "train",
            "--config",
            "configs/train/pix2pix.default.yml",
            "--dry-run",
            "--set",
            f"output_root={tmp_path.as_posix()}",
        ]
    )

    run_dir = next(tmp_path.iterdir())
    report = json.loads((run_dir / "training_report.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["schema_version"] == "microi2i.training_report.v1"
    assert (run_dir / "metrics_log.csv").exists()
    assert (run_dir / "metrics_log.jsonl").exists()
    assert (run_dir / "training_summary.html").exists()
    assert (run_dir / "validation_monitor_manifest.json").exists()
    manifest = json.loads((run_dir / "validation_monitor_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "microi2i.validation_monitor_manifest.v1"


def test_smoke_training_overrides_force_cpu_and_tiny_dataset() -> None:
    args = ["--dataroot", "data", "--gpu_ids", "0", "--n_epochs", "100"]

    effective = apply_smoke_training_overrides(
        args,
        {"enabled": True, "max_epochs": 1, "max_dataset_size": 2, "image_size": 32},
    )
    parsed = parse_legacy_args(effective)

    assert parsed["gpu_ids"] == "-1"
    assert parsed["n_epochs"] == "1"
    assert parsed["n_epochs_decay"] == "0"
    assert parsed["max_dataset_size"] == "2"
    assert parsed["load_size"] == "32"
    assert parsed["crop_size"] == "32"
    assert parsed["no_html"] is True


def test_parse_legacy_loss_log_and_package_training_outputs(tmp_path) -> None:
    experiment = tmp_path / "checkpoints" / "demo"
    images = experiment / "web" / "images"
    images.mkdir(parents=True)
    (experiment / "loss_log.txt").write_text(
        "================ Training Loss ================\n"
        "(epoch: 1, iters: 2, time: 0.001, data: 0.002) G_GAN: 1.250 D_real: 0.500\n",
        encoding="utf-8",
    )
    from PIL import Image

    Image.new("RGB", (8, 8), (10, 20, 30)).save(images / "epoch001_fake_B.png")
    rows = parse_legacy_loss_log(experiment / "loss_log.txt")

    outputs = package_training_outputs(
        tmp_path / "run",
        {
            "paths": {"checkpoints_dir": str(tmp_path / "checkpoints")},
            "experiment_name": "demo",
        },
    )

    assert rows[0]["loss_G_GAN"] == 1.25
    assert outputs["loss_rows"] == 1
    assert outputs["validation_sample_count"] == 1
    assert (tmp_path / "run" / "metrics_log.csv").exists()
    assert (tmp_path / "run" / "validation_samples.html").exists()
    assert (tmp_path / "run" / "loss_curves.csv").exists()
    assert (tmp_path / "run" / "loss_curves.svg").exists()


def test_loss_curve_artifacts_are_stable(tmp_path) -> None:
    paths = write_loss_curve_artifacts(
        tmp_path,
        [
            {"epoch": 1, "iteration": 1, "loss_G": 2.0, "loss_D": 1.0},
            {"epoch": 1, "iteration": 2, "loss_G": 1.5, "loss_D": 0.8},
        ],
    )

    assert [path.name for path in paths] == ["loss_curves.csv", "loss_curves.svg"]
    assert "loss_D" in (tmp_path / "loss_curves.csv").read_text(encoding="utf-8")
    assert "Training Loss Curves" in (tmp_path / "loss_curves.svg").read_text(encoding="utf-8")


def test_validation_monitor_selects_explicit_fixed_images(tmp_path) -> None:
    val_dir = tmp_path / "data" / "val"
    val_dir.mkdir(parents=True)
    from PIL import Image

    for name in ("a.png", "b.png", "c.png"):
        Image.new("RGB", (8, 8), (10, 10, 10)).save(val_dir / name)

    pool = discover_validation_pool(
        repo_root=tmp_path,
        legacy_options={"dataroot": "data", "dataset_mode": "aligned"},
        monitor_config={"fixed_images": ["b"]},
    )

    assert [row["sample_id"] for row in pool["fixed_samples"]] == ["b"]


def test_validation_monitor_uses_first_five_as_default_fixed_set(tmp_path) -> None:
    val_dir = tmp_path / "data" / "val"
    val_dir.mkdir(parents=True)
    from PIL import Image

    for index in range(7):
        Image.new("RGB", (8, 8), (index, index, index)).save(val_dir / f"{index:02d}.png")

    manifest = build_validation_monitor_manifest(
        repo_root=tmp_path,
        run_dir=tmp_path / "run",
        resolved_config={
            "runtime": {"seed": 123},
            "training": {
                "validation_monitor": {},
                "legacy_args": ["--dataroot", "data", "--dataset_mode", "aligned", "--name", "demo"],
            },
        },
        command=["python", "train.py", "--dataroot", "data", "--dataset_mode", "aligned", "--name", "demo"],
        dry_run=True,
    )

    assert [row["sample_id"] for row in manifest["fixed_samples"]] == ["00", "01", "02", "03", "04"]


def test_validation_monitor_keeps_fixed_samples_and_adds_deterministic_random() -> None:
    samples = [{"sample_id": str(index), "index": index, "input_path": f"{index}.png"} for index in range(8)]
    fixed = samples[:5]

    first = select_epoch_samples(samples, fixed, total_count=7, seed=42, epoch=3)
    second = select_epoch_samples(samples, fixed, total_count=7, seed=42, epoch=3)
    other_epoch = select_epoch_samples(samples, fixed, total_count=7, seed=42, epoch=4)

    assert [row["sample_id"] for row in first[:5]] == ["0", "1", "2", "3", "4"]
    assert first == second
    assert first != other_epoch
    assert all(row["selection_role"] == "fixed" for row in first[:5])
    assert all(row["selection_role"] == "random" for row in first[5:])


def test_validation_monitor_missing_pool_warns_without_blocking(tmp_path) -> None:
    manifest = build_validation_monitor_manifest(
        repo_root=tmp_path,
        run_dir=tmp_path / "run",
        resolved_config={
            "training": {
                "validation_monitor": {"enabled": True},
                "legacy_args": ["--dataroot", "missing", "--dataset_mode", "unaligned"],
            }
        },
        command=["python", "train.py", "--dataroot", "missing", "--dataset_mode", "unaligned"],
        dry_run=True,
    )

    assert manifest["enabled"] is True
    assert manifest["pool"]["sample_count"] == 0
    assert manifest["warnings"]


def test_validation_monitor_epoch_hook_writes_paired_metrics_and_html(tmp_path, monkeypatch) -> None:
    from PIL import Image

    val_dir = tmp_path / "data" / "val"
    val_dir.mkdir(parents=True)
    for index in range(2):
        left = Image.new("RGB", (8, 8), (20 + index, 20, 20))
        right = Image.new("RGB", (8, 8), (40 + index, 40, 40))
        combined = Image.new("RGB", (16, 8))
        combined.paste(left, (0, 0))
        combined.paste(right, (8, 0))
        combined.save(val_dir / f"sample_{index}.png")

    class Opt:
        dataroot = str(tmp_path / "data")
        phase = "train"
        dataset_mode = "aligned"
        max_dataset_size = 10
        load_size = 8
        crop_size = 8
        direction = "AtoB"
        input_nc = 3
        output_nc = 3
        preprocess = "resize_and_crop"
        no_flip = True
        serial_batches = True
        batch_size = 1
        num_threads = 0
        isTrain = True

    class FakeModel:
        model_names: list[str] = []

        def eval(self) -> None:
            self.was_eval = True

        def set_input(self, data) -> None:
            self.data = data

        def test(self) -> None:
            return None

        def get_current_visuals(self):
            return {"real_A": self.data["A"], "fake_B": self.data["B"], "real_B": self.data["B"]}

    manifest = build_validation_monitor_manifest(
        repo_root=tmp_path,
        run_dir=tmp_path / "run",
        resolved_config={
            "training": {
                "validation_monitor": {"fixed_count": 1, "total_count": 1, "export_html": True},
                "legacy_args": ["--dataroot", str(tmp_path / "data"), "--dataset_mode", "aligned"],
            }
        },
        command=["python", "train.py", "--dataroot", str(tmp_path / "data"), "--dataset_mode", "aligned"],
        dry_run=False,
    )
    manifest_path = write_validation_monitor_manifest(tmp_path / "run", manifest)
    monkeypatch.setenv(MONITOR_ENV_VAR, str(manifest_path))

    run_epoch_validation_monitor(FakeModel(), Opt(), 1)

    report = json.loads((tmp_path / "run" / "validation_monitor" / "report.json").read_text(encoding="utf-8"))
    assert report["epochs"][0]["metrics"]["status"] == "computed"
    assert report["epochs"][0]["metrics"]["aggregate"]["mae_mean"] == 0.0
    assert (tmp_path / "run" / "validation_monitor" / "index.html").exists()
    assert "Fixed Sample Progression" in (
        tmp_path / "run" / "validation_monitor" / "index.html"
    ).read_text(encoding="utf-8")
    assert (tmp_path / "run" / "validation_monitor" / "epoch_001" / "index.html").exists()
