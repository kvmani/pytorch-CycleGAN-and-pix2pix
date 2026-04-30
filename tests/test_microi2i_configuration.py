from __future__ import annotations

from microi2i.io.configuration import apply_overrides, load_config


def test_load_config_and_apply_dotted_overrides() -> None:
    cfg = load_config("configs/train/pix2pix.default.yml")

    updated = apply_overrides(
        cfg,
        [
            "training.legacy_args=[]",
            "runtime.gpu_ids=-1",
            "dry_run=true",
            "new_section.value=12",
        ],
    )

    assert updated["training"]["legacy_args"] == []
    assert updated["runtime"]["gpu_ids"] == -1
    assert updated["dry_run"] is True
    assert updated["new_section"]["value"] == 12
    assert cfg["training"]["legacy_args"] != []
