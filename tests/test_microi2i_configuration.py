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


def test_apply_dotted_overrides_supports_list_indexes() -> None:
    cfg = {"domain": {"legacy_args": ["--input_folder", "old", "--output_folder", "old_out"]}}

    updated = apply_overrides(
        cfg,
        [
            "domain.legacy_args.1=new_input",
            "domain.legacy_args.3=new_output",
        ],
    )

    assert updated["domain"]["legacy_args"][1] == "new_input"
    assert updated["domain"]["legacy_args"][3] == "new_output"
    assert cfg["domain"]["legacy_args"][1] == "old"
