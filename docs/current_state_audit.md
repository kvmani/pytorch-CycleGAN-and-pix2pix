# Current State Audit

## Repository Baseline

The repository currently combines:

- Upstream PyTorch CycleGAN/pix2pix structure.
- Local EBSD and Kikuchi data preparation and inference scripts.
- Shell scripts for training/inference on specific datasets.
- Generated outputs, checkpoints, and local experiment folders.

## Existing Core Capabilities To Preserve

- `train.py`: general training for `cycle_gan`, `pix2pix`, and related models.
- `test.py`: model inference and HTML image output generation.
- `data/`: aligned, unaligned, single-image, colorization, and template datasets.
- `models/`: CycleGAN, pix2pix, test, colorization, network definitions, and base model behavior.
- `options/`: command-line option parsing for training and testing.
- `util/`: image conversion, HTML writing, visualizer support, and helper utilities.
- `EBSD/`: EBSD data processing, ML input generation, inference, and statistical scripts.
- `kikuchi/`: Kikuchi conversion and CycleGAN dataset preparation scripts.
- Root scripts such as `prepareCycleGanData.py`, `run_ebsd_inference.py`, `run_kikuchi_inference.py`, and result summarization helpers.

## Current Risks

- Workflows rely on ad hoc scripts and hardcoded paths.
- Metadata is not consistently machine-readable.
- Dataset provenance and split policies are not standardized.
- Documentation is mostly inherited from upstream and does not fully describe microscopy usage.
- Generated data and experiments are mixed near source files.
- There is no canonical model registry or checkpoint lifecycle.

## Restructure Goal

Create a package-first scientific platform while preserving the useful behavior above through documented, tested, config-driven workflows.
