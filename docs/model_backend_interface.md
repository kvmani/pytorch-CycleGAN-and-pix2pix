# Model Backend Interface

In MicroI2I, "backend" means a model execution adapter. It does not mean a web server or web app.
The repository remains a CLI-first research platform.

## Responsibilities

Every model backend must provide:

- training command construction or native training execution,
- inference command construction or native inference execution,
- metadata for manifests and reports,
- runtime device handling for CPU and CUDA execution,
- documentation of assumptions, failure modes, and intended scientific use.

## Current Backends

The first adapters wrap the existing legacy scripts:

- `legacy_pix2pix`: wraps upstream pix2pix-style training/inference.
- `legacy_cyclegan`: wraps upstream CycleGAN-style training/inference.

These adapters preserve the proven legacy internals while moving canonical workflow control into
`microi2i`.

## Runtime Device Policy

Configs use:

```yaml
runtime:
  device: auto
  gpu_ids: "0"
  require_cuda: false
```

- `cpu` forces `--gpu_ids -1`.
- `cuda` fails clearly when CUDA is unavailable.
- `auto` uses CUDA only when available and `gpu_ids` is not `-1`.
- `require_cuda: true` fails when no CUDA device is available.

## Adding A New Backend

A new backend must:

- register a unique `model_backend` ID,
- implement train/infer command or execution behavior,
- emit backend metadata into run reports,
- add or update model registry records that reference the backend ID,
- include config presets,
- include unit tests and CLI dry-run integration tests,
- include smoke behavior when computationally feasible,
- update the model registry documentation and Sphinx docs.

Future candidates include CUT, Pix2PixHD, ESRGAN/Real-ESRGAN adapters, diffusion or one-step
translation adapters, and non-GAN restoration baselines.
