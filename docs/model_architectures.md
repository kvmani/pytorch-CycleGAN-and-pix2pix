# Model Architectures

MicroI2I currently preserves the original pix2pix and CycleGAN model families while the package architecture is migrated.

```{image} diagrams/model_architectures.svg
:alt: pix2pix and CycleGAN model architectures
:class: architecture-diagram
```

## pix2pix

pix2pix uses a conditional generator, often a U-Net, and a PatchGAN discriminator.
The U-Net skip connections help preserve local spatial structure, which matters for microscopy features such as grain boundaries, bands, pores, and phase contrast.

## CycleGAN

CycleGAN uses two generators and two discriminators.
It is suited for unpaired domain transfer, such as simulated-to-experimental EBSD/Kikuchi refinement, but it requires careful validation because no paired ground truth constrains each sample.

## Extension Direction

Future model families should be registered through `frozen_checkpoints/model_registry.json` and documented with:

- Architecture family.
- Publication citation.
- Internal modifications.
- Training data assumptions.
- Valid scientific use cases.
- Known failure modes.
