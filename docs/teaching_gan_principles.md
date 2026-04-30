# GAN Principles For Microscopy Image Translation

Generative adversarial networks (GANs) learn by placing two networks in competition.
The generator \(G\) tries to synthesize realistic outputs, while the discriminator \(D\) tries to distinguish generated images from real images.

```{image} diagrams/gan_training_loop.svg
:alt: GAN training loop
:class: architecture-diagram
```

## Why GANs Matter In Microscopy

Microscopy often contains structured signals where simple pixel losses can produce blurry outputs.
GAN objectives encourage outputs that live on the manifold of realistic images, which can help with:

- EBSD and Kikuchi pattern refinement.
- SEM or optical microscopy denoising.
- Super-resolution from lower-resolution acquisitions.
- Translation between simulated and experimental domains.
- Boundary or mask synthesis when paired labels exist.

## Paired Versus Unpaired Learning

pix2pix is paired.
It learns from aligned examples \((x, y)\), where \(x\) is the input image and \(y\) is the target image.
This is appropriate when exact source-target pairs exist.

CycleGAN is unpaired.
It learns from domain \(A\) and domain \(B\) without one-to-one correspondence.
This is useful when simulated and experimental images exist but cannot be paired exactly.

## Scientific Caution

GANs can hallucinate plausible-looking structures.
For scientific microscopy, visual quality is not enough.
Outputs should be validated with paired metrics where possible, downstream task metrics, expert inspection, and documented failure modes.
