# GAN And Image Translation Formulations

This page summarizes the mathematical objectives used by the model families in this repository.

## Conditional GAN Objective

pix2pix learns a mapping \(G: X \rightarrow Y\) conditioned on an input image \(x\).
The discriminator receives both the input and output pair.

\[
\mathcal{L}_{cGAN}(G,D) =
\mathbb{E}_{x,y}[\log D(x,y)] +
\mathbb{E}_{x,z}[\log(1 - D(x, G(x,z)))]
\]

In practice the generator is also trained with a reconstruction loss:

\[
\mathcal{L}_{L1}(G) = \mathbb{E}_{x,y,z}[\|y - G(x,z)\|_1]
\]

The pix2pix objective is:

\[
G^\* = \arg\min_G \max_D
\mathcal{L}_{cGAN}(G,D) + \lambda \mathcal{L}_{L1}(G)
\]

## CycleGAN Objective

CycleGAN learns two mappings:

\[
G: A \rightarrow B, \qquad F: B \rightarrow A
\]

The adversarial loss for \(G\) and discriminator \(D_B\) is:

\[
\mathcal{L}_{GAN}(G,D_B,A,B) =
\mathbb{E}_{y \sim p_{data}(y)}[\log D_B(y)] +
\mathbb{E}_{x \sim p_{data}(x)}[\log(1 - D_B(G(x)))]
\]

Cycle consistency forces translated images to map back to the original domain:

\[
\mathcal{L}_{cyc}(G,F) =
\mathbb{E}_{x \sim p_{data}(x)}[\|F(G(x)) - x\|_1] +
\mathbb{E}_{y \sim p_{data}(y)}[\|G(F(y)) - y\|_1]
\]

The full objective combines both adversarial directions and cycle consistency:

\[
\mathcal{L}(G,F,D_A,D_B) =
\mathcal{L}_{GAN}(G,D_B,A,B) +
\mathcal{L}_{GAN}(F,D_A,B,A) +
\lambda \mathcal{L}_{cyc}(G,F)
\]

## Interpretation For Microscopy

The adversarial terms encourage domain realism.
The reconstruction or cycle terms constrain the mapping so scientific structures are less likely to drift arbitrarily.
These objectives do not prove scientific correctness by themselves.
They must be paired with validation metrics and domain review.
