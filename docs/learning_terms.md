# Learning Terms

## Epoch

An epoch is one full pass through the training dataset.
If a dataset has 1,000 samples and the batch size is 10, one epoch contains 100 parameter-update steps.

## Batch Size

Batch size is the number of samples processed before one optimizer update.
Small batches may fit limited GPU memory but can produce noisier gradients.

## Learning Rate

The learning rate \(\alpha\) controls the optimizer step size.
If \(\alpha\) is too high, training may diverge.
If \(\alpha\) is too low, training may be stable but slow.

## Adam \(\beta_1\) And \(\beta_2\)

Adam keeps moving averages of gradients and squared gradients.
\(\beta_1\) controls momentum in the first moment estimate, and \(\beta_2\) controls smoothing of the second moment estimate.

## Generator

The generator \(G\) creates translated images.
In pix2pix it maps input images to target-domain images.
In CycleGAN there are two generators, one for each direction.

## Discriminator

The discriminator \(D\) learns to distinguish real images from generated images.
Its feedback pushes the generator toward more realistic outputs.

## Loss Weight

Loss weights, such as \(\lambda\) in pix2pix or CycleGAN, control the balance between objectives.
For scientific images, these weights can change the tradeoff between realism and structure preservation.
