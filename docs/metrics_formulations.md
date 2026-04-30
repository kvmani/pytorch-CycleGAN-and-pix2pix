# Metrics And Formulations

This page documents the initial metrics used in `microi2i evaluate`.

Let \(I\) be a target image and \(\hat{I}\) be the predicted image with \(N\) pixels.

## Mean Absolute Error

\[
\mathrm{MAE}(I,\hat{I}) = \frac{1}{N}\sum_{i=1}^{N}|I_i - \hat{I}_i|
\]

MAE is easy to interpret but does not capture structural similarity.

## Root Mean Squared Error

\[
\mathrm{RMSE}(I,\hat{I}) =
\sqrt{\frac{1}{N}\sum_{i=1}^{N}(I_i - \hat{I}_i)^2}
\]

RMSE penalizes large errors more strongly than MAE.

## Peak Signal-To-Noise Ratio

For 8-bit images with maximum intensity \(L=255\):

\[
\mathrm{PSNR} = 20 \log_{10}\left(\frac{L}{\mathrm{RMSE}}\right)
\]

Higher PSNR often indicates better pixel fidelity, but it can favor overly smooth outputs.

## Structural Similarity Index

SSIM compares local luminance, contrast, and structure:

\[
\mathrm{SSIM}(x,y) =
\frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}
{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
\]

where \(\mu_x,\mu_y\) are local means, \(\sigma_x^2,\sigma_y^2\) are local variances, and \(\sigma_{xy}\) is local covariance.

## Gradient Correlation

Let \(G(I)=\sqrt{(\partial_x I)^2 + (\partial_y I)^2}\) be the gradient magnitude image.

\[
\rho_G =
\frac{\mathrm{cov}(G(I),G(\hat{I}))}
{\sigma_{G(I)}\sigma_{G(\hat{I})}}
\]

High gradient correlation indicates that translated images preserve major edges and boundaries.
It can be undefined for nearly constant images.

## Edge Mean Absolute Error

\[
\mathrm{EdgeMAE} =
\frac{1}{N}\sum_{i=1}^{N}|G(I)_i - G(\hat{I})_i|
\]

Lower values indicate closer local edge strength. In microscopy, this is useful for detecting
over-smoothing or artificial boundary sharpening.

## Histogram L1 Distance

For normalized intensity histograms \(h_I\) and \(h_{\hat{I}}\):

\[
D_\mathrm{hist} = \frac{1}{2}\sum_b |h_I(b)-h_{\hat{I}}(b)|
\]

The value is zero when the binned intensity distributions match.

## Contrast-To-Noise Proxy Delta

MicroI2I uses a simple robust contrast proxy:

\[
\mathrm{CNR}_{proxy}(I) =
\frac{P_{95}(I)-P_{5}(I)}{\sigma_I+\epsilon}
\]

and reports:

\[
\Delta\mathrm{CNR}_{proxy}=|\mathrm{CNR}_{proxy}(I)-\mathrm{CNR}_{proxy}(\hat{I})|
\]

This is not a substitute for instrument-aware CNR, but it is useful for automated screening.

## High-Frequency Energy Ratio

\[
R_{HF} = \frac{\mathrm{mean}(G(\hat{I}))}{\mathrm{mean}(G(I))}
\]

Values much below one suggest smoothing; values much above one may indicate noise amplification or
hallucinated texture.

## Laplacian Sharpness Ratio

Using a discrete Laplacian operator \(\nabla^2\):

\[
R_{\nabla^2} =
\frac{\mathrm{var}(\nabla^2\hat{I})}
{\mathrm{var}(\nabla^2 I)}
\]

This proxy helps screen denoising and super-resolution outputs for excessive blur or artificial
sharpening.

## Scientific Use

Metrics must be interpreted alongside visual review and task-specific downstream checks.
For EBSD/Kikuchi workflows, future metrics should include band sharpness, band contrast, and downstream indexing quality where available.
