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

## Scientific Use

Metrics must be interpreted alongside visual review and task-specific downstream checks.
For EBSD/Kikuchi workflows, future metrics should include band sharpness, band contrast, and downstream indexing quality where available.
