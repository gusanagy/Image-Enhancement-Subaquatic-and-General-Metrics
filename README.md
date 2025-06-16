# Image-Enhancement-Subaquatic-and-General-Metrics
This repository contains implementations of various quantitative metrics used to evaluate image enhancement techniques, intended for research and academic use.
# Image Enhancement Metrics

This repository implements a collection of image enhancement quality metrics for research purposes.

| **Category** | **Metric** | **Description** | **Supervision** | **Underwater-specific** | **Implemented** |
|--------------|------------|-----------------|------------------|--------------------------|------------------|
| Objective Quality Metrics | SSIM (Structural Similarity Index Measure) | Measures perceptual similarity between two images. | ✅ | ❌ | ☐ |
| Objective Quality Metrics | PSNR (Peak Signal-to-Noise Ratio) | Measures signal quality based on noise ratio. | ✅ | ❌ | ☐ |
| Objective Quality Metrics | LPIPS | Learned perceptual similarity using deep networks. | ✅ | ❌ | ☐ |
| Objective Quality Metrics | FSIM | Measures similarity based on phase congruency and gradient magnitude. | ✅ | ❌ | ☐ |
| Contrast & Sharpness Metrics | CII | Evaluates global contrast improvement. | ❌ | ❌ | ☐ |
| Contrast & Sharpness Metrics | Tenenbaum Sharpness | Sharpness based on image gradients. | ❌ | ❌ | ☐ |
| Contrast & Sharpness Metrics | Michelson Contrast | Ratio between bright and dark regions. | ❌ | ❌ | ☐ |
| Contrast & Sharpness Metrics | Tenengrad Measure | Gradient-based sharpness measure. | ❌ | ❌ | ☐ |
| Contrast & Sharpness Metrics | Variance of Laplacian | Sharpness via Laplacian variance. | ❌ | ❌ | ☐ |
| Contrast & Sharpness (No Reference) | Spatial Frequency (SF) | Measures level of detail via intensity variations. | ❌ | ❌ | ☐ |
| Contrast & Sharpness (No Reference) | Spectral Residual Sharpness (SRS) | Frequency-based sharpness measure. | ❌ | ❌ | ☐ |
| Color & Illumination Metrics | Colorfulness Index | Measures color saturation and diversity. | ❌ | ❌ | ☐ |
| Color & Illumination Metrics | NIQE | No-reference natural image quality. | ❌ | ❌ | ☐ |
| Color & Illumination Metrics | Entropy | Measures amount of information in the image. | ❌ | ❌ | ☐ |
| Color & Illumination Metrics | Luminance Contrast Ratio | Measures difference between light and dark areas. | ❌ | ❌ | ☐ |
| Color & Illumination (No Reference) | Brightness Measure (BM) | Mean brightness intensity. | ❌ | ❌ | ☐ |
| Color & Illumination (No Reference) | Statistical Naturalness (SN) | Evaluates how natural the illumination appears. | ❌ | ❌ | ☐ |
| Color & Illumination (No Reference) |  no-reference low-light image enhancement evaluation (NLIEE). | Predict the quality of light-enhanced images | ❌ | ❌ | ✅ |
| General Quality (No Reference) | BRISQUE | Blind quality evaluation using natural scene statistics. | ❌ | ❌ | ☐ |
| General Quality (No Reference) | PIQE | Distortion-aware perceptual quality estimator. | ❌ | ❌ | ☐ |
| General Quality (No Reference) | BIQI | Blind quality index based on natural image statistics. | ❌ | ❌ | ☐ |
| General Quality (No Reference) | IL-NIQE | Local version of NIQE for spatial quality. | ❌ | ❌ | ☐ |
| Underwater-Specific Metrics | EUV | Evaluates visibility of edges in underwater images. | ❌ | ✅ | ☐ |
| Underwater-Specific Metrics | HLD | Measures haze-line distance reduction. | ❌ | ✅ | ☐ |
| Underwater-Specific Metrics | UIQM | Underwater Image Quality Measure. | ❌ | ✅ | ✅ |
| Underwater-Specific Metrics | UCIQE | Underwater Color Image Quality Evaluation. | ❌ | ✅ | ✅ |
| Underwater-Specific Metrics | UIEM | Evaluates color recovery and visibility. | ❌ | ✅ | ☐ |


