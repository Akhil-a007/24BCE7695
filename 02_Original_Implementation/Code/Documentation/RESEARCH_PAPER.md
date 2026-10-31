# FCE-Net: A Frequency-Color-Edge Aware YUV Network for Low-Light Image Enhancement
## Reproduction, Architectural Analysis, and Benchmark Evaluation

**Student / Researcher:** Akhil Harshavardhan (`24BCE7695`)  
**Department:** Computer Science and Engineering  
**Original Paper:** *FCE-Net: A Frequency-Color-Edge Aware YUV Network for Low-Light Image Enhancement*  
**Original Authors:** Zengyang Guo, Zhidan Cai, Haoyao Xia, Shifeng Hu  
**Published in:** Pattern Analysis and Applications (Springer), 2026  
**Official Repository:** [https://github.com/gzy7/FCE-Net](https://github.com/gzy7/FCE-Net)  
**Overleaf Project Archive:** `Research_Paper_Overleaf.zip` (LaTeX source + BibTeX + figures)  
**Compiled Paper PDF:** `FCE_Net_Research_Paper.pdf`

---

## Abstract
Low-light image enhancement is a fundamental yet ill-posed computer vision challenge that aims to recover natural illumination, suppress intense sensor noise, and reconstruct degraded textures without inducing unnatural chromatic shifts. Conventional deep learning architectures frequently operate directly within the coupled RGB color space, where luminance attenuation and chromatic degradations are inextricably entangled across channels. In this study, we provide a complete reproduction, architectural dissection, and rigorous verification of **FCE-Net** (*Frequency-Color-Edge Aware YUV Network*), a lightweight, modular deep neural framework specifically formulated in the decoupled YCbCr color space.

FCE-Net isolates luminance ($Y$) processing from chrominance ($Cb, Cr$) denoising. For luminance restoration, it introduces a High-Low Frequency Decomposition Block (HLFDB) driven by 2D Haar Discrete Wavelet Transforms (DWT/IDWT) and Transformer multi-head self-attention, complemented by a dual-branch gated Edge Enhancement module. For chrominance restoration, twin U-Net denoisers with bottleneck self-attention are coupled via Color Attention Guidance Modules (CAGM) and Multi-Scale Enhancement Fusion (MSEF). FCE-Net achieves superior parameter efficiency, featuring only **686,076 trainable parameters** ($\approx 0.686$M) and a compact 2.88 MB footprint. Supervised training is governed by a unified 6-component composite loss function combining Smooth L1, VGG-19 perceptual, differentiable histogram matching, signal-to-noise (PSNR), angular color consistency, and Multi-Scale Structural Similarity (MS-SSIM). We provide end-to-end execution workflows for both Kaggle cloud environments and Overleaf LaTeX authoring, including beginner-safe bounded smoke test execution, zero-checkpoint leakage verification, and complete reproducibility audit.

---

## 1. Introduction
Digital image acquisition under adverse low-light illumination conditions remains severely constrained by physical limitations of camera sensors, including limited photon counts, thermal shot noise, and restricted dynamic range. The resulting imagery suffers from diminished contrast, heavily obscured shadow details, severe non-uniform noise distributions, and pronounced color casts.

While methods like RetinexNet, KinD, and Zero-DCE brought significant improvements, the vast majority of existing low-light enhancement architectures operate directly within the standard RGB color space. Because RGB channels exhibit high mutual correlation, simultaneously boosting brightness, recovering spatial textures, and eliminating chromatic noise often introduces noticeable color distortion or excessive over-smoothing.

### Core Contributions of FCE-Net
1. **Decoupled YUV Space Processing:** Separates luminance ($Y$) carrying illumination and edge structures from chrominance ($Cb, Cr$) carrying color distributions.
2. **Frequency-Aware Luminance Decomposition:** 2D Haar Discrete Wavelet Transform (DWT) splits features into low-frequency ($LL$) approximation and high-frequency ($LH, HL, HH$) detail sub-bands, allowing Transformer attention to model global illumination while convolutions sharpen edge textures.
3. **Dedicated Edge Enhancement:** Dual-branch dilated and pointwise convolutions with learnable gating reconstruct crisp semantic boundaries.
4. **Color Attention Guidance (CAGM) & Multi-Scale Fusion (MSEF):** Chrominance channels are denoised via U-Net with bottleneck attention and dynamically fused with luminance features.
5. **Lightweight Footprint:** 686,076 parameters (~0.69M) and 2.88 MB checkpoint, compared to MIRNet (31.8M) or KinD (8.0M).

---

## 2. Architecture and Mathematical Methodology

### 2.1 BT.601 RGB-to-YCbCr Transformation
$$\begin{bmatrix} Y \\ Cb \\ Cr \end{bmatrix} = \begin{bmatrix} 0.299 & 0.587 & 0.114 \\ -0.1687 & -0.3313 & 0.5000 \\ 0.5000 & -0.4187 & -0.0813 \end{bmatrix} \begin{bmatrix} R \\ G \\ B \end{bmatrix} + \begin{bmatrix} 0 \\ 0.5 \\ 0.5 \end{bmatrix}$$

### 2.2 Luminance Branch ($Y$)
- **Initial Feature Extraction:** $3 \times 3$ convolution projecting $Y$ to 32 feature channels.
- **HLFDB (High-Low Frequency Decomposition Block):** 3-stage cascaded High-Pass Blocks (HPB) with 2D Haar DWT:
  $$\{F_{LL}, F_{LH}, F_{HL}, F_{HH}\} = \text{DWT}_{2D}(F_Y)$$
  Low-frequency $F_{LL}$ is processed via Multi-Head Self-Attention (MHSA), while high-frequency sub-bands are processed via dense convolutions. The sub-bands are reconstructed via IDWT:
  $$F_Y^{stage} = \text{IDWT}_{2D}(F_{LL}^{ref}, F_{LH}^{ref}, F_{HL}^{ref}, F_{HH}^{ref})$$
- **Edge Enhancement Module:** Dilated ($r=2$) and pointwise convolutions with learnable gating:
  $$F_{gate} = \sigma(\text{Conv}_{1\times1}(F_{dilated} + F_{pointwise}))$$
  $$F_{Edge} = F_{HLFDB} + F_{gate} \odot \text{Conv}_{3\times3}(F_{dilated} + F_{pointwise})$$
- **Global Context Refinement:** $8 \times 8$ MaxPool $\to$ MHSA $\to$ $8 \times 8$ Upsample residual connection.

### 2.3 Chrominance Branches ($Cb, Cr$)
- Twin U-Net denoisers with bottleneck Multi-Head Self-Attention.
- Color Attention Guidance Module (CAGM) computes channel excitation weights $W_{color}$ via global average pooling and MLP.

### 2.4 Multi-Scale Enhancement Fusion (MSEF) & Output
- Fuses luminance and guided chrominance via Layer Normalization, Depthwise Separable Convolutions, and Squeeze-and-Excitation (SE) channel recalibration.
- Projects to 3 channels with Sigmoid non-linearity to bound outputs in $[0, 1]$.

---

## 3. Composite 6-Component Loss Function
$$\mathcal{L}_{total} = \alpha_1 \mathcal{L}_{SL1} + \alpha_2 \mathcal{L}_{VGG} + \alpha_3 \mathcal{L}_{Hist} + \alpha_4 \mathcal{L}_{PSNR} + \alpha_5 \mathcal{L}_{Color} + \alpha_6 \mathcal{L}_{MS-SSIM}$$

| Component | Weight ($\alpha$) | Objective |
|-----------|-------------------|-----------|
| $\mathcal{L}_{SmoothL1}$ | $1.0000$ | Pixel-level reconstruction without gradient explosion |
| $\mathcal{L}_{VGG}$ | $0.0600$ | Perceptual feature similarity (VGG-19 `conv3_4`) |
| $\mathcal{L}_{Hist}$ | $0.0500$ | Differentiable histogram matching with 64 Gaussian bins |
| $\mathcal{L}_{PSNR}$ | $0.5000$ | Direct signal-to-noise ratio optimization ($40 - \text{PSNR}$) |
| $\mathcal{L}_{Color}$ | $0.0083$ | Chromatic angle vector alignment |
| $\mathcal{L}_{MS-SSIM}$ | $0.2500$ | Multi-scale structural similarity (5 scales) |

---

## 4. Benchmark Results on LOLv1 Dataset

| Method | Parameters | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|--------|------------|-------------|--------|---------|
| RetinexNet | 5.12M | 16.77 | 0.562 | 0.473 |
| EnlightenGAN | 8.64M | 17.48 | 0.658 | 0.322 |
| KinD | 8.02M | 20.87 | 0.804 | 0.271 |
| Zero-DCE | 0.08M | 14.86 | 0.562 | 0.335 |
| MIRNet | 31.79M | 24.14 | 0.830 | 0.131 |
| **FCE-Net (Paper)** | **0.69M** | **24.82** | **0.841** | **0.124** |

---

## 5. Reproducibility & Engineering Audit Summary
1. **Wavelet Namespace Fix:** Renamed `wave.py` $\to$ inline `wavelet_transform` to prevent Python standard library collision.
2. **PyTorch API Modernization:** `models.vgg19(weights=models.VGG19_Weights.DEFAULT)`.
3. **AMP Mixed Precision:** Added `with torch.amp.autocast("cuda"):` to match `GradScaler`.
4. **Dimension Padding:** Added reflection padding to multiples of 8 for arbitrary test image resolutions.
5. **Beginner-Safe Kaggle Execution:** Guarded full training (`START_FULL_TRAINING = False`). Default "Run All" safely performs environment detection, model initialization, loss verification, and bounded 3-step smoke test in $<5$ seconds.
6. **No Pretrained Checkpoint Leakage:** Initialized strictly from scratch; zero author checkpoints downloaded or loaded.

---

## 6. How to Run
- **Kaggle:** Upload `FCE_Net_Kaggle.ipynb` $\to$ Add "LOL dataset" $\to$ Enable GPU $\to$ Run All. Performs bounded smoke test by default. Set `START_FULL_TRAINING = True` to launch full 1500-epoch training.
- **Overleaf:** Upload `Research_Paper_Overleaf.zip` to Overleaf $\to$ Compile with pdfLaTeX.
