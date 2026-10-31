# FCE-Net Project Plan — 24BCE7695

## Paper
**FCE-Net: A Frequency-Color-Edge Aware YUV Network for Low-Light Image Enhancement**
- Official repo: https://github.com/gzy7/FCE-Net

## Architecture Summary
- Converts RGB → YCbCr (YUV) in the model
- **Y (luminance) branch**: Processed through HLFDB (High-Low Frequency Decomposition Block) using Haar wavelet DWT/IDWT, with Transformer attention blocks and Dense connections; followed by EdgeEnhance module
- **Cb/Cr (chrominance) branches**: Each processed through a Denoiser with multi-head self-attention bottleneck
- **ColorAttentionGuidance (CAGM)**: Fuses Cb and Cr features with channel attention
- **Luminance refinement**: MaxPool → MHSA → Upsample pathway added to luminance
- **MSEFBlock**: Combines features via layer norm + depthwise conv + SE attention
- **Final**: Recombine luminance + color features → 3-channel RGB output with sigmoid

## Loss Functions (CombinedLoss)
1. Smooth L1 Loss (α₁ = 1.00)
2. VGG Perceptual Loss — VGG19 features[:16] (α₂ = 0.06)
3. Histogram Loss — differentiable via Gaussian kernels (α₃ = 0.05)
4. PSNR Loss — 40 − PSNR (α₄ = 0.5)
5. Color Loss — mean absolute color difference (α₅ = 0.0083)
6. Multi-Scale SSIM Loss (α₆ = 0.25)

## Training Configuration
- Optimizer: Adam, LR = 2e-4
- Scheduler: CosineAnnealingLR (T_max = num_epochs)
- Epochs: 1500
- Batch size: 1
- Crop size: 256×256 (training)
- Mixed precision: torch.cuda.amp.GradScaler
- Best model saved by validation PSNR

## Datasets
- **LOLv1** (paired): Google Drive link in README
  - Structure: `train/low/`, `train/high/`, `eval/low/`, `eval/high/`
  - 485 training pairs, 15 eval pairs
- **LOLv2** (paired): Google Drive link in README
- Unpaired test sets: DICM, LIME, MEF, NPE, VV

## Dependencies
- Python 3.9
- PyTorch with CUDA
- torchvision, torchmetrics
- pytorch_msssim (pip: pytorch-msssim)
- PyWavelets (pywt)
- einops
- lpips (for metric.py)
- PIL/Pillow

## Key Issues Found in Original Code
1. `train.py`: Dataset paths are empty strings ('') — must be filled in
2. `test.py`: Dataset and weights paths are empty strings
3. `metric.py`: Prediction/GT directory paths are empty strings
4. `losses.py`: `models.vgg19(weights=True)` — should be `weights='DEFAULT'` or `pretrained=True`
5. `wave.py`: File name conflicts with Python stdlib `wave` module — needs renaming
6. `train.py`: Uses deprecated `torch.cuda.amp.GradScaler()` — needs update for newer PyTorch
7. Input images must be divisible by 8 (due to MaxPool2d(8) + Upsample(8))

## Task Checklist
- [x] A. Clone & inspect repository
- [x] B. Create project folder structure
- [x] C. Create Kaggle notebook (FCE_Net_Kaggle.ipynb) — 41 cells, 20 code, all syntax-valid
- [x] D. Debug & verify (smoke tests) — 7/7 unit + 5/5 integration tests PASS
- [x] E. Dataset & training configuration — LOLv1 auto-detection, 1500-epoch config
- [x] F. Final deliverables — README.md, AUDIT.md, PROJECT_PLAN.md, DATASET_INFO.md, presentation outline

## Progress Log
- 2026-10-29 18:56: Repository cloned, all 8 source files inspected thoroughly
- 2026-10-29 18:57: Project structure created (01–04 folders), original code preserved
- 2026-10-29 19:44: Kaggle notebook generated — 41 cells, all 20 code cells pass AST syntax check
- 2026-10-29 19:49: Smoke tests PASSED (7/7): imports, DWT/IDWT, Denoiser, MHSA, MSEFBlock, losses, backward, checkpoint
- 2026-10-29 19:50: README.md and AUDIT.md created
- 2026-10-29 19:55: Integration test PASSED (5/5): module loading, forward pass, loss+backward, checkpoint round-trip
- 2026-10-29 20:30: Deep architecture comparison (27-point diff) — 0 critical issues
- 2026-10-29 21:26: Independent 10-point audit completed — all 10 checks PASS
- 2026-10-29 21:26: Model confirmed: 686,076 params (~0.69M), 2.88 MB checkpoint
- 2026-10-29 21:26: **Status: READY FOR KAGGLE SMOKE TEST**

