# Checkpoint Provenance & Reproducibility Audit — LOLv1 Local Model

**Student Registration:** 24BCE7695  
**Presenter:** Akhil Harshavardhan  
**Model Architecture:** FCE-Net (`filters=32`, 686,076 parameters, 330 parameter tensors)  
**Execution Platform:** Local NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM, CUDA 12.4, TF32)  
**Training Set:** Complete LOLv1 our485 paired split (485 low/high images)  
**Evaluation Set:** Complete LOLv1 eval15 paired split (15 low/high images)  
**Trained From:** Random parameter initialization (ZERO third-party checkpoints)  
**Loss Function:** 6-component composite loss (Smooth L1 + VGG Perceptual + MS-SSIM + DWT Edge + Histogram + Color)  
**Optimizer:** Adam (`lr=2e-4`) with CosineAnnealingLR (`eta_min=1e-6`)  
**SHA-256 Checksum:** `8103af068298afb8c241e08e29163b74da396e264d89901fc80a183bdc89759d`  

---

## Quantitative Benchmark on eval15 (15 Images)
- **Mean PSNR:** **20.1037 dB**
- **Mean SSIM:** **0.7698**
- **Mean LPIPS:** **0.3125**
- **Best Checkpoint Epoch:** Epoch 19 / 20
- **Parameters:** Exactly 686,076 float32 weights (2.88 MB)

Generated on: 2026-10-30 22:09:16
