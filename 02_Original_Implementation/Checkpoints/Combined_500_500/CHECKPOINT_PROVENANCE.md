# Checkpoint Provenance & Reproducibility Audit — Joint 500+500 Model

**Student Registration:** 24BCE7695  
**Presenter:** Akhil Harshavardhan  
**Model Architecture:** FCE-Net (`filters=32`, 686,076 parameters, 330 parameter tensors)  
**Execution Platform:** Local NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM, CUDA 12.4, TF32)  
**Training Set:** Joint 1,000 paired low/high images (500 from LOLv1 our485 + 500 from LOLv2 Real-Captured)  
**Evaluation Set:** 115 paired images (15 from LOLv1 eval15 + 100 from LOLv2 Test)  
**Trained From:** Random parameter initialization (ZERO third-party checkpoints)  
**Loss Function:** 6-component composite loss (Smooth L1 + VGG Perceptual + MS-SSIM + Differentiable Histogram + PSNR + Angular Color)  
**Optimizer:** Adam (`lr=2e-4`) with CosineAnnealingLR (`eta_min=1e-6`)  
**SHA-256 Checksum:** `613620e18fbe0fe5667767793283d0520580c823ad1dd969a4eb0fe93bd4c7f8`  

---

## Benchmark Scores
- **LOLv1 eval15 (15 Images):**
  - PSNR:  **20.5268 dB**
  - SSIM:  **0.7901**
  - LPIPS: **0.2363**
- **LOLv2 Test (100 Images):**
  - PSNR:  **19.7629 dB**
  - SSIM:  **0.8046**
  - LPIPS: **0.2541**
- **Combined Dual Benchmark (115 Images):**
  - Aggregate PSNR:  **19.8626 dB**
  - Aggregate SSIM:  **0.8027**
  - Aggregate LPIPS: **0.2518**

Generated on: 2026-10-30 21:07:04
