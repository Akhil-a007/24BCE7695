# LOLv2 Checkpoint Provenance & Model Audit

**Student Registration:** 24BCE7695  
**Presenter:** Akhil Harshavardhan  
**Model:** FCE-Net (Frequency-Color-Edge Aware YUV Network)  
**Trained On:** LOLv2 Real-Captured Dataset (All 689 pairs: complete dataset)  
**Evaluated On:** LOLv2 Real-Captured Test Set (100 images)  
**Hardware Platform:** NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM, CUDA 12.4)  
**Runtime:** PyTorch 2.5.1+cu124, TF32 Acceleration  
**Training Epochs:** 20  

---

## Performance Summary

| Metric | Score | Paper Benchmark (Guo et al. 2026) |
|---|---|---|
| **PSNR** | **18.9829 dB** | 24.82 dB |
| **SSIM** | **0.7803** | 0.841 |
| **LPIPS** | **0.3426** | 0.124 |
| **Parameters** | **686,076** (~0.69M) | 0.69M |

---

## Checkpoint Files

1. `lolv2_best_model.pth` (and `best_model.pth`) — Best model checkpoint by validation PSNR (22.1767 dB, Epoch 15).
2. `lolv2_final_model.pth` (and `final_model.pth`) — Final checkpoint recorded at Epoch 20.
