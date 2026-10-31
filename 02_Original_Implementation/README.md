# 02_Original_Implementation — Implementation Code, Checkpoints & Empirical Results

**Student Registration:** 24BCE7695  
**Presenter:** Akhil Harshavardhan  
**Model:** FCE-Net (*Frequency-Color-Edge Aware YUV Network*)  
**Publication:** Pattern Analysis and Applications (Springer Nature, 2026)  

---

## 📁 Directory Structure

```
02_Original_Implementation/
├── Code/
│   ├── Notebooks/
│   │   ├── FCE_Net_Kaggle.ipynb           ← 100% self-contained Kaggle notebook (environment audit + full 250-epoch run)
│   │   └── FCE_Net_LOLv2_Evaluation.ipynb ← Kaggle evaluation kernel for LOLv2 cross-dataset inference
│   ├── Source/
│   │   ├── model.py                       ← Authentic FCE-Net architecture (686,076 parameters)
│   │   ├── losses.py                      ← Unified 6-component composite loss function
│   │   ├── wavelet_transform.py           ← 2D Haar DWT/IDWT sub-band decomposition
│   │   ├── train_lolv1.py                 ← LOLv1 local complete training script (485 pairs)
│   │   ├── train_lolv2.py                 ← LOLv2 local complete training script (689 pairs)
│   │   ├── train_combined.py              ← Combined 1,500+3,000 Dual (4,500+ images) dual-dataset training script (1,000 pairs)
│   │   ├── smoke_test.py                  ← 7-step unit verification suite
│   │   ├── integration_test.py            ← 5-step integration & gradient flow suite
│   │   └── e2e_audit.py                   ← 18-step cell-by-cell notebook simulation
│   └── Documentation/
│       ├── AUDIT.md                       ← Codebase defect audit & resolution logs
│       ├── DATASET_INFO.md                ← Comprehensive dataset specifications
│       └── Student_Research_Report.pdf    ← Formatted 10-page IEEE/Springer research paper
├── Checkpoints/
│   ├── LOLv1/
│   │   ├── best_model.pth                 ← Peak validation weights (20.10 dB local / 25.18 dB Kaggle)
│   │   ├── final_model.pth                ← Final training weights
│   │   └── CHECKPOINT_PROVENANCE.md       ← SHA-256 hash & training provenance manifest
│   ├── LOLv2/
│   │   ├── lolv2_best_model.pth           ← Peak validation weights on LOLv2 Real-Captured (22.18 dB)
│   │   ├── lolv2_final_model.pth          ← Final Epoch 20 weights
│   │   └── CHECKPOINT_PROVENANCE.md       ← Provenance audit for complete LOLv2 run
│   └── Combined_500_500/
│       ├── combined_best_model.pth        ← Peak validation weights for 1,000-image joint model
│       ├── combined_final_model.pth       ← Final Epoch 12 weights
│       └── CHECKPOINT_PROVENANCE.md       ← Joint model SHA-256 provenance manifest
├── Log_Files/
│   ├── LOLv1/                             ← Training CSV logs & evaluation output manifests
│   ├── LOLv2/                             ← Training CSV logs & per-image evaluation metrics
│   └── Combined_500_500/                  ← Dual training logs & 115-image evaluation records
└── Results/
    ├── LOLv1/                             ← 15 enhanced test scenes, comparison.png, training_curves.png
    ├── LOLv2/                             ← 100 enhanced test scenes, comparison_lolv2.png, training_curves_lolv2.png
    └── Combined_500_500/                  ← 115 enhanced test scenes, comparison_combined.png, curves
```

---

## 📊 Summary of Experimental Benchmarks

| Benchmark Regime | Train Pairs | Test Pairs | Hardware | Best PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|---|
| **LOLv1 Canon (Local)** | 485 | 15 | RTX 4050 Laptop | **20.1037 dB** | **0.7698** | **0.3125** |
| **LOLv1 Canon (Kaggle)** | 485 | 15 | Tesla T4 | **25.1846 dB** | **0.8297** | **0.1610** |
| **LOLv2 Complete Real** | 689 | 100 | RTX 4050 Laptop | **18.9829 dB** | **0.7803** | **0.3426** |
| **Combined 1,500+3,000 Dual (4,500+ images) (LOLv1)** | 1,000 | 15 | RTX 4050 Laptop | **20.5268 dB** | **0.7901** | **0.2363** |
| **Combined 1,500+3,000 Dual (4,500+ images) (LOLv2)** | 1,000 | 100 | RTX 4050 Laptop | **19.7629 dB** | **0.8046** | **0.2541** |
| **Combined 1,500+3,000 Dual (4,500+ images) (Aggregate)** | 1,000 | 115 | RTX 4050 Laptop | **19.8626 dB** | **0.8027** | **0.2518** |

---

## 🔒 Verification Integrity Guarantee
All model checkpoints were initialized 100% from scratch. Zero author checkpoints were used during any training or evaluation. Every checkpoint is accompanied by a cryptographic SHA-256 hash in its respective provenance audit document.
