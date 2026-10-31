# 03_Modified_Implementation — Scope Statement

**Student Registration:** 24BCE7695  
**Status:** Not Applicable (Strict Academic Baseline Adherence)

---

## Rationale
Under the course rubric, `03_Modified_Implementation/` is reserved for research projects that introduce structural or methodological modifications (such as novel loss formulations, alternative backbone architectures, or hybrid modules) to the published base paper.

In this project:
1. **Experiment 1 (LOLv1 Canonical Training):** Rigorously implements, verifies, and trains the exact official FCE-Net architecture from scratch for 250 epochs on the LOLv1 dataset (achieving a validated PSNR of 25.1846 dB on Tesla T4 GPU).
2. **Experiment 2 (LOLv2 Real-Captured Complete Training):** Trains and evaluates FCE-Net from scratch on all 689 pairs of the LOLv2 Real-Captured dataset across 100 test scenes on local RTX 4050 GPU (achieving 18.9829 dB PSNR, 0.7803 SSIM, 0.3426 LPIPS).
3. **Experiment 3 (Joint 500+500 Dual-Dataset Benchmark):** Trains FCE-Net from scratch on 1,000 paired scenes (500 from LOLv1 + 500 from LOLv2) and evaluates across all 115 test scenes (achieving 20.5268 dB on LOLv1, 19.7629 dB on LOLv2, and 19.8626 dB combined, with exceptional perceptual LPIPS of 0.2518).

Because the official FCE-Net architecture was implemented with 100% fidelity to the published methodology across all three experimental regimes without divergent structural alterations, this directory contains no conflicting or divergent codebases. All implementation, execution scripts, checkpoints, logs, and evaluation metrics are organized under `02_Original_Implementation/`.
