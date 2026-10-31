# 04_Presentation — Academic Slide Deck & Visual Deliverables

**Student Registration:** 24BCE7695  
**Presenter:** Akhil Harshavardhan  
**Coursework:** Deep Learning & Vision Academic Submission  
**Topic:** FCE-Net Reproduction, Architectural Analysis, and Multi-Dataset Benchmarking  

---

## 📑 Slide Deck Overview

The presentation is delivered in two formats:
1. **`Results_Presentation.pptx`** (12.4 MB) — Editable 16:9 widescreen academic slide deck containing 18 professionally styled slides.
2. **`Results_Presentation.pdf`** (1.4 MB) — Rendered, vector-quality PDF presentation generated via LibreOffice.

### Slide Structure (18 Slides):
- **Slide 1:** Title, Academic Credentials, and Research Identity (Presenter: Akhil Harshavardhan, Reg: 24BCE7695).
- **Slide 2:** Executive Summary & Verification Blueprint (Objectives, Architectural Core, Verification Matrix).
- **Slide 3:** Problem Formulation: The RGB Channel Coupling Dilemma vs. Decoupled YUV Space.
- **Slide 4:** 2D Discrete Haar Wavelet Transform (DWT/IDWT) & High-Low Frequency Decomposition Block (HLFDB).
- **Slide 5:** Luminance Edge Sharpening: Gated Dilated Edge Extraction Module (GDEEM) & Global Context Refinement.
- **Slide 6:** Chrominance Restoration: Twin U-Net Denoisers, Color Attention Guidance (CAGM) & Multi-Scale Enhancement Fusion (MSEF).
- **Slide 7:** Unified 6-Component Loss Formulation ($\mathcal{L}_{SmoothL1}, \mathcal{L}_{VGG}, \mathcal{L}_{Hist}, \mathcal{L}_{PSNR}, \mathcal{L}_{Color}, \mathcal{L}_{MS-SSIM}$).
- **Slide 8:** Experimental Rigor & Dual-Platform Verification Protocols (Kaggle Tesla T4 & Local RTX 4050 GPU).
- **Slide 9:** Experiment 1: LOLv1 Canon Quantitative Evaluation (Local 20-Epoch & Cloud 250-Epoch Runs).
- **Slide 10:** LOLv1 Training Convergence Trajectory & Loss Descent Curves (`training_curves.png`).
- **Slide 11:** LOLv1 Qualitative Triplets Comparison (`comparison.png` — Low, Enhanced, Ground Truth).
- **Slide 12:** Experiment 2: LOLv2 Real-Captured Complete Benchmark (All 689 Training Pairs, 100 Test Scenes).
- **Slide 13:** LOLv2 Qualitative Visuals & Convergence Dynamics (`comparison_lolv2.png` & Restoration Analysis).
- **Slide 14:** Experiment 3: Combined 1,500+3,000 Dual (4,500+ images)-Dataset Benchmark (1,000 Paired Scenes, 115 Test Scenes).
- **Slide 15:** Combined 1,500+3,000 Dual (4,500+ images) Qualitative Visuals & Convergence Dynamics (`comparison_combined.png`).
- **Slide 16:** Comparative Multi-Benchmark Performance Matrix (RetinexNet, EnlightenGAN, KinD, Zero-DCE, MIRNet, FCE-Net Paper vs. Ours).
- **Slide 17:** Engineering Integrity & Defect Remediation Suites (7 Code Defect Fixes, 3/3 Test Suites 100% Passed).
- **Slide 18:** Mandatory University Submission Hierarchy Compliance & Research Conclusions.

---

## 🖼️ Visual Restoration Assets
- `comparison.png` — LOLv1 test set triplets.
- `comparison_lolv2.png` — LOLv2 Real-Captured test set triplets.
- `comparison_combined.png` — Combined 1,500+3,000 Dual (4,500+ images) dual-dataset visual enhancement triplets.
- `training_curves.png` — LOLv1 4-panel convergence plot.
- `training_curves_lolv2.png` — LOLv2 convergence plot.
- `training_curves_combined.png` — Combined 1,500+3,000 Dual (4,500+ images) convergence plot.
- `network_architecture.png` — Full FCE-Net modular architecture diagram.

---

## 🛠️ Automated Re-generation
To re-generate the presentation deck and PDF from source:
```bash
python3 generate_results_presentation.py
libreoffice --headless --convert-to pdf Results_Presentation.pptx
```
