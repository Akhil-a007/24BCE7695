# AUDIT.md — FCE-Net Implementation & Research Paper Verification Log

**Project:** 24BCE7695  
**Paper:** FCE-Net: A Frequency-Color-Edge Aware YUV Network for Low-Light Image Enhancement  
**Authors:** Zengyang Guo, Zhidan Cai, Haoyao Xia, Shifeng Hu (Springer, 2026)  
**Student Researcher:** Akhil Harshavardhan (`24BCE7695`)  
**Last updated:** 2026-10-29 23:20 IST  
**Status:** ✅ COMPLETE & VERIFIED — Ready for Kaggle Execution & Overleaf Authoring

---

## 1. Deliverables Manifest

| Deliverable | Status | Location / Output Path | Details |
|-------------|--------|------------------------|---------|
| **PowerPoint Presentation**| ✅ Generated| `04_Presentation/FCE_Net_Presentation.pptx` & `./FCE_Net_Presentation.pptx` | 12-slide modern 16:9 widescreen presentation deck (294 KB) |
| **Presentation PDF** | ✅ Exported | `04_Presentation/FCE_Net_Presentation.pdf` & `./FCE_Net_Presentation.pdf` | Clean PDF export via LibreOffice (544 KB, 12 slides) |
| **Presentation Outline** | ✅ Complete | `04_Presentation/presentation_outline.md` | Comprehensive 12-slide presentation content outline |
| **Research Paper (LaTeX)** | ✅ Complete | `01_Base_Paper/Research_Paper/main.tex` | 7 pages, two-column publication layout, Overleaf/pdfLaTeX ready |
| **Research Paper (PDF)** | ✅ Compiled | `FCE_Net_Research_Paper.pdf` & `01_Base_Paper/FCE_Net_Research_Paper.pdf` | Clean compilation via `pdflatex` (628 KB, 7 pages) |
| **Overleaf ZIP Archive** | ✅ Packaged | `Research_Paper_Overleaf.zip` & `01_Base_Paper/Research_Paper_Overleaf.zip` | Includes `main.tex`, `references.bib`, `figures/network.png`, `README.md` |
| **Research Paper (Markdown)**| ✅ Complete | `RESEARCH_PAPER.md` & `01_Base_Paper/RESEARCH_PAPER.md` | Full markdown reproduction report with equations and tables |
| **Kaggle Notebook** | ✅ Hardened | `FCE_Net_Kaggle.ipynb` | 41 cells, beginner-safe bounded smoke test, guarded full training |
| **Notebook Generator** | ✅ Verified | `_build_notebook.py` | Programmatic generator with `splitlines(keepends=True)` |
| **Base Code (GitHub)** | ✅ Preserved | `01_Base_Paper/Base_Code/` | Byte-for-byte untouched clone of official repo |
| **Source Implementation** | ✅ Hardened | `02_Original_Implementation/Code/` | Fixed wave stdlib conflict, VGG weights, autocast AMP |
| **Architecture Diagram** | ✅ High-Res | `04_Presentation/network_architecture.png` | 2921×949 high-resolution diagram from paper |
| **Dataset Documentation** | ✅ Complete | `DATASET_INFO.md` | LOLv1 layout, splits, auto-detection paths |
| **Unit Smoke Test** | ✅ 7/7 PASS | `smoke_test.py` | Modules, DWT, Losses, Shapes, Checkpoint |
| **Integration Test** | ✅ 5/5 PASS | `integration_test.py` | Full forward pass, 330/330 param gradients, round-trip |
| **E2E Sequential Audit** | ✅ 19/19 PASS | `e2e_audit.py` | Full sequential simulation of Kaggle kernel |

---

## 2. Beginner-Safe Execution Verification (Kaggle Audit)

### 2.1 Default "Run All" Behavior
- When a user imports `FCE_Net_Kaggle.ipynb` to Kaggle and clicks **"Run All"**, it executes:
  1. Dependencies check & device detection (CUDA or CPU).
  2. Inlined architecture definitions (DWT, HLFDB, Denoisers, CAGM, MSEF, 6 loss terms).
  3. Dataset auto-detection across 6 standard paths (falls back gracefully to synthetic if dataset not yet attached).
  4. Dataset validation (skips cleanly without confusing tracebacks if dataset not found).
  5. Model initialization: fresh weights initialized from scratch (686,076 parameters, 2.74 MB float32).
  6. Shape checks: $128 \times 128$, $256 \times 256$, $400 \times 600$ verified.
  7. Loss component verification: Smooth L1, PSNR, Color, MS-SSIM, Histogram, Perceptual, and backward pass verified.
  8. **Bounded Smoke Test:** executes a 3-step mini-train on device (taking ~4–9 seconds).
  9. **Guarded Full Training (Cell 28):** `START_FULL_TRAINING = False` stops execution with an informative banner. Multi-hour training cannot start accidentally.
  10. Checkpoint save/load round-trip: verified using temporary state dict ($0.00$ numerical difference).
  11. Post-training cells (Inference, Evaluation, Visualization, Curves): gracefully notify user that full training was skipped and exit cleanly.
  12. Export Artifacts: creates `RUN_SUMMARY.txt` manifest and packages `FCE_Net_Results.zip` in `/kaggle/working/`.

### 2.2 Enabling Full Training
To run full training (1500 epochs, ~3–6 hours on GPU):
1. User attaches the LOL dataset via Kaggle UI (`Add Data` → search `LOL dataset` by Soumik Rakshit).
2. User selects an accelerator (`Settings` → `Accelerator` → `GPU T4 x2` or `P100`).
3. User changes `START_FULL_TRAINING = True` in Section 9 (Cell 28) and runs it.
4. Subsequent cells (11–14) will then automatically run inference on test images, compute PSNR/SSIM/LPIPS, display comparison plots, plot training curves, and package all checkpoints and results into `FCE_Net_Results.zip`.

### 2.3 Confirmation of Zero Checkpoint Leakage
- **Audit finding:** The notebook **never** downloads or loads author-provided pretrained checkpoints.
- The FCE model is initialized strictly from scratch using standard normal/kaiming initialization (`_init_weights`).
- The only weights loaded from external sources are torchvision's standard VGG-19 features (`models.vgg19(weights=VGG19_Weights.DEFAULT).features[:16]`), which is the universal feature extractor for perceptual loss.
- All checkpoints saved and loaded during training are generated locally in `/kaggle/working/checkpoints/`.

---

## 3. Local Verification Results & Test Evidence

### 3.1 Unit Test Suite (`smoke_test.py`)
```
============================================================
  FCE-Net Local Smoke Test
============================================================
[1/7] Testing imports... (torch 2.15.0+cpu, pywt, einops)       ✅ PASS
[2/7] Testing Wavelet Transform (DWT/IDWT sub-bands)            ✅ PASS
[3/7] Building FCE model (Denoiser, MHSA, MSEFBlock shapes)     ✅ PASS
[4/7] Parameter count (Denoiser 8,363 params)                  ✅ PASS
[5/7] Testing loss functions (SL1, PSNR, MSSSIM, Color, Hist)   ✅ PASS
[6/7] Testing backward pass (gradients flow to input)          ✅ PASS
[7/7] Testing checkpoint save/load (diff: 0.00e+00)             ✅ PASS
============================================================
  ALL SMOKE TESTS PASSED ✅ (7/7)
============================================================
```

### 3.2 Full Integration Test (`integration_test.py`)
```
============================================================
  FCE-Net Full Integration Test (CPU)
============================================================
[1/5] Defining all modules from notebook                        ✅ PASS
[2/5] Full model construction (686,076 parameters, 2.74 MB)    ✅ PASS
[3/5] Forward pass with 128x128 input (range [0, 1], 0.22s)     ✅ PASS
[4/5] Loss computation + backward pass (330/330 params grad)   ✅ PASS
[5/5] Checkpoint round-trip (size: 2.88 MB, diff: 0.00e+00)    ✅ PASS
============================================================
  ALL INTEGRATION TESTS PASSED ✅ (5/5)
============================================================
```

### 3.3 End-to-End Sequential Kernel Audit (`e2e_audit.py`)
Sequential simulation of Jupyter kernel state executing all 20 code cells in `FCE_Net_Kaggle.ipynb`:

| Cell | Notebook Section | Execution Result | Behavior |
|------|------------------|------------------|----------|
| 2 | Section 1: Dependencies | ⏭️ SKIP | Handled via pre-installed packages |
| 4 | Section 2: Device Check | ✅ PASS | Detects hardware, prints device details |
| 7 | Section 3a: Wavelet | ✅ PASS | DWT_2D and IDWT_2D classes initialized |
| 9 | Section 3b: HLFDB | ✅ PASS | High-Low Frequency Decomposition block |
| 11 | Section 3c: FCE Model | ✅ PASS | Full model architecture constructed |
| 13 | Section 3d: Losses | ✅ PASS | CombinedLoss with all 6 terms |
| 15 | Section 3e: Dataloader | ✅ PASS | Paired dataset with random cropping |
| 17 | Section 4: Dataset Detect | ✅ PASS | Auto-detects 6 candidate paths |
| 19 | Section 5: Dataset Valid | ✅ PASS | Gracefully skips when dataset absent |
| 21 | Section 6: Model Init | ✅ PASS | 686,076 params, forward pass validated |
| 23 | Section 7: Loss Verify | ✅ PASS | All 6 components + backward pass validated |
| 25 | Section 8: Smoke Test | ✅ PASS | 3-step mini-train completes in 9.5s (loss: 16.18 → 14.81) |
| 27 | Section 9a: Train Config | ✅ PASS | Defines hyperparameters, Adam, CosineAnnealing |
| 28 | Section 9b: Full Training | ✅ PASS | Guarded: `START_FULL_TRAINING=False` prints banner & skips |
| 30 | Section 10: Checkpoint Test | ✅ PASS | Temporary checkpoint save/load diff: 0.00e+00 |
| 32 | Section 11: Inference | ✅ PASS | Gracefully skips without checkpoint |
| 34 | Section 12: Evaluation | ✅ PASS | Gracefully skips without result images |
| 36 | Section 13a: Visualization | ✅ PASS | Gracefully skips without images |
| 38 | Section 13b: Curves | ✅ PASS | Gracefully skips without CSV rows |
| 40 | Section 14: Export | ✅ PASS | Manifest written & `FCE_Net_Results.zip` created |

**Result:** `19/19 executable PASS, 1 skip (pip install)` — **100% SUCCESS.**

---

## 4. Overleaf LaTeX Compilation Evidence

```
$ cd 01_Base_Paper/Research_Paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
This is pdfTeX, Version 3.141592653-2.6-1.40.26 (TeX Live 2025/dev/Debian)
Output written on main.pdf (7 pages, 628,493 bytes).
Transcript written on main.log.
```
- **Total Pages:** 7 pages.
- **Figures:** Embedded `figures/network.png` (2921×949 high-resolution diagram).
- **Citations:** Full BibTeX resolution with 12 cited papers (Guo 2026, Wei 2018, Zhang 2019, Guo 2020, Jiang 2021, Zamir 2020, Mallat 1989, Vaswani 2017, Johnson 2016, Wang 2003, Zhang 2018, Hu 2018).
- **Overleaf Archive:** `Research_Paper_Overleaf.zip` verified and unpackable into Overleaf with zero configuration.

---

## 5. Architectural & Methodological Integrity

- **Color Space:** Decoupled YCbCr (BT.601) strictly preserved.
- **Luminance Branch:** Haar DWT/IDWT 3-stage cascaded HLFDB with MHSA, EdgeEnhance (dilated + pointwise gated), and $8\times8$ MaxPool/Upsample global illumination refinement strictly preserved.
- **Chrominance Branches:** Twin U-Net denoisers with bottleneck self-attention strictly preserved.
- **Fusion:** Color Attention Guidance (CAGM) + Multi-Scale Enhancement Fusion (MSEF) with depthwise separable conv and SE attention strictly preserved.
- **Loss Weights:** Exactly matching paper:
  - $\alpha_1 = 1.0000$ (Smooth L1)
  - $\alpha_2 = 0.0600$ (VGG-19 Perceptual)
  - $\alpha_3 = 0.0500$ (Histogram Loss)
  - $\alpha_4 = 0.5000$ (PSNR Loss)
  - $\alpha_5 = 0.0083$ (Color Loss)
  - $\alpha_6 = 0.2500$ (Multi-Scale SSIM)
- **Training Hyperparameters:** Adam lr=$2\times10^{-4}$, CosineAnnealingLR $T_{max}=1500$, batch size 1, crop size 256.

---

## 6. Summary for User

The user can import `FCE_Net_Kaggle.ipynb` into Kaggle and click **"Run All"**:
- It will execute environment setup, model initialization, loss verification, and the bounded smoke test cleanly.
- It will **NOT** accidentally start the 1500-epoch full training.
- It will save all outputs and manifests to Kaggle's `/kaggle/working/` and generate a downloadable `FCE_Net_Results.zip`.
- To train, the user simply sets `START_FULL_TRAINING = True` in Section 9 and runs that cell.
- The research paper is complete, compiled to PDF (`FCE_Net_Research_Paper.pdf`), and ready to upload to Overleaf (`Research_Paper_Overleaf.zip`).
