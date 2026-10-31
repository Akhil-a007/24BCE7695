# FCE-Net Implementation & Research Paper — README

## Project: 24BCE7695
**Paper:** FCE-Net: A Frequency-Color-Edge Aware YUV Network for Low-Light Image Enhancement  
**Authors:** Zengyang Guo, Zhidan Cai, Haoyao Xia, Shifeng Hu  
**Published:** Pattern Analysis and Applications (Springer), 2026  
**Official Repository:** https://github.com/gzy7/FCE-Net  
**Student Researcher:** Akhil Harshavardhan (`24BCE7695`)

---

## Deliverables Summary

| Artifact | File / Path | Purpose |
|----------|-------------|---------|
| **PowerPoint Presentation** | `FCE_Net_Presentation.pptx` | Professional 16:9 widescreen presentation deck (12 slides) |
| **Presentation PDF** | `FCE_Net_Presentation.pdf` | Exported presentation in PDF format |
| **Kaggle Notebook** | `FCE_Net_Kaggle.ipynb` | Fully self-contained notebook (smoke test by default; guarded full training) |
| **Research Paper (PDF)** | `FCE_Net_Research_Paper.pdf` | 7-page publication-ready academic paper (compiled with pdfLaTeX) |
| **Overleaf LaTeX Package** | `Research_Paper_Overleaf.zip` | Complete archive for Overleaf (`main.tex`, `references.bib`, figures, README) |
| **Research Paper (Markdown)**| `RESEARCH_PAPER.md` | Full academic reproduction report in Markdown format |
| **Verification Log** | `AUDIT.md` | Verification record with unit, integration, and E2E kernel audit evidence |

---

## Quick Start — Running on Kaggle

### Step 1: Upload the Notebook
1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **"+ New Notebook"**
3. Click **File → Import Notebook**
4. Upload `FCE_Net_Kaggle.ipynb` from this project folder

### Step 2: Add the Dataset (Optional for Smoke Test, Required for Training)
1. In the notebook sidebar, click **"Add Data"** (or the **+** icon on the right panel)
2. Search for **"LOL dataset"** (by Soumik Rakshit) — this is LOLv1
3. Click **"Add"** to attach it to your notebook
4. The notebook auto-detects it at `/kaggle/input/lol-dataset/`

### Step 3: Enable GPU Accelerator
1. Click **Settings** (gear icon) on the right panel
2. Under **Accelerator**, select **GPU T4 x2** or **GPU P100**
3. The notebook auto-detects CUDA and configures mixed-precision (AMP)

### Step 4: Run the Notebook

#### Default Execution — Bounded Smoke Test (~1–2 Minutes)
- Click **"Run All"**.
- By default, the notebook performs:
  1. Dependencies check and GPU detection
  2. Source code definitions (all modules inline)
  3. Dataset path detection (or informative fallback if not yet added)
  4. Model initialization (686,076 parameters, 0 pretrained checkpoints)
  5. Shape checks ($128 \times 128$, $256 \times 256$, $400 \times 600$)
  6. All 6 loss components test + backward gradient check
  7. **Bounded Smoke Test:** runs a 3-step mini-train to verify the entire pipeline
  8. **Guarded Training:** Cell 28 safely stops because `START_FULL_TRAINING = False` (prevents multi-hour accidental training)
  9. Exports a verification manifest to `/kaggle/working/FCE_Net_Results.zip`

#### Full Training (1500 Epochs, ~3–6 Hours on GPU)
- When ready to train the full model:
  1. In Section 9 (Cell 28), change:
     ```python
     START_FULL_TRAINING = True
     ```
  2. Run Cell 28 manually.
  3. The model trains for 1500 epochs, saving checkpoints to `/kaggle/working/checkpoints/` and logs to `/kaggle/working/training_log.csv`.
  4. Cells 11–14 then execute inference, compute PSNR/SSIM/LPIPS, plot curves, and package all trained artifacts into `FCE_Net_Results.zip`.

---

## Overleaf Quick Start

1. Go to [Overleaf](https://www.overleaf.com/).
2. Click **New Project** → **Upload Project**.
3. Select `Research_Paper_Overleaf.zip` from this directory.
4. Ensure compiler is set to **pdfLaTeX** (Menu → Compiler → pdfLaTeX).
5. Click **Recompile**. The complete 7-page paper with equations, tables, figures, and bibliography will compile with zero errors.

---

## Project Structure

```
24BCE7695/
├── FCE_Net_Presentation.pptx        ← Professional 16:9 widescreen presentation deck
├── FCE_Net_Presentation.pdf         ← Exported presentation in PDF format
├── FCE_Net_Kaggle.ipynb              ← Main deliverable (upload to Kaggle)
├── FCE_Net_Research_Paper.pdf        ← Compiled 7-page research paper PDF
├── Research_Paper_Overleaf.zip       ← Overleaf-ready LaTeX package
├── RESEARCH_PAPER.md                 ← Research paper in Markdown format
├── README.md                          ← This guide
├── PROJECT_PLAN.md                    ← Planning & architectural analysis
├── AUDIT.md                           ← Verification evidence & test logs
├── DATASET_INFO.md                    ← LOLv1 dataset documentation
│
├── 01_Base_Paper/
│   ├── FCE_Net_Research_Paper.pdf    ← Copy of compiled paper PDF
│   ├── Research_Paper_Overleaf.zip   ← Copy of Overleaf zip archive
│   ├── RESEARCH_PAPER.md             ← Copy of research paper markdown
│   ├── Research_Paper/               ← Unpacked LaTeX sources (main.tex, references.bib, etc.)
│   └── Base_Code/                     ← Untouched original GitHub source
│
├── 02_Original_Implementation/
│   ├── Code/                          ← Source files with fixes applied
│   ├── Log_Files/                     ← Training logs (populated after Kaggle training)
│   ├── Checkpoints/                   ← Model checkpoints (populated after Kaggle training)
│   └── Results/                       ← Output images (populated after Kaggle testing)
│
├── 03_Modified_Implementation/        ← Reserved for future architectural extensions
│
├── 04_Presentation/
│   ├── FCE_Net_Presentation.pptx     ← Slide deck in PPTX format
│   ├── FCE_Net_Presentation.pdf      ← Slide deck in PDF format
│   ├── presentation_outline.md        ← 12-slide comprehensive presentation outline
│   └── network_architecture.png       ← High-res architecture diagram (2921×949)
│
├── smoke_test.py                      ← Unit test suite (7/7 PASS)
├── integration_test.py                ← Integration test suite (5/5 PASS)
├── e2e_audit.py                       ← End-to-end sequential kernel audit (19/19 PASS)
├── _build_notebook.py                 ← Programmatic notebook generator
└── _original_repo/                    ← Full git clone of official repository
```

---

## Architecture Overview

FCE-Net processes images in YUV (YCbCr BT.601) color space with three main branches:

1. **Luminance ($Y$) branch:** Conv $\to$ HFDM (3-stage Haar wavelet DWT/IDWT + Transformer attention + Dense connections) $\to$ EdgeEnhance (dilated + pointwise conv with gating) $\to$ MaxPool/MHSA/Upsample refinement.
2. **Chrominance ($Cb, Cr$) branches:** Each denoised via U-Net encoder-decoder with MHSA bottleneck.
3. **Fusion:** Color Attention Guidance (CAGM) + Multi-Scale Enhancement Fusion (MSEF) $\to$ Recombine $\to$ Sigmoid $\to$ RGB output.

**Model size:** 686,076 parameters (~0.69M), 2.88 MB checkpoint.

**Loss function:** 6-component weighted sum:
$$\mathcal{L}_{total} = 1.0 \mathcal{L}_{SL1} + 0.06 \mathcal{L}_{VGG} + 0.05 \mathcal{L}_{Hist} + 0.5 \mathcal{L}_{PSNR} + 0.0083 \mathcal{L}_{Color} + 0.25 \mathcal{L}_{MS-SSIM}$$

**Training:** Adam optimizer, lr=$2\times10^{-4}$, CosineAnnealingLR ($T_{max}=1500$), batch size 1, $256\times256$ random crops, mixed precision on GPU.

---

## Local Verification Status (All Passing)

| Test Suite | Command | Result |
|------------|---------|--------|
| Unit Smoke Test | `python3 smoke_test.py` | ✅ 7/7 PASS |
| Integration Test | `python3 integration_test.py` | ✅ 5/5 PASS |
| End-to-End Sequential Kernel Audit | `python3 e2e_audit.py` | ✅ 19/19 PASS (1 skip for pip) |
| LaTeX Paper Compilation | `pdflatex main.tex` | ✅ 7 pages, zero errors |
