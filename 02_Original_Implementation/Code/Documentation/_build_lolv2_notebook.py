#!/usr/bin/env python3
"""
Build FCE_Net_LOLv2_Evaluation.ipynb programmatically.
Cross-Dataset Evaluation on LOLv2 using LOLv1-trained checkpoint.
Registration Number: 24BCE7695
"""
import json, os

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": source.splitlines(keepends=True),
            "execution_count": None, "outputs": []}

with open("FCE_Net_Kaggle.ipynb") as f:
    orig_nb = json.load(f)

# Extract core architecture cells from original verified notebook
wavelet_code = "".join(orig_nb["cells"][7]["source"])
hlfdb_code = "".join(orig_nb["cells"][9]["source"])
model_code = "".join(orig_nb["cells"][11]["source"])

cells = []

# ── Title ─────────────────────────────────────────────────────────────
cells.append(md("""# FCE-Net: LOLv2 Cross-Dataset Evaluation & Inference
## Registration Number: 24BCE7695 | Experiment 2: Zero-Shot Cross-Dataset Benchmark

**Paper:** FCE-Net: A Frequency-Color-Edge Aware YUV Network for Low-Light Image Enhancement  
**Authors:** Zengyang Guo, Zhidan Cai, Haoyao Xia, Shifeng Hu  
**Venue:** Pattern Analysis and Applications (Springer), 2026  
**Official Repository:** https://github.com/gzy7/FCE-Net  
**Student Researcher:** Akhil Harshavardhan (`24BCE7695`)

---

### Experiment 2 Protocol
1. **Zero Training / Zero Fine-Tuning:** The model is evaluated strictly in zero-shot cross-dataset inference mode. No weights are modified or retrained.
2. **Checkpoint Provenance:** Reuses the verified `best_model.pth` trained from scratch on LOLv1 (250 epochs, Tesla T4, Best PSNR = 25.1846 dB).
3. **Dataset Scope:** Evaluates on the **LOLv2** benchmark (supporting `Real_captured` and `Synthetic` splits, up to 250 evaluation pairs).
4. **Target Outputs:**
   - Full-resolution enhanced images saved to `/kaggle/working/results_lolv2/enhanced/`
   - Quantitative evaluation report: PSNR (dB), SSIM, and LPIPS (AlexNet)
   - Visual comparison figure saved to `/kaggle/working/results_lolv2/comparison_lolv2.png`
   - Complete export package: `/kaggle/working/FCE_Net_LOLv2_Results.zip`"""))

# ── 1. Environment Setup ──────────────────────────────────────────────
cells.append(md("## 1. Environment Setup & Dependency Installation"))

cells.append(code("""import subprocess, sys

def pip_install(*pkgs):
    for p in pkgs:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])

try:
    import lpips
except ImportError:
    pip_install("lpips")

try:
    import pytorch_msssim
except ImportError:
    pip_install("pytorch_msssim")

try:
    import torchmetrics
except ImportError:
    pip_install("torchmetrics")

print("✅  All evaluation dependencies verified.")"""))

# ── 2. Hardware / Device Check ────────────────────────────────────────
cells.append(md("## 2. Hardware / Device Check"))

cells.append(code("""import torch
import os
import warnings

warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {props.total_memory / 1e9:.1f} GB")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ No GPU detected — evaluation will run on CPU.")
    print("   Recommendation: Enable Kaggle GPU accelerator for faster inference.")

print(f"Using device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")"""))

# ── 3. Source Code Setup ──────────────────────────────────────────────
cells.append(md("""## 3. Source Code Setup (Inline FCE-Net Architecture)
All neural modules are defined inline for 100% self-contained Kaggle execution.
The architecture matches the LOLv1 training experiment (686,076 parameters)."""))

cells.append(md("### 3a. Wavelet Transform (DWT / IDWT)"))
cells.append(code(wavelet_code))

cells.append(md("### 3b. HLFDB — High-Low Frequency Decomposition Block"))
cells.append(code(hlfdb_code))

cells.append(md("### 3c. FCE-Net Model"))
cells.append(code(model_code))

# ── 4. Dataloader & Image Utilities ───────────────────────────────────
cells.append(md("### 3d. Evaluation Dataset & Image Handling Utilities"))
cells.append(code("""import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class LOLv2EvalDataset(Dataset):
    \"\"\"Dataset for LOLv2 evaluation (supports both paired and unpaired testing).\"\"\"
    def __init__(self, low_paths, high_paths=None, max_images=None):
        self.low_paths = sorted(low_paths)
        self.high_paths = sorted(high_paths) if high_paths else None
        
        if self.high_paths:
            n = min(len(self.low_paths), len(self.high_paths))
            self.low_paths = self.low_paths[:n]
            self.high_paths = self.high_paths[:n]
            
        if max_images and max_images > 0:
            self.low_paths = self.low_paths[:max_images]
            if self.high_paths:
                self.high_paths = self.high_paths[:max_images]

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):
        low_p = self.low_paths[idx]
        low_img = Image.open(low_p).convert('RGB')
        low_tensor = self.to_tensor(low_img)
        fname = os.path.basename(low_p)

        if self.high_paths:
            high_p = self.high_paths[idx]
            high_img = Image.open(high_p).convert('RGB')
            high_tensor = self.to_tensor(high_img)
            return low_tensor, high_tensor, fname
        
        return low_tensor, fname

print("✅  Evaluation dataset class defined.")"""))

# ── 5. Checkpoint Auto-Detection ──────────────────────────────────────
cells.append(md("""## 4. LOLv1 Trained Checkpoint Auto-Detection & Verification
Locates the trained checkpoint from your previous LOLv1 Kaggle run.

**How to connect in Kaggle:**
1. In the right sidebar, click **"+ Add Input"** (or "Add Data").
2. Select **"Notebook Output Files"**.
3. Search for your previous LOLv1 notebook name (e.g. `fce-net-kaggle` or `notebook356021ef66`).
4. Click **"Add"**. Kaggle will mount the output under `/kaggle/input/<notebook-name>/`.
5. Alternatively, if you uploaded `best_model.pth` or `FCE_Net_Results.zip` as a dataset, it will be auto-detected."""))

cells.append(code("""import os
import glob
import zipfile
import torch

CHECKPOINT_PATH = None
EXTRACTED_DIR = "/kaggle/working/imported_checkpoint"

# 1. Search for extracted checkpoint files (.pth)
search_patterns = [
    "/kaggle/input/**/best_model.pth",
    "/kaggle/input/**/final_model.pth",
    "/kaggle/input/**/checkpoint_epoch*.pth",
    "/kaggle/input/**/*model*.pth",
    "/kaggle/working/checkpoints/best_model.pth",
    "/kaggle/working/**/best_model.pth",
    "./**/*best_model*.pth",
    "./**/*model*.pth"
]

found_pths = []
for pattern in search_patterns:
    for p in glob.glob(pattern, recursive=True):
        if os.path.isfile(p) and not p.endswith("vgg19.pth"):
            found_pths.append(p)

# 2. Search for exported ZIP archive from previous run
if not found_pths:
    zip_patterns = [
        "/kaggle/input/**/FCE_Net_Results*.zip",
        "/kaggle/input/**/*Results*.zip",
        "/kaggle/input/**/*checkpoint*.zip",
        "/kaggle/working/FCE_Net_Results.zip",
        "./*.zip"
    ]
    for pattern in zip_patterns:
        for z in glob.glob(pattern, recursive=True):
            if os.path.isfile(z):
                print(f"📦 Found checkpoint archive: {z}")
                os.makedirs(EXTRACTED_DIR, exist_ok=True)
                with zipfile.ZipFile(z, 'r') as zip_ref:
                    zip_ref.extractall(EXTRACTED_DIR)
                print(f"   Extracted to {EXTRACTED_DIR}")
                for p in glob.glob(f"{EXTRACTED_DIR}/**/*.pth", recursive=True):
                    found_pths.append(p)
                break

# 3. Validate candidates against FCE-Net architecture
model_candidate = FCE(filters=32)
expected_keys = set(model_candidate.state_dict().keys())
verified_checkpoint = None

for candidate in found_pths:
    try:
        ckpt = torch.load(candidate, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict):
            state_dict = ckpt
        else:
            continue
            
        common_keys = expected_keys.intersection(set(state_dict.keys()))
        if len(common_keys) >= len(expected_keys) * 0.9:
            verified_checkpoint = candidate
            CHECKPOINT_STATE = state_dict
            print(f"✅ VERIFIED LOLv1 CHECKPOINT: {candidate}")
            print(f"   Architecture match: {len(common_keys)}/{len(expected_keys)} tensors")
            print(f"   File size: {os.path.getsize(candidate) / 1e6:.2f} MB")
            break
    except Exception as e:
        continue

if verified_checkpoint:
    CHECKPOINT_PATH = verified_checkpoint
else:
    print("=" * 68)
    print("ℹ️  NO ATTACHED CHECKPOINT DETECTED YET")
    print("=" * 68)
    print("To attach your LOLv1 checkpoint in Kaggle:")
    print("  1. In the right panel, click '+ Add Input'")
    print("  2. Select 'Notebook Output Files'")
    print("  3. Find your previous executed notebook and click 'Add'")
    print("  4. Or click '+ Add Input' -> 'Upload Dataset' -> upload best_model.pth")
    print("=" * 68)"""))

# ── 6. LOLv2 Dataset Detection ────────────────────────────────────────
cells.append(md("""## 5. LOLv2 Dataset Auto-Discovery & Validation
Auto-scans `/kaggle/input/` for the LOLv2 dataset (Real_captured and/or Synthetic test splits)."""))

cells.append(code("""import os
import glob
from PIL import Image

# Maximum evaluation images to process (set to 250 as requested, or None for all)
MAX_EVAL_IMAGES = 250

LOLV2_TEST_LOW = None
LOLV2_TEST_HIGH = None
DATASET_SPLIT_NAME = "Unknown"

# Search candidates for LOLv2
search_dirs = [
    # LOLv2 Real captured
    ("LOLv2 Real-Captured Test",
     "/kaggle/input/**/LOLv2/Real_captured/Test/Low",
     "/kaggle/input/**/LOLv2/Real_captured/Test/Normal"),
    ("LOLv2 Real-Captured (lowercase)",
     "/kaggle/input/**/real_captured/test/low",
     "/kaggle/input/**/real_captured/test/normal"),
    ("LOLv2 Real Test",
     "/kaggle/input/**/Real/Test/Low",
     "/kaggle/input/**/Real/Test/Normal"),
    # LOLv2 Synthetic
    ("LOLv2 Synthetic Test",
     "/kaggle/input/**/LOLv2/Synthetic/Test/Low",
     "/kaggle/input/**/LOLv2/Synthetic/Test/Normal"),
    ("LOLv2 Synthetic (lowercase)",
     "/kaggle/input/**/synthetic/test/low",
     "/kaggle/input/**/synthetic/test/normal"),
    # Generic Kaggle low/high test folders
    ("Generic LOLv2 / Low-Light Test",
     "/kaggle/input/**/test/low",
     "/kaggle/input/**/test/high"),
    ("Generic LOLv2 Low/Normal",
     "/kaggle/input/**/test/low",
     "/kaggle/input/**/test/normal"),
    ("LOLv2 our / eval",
     "/kaggle/input/**/eval*/low",
     "/kaggle/input/**/eval*/high")
]

for name, low_pat, high_pat in search_dirs:
    low_matches = glob.glob(low_pat, recursive=True)
    high_matches = glob.glob(high_pat, recursive=True)
    if low_matches:
        cand_low = [d for d in low_matches if os.path.isdir(d)]
        cand_high = [d for d in high_matches if os.path.isdir(d)]
        if cand_low:
            LOLV2_TEST_LOW = cand_low[0]
            DATASET_SPLIT_NAME = name
            if cand_high:
                LOLV2_TEST_HIGH = cand_high[0]
            break

# If not found on Kaggle, check local paths or prepare synthetic audit pairs
if not LOLV2_TEST_LOW or not os.path.isdir(str(LOLV2_TEST_LOW)):
    local_cands = glob.glob("./**/Test/Low", recursive=True) + glob.glob("./**/test/low", recursive=True)
    if local_cands:
        LOLV2_TEST_LOW = local_cands[0]
        DATASET_SPLIT_NAME = "Local Test Split"
        high_cands = glob.glob("./**/Test/Normal", recursive=True) + glob.glob("./**/test/high", recursive=True)
        if high_cands:
            LOLV2_TEST_HIGH = high_cands[0]

# Display discovery summary
print("=" * 68)
print("LOLv2 DATASET DISCOVERY REPORT")
print("=" * 68)
if LOLV2_TEST_LOW:
    low_imgs = [f for f in os.listdir(LOLV2_TEST_LOW) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"✅ Split Name      : {DATASET_SPLIT_NAME}")
    print(f"   Low-light Path  : {LOLV2_TEST_LOW} ({len(low_imgs)} images)")
    if LOLV2_TEST_HIGH:
        high_imgs = [f for f in os.listdir(LOLV2_TEST_HIGH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"   Ground Truth    : {LOLV2_TEST_HIGH} ({len(high_imgs)} images)")
        print(f"   Evaluation Mode : Paired Benchmark (PSNR + SSIM + LPIPS)")
    else:
        print(f"   Ground Truth    : None detected (Unpaired inference mode)")
    print(f"   Target Limit    : {MAX_EVAL_IMAGES} images")
else:
    print("ℹ️  No LOLv2 dataset attached yet.")
    print("   To attach LOLv2 in Kaggle:")
    print("     1. Click '+ Add Input'")
    print("     2. Search 'LOL-v2' or 'LOLv2 dataset'")
    print("     3. Click 'Add'")
    print("   A 3-sample synthetic audit set will be used for smoke verification.")
print("=" * 68)"""))

# ── 7. Model Loading & Verification ───────────────────────────────────
cells.append(md("""## 6. Model Loading & Inference Smoke Test
Instantiates FCE-Net and loads the verified checkpoint weights."""))

cells.append(code("""model = FCE(filters=32).to(DEVICE)

if CHECKPOINT_PATH and 'CHECKPOINT_STATE' in globals():
    model.load_state_dict(CHECKPOINT_STATE)
    model.eval()
    print("✅ Model successfully initialized with verified LOLv1 checkpoint.")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
else:
    print("⚠️  Running in unweighted audit mode (checkpoint not yet attached).")
    print("   Attach your LOLv1 checkpoint notebook output before final benchmark.")
    model.eval()

# Smoke test with 1 forward pass
dummy_in = torch.randn(1, 3, 256, 256, device=DEVICE).clamp(0, 1)
with torch.no_grad():
    dummy_out = model(dummy_in)

assert dummy_out.shape == dummy_in.shape, f"Shape mismatch: {dummy_out.shape} vs {dummy_in.shape}"
print("✅  Inference smoke test PASSED")
print(f"    Input  shape: {dummy_in.shape}")
print(f"    Output shape: {dummy_out.shape}")
print(f"    Dynamic range: [{dummy_out.min():.4f}, {dummy_out.max():.4f}]")"""))

# ── 8. LOLv2 Inference Loop ───────────────────────────────────────────
cells.append(md("""## 7. LOLv2 Inference Loop
Runs pure inference across LOLv2 test images and saves full-resolution enhanced outputs."""))

cells.append(code("""import time
from torchvision.utils import save_image

OUTPUT_DIR = "/kaggle/working/results_lolv2/enhanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Prepare image list
if LOLV2_TEST_LOW and os.path.isdir(LOLV2_TEST_LOW):
    valid_exts = ('.png', '.jpg', '.jpeg')
    low_files = sorted([os.path.join(LOLV2_TEST_LOW, f) for f in os.listdir(LOLV2_TEST_LOW) if f.lower().endswith(valid_exts)])
    high_files = None
    if LOLV2_TEST_HIGH and os.path.isdir(LOLV2_TEST_HIGH):
        high_files = sorted([os.path.join(LOLV2_TEST_HIGH, f) for f in os.listdir(LOLV2_TEST_HIGH) if f.lower().endswith(valid_exts)])
else:
    # Synthetic fallback for zero-traceback audit
    print("Creating synthetic demo samples for dry-run verification...")
    synth_dir = "/kaggle/working/synthetic_test/low"
    synth_gt = "/kaggle/working/synthetic_test/high"
    os.makedirs(synth_dir, exist_ok=True)
    os.makedirs(synth_gt, exist_ok=True)
    for i in range(3):
        im_low = Image.new("RGB", (400, 256), color=(20 + i*10, 25 + i*10, 30 + i*10))
        im_high = Image.new("RGB", (400, 256), color=(140 + i*10, 150 + i*10, 160 + i*10))
        im_low.save(os.path.join(synth_dir, f"test_{i+1:03d}.png"))
        im_high.save(os.path.join(synth_gt, f"test_{i+1:03d}.png"))
    low_files = sorted(glob.glob(f"{synth_dir}/*.png"))
    high_files = sorted(glob.glob(f"{synth_gt}/*.png"))

eval_dataset = LOLv2EvalDataset(low_files, high_files, max_images=MAX_EVAL_IMAGES)
print(f"Processing {len(eval_dataset)} evaluation images on {DEVICE}...")

enhanced_records = []
inference_start = time.time()

for idx in range(len(eval_dataset)):
    item = eval_dataset[idx]
    if len(item) == 3:
        low_t, high_t, fname = item
    else:
        low_t, fname = item
        high_t = None
        
    low_batch = low_t.unsqueeze(0).to(DEVICE)
    
    t0 = time.time()
    with torch.no_grad():
        enhanced_batch = model(low_batch)
    latency_ms = (time.time() - t0) * 1000
    
    out_path = os.path.join(OUTPUT_DIR, fname)
    save_image(enhanced_batch.squeeze(0).cpu(), out_path)
    
    enhanced_records.append({
        "filename": fname,
        "out_path": out_path,
        "latency_ms": latency_ms
    })
    
    if (idx + 1) % 50 == 0 or idx == len(eval_dataset) - 1 or idx == 0:
        print(f"  [{idx+1:3d}/{len(eval_dataset)}] Saved: {fname} ({latency_ms:.1f} ms)")

total_inf_time = time.time() - inference_start
print()
print(f"✅ Inference completed: {len(enhanced_records)} images saved to {OUTPUT_DIR}")
print(f"   Total time: {total_inf_time:.2f} s (Avg: {total_inf_time / len(enhanced_records) * 1000:.1f} ms/image)")"""))

# ── 9. Quantitative Evaluation (PSNR, SSIM, LPIPS) ────────────────────
cells.append(md("""## 8. Quantitative Evaluation (PSNR, SSIM, LPIPS)
Computes standard full-reference benchmark metrics against paired LOLv2 ground truth."""))

cells.append(code("""import csv
import json
import numpy as np
import lpips as lpips_lib
from pytorch_msssim import ssim as ssim_fn

METRICS_CSV = "/kaggle/working/results_lolv2/metrics_lolv2.csv"
SUMMARY_JSON = "/kaggle/working/results_lolv2/summary_lolv2.json"

has_gt = eval_dataset.high_paths is not None and len(eval_dataset.high_paths) > 0

if has_gt:
    print("Setting up perceptual loss (LPIPS - AlexNet trunk)...")
    lpips_fn = lpips_lib.LPIPS(net="alex").to(DEVICE)
    lpips_fn.eval()
    
    psnr_list, ssim_list, lpips_list = [], [], []
    detailed_metrics = []
    
    for idx in range(len(eval_dataset)):
        low_t, high_t, fname = eval_dataset[idx]
        enhanced_p = os.path.join(OUTPUT_DIR, fname)
        enhanced_img = Image.open(enhanced_p).convert('RGB')
        enhanced_t = transforms.ToTensor()(enhanced_img).to(DEVICE)
        high_t = high_t.to(DEVICE)
        
        # Calculate PSNR
        mse = torch.mean((enhanced_t - high_t) ** 2).item()
        psnr_val = 10 * np.log10(1.0 / (mse + 1e-10))
        
        # Calculate SSIM
        ssim_val = ssim_fn(enhanced_t.unsqueeze(0), high_t.unsqueeze(0), data_range=1.0).item()
        
        # Calculate LPIPS (requires [-1, 1] range)
        enh_norm = (enhanced_t.unsqueeze(0) * 2.0) - 1.0
        high_norm = (high_t.unsqueeze(0) * 2.0) - 1.0
        with torch.no_grad():
            lpips_val = lpips_fn(enh_norm, high_norm).item()
            
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        lpips_list.append(lpips_val)
        
        detailed_metrics.append({
            "image": fname,
            "psnr": round(psnr_val, 4),
            "ssim": round(ssim_val, 4),
            "lpips": round(lpips_val, 4)
        })
        
        if (idx + 1) % 50 == 0 or idx == len(eval_dataset) - 1 or idx < 5:
            print(f"  [{idx+1:3d}/{len(eval_dataset)}] {fname} -> PSNR: {psnr_val:.4f} dB | SSIM: {ssim_val:.4f} | LPIPS: {lpips_val:.4f}")
            
    summary_results = {
        "dataset_split": DATASET_SPLIT_NAME,
        "checkpoint_used": CHECKPOINT_PATH if CHECKPOINT_PATH else "Unweighted_Baseline",
        "num_evaluated": len(psnr_list),
        "mean_psnr": round(float(np.mean(psnr_list)), 4),
        "std_psnr": round(float(np.std(psnr_list)), 4),
        "mean_ssim": round(float(np.mean(ssim_list)), 4),
        "std_ssim": round(float(np.std(ssim_list)), 4),
        "mean_lpips": round(float(np.mean(lpips_list)), 4),
        "std_lpips": round(float(np.std(lpips_list)), 4)
    }
    
    # Save CSV
    with open(METRICS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "psnr", "ssim", "lpips"])
        writer.writeheader()
        writer.writerows(detailed_metrics)
        
    # Save JSON
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary_results, f, indent=2)
        
    print()
    print("=" * 50)
    print("      LOLv2 QUANTITATIVE BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"  Evaluated Images : {summary_results['num_evaluated']}")
    print(f"  Mean PSNR        : {summary_results['mean_psnr']:.4f} dB (± {summary_results['std_psnr']:.4f})")
    print(f"  Mean SSIM        : {summary_results['mean_ssim']:.4f} (± {summary_results['std_ssim']:.4f})")
    print(f"  Mean LPIPS       : {summary_results['mean_lpips']:.4f} (± {summary_results['std_lpips']:.4f})")
    print("=" * 50)
    print(f"Metrics saved: {METRICS_CSV}")
    print(f"Summary saved: {SUMMARY_JSON}")
else:
    print("ℹ️  Ground truth images not available for this split — full-reference metrics skipped.")"""))

# ── 10. Visualizations ────────────────────────────────────────────────
cells.append(md("""## 9. Visual Comparisons & Metric Distributions
Renders side-by-side visual comparisons and metric distribution charts."""))

cells.append(code("""try:
    import matplotlib.pyplot as plt

    n_samples = min(5, len(eval_dataset))
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        item = eval_dataset[i]
        fname = item[2] if len(item) == 3 else item[1]
        
        low_p = eval_dataset.low_paths[i]
        out_p = os.path.join(OUTPUT_DIR, fname)
        
        low_img = Image.open(low_p).convert('RGB')
        out_img = Image.open(out_p).convert('RGB')
        
        axes[i, 0].imshow(low_img)
        axes[i, 0].set_title(f"Low-Light Input: {fname}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(out_img)
        axes[i, 1].set_title(f"FCE-Net Enhanced (LOLv1 weights)")
        axes[i, 1].axis('off')
        
        if len(item) == 3:
            high_p = eval_dataset.high_paths[i]
            high_img = Image.open(high_p).convert('RGB')
            axes[i, 2].imshow(high_img)
            axes[i, 2].set_title(f"Ground Truth (Normal Light)")
        else:
            axes[i, 2].text(0.5, 0.5, "Unpaired Reference", ha='center', va='center')
        axes[i, 2].axis('off')

    plt.tight_layout()
    comp_path = "/kaggle/working/results_lolv2/comparison_lolv2.png"
    plt.savefig(comp_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅  Visual comparison saved to {comp_path}")
except ImportError:
    print("matplotlib not available.")"""))

# ── 11. Artifact Export ───────────────────────────────────────────────
cells.append(md("""## 10. Export Artifacts for University Submission
Packages all LOLv2 results into a single archive for `24BCE7695` university submission."""))

cells.append(code(r"""import shutil

EXPORT_DIR = "/kaggle/working/results_lolv2"
ZIP_PATH = "/kaggle/working/FCE_Net_LOLv2_Results"

# Write run manifest
manifest_path = os.path.join(EXPORT_DIR, "EVALUATION_REPORT.txt")
with open(manifest_path, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("FCE-Net LOLv2 Cross-Dataset Evaluation Manifest\n")
    f.write("Student Registration : 24BCE7695\n")
    f.write("Experiment           : Experiment 2 (Zero-Shot Cross-Dataset)\n")
    f.write(f"Checkpoint Used      : {CHECKPOINT_PATH}\n")
    f.write(f"Dataset Split        : {DATASET_SPLIT_NAME}\n")
    f.write(f"Device               : {DEVICE}\n")
    f.write("=" * 60 + "\n")

# Make archive
archive_file = shutil.make_archive(ZIP_PATH, "zip", EXPORT_DIR)
zip_size_mb = os.path.getsize(archive_file) / (1024 * 1024)

print("=" * 68)
print("✅  LOLv2 EVALUATION ARTIFACTS PACKAGED SUCCESSFULLY")
print("=" * 68)
print(f"Archive file : {archive_file} ({zip_size_mb:.2f} MB)")
print("Contents:")
print("  - enhanced/          : Enhanced high-resolution output images")
print("  - metrics_lolv2.csv  : Per-image PSNR, SSIM, LPIPS metrics")
print("  - summary_lolv2.json : Aggregate benchmark statistics")
print("  - comparison_lolv2.png : High-resolution side-by-side visual comparison")
print("  - EVALUATION_REPORT.txt: Execution and environment manifest")
print("=" * 68)
print("Next steps:")
print("  1. In Kaggle, open the 'Output' pane on the right.")
print("  2. Download 'FCE_Net_LOLv2_Results.zip'.")
print("  3. Place the extracted files into '02_Original_Implementation/Results/LOLv2/'.")
print("=" * 68)"""))

# ── Assemble notebook ─────────────────────────────────────────────────
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = "FCE_Net_LOLv2_Evaluation.ipynb"
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=2)

print(f"✅ Generated {out_path} ({os.path.getsize(out_path)} bytes, {len(cells)} cells)")
