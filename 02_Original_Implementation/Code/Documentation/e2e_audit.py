import matplotlib
matplotlib.use("Agg")
#!/usr/bin/env python3
"""
End-to-end audit: Execute every notebook cell in sequence, simulating Kaggle.
Skips only the actual training loop (cell 28) and pip installs (cell 2).
Reports PASS/FAIL for each cell with actual error messages.
"""
import sys, os, traceback, json, tempfile, shutil

candidate_nbs = [
    "FCE_Net_Kaggle.ipynb",
    os.path.join(os.path.dirname(__file__), "FCE_Net_Kaggle.ipynb"),
    os.path.join(os.path.dirname(__file__), "../Notebooks/FCE_Net_Kaggle.ipynb"),
    "/home/charizard/24BCE7695/FINAL_SUBMISSION/24BCE7695/02_Original_Implementation/Code/Notebooks/FCE_Net_Kaggle.ipynb"
]
NOTEBOOK = next((p for p in candidate_nbs if os.path.exists(p)), None)
assert NOTEBOOK is not None, "FCE_Net_Kaggle.ipynb not found"

# Create a temp dir to simulate /kaggle/working
WORK_DIR = tempfile.mkdtemp(prefix="kaggle_working_")
os.makedirs(os.path.join(WORK_DIR, "checkpoints"), exist_ok=True)

# Shared namespace simulating Jupyter kernel state
ns = {"__builtins__": __builtins__}

with open(NOTEBOOK) as f:
    nb = json.load(f)

code_cells = [(i, ''.join(c['source'])) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']

results = []

import torch

def run_cell(idx, src, label="", skip=False):
    """Execute one cell in the shared namespace."""
    tag = f"Cell {idx:2d}"
    if skip:
        results.append((tag, "SKIP", label))
        print(f"  [{tag}] SKIP — {label}")
        return True
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        code = compile(src, f"<cell_{idx}>", "exec")
        exec(code, ns)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        results.append((tag, "PASS", label))
        print(f"  [{tag}] PASS — {label}")
        return True
    except Exception as e:
        tb = traceback.format_exc().strip().split("\n")
        short = tb[-1][:120]
        results.append((tag, "FAIL", f"{label}: {short}"))
        print(f"  [{tag}] FAIL — {label}")
        print(f"         {short}")
        return False

print("=" * 70)
print("  FCE-Net Notebook — End-to-End Sequential Audit")
print(f"  Simulated working dir: {WORK_DIR}")
print("=" * 70)

for idx, src in code_cells:
    first_line = src.strip().split("\n")[0][:70]

    # Cell 2: pip install — skip (deps already installed locally)
    if idx == 2:
        run_cell(idx, src, "pip installs (skip — deps already present)", skip=True)
        continue

    # Cell 4: device check — just run it
    if idx == 4:
        run_cell(idx, src, "GPU/device detection")
        continue

    # Cell 7,9,11,13,15: source code definitions
    if idx in (7, 9, 11, 13, 15):
        run_cell(idx, src, f"Source code: {first_line}")
        continue

    # Cell 17: dataset detection — will report "not found" on local machine (expected)
    if idx == 17:
        run_cell(idx, src, "Dataset path detection")
        # Check variables are set
        tl = ns.get("TRAIN_LOW")
        tst = ns.get("TEST_LOW")
        if tl:
            print(f"         → Train: {tl}")
        else:
            print(f"         → Dataset not found (expected on local machine)")
        continue

    # Cell 19: dataset visualization — will fail if no dataset, so guard
    if idx == 19:
        # Patch to skip display if no dataset
        if not ns.get("TRAIN_LOW"):
            run_cell(idx, src, "Dataset visualization (no dataset — expecting graceful skip)", skip=False)
            continue
        run_cell(idx, src, "Dataset visualization")
        continue

    # Cell 21: model init
    if idx == 21:
        run_cell(idx, src, "Model initialization + shape check")
        continue

    # Cell 23: loss verification
    if idx == 23:
        run_cell(idx, src, "Loss function verification")
        continue

    # Cell 25: smoke test
    if idx == 25:
        run_cell(idx, src, "Smoke test (1-epoch mini-train)")
        continue

    # Cell 27: training config (defines constants, helpers — does NOT train)
    if idx == 27:
        # Patch output paths to our temp dir
        src_patched = src.replace('"/kaggle/working/checkpoints"', f'"{WORK_DIR}/checkpoints"')
        src_patched = src_patched.replace('"/kaggle/working/training_log.csv"', f'"{WORK_DIR}/training_log.csv"')
        run_cell(idx, src_patched, "Training config + helpers (patched paths)")
        continue

    # Cell 28: FULL TRAINING LOOP — now guarded with START_FULL_TRAINING = False by default
    if idx == 28:
        src_patched = src.replace('"/kaggle/working/checkpoints"', f'"{WORK_DIR}/checkpoints"')
        src_patched = src_patched.replace('"/kaggle/working/training_log.csv"', f'"{WORK_DIR}/training_log.csv"')
        run_cell(idx, src_patched, "Guarded training loop (START_FULL_TRAINING=False verifies safe default skip)")
        continue

    # Cell 30: checkpoint save/load test
    if idx == 30:
        run_cell(idx, src, "Checkpoint save/load verification")
        continue

    # Cell 32: inference — needs checkpoint + dataset; test definition only
    if idx == 32:
        # Patch result dir
        src_patched = src.replace('"/kaggle/working/results"', f'"{WORK_DIR}/results"')
        run_cell(idx, src_patched, "Inference cell (no checkpoint — expecting graceful skip)")
        continue

    # Cell 34: evaluation
    if idx == 34:
        src_patched = src.replace('RESULT_DIR', f'"{WORK_DIR}/results"')
        run_cell(idx, src_patched, "Evaluation cell (no results — expecting graceful skip)")
        continue

    # Cell 36: result visualization
    if idx == 36:
        src_patched = src.replace('RESULT_DIR', f'"{WORK_DIR}/results"')
        run_cell(idx, src_patched, "Result visualization (graceful skip when no results)")
        continue

    # Cell 38: training curves
    if idx == 38:
        src_patched = src.replace('LOG_FILE', f'"{WORK_DIR}/training_log.csv"')
        run_cell(idx, src_patched, "Training curves (graceful skip when no log file)")
        continue

    # Cell 40: export
    if idx == 40:
        src_patched = src.replace("/kaggle/working", WORK_DIR)
        run_cell(idx, src_patched, "Export cell (patched paths)")
        continue

    # Fallback
    run_cell(idx, src, first_line)

# Summary
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
passed = sum(1 for _, s, _ in results if s == "PASS")
failed = sum(1 for _, s, _ in results if s == "FAIL")
skipped = sum(1 for _, s, _ in results if s == "SKIP")
total = len(results)

for tag, status, desc in results:
    icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️"}[status]
    print(f"  {icon} [{tag}] {status}: {desc}")

print(f"\n  Total: {total}  |  PASS: {passed}  |  FAIL: {failed}  |  SKIP: {skipped}")

if failed > 0:
    print("\n  ⚠️  Some cells FAILED — review errors above")
    sys.exit(1)
else:
    print("\n  ✅  All executable cells PASSED")
    sys.exit(0)

