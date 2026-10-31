#!/usr/bin/env python3
"""Full model integration test — builds the complete FCE network and runs
a forward pass + loss backward on CPU with a small dummy input."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("  FCE-Net Full Integration Test (CPU)")
print("=" * 60)

# Execute notebook cells 3a-3e inline by running the notebook code
# We'll extract and exec the code cells from the notebook
import json

candidate_paths = [
    os.path.join(os.path.dirname(__file__), "FCE_Net_Kaggle.ipynb"),
    os.path.join(os.path.dirname(__file__), "../Notebooks/FCE_Net_Kaggle.ipynb"),
    os.path.join(os.path.dirname(__file__), "Notebooks/FCE_Net_Kaggle.ipynb"),
    "/home/charizard/24BCE7695/FINAL_SUBMISSION/24BCE7695/02_Original_Implementation/Code/Notebooks/FCE_Net_Kaggle.ipynb"
]
nb_path = next((p for p in candidate_paths if os.path.exists(p)), None)
assert nb_path is not None, "FCE_Net_Kaggle.ipynb not found"
nb = json.load(open(nb_path))

# Find code cells (cells 3a through 3e: indices for wavelet, HLFDB, model, losses, dataloader)
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']

# We need cells 2-6 (0-indexed): wavelet, HLFDB, model, losses, dataloader
# Cell 0 = pip install (skip)
# Cell 1 = device check
# Cell 2 = wavelet
# Cell 3 = HLFDB
# Cell 4 = model
# Cell 5 = losses
# Cell 6 = dataloader

print("\n[1/5] Defining all modules from notebook...")
import torch
DEVICE = torch.device('cpu')

for cell_idx in [2, 3, 4, 5, 6]:
    src = ''.join(code_cells[cell_idx]['source'])
    try:
        exec(compile(src, f'<cell_{cell_idx}>', 'exec'), globals())
    except Exception as e:
        print(f"  ❌ Error in code cell {cell_idx}: {e}")
        sys.exit(1)

print("\n[2/5] Full model construction...")
model = FCE(filters=32)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params:,}")
print(f"  Model size: {total_params * 4 / 1e6:.2f} MB (float32)")

print("\n[3/5] Forward pass with 128x128 input...")
start = time.time()
dummy = torch.randn(1, 3, 128, 128)
with torch.no_grad():
    out = model(dummy)
elapsed = time.time() - start
print(f"  Input:  {dummy.shape}")
print(f"  Output: {out.shape}")
print(f"  Range:  [{out.min().item():.4f}, {out.max().item():.4f}]")
print(f"  Time:   {elapsed:.2f}s")
assert out.shape == (1, 3, 128, 128), f"Shape mismatch: {out.shape}"
assert 0 <= out.min().item() and out.max().item() <= 1, "Output out of [0,1] range!"
print("  ✅ Forward pass OK")

print("\n[4/5] Loss computation + backward pass...")
criterion = CombinedLoss('cpu')
# MS-SSIM requires images >= 160px due to 4 downsamplings
target = torch.rand(1, 3, 256, 256)
model.train()
output = model(torch.rand(1, 3, 256, 256))
loss = criterion(output, target)
print(f"  Loss value: {loss.item():.6f}")
loss.backward()
# Check gradients exist
has_grad = sum(1 for p in model.parameters() if p.grad is not None)
total_p = sum(1 for p in model.parameters())
print(f"  Params with gradients: {has_grad}/{total_p}")
print("  ✅ Backward pass OK")

print("\n[5/5] Checkpoint round-trip...")
import tempfile
ckpt_path = os.path.join(tempfile.gettempdir(), "fce_full_test.pth")
torch.save(model.state_dict(), ckpt_path)
size_mb = os.path.getsize(ckpt_path) / 1e6
model2 = FCE(filters=32)
model2.load_state_dict(torch.load(ckpt_path, weights_only=True))
model.eval()
model2.eval()
test_in = torch.randn(1, 3, 128, 128)
with torch.no_grad():
    o1 = model(test_in)
    o2 = model2(test_in)
diff = (o1 - o2).abs().max().item()
print(f"  Checkpoint size: {size_mb:.2f} MB")
print(f"  Max output diff: {diff:.2e}")
assert diff < 1e-5
os.remove(ckpt_path)
print("  ✅ Checkpoint round-trip OK")

print("\n" + "=" * 60)
print("  ALL INTEGRATION TESTS PASSED ✅")
print(f"  Model: {total_params:,} params, checkpoint: {size_mb:.2f} MB")
print("=" * 60)

