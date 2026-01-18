import torch
from pathlib import Path
import numpy as np

ds_dir = Path("data/dataset_pt")
pts = sorted(ds_dir.glob("*.pt"))
assert pts, f"No .pt files found in {ds_dir}"

bad = 0
counts = np.zeros(3, dtype=np.int64)

for p in pts:
    obj = torch.load(p, map_location="cpu", weights_only=True)
    img = obj.get("image", None)
    m = obj.get("mask", None)

    ok = True
    if img is None or m is None:
        ok = False
    else:
        if tuple(img.shape) != (10, 512, 512): ok = False
        if tuple(m.shape) != (512, 512): ok = False
        if img.dtype not in (torch.float32, torch.float16): ok = False
        if m.dtype not in (torch.uint8, torch.int64, torch.int32): ok = False

        u, c = torch.unique(m.to(torch.int64), return_counts=True)
        for ui, ci in zip(u.tolist(), c.tolist()):
            if 0 <= ui <= 2:
                counts[ui] += ci
            else:
                ok = False

    if not ok:
        bad += 1

total = counts.sum()
print("PT samples:", len(pts))
print("Bad samples:", bad)
print("Pixel counts (0,1,2):", counts.tolist())
print("Pixel % (0,1,2):", (counts / max(total, 1) * 100).round(4).tolist())

# Also check value range sanity for imagery (quick stats on 3 samples)
for p in pts[:3]:
    obj = torch.load(p, map_location="cpu", weights_only=True)
    img = obj["image"]
    print(f"\n{p.name} image stats:")
    print("  dtype:", img.dtype, "min:", float(img.min()), "max:", float(img.max()), "mean:", float(img.mean()))