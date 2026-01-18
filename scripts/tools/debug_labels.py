import os
import numpy as np

labels_dir = "data/labels"
files = [f for f in os.listdir(labels_dir) if f.endswith(".npy")]

print("Label .npy files:", len(files))
if not files:
    raise SystemExit("No label .npy files found in data/labels")

sample_path = os.path.join(labels_dir, files[0])
m = np.load(sample_path)

print("Sample:", files[0])
print("dtype:", m.dtype, "shape:", m.shape)
print("min:", m.min(), "max:", m.max())
u = np.unique(m)
print("unique count:", len(u))
print("unique(sample):", u[:20])
