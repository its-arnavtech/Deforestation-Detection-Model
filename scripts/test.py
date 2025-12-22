import os
import torch

dataset_dir = "data/dataset_pt"
files = [f for f in os.listdir(dataset_dir) if f.endswith(".pt")]

sample_path = os.path.join(dataset_dir, files[0])
sample = torch.load(sample_path)

x = sample["image"]
y = sample["mask"]

print("Num samples:", len(files))
print("X shape:", x.shape)
print("Y shape:", y.shape)
print("X dtype:", x.dtype, "Y dtype:", y.dtype)
print("Y unique classes:", torch.unique(y)[:10], "...")  # show first few unique values
