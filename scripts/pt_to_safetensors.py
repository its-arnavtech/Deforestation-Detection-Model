from pathlib import Path
import torch
from safetensors.torch import save_file

PT_PATH = Path("checkpoints/best.pt")
ST_PATH = Path("checkpoints/best.safetensors")

ckpt = torch.load(PT_PATH, map_location="cpu", weights_only=False)

state_dict = ckpt["model_state"]

# Save ONLY tensors → safe, non-pickle format
save_file(
    {k: v.detach().cpu() for k, v in state_dict.items()},
    ST_PATH.as_posix(),
)

print("✅ Saved:", ST_PATH)
