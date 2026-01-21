from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))  # so `import models...` works

from models.unet import UNet  # now resolves

def main():
    root = Path(__file__).resolve().parent
    ckpt_path = root / "checkpoints" / "best.pt"
    out_path = root / "artifacts" / "unet_deforestation.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = UNet(in_channels=10, num_classes=2)
    model.eval()

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Handle common checkpoint formats
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    # Strip common prefixes
    cleaned = {}
    for k, v in state.items():
        for prefix in ("model.", "module."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        cleaned[k] = v

    model.load_state_dict(cleaned, strict=False)

    dummy = torch.randn(1, 10, 512, 512)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
    )

    print("Wrote:", out_path)
