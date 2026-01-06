from pathlib import Path

import torch
from safetensors.torch import load_file as safetensors_load_file

from src.models.unet import UNet

# Use the secure weights file produced by train.py
WEIGHTS_PATH = Path("checkpoints/best.safetensors")

OUT_DIR = Path("artifacts")
OUT_PATH = OUT_DIR / "unet_deforestation.onnx"

IN_CHANNELS = 10
NUM_CLASSES = 3
BASE = 32


def main():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH.as_posix()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build model
    model = UNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, base=BASE)

    # Load safetensors weights (no pickle)
    state = safetensors_load_file(str(WEIGHTS_PATH))  # dict[str, Tensor]
    model.load_state_dict(state, strict=True)
    model.eval()

    # fixed-shape dummy input for export
    dummy = torch.randn(1, IN_CHANNELS, 512, 512, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        OUT_PATH.as_posix(),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes=None,  # fixed 512x512 for now
    )

    print("Saved ONNX:", OUT_PATH.as_posix())


if __name__ == "__main__":
    main()
