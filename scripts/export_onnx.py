from pathlib import Path
import torch

from src.models.unet import UNet

CKPT_PATH = Path("checkpoints/best.pt")
OUT_DIR = Path("artifacts")
OUT_PATH = OUT_DIR / "unet_deforestation.onnx"

IN_CHANNELS = 10
NUM_CLASSES = 3
BASE = 32

def main():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH.as_posix()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = UNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, base=BASE)
    model.load_state_dict(ckpt["model_state"])
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
