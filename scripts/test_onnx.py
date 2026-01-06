import numpy as np
import torch
from pathlib import Path
import onnxruntime as ort

PT_PATH = sorted(Path("data/dataset_pt").glob("*.pt"))[0]
ONNX_PATH = Path("artifacts/unet_deforestation.onnx")

def main():
    sample = torch.load(PT_PATH, map_location="cpu", weights_only=True)
    x = sample["image"].unsqueeze(0).numpy().astype(np.float32)

    print("Available providers:", ort.get_available_providers())

    # Force CUDA if present; otherwise CPU
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(ONNX_PATH.as_posix(), providers=providers)

    # Confirm what it actually picked
    print("Session providers:", sess.get_providers())

    y = sess.run(["logits"], {"image": x})[0]

    print("input:", x.shape, x.dtype)
    print("output:", y.shape, y.dtype)
    print("logits stats: min/max/mean =", float(y.min()), float(y.max()), float(y.mean()))

if __name__ == "__main__":
    main()
