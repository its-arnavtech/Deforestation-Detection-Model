import time
import numpy as np
import torch
from pathlib import Path
import onnxruntime as ort

PT_PATH = sorted(Path("data/dataset_pt").glob("*.pt"))[0]
ONNX_PATH = Path("artifacts/unet_deforestation.onnx")

def main():
    sample = torch.load(PT_PATH, map_location="cpu", weights_only=True)
    x = sample["image"].unsqueeze(0).numpy().astype(np.float32)

    sess = ort.InferenceSession(
        ONNX_PATH.as_posix(),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # Warmup
    for _ in range(10):
        _ = sess.run(["logits"], {"image": x})[0]

    # Timed runs
    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = sess.run(["logits"], {"image": x})[0]
    t1 = time.perf_counter()

    ms = (t1 - t0) * 1000 / iters
    print(f"ONNX Runtime (CUDA) avg latency: {ms:.2f} ms per tile (1x10x512x512)")

if __name__ == "__main__":
    main()
