import numpy as np
import torch
from pathlib import Path
import onnxruntime as ort

PT_PATH = sorted(Path("data/dataset_pt").glob("*.pt"))[0]
ONNX_PATH = Path("artifacts/unet_deforestation.onnx")
OUT_PATH = Path("artifacts/pred_mask.npy")

def main():
    sample = torch.load(PT_PATH, map_location="cpu")
    image = sample["image"].unsqueeze(0).numpy().astype(np.float32)  # [1,10,512,512]
    gt_mask = sample["mask"].numpy()                                 # [512,512]

    sess = ort.InferenceSession(
        ONNX_PATH.as_posix(),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    logits = sess.run(["logits"], {"image": image})[0]   # [1,3,512,512]
    pred = np.argmax(logits, axis=1)[0].astype(np.uint8)

    np.save(OUT_PATH, pred)

    print("Saved prediction:", OUT_PATH.as_posix())
    print("GT classes:", np.unique(gt_mask))
    print("Pred classes:", np.unique(pred))

    # Simple pixel accuracy
    acc = (pred == gt_mask).mean()
    print(f"Pixel accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
