from pathlib import Path
import numpy as np
from PIL import Image

MASK_PATH = Path("artifacts/pred_mask.npy")
OUT_PATH  = Path("artifacts/pred_mask.png")

def main():
    mask = np.load(MASK_PATH)  # (512,512), uint8 classes 0/1/2
    # scale 0/1/2 -> 0/127/255 so you can see it
    img = (mask.astype(np.uint8) * 127).clip(0, 255)
    Image.fromarray(img, mode="L").save(OUT_PATH)
    print("Saved:", OUT_PATH.as_posix(), "| unique:", np.unique(mask).tolist())

if __name__ == "__main__":
    main()
