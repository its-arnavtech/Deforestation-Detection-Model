from pathlib import Path
import numpy as np
import rasterio
from PIL import Image

TILE_PATH = Path("data/tiles_sentinel/sentinel10bands_median_amazon_0_0.tif")
MASK_PATH = Path("artifacts/pred_mask.npy")
OUT_PATH  = Path("artifacts/overlay.png")

def norm01(x):
    x = x.astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return x

def main():
    mask = np.load(MASK_PATH).astype(np.uint8)

    with rasterio.open(TILE_PATH) as src:
        img = src.read().astype(np.float32)  # (C,H,W)

    # take first 3 bands for a quick "RGB-like" view
    rgb = img[:3] / 10000.0
    rgb = np.stack([norm01(rgb[0]), norm01(rgb[1]), norm01(rgb[2])], axis=-1)  # (H,W,3)
    base = (rgb * 255).astype(np.uint8)

    # overlay: class 1 + class 2 highlighted (simple)
    overlay = base.copy()
    overlay[mask == 1] = (overlay[mask == 1] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    overlay[mask == 2] = (overlay[mask == 2] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)

    Image.fromarray(overlay).save(OUT_PATH)
    print("Saved:", OUT_PATH.as_posix(), "| mask unique:", np.unique(mask).tolist())

if __name__ == "__main__":
    main()
