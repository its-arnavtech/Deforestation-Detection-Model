import os
from pathlib import Path

import torch
import numpy as np
import rasterio

TILE_DIR = Path("data/tiles_sentinel")
LABEL_DIR = Path("data/labels")
OUT_DIR = Path("data/dataset_pt")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Read ONLY the first 10 bands (Sentinel-2 10m bands you exported)
BANDS_1_TO_10 = list(range(1, 11))

def load_tile(path: Path) -> np.ndarray:
    """Load a Sentinel tile and normalize to ~[0,1] by /10000."""
    with rasterio.open(path) as src:
        # Force 10 bands (drop band 11 / QA-like band)
        img = src.read(BANDS_1_TO_10).astype(np.float32) / 10000.0

    # Safety check: expect (10, 512, 512)
    if img.shape[0] != 10:
        raise ValueError(f"Expected 10 bands, got {img.shape[0]} for {path.name}")
    return img

def load_label(tile_stem: str) -> np.ndarray:
    """Load corresponding label mask as int64."""
    label_path = LABEL_DIR / f"{tile_stem}_label.npy"
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label for {tile_stem}: {label_path.as_posix()}")

    mask = np.load(label_path).astype(np.int64)

    # Safety check: expect (512, 512)
    if mask.ndim != 2:
        raise ValueError(f"Mask is not 2D for {tile_stem}: shape={mask.shape}")
    return mask

if __name__ == "__main__":
    tiles = sorted([p for p in TILE_DIR.glob("*.tif")])

    if not tiles:
        raise FileNotFoundError(f"No .tif tiles found in {TILE_DIR.as_posix()}")

    for tile_path in tiles:
        name = tile_path.stem

        img = load_tile(tile_path)
        mask = load_label(name)

        img_tensor = torch.from_numpy(img)          # float32, [10,512,512]
        mask_tensor = torch.from_numpy(mask)        # int64,  [512,512]

        out_path = OUT_DIR / f"{name}.pt"

        torch.save(
            {"image": img_tensor, "mask": mask_tensor},
            out_path
        )

        print("Saved:", out_path.as_posix())
