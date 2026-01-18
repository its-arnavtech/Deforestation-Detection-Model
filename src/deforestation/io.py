from __future__ import annotations

from pathlib import Path
import numpy as np
import rasterio


def read_tile_tif(path: str | Path, expected_channels: int = 10) -> np.ndarray:
    """
    Reads a multiband GeoTIFF tile -> float32 [C,H,W] normalized to ~0..1 using /10000.
    If the file has extra bands, keeps the first `expected_channels`.
    If it has fewer bands, errors.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tile not found: {path.as_posix()}")

    with rasterio.open(path) as src:
        arr = src.read().astype("float32")  # [C,H,W]

    c = arr.shape[0]
    if c < expected_channels:
        raise ValueError(f"Tile has {c} bands, but expected at least {expected_channels}.")
    if c > expected_channels:
        # Keep first N bands (common case: extra metadata band at the end)
        arr = arr[:expected_channels, :, :]

    arr /= 10000.0
    return arr


def save_mask_npy(mask_hw: np.ndarray, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path.as_posix(), mask_hw)


def save_mask_png(mask_hw: np.ndarray, out_path: str | Path) -> None:
    """
    Saves mask as a grayscale PNG (0/1/2). No colormap by design (simple + dependency-free).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image  # optional
    except Exception as e:
        raise RuntimeError("Pillow not installed. Install with: pip install pillow") from e

    img = Image.fromarray(mask_hw.astype(np.uint8), mode="L")
    img.save(out_path.as_posix())
