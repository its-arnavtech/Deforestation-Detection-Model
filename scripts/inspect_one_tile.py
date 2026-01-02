from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

TILES_DIR = Path("data/tiles_sentinel")
TREECOVER = Path("data/hansen/treecover2000.tif")
LOSSYEAR = Path("data/hansen/lossyear.tif")

FOREST_THRESHOLD = 30

def warp_to_tile(src_ds, tile_ds, resampling):
    dst = np.zeros((tile_ds.height, tile_ds.width), dtype=np.float32)
    reproject(
        source=rasterio.band(src_ds, 1),
        destination=dst,
        src_transform=src_ds.transform,
        src_crs=src_ds.crs,
        dst_transform=tile_ds.transform,
        dst_crs=tile_ds.crs,
        dst_nodata=0,
        resampling=resampling,
    )
    return dst

def stats(name, arr):
    arr_np = np.asarray(arr)
    print(f"\n{name}:")
    print("  dtype:", arr_np.dtype)
    print("  min/max:", float(np.min(arr_np)), float(np.max(arr_np)))
    # show uniques only for small-ish categorical ranges
    if arr_np.dtype.kind in "iu" or (np.max(arr_np) <= 30 and np.min(arr_np) >= 0):
        u = np.unique(arr_np)
        print("  unique (first 30):", u[:30], ("..." if len(u) > 30 else ""))

def main():
    tile_paths = sorted(TILES_DIR.glob("*.tif"))
    if not tile_paths:
        raise FileNotFoundError(f"No tiles found in {TILES_DIR}")

    tile_path = tile_paths[0]  # first tile
    print("Using tile:", tile_path.name)

    with rasterio.open(tile_path) as tds, rasterio.open(TREECOVER) as tc_ds, rasterio.open(LOSSYEAR) as ly_ds:
        # Warp Hansen to tile grid
        tc = warp_to_tile(tc_ds, tds, Resampling.bilinear)
        ly = warp_to_tile(ly_ds, tds, Resampling.nearest)

        # Clean up
        tc = np.clip(tc, 0, 100)
        ly_int = np.round(ly).astype(np.uint8)

        stats("treecover_warped", tc)
        stats("lossyear_warped_float", ly)
        stats("lossyear_warped_uint8", ly_int)

        # Apply label logic
        forest_like = tc >= FOREST_THRESHOLD
        deforested = forest_like & (ly_int > 0)
        forest = forest_like & (ly_int == 0)

        mask = np.zeros((tds.height, tds.width), dtype=np.uint8)
        mask[forest] = 1
        mask[deforested] = 2

        stats("mask", mask)
        counts = np.bincount(mask.flatten(), minlength=3)
        print("\nmask counts:", counts.tolist(), " (0,1,2)")
        print("mask %:", (counts / max(counts.sum(),1) * 100).round(4).tolist())

if __name__ == "__main__":
    main()
