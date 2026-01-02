from pathlib import Path
import rasterio

TILE = sorted(Path("data/tiles_sentinel").glob("*.tif"))[0]
TC = Path("data/hansen/treecover2000.tif")
LY = Path("data/hansen/lossyear.tif")

def sample(ds, x, y):
    # ds.sample expects (x,y) in ds CRS coordinates
    v = next(ds.sample([(x, y)]))[0]
    return v

with rasterio.open(TILE) as tds, rasterio.open(TC) as tc, rasterio.open(LY) as ly:
    tb = tds.bounds
    cx = (tb.left + tb.right) / 2
    cy = (tb.bottom + tb.top) / 2

    print("Tile:", TILE.name)
    print("Tile CRS:", tds.crs)
    print("Tile bounds:", tb)
    print("Tile center:", (cx, cy))

    print("\nTreecover CRS:", tc.crs, "bounds:", tc.bounds)
    print("Lossyear CRS:", ly.crs, "bounds:", ly.bounds)

    print("\nSample at tile center:")
    print(" treecover2000:", sample(tc, cx, cy))
    print(" lossyear:", sample(ly, cx, cy))
