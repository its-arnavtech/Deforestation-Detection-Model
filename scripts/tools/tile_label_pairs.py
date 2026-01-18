import os

TILE_DIR = "data/tiles_sentinel"
LABEL_DIR = "data/labels"

# Collect tile filenames
tile_files = [f for f in os.listdir(TILE_DIR) if f.endswith(".tif")]
tile_stems = set(os.path.splitext(f)[0] for f in tile_files)

# Collect label filenames AND STRIP "_label"
label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith(".npy")]
label_stems = set(
    os.path.splitext(f)[0].replace("_label", "")
    for f in label_files
)

print("Tiles:", len(tile_files))
print("Labels:", len(label_files))
print("Matched stems:", len(tile_stems & label_stems))

missing_labels = sorted(tile_stems - label_stems)
missing_tiles = sorted(label_stems - tile_stems)

print("\nTiles missing labels:", len(missing_labels))
print("Example:", missing_labels[:5])

print("\nLabels missing tiles:", len(missing_tiles))
print("Example:", missing_tiles[:5])
