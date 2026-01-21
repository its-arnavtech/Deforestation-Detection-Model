# Deforestation Detection (ONNX Inference)

A **Geospatial machine learning inference pipeline** for detecting deforestation using **Sentinel-2 imagery** and a **U-Net segmentation model exported to ONNX**.

This repository contains **code only** (CLI + Python API).  
The trained model is hosted **separately on Hugging Face**, and must be downloaded explicitly by the user.

---

## Project Overview

This project performs **semantic segmentation** on Sentinel-2 tiles to classify land cover into three categories:

| Class ID | Name        | Description |
|---------:|-------------|-------------|
| 0 | Non-forest | Areas not forested in the year 2000 |
| 1 | Forest | Forest in 2000 with no recorded loss |
| 2 | Deforested | Forest in 2000 with a recorded loss event (2001–2024) |

The model follows **Hansen Global Forest Change (GFC)** semantics and is designed for **inference only**.

---

## Repository Scope (Important)

This repository includes the following:
- CLI tool(`deforestation`)
- Python importable package
- ONNX inference logic
- GeoTIFF I/O and preprocessing
- **NO model weights**
- **NO training code**
- **NO datasets**

The trained model is distributed separately to keep the package lightweight and reproducible.

---

## Installation

### Requirements
- Python **3.11**
- Windows (tested), Linux/macOS likely compatible
- No GPU required (CPU inference via ONNX Runtime)

### Install from PyPI
```bash
pip install deforestation-detection
```

### Download the Model (Required)

The ONNX model is hosted on Hugging Face:
https://huggingface.co/ItsArca/deforestation-unet-onnx

## Download it using the Hugging Face CLI:

pip install -U huggingface_hub
hf download ItsArca/deforestation-unet-onnx unet_deforestation.onnx --local-dir models

## This will create:

models/
  unet_deforestation.onnx

### Input Data Requirements
## Tile format

File type: GeoTIFF (.tif)
Shape: 512 × 512
Bands: 10 spectral bands
dtype: float32

### Important note on band count

Some Sentinel-derived tiles include extra bands (e.g. QA, masks, metadata).
You must supply only the 10 spectral bands used during training.
Extra bands must be removed before inference.

### Output
pred_mask.npy
Shape: (512, 512)
dtype: integer (uint8)
Values: {0, 1, 2}

### Output Class Semantics (Authoritative)
Class ID	Name	Meaning
0	Non-forest	Not forested in year 2000 (treecover2000 < 30%)
1	Forest	Forest in 2000 with no recorded loss
2	Deforested	Forest in 2000 with a recorded loss event (2001–2024)

## Important semantic detail: 
Class 2 does NOT mean "no trees today".

It means:
1. The pixel was forested in 2000, and
2. A loss event was recorded at some point after 2000

Areas with regrowth may still be labeled as 2.
This exactly matches Hansen GFC conventions.

### Python API Usage

You may also use the package programmatically:

from deforestation.infer import predict

mask = predict(
    onnx_path="models/unet_deforestation.onnx",
    tile_path="tile_10band.tif"
)

print(mask.shape)  # (512, 512)

### All end-to-end testing was performed in a completely separate repository and virtual environment, simulating a real user.

Hardware Notes

Inference is performed via ONNX Runtime:
CPU execution provider by default
GPU / OpenVINO support can be added later
No GPU is required to run this project.

### License

MIT License

### Citation / Attribution

If you use this work in research or downstream projects, please cite:
Hansen Global Forest Change (GFC)
Sentinel-2 (ESA / Copernicus)
ONNX Runtime
