---
license: mit
tags:
- onnx
- segmentation
- satellite-imagery
- sentinel-2
library_name: onnxruntime
---

# Deforestation U-Net (ONNX)

This repository contains an **ONNX-exported U-Net segmentation model** intended for **deforestation detection** on Sentinel-2 imagery tiles.

It is designed to be used with the PyPI package **`deforestation-detection`**.

## Files
- `unet_deforestation.onnx` — ONNX model file

## Input
- **Type:** float32
- **Shape:** `[1, 10, 512, 512]` (NCHW)
- **Meaning:** 10-band Sentinel-2 tile (512×512)
- **Band order / normalization:** *[TODO: document your exact band order + scaling]*

## Output

The model outputs per-pixel **class logits**, which should be converted to a segmentation mask using `argmax` over the class dimension.

- **Logits shape:** `[1, 3, 512, 512]`
- **Output mask shape:** `[512, 512]`
- **Mask dtype:** integer (typically `uint8`)
- **Valid class IDs:** `{0, 1, 2}`

### Class definitions (authoritative)

| Class ID | Name        | Description |
|---------:|-------------|-------------|
| 0 | Non-forest | Areas **not forested in the year 2000** (`treecover2000 < 30%`). Includes water, urban areas, agriculture, bare ground, and naturally non-forested land. |
| 1 | Forest | Areas **forested in 2000** with **no recorded loss** up to the target year (`treecover2000 ≥ 30%` and `lossyear == 0`). |
| 2 | Deforested | Areas **forested in 2000** that experienced **forest loss between 2001–2024** (`treecover2000 ≥ 30%` and `lossyear > 0`). |

### Important note on Hansen GFC semantics

Class **2 (Deforested)** does **not** necessarily mean “no trees today”.

It means:
- The pixel was forested in 2000, and  
- A loss event was recorded at some point after 2000.

Areas with regrowth may still be labeled as **Deforested (2)**.  
This behavior is **intentional** and follows Hansen Global Forest Change conventions.

## Usage with the CLI

Install the inference package:
```bash
pip install deforestation-detection

- **Band order / normalization:** Matches the training pipeline.  
  If tiles contain extra bands (e.g. QA or mask bands), users must supply only the 10 spectral bands used during training.
