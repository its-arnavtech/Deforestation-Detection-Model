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
