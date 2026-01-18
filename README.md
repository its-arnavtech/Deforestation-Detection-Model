# Deforestation Detection (ONNX)
CLI + Python package to run deforestation segmentation on 512x512 Sentinel tiles using a U-Net exported to ONNX.

## Install
pip install deforestation-detection

## Download model
Get the ONNX model from Hugging Face: <YOUR_LINK_HERE>

## Predict
deforestation predict --onnx path/to/unet_deforestation.onnx --tile path/to/tile.tif --out pred_mask.npy

CPU-only by default (onnxruntime)
GPU support via pip install deforestation-detection[gpu] (onnxruntime-gpu)