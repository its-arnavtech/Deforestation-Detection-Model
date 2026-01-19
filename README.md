# 🌲 Deforestation Detection (ONNX)

A **CLI tool and Python package** for running **deforestation segmentation** on **512×512 Sentinel-2 tiles** using a **U-Net model exported to ONNX**.

This repository contains:
- Inference-ready CLI (`deforestation`)
-  Clean Python API
-  ONNX Runtime (CPU & GPU)
-  Secure model loading (no pickled models)
-  No bundled model weights (download separately)

> **Models are distributed separately via Hugging Face** to keep this package lightweight and secure.

---

## Features

-  Remote sensing–focused (Sentinel-2, 10 bands)
-  U-Net semantic segmentation
-  ONNX Runtime for fast inference
-  CPU-only by default
-  Optional GPU acceleration (CUDA)
-  No unsafe `torch.load` usage
-  PyPI-published, production-ready

---

## Installation

### CPU (default)
```bash
pip install deforestation-detection
```

### GPU (default)
```bash
pip install deforestation-detection[gpu]
