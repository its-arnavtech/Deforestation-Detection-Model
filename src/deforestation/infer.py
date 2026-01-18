from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import onnxruntime as ort
import rasterio


# Keep runtime output clean for a GitHub package:
# 0=verbose, 1=info, 2=warning, 3=error, 4=fatal
ort.set_default_logger_severity(4)


# ---- Model I/O assumptions ----
# Your ONNX expects: (1, 10, 512, 512) float32
IN_CHANNELS = 10
TILE_SIZE = 512

# Sentinel-2 style tiles you inspected had 11 bands.
# Commonly the *last* band is a derived layer / QA / extra.
# We will take the first 10 bands by default.
BAND_INDICES_0BASED = list(range(10))  # [0..9]


def load_tile_chw(tile_path: str | Path, band_indices: Sequence[int] = BAND_INDICES_0BASED) -> np.ndarray:
    """
    Load a GeoTIFF tile into CHW float32, normalized to roughly [0..1].

    - Reads bands (C,H,W)
    - Selects the first 10 bands by default
    - Normalizes by 10000.0 (Sentinel-style scaling)
    """
    tile_path = Path(tile_path)
    with rasterio.open(tile_path) as src:
        img = src.read().astype(np.float32)  # (C,H,W)

    if img.ndim != 3:
        raise ValueError(f"Expected 3D raster (C,H,W), got shape {img.shape} from {tile_path}")

    if img.shape[1] != TILE_SIZE or img.shape[2] != TILE_SIZE:
        raise ValueError(
            f"Expected tile size {TILE_SIZE}x{TILE_SIZE}, got {img.shape[1]}x{img.shape[2]} from {tile_path}"
        )

    if img.shape[0] < max(band_indices) + 1:
        raise ValueError(
            f"Tile has {img.shape[0]} bands, but band_indices require >= {max(band_indices)+1} bands."
        )

    img = img[np.array(band_indices), :, :]  # (10,H,W)

    # Normalize
    img = img / 10000.0

    # Sanity: ensure correct channels
    if img.shape[0] != IN_CHANNELS:
        raise ValueError(f"Expected {IN_CHANNELS} channels after selection, got {img.shape[0]}")

    return img.astype(np.float32)


def create_session(onnx_path: str | Path, prefer_cuda: bool = False) -> ort.InferenceSession:
    """
    Create an ONNX Runtime session.

    Default is CPU (best for GitHub releases).
    If prefer_cuda=True, try CUDA first, then fall back to CPU.
    """
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path.as_posix()}")

    sess_opts = ort.SessionOptions()
    # You can tune these later if you want:
    sess_opts.intra_op_num_threads = 0
    sess_opts.inter_op_num_threads = 0

    providers = ["CPUExecutionProvider"]
    if prefer_cuda:
        # Try CUDA first, then CPU fallback
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    try:
        sess = ort.InferenceSession(onnx_path.as_posix(), sess_options=sess_opts, providers=providers)
    except Exception:
        # If CUDA init blows up, fall back cleanly to CPU
        sess = ort.InferenceSession(onnx_path.as_posix(), sess_options=sess_opts, providers=["CPUExecutionProvider"])

    return sess


def predict_logits(session: ort.InferenceSession, image_chw: np.ndarray) -> np.ndarray:
    """
    Run ONNX model and return logits as NCHW float32: (1, num_classes, 512, 512)
    """
    if image_chw.shape != (IN_CHANNELS, TILE_SIZE, TILE_SIZE):
        raise ValueError(f"Expected image CHW {(IN_CHANNELS, TILE_SIZE, TILE_SIZE)}, got {image_chw.shape}")

    x = np.expand_dims(image_chw, axis=0).astype(np.float32)  # (1,10,512,512)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    (logits,) = session.run([output_name], {input_name: x})
    return logits.astype(np.float32)


def predict_mask(session: ort.InferenceSession, tile_path: str | Path, prefer_cuda: bool = False) -> np.ndarray:
    """
    Convenience: load tile -> run model -> argmax mask (H,W) uint8.
    """
    image_chw = load_tile_chw(tile_path)
    logits = predict_logits(session, image_chw)
    mask = np.argmax(logits, axis=1)[0].astype(np.uint8)  # (H,W)
    return mask
