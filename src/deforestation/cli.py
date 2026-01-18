from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .infer import create_session, load_tile_chw, predict_logits


def cmd_predict(args: argparse.Namespace) -> int:
    onnx_path = Path(args.onnx)
    tile_path = Path(args.tile)
    out_path = Path(args.out)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path.as_posix()}")
    if not tile_path.exists():
        raise FileNotFoundError(f"Tile not found: {tile_path.as_posix()}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Create session (default CPU; --cuda opts in)
    sess = create_session(onnx_path, prefer_cuda=bool(args.cuda))

    # Helpful but clean: tell user what actually got used
    used = sess.get_providers()
    if args.cuda and ("CUDAExecutionProvider" not in used):
        print("⚠️  CUDA requested, but CUDA provider not available. Falling back to CPU.")
    print("ONNX providers:", used)

    # Load + run
    image_chw = load_tile_chw(tile_path)
    logits = predict_logits(sess, image_chw)

    mask = np.argmax(logits, axis=1)[0].astype(np.uint8)  # (512,512)

    np.save(out_path.as_posix(), mask)

    uniq = np.unique(mask)
    print(f"Saved: {out_path.as_posix()} | classes: {uniq.tolist()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="deforestation", description="Deforestation segmentation CLI (ONNX)")
    sub = p.add_subparsers(dest="command", required=True)

    p_predict = sub.add_parser("predict", help="Predict a mask from a GeoTIFF tile using an ONNX model")
    p_predict.add_argument("--onnx", required=True, help="Path to ONNX model")
    p_predict.add_argument("--tile", required=True, help="Path to input tile GeoTIFF")
    p_predict.add_argument("--out", required=True, help="Path to output .npy mask")
    p_predict.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDAExecutionProvider if available (falls back to CPU if not).",
    )
    p_predict.set_defaults(func=cmd_predict)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
