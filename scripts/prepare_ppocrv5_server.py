#!/usr/bin/env python3
"""
Prepare PP-OCRv5 server detection and recognition checkpoints for AutoKernel.

This script converts Paddle checkpoints into PyTorch `.pth` files using the
converter implementations that already live in PaddleOCR2Pytorch.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
AUTOKERNEL_ROOT = SCRIPT_DIR.parent
if str(AUTOKERNEL_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOKERNEL_ROOT))

from models.ppocrv5_server import (  # noqa: E402
    WORKSPACE_DIR,
    default_server_det_weights_path,
    default_server_rec_weights_path,
    ensure_ppocr_on_path,
    get_ppocr_root,
    server_det_architecture,
    server_rec_architecture,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PP-OCRv5 server Paddle checkpoints into .pth files.",
    )
    parser.add_argument(
        "--det-src",
        default=os.getenv("AUTOKERNEL_PPOCRV5_SERVER_DET_PDPARAMS"),
        help="Source Paddle checkpoint for PP-OCRv5 server detection.",
    )
    parser.add_argument(
        "--rec-src",
        default=os.getenv("AUTOKERNEL_PPOCRV5_SERVER_REC_PDPARAMS"),
        help="Source Paddle checkpoint for PP-OCRv5 server recognition.",
    )
    parser.add_argument(
        "--det-dst",
        default=str(default_server_det_weights_path()),
        help="Destination .pth path for the converted server detector.",
    )
    parser.add_argument(
        "--rec-dst",
        default=str(default_server_rec_weights_path()),
        help="Destination .pth path for the converted server recognizer.",
    )
    return parser.parse_args()


def _require_source(path_str: str | None, label: str) -> Path:
    if not path_str:
        raise ValueError(
            f"Missing {label} source checkpoint. Pass --{label}-src or set the "
            f"AUTOKERNEL_PPOCRV5_SERVER_{label.upper()}_PDPARAMS environment variable."
        )
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} source checkpoint not found: {path}")
    return path


def main() -> None:
    args = parse_args()
    ensure_ppocr_on_path()

    try:
        from converter.ppocr_v5_det_converter import PPOCRv5DetConverter  # type: ignore
        from converter.ppocr_v5_rec_converter import (  # type: ignore
            PPOCRv5RecConverter,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Failed to import PaddleOCR2Pytorch PP-OCRv5 converters. "
            "Make sure the PP-OCR stack is installed, including Paddle for "
            "reading Paddle checkpoints."
        ) from exc

    det_src = _require_source(args.det_src, "det")
    rec_src = _require_source(args.rec_src, "rec")
    det_dst = Path(args.det_dst).expanduser().resolve()
    rec_dst = Path(args.rec_dst).expanduser().resolve()
    det_dst.parent.mkdir(parents=True, exist_ok=True)
    rec_dst.parent.mkdir(parents=True, exist_ok=True)
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    det_cfg = server_det_architecture()
    rec_cfg, _ = server_rec_architecture()

    print(f"Using PaddleOCR2Pytorch root: {get_ppocr_root()}")
    print(f"Converting detector checkpoint:   {det_src}")
    det_converter = PPOCRv5DetConverter(det_cfg, str(det_src))
    det_converter.save_pytorch_weights(str(det_dst))

    print(f"Converting recognizer checkpoint: {rec_src}")
    rec_converter = PPOCRv5RecConverter(rec_cfg, str(rec_src))
    rec_converter.save_pytorch_weights(str(rec_dst))

    print(f"Detector weights written to:   {det_dst}")
    print(f"Recognizer weights written to: {rec_dst}")


if __name__ == "__main__":
    main()
