"""Detect and recognise handwritten/printed digits in an image (CPU only).

Usage:
    python detect.py photo.jpg
    python detect.py photo.jpg --json
    python detect.py photo.jpg --debug debug_out          # dumps every intermediate image
    python detect.py photo.jpg --model models/mnist_cnn.tflite
    python detect.py --help                               # all preprocessing knobs

Prints one line of text per detected text line (digits in reading order; a
low-confidence digit is printed as "?"). Always writes an annotated image.
Exit codes: 0 = digits found, 2 = no digits found, 1 = error.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from preprocess import Segmentation, add_preprocess_args, load_image, params_from_args, save_image, segment_image

DEFAULT_MODEL = Path("models") / "mnist_cnn.keras"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Detect digits in an image with the MNIST CNN.")
    ap.add_argument("image", help="input image (png, jpg, bmp, tif, webp)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help=f"model file: .keras, .tflite or .onnx (default {DEFAULT_MODEL})")
    ap.add_argument("--min-conf", type=float, default=0.5, help="digits below this softmax confidence are printed as ? (default 0.5)")
    ap.add_argument("--debug", default=None, help="directory for intermediate images (gray, binary, boxes, crops)")
    ap.add_argument("--out", default=None, help="path of the annotated output image (default: <image>_detected.png)")
    ap.add_argument("--json", action="store_true", help="print a JSON result instead of plain text")
    ap.add_argument("--verbose", action="store_true", help="print per-digit confidences and timing")
    add_preprocess_args(ap)
    return ap.parse_args(argv)


def recognise(seg: Segmentation, predictor, min_conf: float) -> list[dict]:
    """Classify every crop in one batch and rebuild each text line."""
    batch = np.stack([c.tensor for c in seg.crops]).astype(np.float32)
    probs = predictor.predict(batch)
    labels = probs.argmax(axis=1)
    confs = probs.max(axis=1)
    lines: dict[int, dict] = {}
    for crop, label, conf in zip(seg.crops, labels, confs):
        accepted = bool(conf >= min_conf)
        entry = lines.setdefault(crop.line, {"line": crop.line, "text": "", "digits": []})
        entry["text"] += str(int(label)) if accepted else "?"
        entry["digits"].append({
            "x": crop.x, "y": crop.y, "w": crop.w, "h": crop.h,
            "label": int(label), "confidence": round(float(conf), 4), "accepted": accepted,
        })
    return [lines[k] for k in sorted(lines)]


def annotate(image: np.ndarray, lines: list[dict], rejected=None) -> np.ndarray:
    vis = image.copy()
    if rejected:
        for r in rejected:
            cv2.rectangle(vis, (r.x, r.y), (r.x + r.w, r.y + r.h), (0, 0, 255), 1)
    for ln in lines:
        for d in ln["digits"]:
            color = (0, 180, 0) if d["accepted"] else (0, 140, 255)
            thickness = max(1, int(round(d["h"] / 60)))
            cv2.rectangle(vis, (d["x"], d["y"]), (d["x"] + d["w"], d["y"] + d["h"]), color, thickness)
            scale = max(0.5, d["h"] / 70.0)
            text = f"{d['label']} {d['confidence']:.2f}" if d["accepted"] else f"? {d['confidence']:.2f}"
            org = (d["x"], max(int(20 * scale), d["y"] - int(6 * scale)))
            cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, max(1, thickness), cv2.LINE_AA)
    return vis


def output_path(args) -> Path:
    src = Path(args.image)
    if args.out:
        return Path(args.out)
    if args.debug:
        return Path(args.debug) / f"{src.stem}_detected.png"
    return src.with_name(f"{src.stem}_detected.png")


def main(argv=None) -> int:
    args = parse_args(argv)
    timing: dict[str, float] = {}
    try:
        image = load_image(args.image)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    params = params_from_args(args)
    t0 = time.perf_counter()
    seg = segment_image(image, params, debug_dir=args.debug)
    timing["segmentation_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    out_path = output_path(args)
    if not seg.crops:
        save_image(str(out_path), annotate(image, [], seg.rejected))
        msg = {"image": args.image, "text": "", "lines": [], "rejected": [r.__dict__ for r in seg.rejected], "annotated": str(out_path), "timing_ms": timing}
        print(json.dumps(msg, indent=2) if args.json else f"No digits found. ({len(seg.rejected)} blobs rejected; try --debug to inspect)")
        return 2

    if not Path(args.model).is_file():
        print(f"error: model file not found: {args.model} (run train.py first, or pass --model)", file=sys.stderr)
        return 1
    t0 = time.perf_counter()
    from predictor import load_predictor

    predictor = load_predictor(args.model)
    timing["model_load_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    lines = recognise(seg, predictor, args.min_conf)
    timing["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    timing["digits"] = len(seg.crops)

    save_image(str(out_path), annotate(image, lines, seg.rejected if args.debug else None))
    text = "\n".join(ln["text"] for ln in lines)
    if args.json:
        result = {
            "image": args.image, "model": args.model, "backend": predictor.backend,
            "text": text, "lines": lines,
            "rejected": [r.__dict__ for r in seg.rejected],
            "annotated": str(out_path), "timing_ms": timing,
        }
        print(json.dumps(result, indent=2))
    else:
        print(text)
        if args.verbose:
            for ln in lines:
                for d in ln["digits"]:
                    flag = "" if d["accepted"] else "  (below --min-conf)"
                    print(f"  line {ln['line']}  x={d['x']:5d} y={d['y']:5d} w={d['w']:4d} h={d['h']:4d}  -> {d['label']}  conf {d['confidence']:.3f}{flag}")
            print(f"  backend {predictor.backend} | {timing}")
            print(f"  annotated image: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
