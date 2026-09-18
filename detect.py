"""Detect and recognise handwritten/printed digits in an image (CPU only).

Usage:
    python detect.py photo.jpg
    python detect.py photo.jpg --paper                    # find the sheet of paper first, read digits only on it
    python detect.py photo.jpg --json
    python detect.py photo.jpg --debug debug_out          # dumps every intermediate image
    python detect.py photo.jpg --model models/mnist_cnn.tflite
    python detect.py --help                               # all preprocessing knobs

Prints one line of text per detected text line (digits in reading order; a
low-confidence digit is printed as "?"). Always writes an annotated image.
Exit codes: 0 = digits found, 2 = no digits found, 1 = error.

run_pipeline() is the single entry point shared by this CLI, server.py and
camera.py.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from paper import Paper, PaperParams, draw_paper, find_paper
from preprocess import PreprocessParams, Segmentation, add_preprocess_args, load_image, params_from_args, save_image, segment_image

DEFAULT_MODEL = Path("models") / "mnist_cnn.keras"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Detect digits in an image with the MNIST CNN.")
    ap.add_argument("image", help="input image (png, jpg, bmp, tif, webp)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help=f"model file: .keras, .tflite or .onnx (default {DEFAULT_MODEL})")
    ap.add_argument("--min-conf", type=float, default=0.5, help="digits below this softmax confidence are printed as ? (default 0.5)")
    ap.add_argument("--paper", action="store_true", help="locate the sheet of paper first and detect digits only on it (ignores desk, hands, background)")
    ap.add_argument("--debug", default=None, help="directory for intermediate images (paper, gray, binary, boxes, crops)")
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


def run_pipeline(image: np.ndarray, params: PreprocessParams, predictor, min_conf: float, use_paper: bool = False,
                 paper_params: Optional[PaperParams] = None, debug_dir: Optional[str] = None) -> tuple[dict, Optional[Paper], Segmentation]:
    """Full detection on a BGR image. With use_paper the sheet of paper is
    located first and digits are searched only there; boxes are returned in
    the coordinates of the original image (with the exact projected
    quadrilateral as 'quad' when the sheet was perspective-corrected).

    Returns (result_dict, paper_or_None, segmentation)."""
    timing: dict[str, float] = {}
    paper = None
    if use_paper:
        t0 = time.perf_counter()
        paper = find_paper(image, paper_params)
        timing["paper_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if debug_dir and paper is not None and not paper.fills_frame:
            save_image(str(Path(debug_dir) / "00_paper.png"), paper.warped)
    source = paper.warped if paper is not None else image

    t0 = time.perf_counter()
    seg = segment_image(source, params, debug_dir=debug_dir)
    timing["segmentation_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    lines = recognise(seg, predictor, min_conf) if seg.crops else []
    timing["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    rejected = [{"x": r.x, "y": r.y, "w": r.w, "h": r.h, "reason": r.reason} for r in seg.rejected]
    if paper is not None and not paper.fills_frame:
        for ln in lines:
            for d in ln["digits"]:
                d.update(paper.map_box(d["x"], d["y"], d["w"], d["h"]))
        rejected = [dict(r, **paper.map_box(r["x"], r["y"], r["w"], r["h"])) for r in rejected]

    paper_info = None
    if use_paper:
        paper_info = paper.info() if paper is not None else {"found": False}
    result = {
        "text": "\n".join(ln["text"] for ln in lines),
        "lines": lines,
        "digit_count": len(seg.crops),
        "rejected": rejected,
        "paper": paper_info,
        "timing_ms": timing,
    }
    return result, paper, seg


def _poly(vis: np.ndarray, quad, color, thickness: int) -> None:
    cv2.polylines(vis, [np.array(quad, dtype=np.int32).reshape(-1, 1, 2)], True, color, thickness, cv2.LINE_AA)


def annotate(image: np.ndarray, lines: list[dict], rejected=None, paper: Optional[Paper] = None) -> np.ndarray:
    """Draw the paper outline (orange), accepted digits (green), low-confidence
    digits (amber) and, if given, rejected blobs (thin red)."""
    vis = image.copy()
    draw_paper(vis, paper)
    if rejected:
        for r in rejected:
            rx, ry, rw, rh = (r["x"], r["y"], r["w"], r["h"]) if isinstance(r, dict) else (r.x, r.y, r.w, r.h)
            quad = r.get("quad") if isinstance(r, dict) else None
            if quad:
                _poly(vis, quad, (0, 0, 255), 1)
            else:
                cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 1)
    for ln in lines:
        for d in ln["digits"]:
            color = (0, 180, 0) if d["accepted"] else (0, 140, 255)
            thickness = max(1, int(round(d["h"] / 60)))
            if d.get("quad"):
                _poly(vis, d["quad"], color, thickness)
            else:
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
    try:
        image = load_image(args.image)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not Path(args.model).is_file():
        print(f"error: model file not found: {args.model} (run train.py first, or pass --model)", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    from predictor import load_predictor

    predictor = load_predictor(args.model)
    load_ms = round((time.perf_counter() - t0) * 1000, 1)

    params = params_from_args(args)
    result, paper, seg = run_pipeline(image, params, predictor, args.min_conf, use_paper=args.paper, debug_dir=args.debug)
    result["timing_ms"]["model_load_ms"] = load_ms
    result["timing_ms"]["digits"] = result["digit_count"]

    out_path = output_path(args)
    save_image(str(out_path), annotate(image, result["lines"], result["rejected"] if args.debug else None, paper))
    payload = {"image": args.image, "model": args.model, "backend": predictor.backend, **result, "annotated": str(out_path)}

    if not seg.crops:
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            where = "" if not args.paper else (" on the detected paper" if paper is not None else "; no sheet of paper found either")
            print(f"No digits found{where}. ({len(result['rejected'])} blobs rejected; try --debug to inspect)")
        return 2

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    print(result["text"])
    if args.verbose:
        if args.paper:
            p = result["paper"]
            print("  paper:", "whole image" if p.get("fills_frame") else (f"found, {p['area_fraction'] * 100:.0f}% of the image, corners {p['quad']}" if p.get("found") else "not found (whole image used)"))
        for ln in result["lines"]:
            for d in ln["digits"]:
                flag = "" if d["accepted"] else "  (below --min-conf)"
                print(f"  line {ln['line']}  x={d['x']:5d} y={d['y']:5d} w={d['w']:4d} h={d['h']:4d}  -> {d['label']}  conf {d['confidence']:.3f}{flag}")
        print(f"  backend {predictor.backend} | {result['timing_ms']}")
        print(f"  annotated image: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
