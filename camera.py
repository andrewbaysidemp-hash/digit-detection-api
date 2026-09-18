"""Real-time digit detection from a webcam, a video file or a still image.

    python camera.py                                   # webcam 0, ONNX model if exported, else .keras
    python camera.py --source 1                        # second camera
    python camera.py --source clip.mp4                 # any video file OpenCV can open
    python camera.py --source samples/clean_00.png --no-window --frames 3   # headless self-test

Keys in the window:
    q / Esc   quit                 space   pause / resume
    s         save snapshot (frame, annotated frame, JSON) to captures/
    r         reset the stabilised reading
    + / -     raise / lower the confidence threshold
    o         toggle global Otsu threshold (thick marker strokes)
    [ / ]     shrink / grow the searched area (the green guide box)

How it works: the capture loop shows frames at camera speed while a worker
thread processes the NEWEST frame only (older frames are dropped), so the
display never stalls. By default the sheet of paper is located first (orange
outline) and digits are searched only on it, never on the desk, hands or
background; --no-paper detects on the whole guide box instead. Detection runs
at a reduced working resolution, and the reading is stabilised with a
majority vote over the last few results.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np

from detect import annotate, run_pipeline
from preprocess import PreprocessParams, load_image, save_image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def default_model() -> str:
    for name in ("mnist_cnn.onnx", "mnist_cnn.keras", "mnist_cnn.tflite"):
        p = Path("models") / name
        if p.is_file():
            return str(p)
    return str(Path("models") / "mnist_cnn.keras")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Real-time digit detection from a camera or video.")
    ap.add_argument("--source", default="0", help="camera index (0, 1, ...), video file or image file (default 0)")
    ap.add_argument("--model", default=default_model(), help="model file (.onnx recommended for lowest latency)")
    ap.add_argument("--width", type=int, default=1280, help="requested camera width (default 1280)")
    ap.add_argument("--height", type=int, default=720, help="requested camera height (default 720)")
    ap.add_argument("--max-side", type=int, default=640, help="working resolution of the detection area (default 640; lower = faster)")
    ap.add_argument("--roi", type=float, default=None, help="fraction of the frame (centre) searched, 0.2-1.0 (default: 1.0 with paper detection, 0.7 without)")
    ap.add_argument("--no-paper", action="store_true", help="do not look for the sheet of paper first (detect on the whole area)")
    ap.add_argument("--min-conf", type=float, default=0.6, help="confidence below which a digit is shown as ? (default 0.6)")
    ap.add_argument("--history", type=int, default=8, help="number of results in the majority vote (default 8)")
    ap.add_argument("--otsu", action="store_true", help="start with the global Otsu threshold")
    ap.add_argument("--keep-lines", action="store_true", help="do not remove ruled lines")
    ap.add_argument("--save-dir", default="captures", help="where snapshots go (default captures/)")
    ap.add_argument("--frames", type=int, default=0, help="stop after this many frames (0 = run until quit)")
    ap.add_argument("--no-window", action="store_true", help="headless: print readings instead of showing a window")
    ap.add_argument("--sync", action="store_true", help="detect every frame in the main thread (implied by --no-window)")
    return ap.parse_args(argv)


# --------------------------------------------------------------------------- #
# Small pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
class Stabilizer:
    """Majority vote over the last `size` readings. stable() returns
    (text, agreement) where agreement is the fraction of votes for the winner."""

    def __init__(self, size: int = 8):
        self.size = max(1, size)
        self.votes: collections.deque[str] = collections.deque(maxlen=self.size)

    def push(self, text: str) -> None:
        self.votes.append(text)

    def reset(self) -> None:
        self.votes.clear()

    def stable(self) -> tuple[str, float]:
        if not self.votes:
            return "", 0.0
        counts = collections.Counter(self.votes)
        text, n = counts.most_common(1)[0]
        return text, n / len(self.votes)


def roi_rect(shape: tuple[int, ...], frac: float) -> tuple[int, int, int, int]:
    """Centred rectangle covering `frac` of the width and height: (x0, y0, x1, y1)."""
    h, w = shape[:2]
    frac = min(1.0, max(0.2, frac))
    rw, rh = int(round(w * frac)), int(round(h * frac))
    x0, y0 = (w - rw) // 2, (h - rh) // 2
    return x0, y0, x0 + rw, y0 + rh


def detect_frame(frame: np.ndarray, predictor, params: PreprocessParams, roi: float, min_conf: float, use_paper: bool = True) -> dict:
    """Run the pipeline on the centre area of one frame (finding the sheet of
    paper first when use_paper). Boxes and the paper outline are returned in
    full-frame coordinates."""
    x0, y0, x1, y1 = roi_rect(frame.shape, roi)
    t0 = time.perf_counter()
    result, paper, _seg = run_pipeline(frame[y0:y1, x0:x1], params, predictor, min_conf, use_paper=use_paper)
    lines = result["lines"]
    for ln in lines:
        for d in ln["digits"]:
            d["x"] += x0
            d["y"] += y0
            if d.get("quad"):
                d["quad"] = [[qx + x0, qy + y0] for qx, qy in d["quad"]]
    paper_quad = None
    if paper is not None and not paper.fills_frame:
        paper_quad = [[int(round(float(qx))) + x0, int(round(float(qy))) + y0] for qx, qy in paper.quad]
    return {
        "lines": lines, "text": " | ".join(ln["text"] for ln in lines), "digits": result["digit_count"],
        "roi": (x0, y0, x1, y1), "paper_quad": paper_quad,
        "paper_found": bool(paper is not None) if use_paper else None,
        "ms": (time.perf_counter() - t0) * 1000, "ts": time.time(),
    }


# --------------------------------------------------------------------------- #
# Background detector
# --------------------------------------------------------------------------- #
class Detector(threading.Thread):
    """Takes the newest submitted frame, detects, publishes the result."""

    def __init__(self, predictor, params: PreprocessParams, roi: float, min_conf: float, use_paper: bool = True):
        super().__init__(daemon=True)
        self.predictor, self.params, self.roi, self.min_conf, self.use_paper = predictor, params, roi, min_conf, use_paper
        self._frame = None
        self._cond = threading.Condition()
        self.result: dict | None = None
        self.running = True

    def submit(self, frame: np.ndarray) -> None:
        with self._cond:
            self._frame = frame  # newest wins; older unprocessed frames are dropped
            self._cond.notify()

    def stop(self) -> None:
        self.running = False
        with self._cond:
            self._cond.notify()

    def run(self) -> None:
        while self.running:
            with self._cond:
                while self._frame is None and self.running:
                    self._cond.wait(timeout=0.5)
                frame, self._frame = self._frame, None
            if frame is None:
                continue
            try:
                self.result = detect_frame(frame, self.predictor, self.params, self.roi, self.min_conf, self.use_paper)
            except Exception as exc:  # keep the window alive on a bad frame
                self.result = {"lines": [], "text": "", "digits": 0, "roi": roi_rect(frame.shape, self.roi), "paper_quad": None, "paper_found": None, "ms": 0.0, "ts": time.time(), "error": str(exc)}


# --------------------------------------------------------------------------- #
# Drawing and I/O
# --------------------------------------------------------------------------- #
def draw_overlay(frame: np.ndarray, result: dict | None, roi: float, stable_text: str, agreement: float, fps: float, min_conf: float, paused: bool, otsu: bool) -> np.ndarray:
    vis = annotate(frame, result["lines"]) if result else frame.copy()
    x0, y0, x1, y1 = roi_rect(frame.shape, roi)
    if roi < 1.0:
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 200, 0), 1)
    if result and result.get("paper_quad"):
        cv2.polylines(vis, [np.array(result["paper_quad"], dtype=np.int32).reshape(-1, 1, 2)], True, (255, 160, 0), 2, cv2.LINE_AA)
    h, w = vis.shape[:2]
    bar_h = 56
    cv2.rectangle(vis, (0, 0), (w, bar_h), (0, 0, 0), -1)
    reading = stable_text if stable_text else "-"
    color = (0, 255, 0) if agreement >= 0.6 else (0, 200, 255)
    cv2.putText(vis, reading, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)
    det_ms = result["ms"] if result else 0.0
    paper_state = ""
    if result and result.get("paper_found") is not None:
        paper_state = "paper: found  " if result["paper_found"] else "paper: NOT FOUND  "
    info = f"{paper_state}{fps:4.1f} fps  detect {det_ms:5.1f} ms  agree {agreement * 100:3.0f}%  conf>={min_conf:.2f}  {'OTSU ' if otsu else ''}{'PAUSED' if paused else ''}"
    cv2.putText(vis, info, (w - 8 - int(len(info) * 9.6), 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(vis, "q quit  space pause  s snapshot  r reset  +/- conf  o otsu  [ ] area", (w - 8 - 70 * 9, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    return vis


def open_source(source: str, width: int, height: int):
    """Return (capture_or_None, still_image_or_None)."""
    if source.isdigit():
        index = int(source)
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera {index}; try --source 1 or check camera privacy settings")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap, None
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"source not found: {source}")
    if path.suffix.lower() in IMAGE_EXTS:
        return None, load_image(str(path))
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {source}")
    return cap, None


def save_snapshot(save_dir: str, frame: np.ndarray, result: dict | None, stable_text: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = Path(save_dir) / f"capture_{stamp}"
    save_image(f"{base}.png", frame)
    if result:
        save_image(f"{base}_detected.png", annotate(frame, result["lines"]))
        with open(f"{base}.json", "w", encoding="utf-8") as f:
            json.dump({"stable_text": stable_text, **{k: v for k, v in result.items() if k != "ts"}}, f, indent=2)
    return str(base)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not Path(args.model).is_file():
        print(f"error: model not found: {args.model} (run train.py, optionally export_onnx.py)", file=sys.stderr)
        return 1
    from predictor import load_predictor

    predictor = load_predictor(args.model)
    params = PreprocessParams(max_side=args.max_side, use_otsu=args.otsu, remove_lines=not args.keep_lines)
    try:
        cap, still = open_source(args.source, args.width, args.height)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    sync = args.sync or args.no_window
    use_paper = not args.no_paper
    roi = args.roi if args.roi is not None else (1.0 if use_paper else 0.7)
    min_conf = args.min_conf
    detector = None
    if not sync:
        detector = Detector(predictor, params, roi, min_conf, use_paper)
        detector.start()
    stab = Stabilizer(args.history)
    last_result_ts = 0.0
    result = None
    paused = False
    frames = 0
    fps, t_prev = 0.0, time.perf_counter()
    last_printed = None
    print(f"source {args.source} | model {args.model} ({predictor.backend}) | paper detection {'on' if use_paper else 'off'} | press q to quit", file=sys.stderr)

    try:
        while True:
            if still is not None:
                frame = still.copy()
            else:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("end of stream", file=sys.stderr)
                    break
            frames += 1
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - t_prev)) if fps else 1.0 / max(1e-6, now - t_prev)
            t_prev = now

            if not paused:
                if sync:
                    result = detect_frame(frame, predictor, params, roi, min_conf, use_paper)
                else:
                    detector.submit(frame)
                    result = detector.result
                if result and result["ts"] != last_result_ts:
                    last_result_ts = result["ts"]
                    stab.push(result["text"])
            stable_text, agreement = stab.stable()

            if args.no_window:
                if result and (result["text"], stable_text) != last_printed:
                    paper_note = "" if result.get("paper_found") is None else (" paper=found" if result["paper_found"] else " paper=none")
                    print(f"frame {frames}: now={result['text']!r} stable={stable_text!r} agree={agreement:.2f} digits={result['digits']}{paper_note} {result['ms']:.1f} ms")
                    last_printed = (result["text"], stable_text)
            else:
                cv2.imshow("Digit detection", draw_overlay(frame, result, roi, stable_text, agreement, fps, min_conf, paused, params.use_otsu))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused = not paused
                elif key == ord("s"):
                    print("saved", save_snapshot(args.save_dir, frame, result, stable_text), file=sys.stderr)
                elif key == ord("r"):
                    stab.reset()
                elif key in (ord("+"), ord("=")):
                    min_conf = min(1.0, min_conf + 0.05)
                elif key == ord("-"):
                    min_conf = max(0.0, min_conf - 0.05)
                elif key == ord("o"):
                    params.use_otsu = not params.use_otsu
                elif key == ord("["):
                    roi = max(0.2, roi - 0.05)
                elif key == ord("]"):
                    roi = min(1.0, roi + 0.05)
                if detector is not None:
                    detector.roi, detector.min_conf = roi, min_conf
            if args.frames and frames >= args.frames:
                break
    finally:
        if detector is not None:
            detector.stop()
        if cap is not None:
            cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()
    if args.no_window:
        stable_text, agreement = stab.stable()
        print(f"final reading: {stable_text!r} (agreement {agreement:.2f}, {frames} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
