"""Tests for the digit detector.

    python -m pytest tests -q

The preprocessing tests need only OpenCV. The end-to-end tests are skipped
until models/mnist_cnn.keras and samples/ground_truth.json exist
(run train.py and make_samples.py first).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import preprocess as pp  # noqa: E402

MODEL = ROOT / "models" / "mnist_cnn.keras"
SAMPLES = ROOT / "samples"
TRUTH = SAMPLES / "ground_truth.json"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def draw_text(text: str, dark_on_light: bool = True, scale: float = 3.0, thickness: int = 6) -> np.ndarray:
    """Render digits with OpenCV's Hershey font on a plain background (BGR)."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    paper, ink = ((240, 240, 240), (30, 30, 30)) if dark_on_light else ((25, 25, 25), (230, 230, 230))
    img = np.full((h + base + 60, w + 60, 3), paper, np.uint8)
    cv2.putText(img, text, (30, 30 + h), cv2.FONT_HERSHEY_SIMPLEX, scale, ink, thickness, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------- #
# unit tests: no model needed
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dark_on_light", [True, False])
def test_polarity_autodetect(dark_on_light):
    img = draw_text("123", dark_on_light=dark_on_light)
    binary = pp.to_binary(pp.to_gray(img))
    ink_fraction = cv2.countNonZero(binary) / binary.size
    assert 0.01 < ink_fraction < 0.4, "ink must be the minority regardless of polarity"
    crops = pp.segment_digits(img)
    assert len(crops) == 3
    assert [c.line for c in crops] == [0, 0, 0]
    assert crops[0].x < crops[1].x < crops[2].x


def test_holes_do_not_create_extra_boxes():
    crops = pp.segment_digits(draw_text("0808"))
    assert len(crops) == 4


def test_detached_top_bar_is_merged():
    """A '5' drawn as a body plus a separate top bar must become ONE box."""
    img = np.full((220, 160, 3), 245, np.uint8)
    cv2.rectangle(img, (40, 70), (110, 190), (20, 20, 20), 8)          # body (a box shape is fine)
    cv2.line(img, (40, 40), (125, 40), (20, 20, 20), 8)                # detached top bar, 30 px above
    crops = pp.segment_digits(img)
    assert len(crops) == 1
    assert crops[0].y <= 45 and crops[0].y + crops[0].h >= 185


def test_touching_digits_are_split():
    img = draw_text("0", scale=3.0, thickness=6)
    # paste a second copy so that the two zeros touch
    gray = pp.to_gray(img)
    binary = pp.to_binary(gray)
    ys, xs = np.nonzero(binary)
    one = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = one.shape
    pair = np.zeros((h + 40, 2 * w + 40 - 3), np.uint8)
    pair[20:20 + h, 20:20 + w] = one
    pair[20:20 + h, 20 + w - 3:20 + 2 * w - 3] = np.maximum(pair[20:20 + h, 20 + w - 3:20 + 2 * w - 3], one)
    boxes = pp.find_digit_boxes(pair)
    assert len(boxes) == 2


def test_underline_is_removed():
    img = draw_text("4567")
    h, w = img.shape[:2]
    cv2.line(img, (20, h - 25), (w - 20, h - 25), (20, 20, 20), 5)
    crops = pp.segment_digits(img)
    assert len(crops) == 4


def test_two_lines_are_ordered_top_to_bottom_then_left_to_right():
    boxes = [(300, 10, 40, 60), (10, 12, 40, 60), (150, 8, 40, 60), (20, 200, 40, 60), (200, 205, 40, 60)]
    ordered = pp.order_boxes(boxes)
    assert [line for line, _ in ordered] == [0, 0, 0, 1, 1]
    assert [b[0] for _, b in ordered] == [10, 150, 300, 20, 200]


def test_crop_to_mnist_properties():
    mask = np.zeros((120, 60), np.uint8)
    cv2.ellipse(mask, (30, 60), (22, 50), 0, 0, 360, 255, 6)
    out = pp.crop_to_mnist(mask)
    assert out.shape == (28, 28) and out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0 and out.max() > 0.9
    m = cv2.moments(out)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    assert abs(cx - 13.5) <= 1.0 and abs(cy - 13.5) <= 1.0, "centre of mass must be centred"
    ys, xs = np.nonzero(out > 0.1)
    assert ys.max() - ys.min() + 1 <= 21 and xs.max() - xs.min() + 1 <= 21, "digit must fit in the 20x20 box"


def test_thin_strokes_are_thickened():
    mask = np.zeros((100, 60), np.uint8)
    cv2.line(mask, (30, 5), (30, 95), 255, 1)                          # 1 px wide stroke
    out = pp.crop_to_mnist(mask)
    assert out.max() > 0.9, "a hairline stroke must still be strongly visible after downscaling"


def test_unicode_paths_roundtrip(tmp_path):
    img = draw_text("7")
    path = tmp_path / "tëst ünïcode 图" / "digit 7.png"
    pp.save_image(str(path), img)
    assert path.is_file()
    back = pp.load_image(str(path))
    assert back.shape == img.shape and back.dtype == np.uint8


def test_load_image_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        pp.load_image(str(tmp_path / "missing.png"))
    bad = tmp_path / "bad.png"
    bad.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError):
        pp.load_image(str(bad))


def test_blank_images_give_no_digits():
    assert pp.segment_digits(np.full((200, 300, 3), 255, np.uint8)) == []
    assert pp.segment_digits(np.zeros((200, 300, 3), np.uint8)) == []


# --------------------------------------------------------------------------- #
# end-to-end tests: need the trained model and the generated samples
# --------------------------------------------------------------------------- #
needs_model = pytest.mark.skipif(not (MODEL.is_file() and TRUTH.is_file()), reason="run train.py and make_samples.py first")

# With 6 images per category one miss already costs 17 %, so the hard limits
# are on the totals (54 numbers, ~270 digits); per category only a sanity floor.
MIN_DIGIT_ACCURACY = 0.95      # over all digits of all samples
MIN_SEQUENCE_ACCURACY = 0.80   # exact number, over all samples
MIN_CATEGORY_ACCURACY = 0.50   # exact number, per category


@pytest.fixture(scope="module")
def predictor():
    from predictor import load_predictor

    return load_predictor(str(MODEL))


@pytest.fixture(scope="module")
def sample_results(predictor):
    from detect import recognise

    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    results = {}
    for name, gt in truth.items():
        seg = pp.segment_image(pp.load_image(str(SAMPLES / name)))
        lines = recognise(seg, predictor, min_conf=0.0) if seg.crops else []
        results[name] = {"truth": gt["lines"], "pred": [ln["text"] for ln in lines], "category": gt["category"]}
    return results


@needs_model
def test_per_category_sequence_accuracy(sample_results):
    per_cat: dict[str, list[bool]] = {}
    digit_ok = digit_total = 0
    for r in sample_results.values():
        per_cat.setdefault(r["category"], []).append(r["pred"] == r["truth"])
        for t, p in zip(r["truth"], r["pred"]):
            for tc, pc in zip(t, p):
                digit_ok += tc == pc
            digit_total += len(t)
    print("\nexact-number accuracy per category:")
    failures = []
    for cat, oks in sorted(per_cat.items()):
        acc = sum(oks) / len(oks)
        print(f"  {cat:14s} {acc:.2f}  ({sum(oks)}/{len(oks)})")
        if acc < MIN_CATEGORY_ACCURACY:
            failures.append(f"{cat}: {acc:.2f} < {MIN_CATEGORY_ACCURACY}")
    all_ok = [ok for oks in per_cat.values() for ok in oks]
    seq_acc = sum(all_ok) / len(all_ok)
    digit_acc = digit_ok / max(1, digit_total)
    print(f"  overall exact-number accuracy: {seq_acc:.3f} ({sum(all_ok)}/{len(all_ok)})")
    print(f"  overall per-digit accuracy:    {digit_acc:.4f} ({digit_ok}/{digit_total})")
    if seq_acc < MIN_SEQUENCE_ACCURACY:
        failures.append(f"exact-number accuracy {seq_acc:.3f} < {MIN_SEQUENCE_ACCURACY}")
    if digit_acc < MIN_DIGIT_ACCURACY:
        failures.append(f"per-digit accuracy {digit_acc:.4f} < {MIN_DIGIT_ACCURACY}")
    assert not failures, failures


@needs_model
def test_cli_plain_and_json(tmp_path):
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    name = next(n for n, g in truth.items() if g["category"] == "clean")
    py = sys.executable
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="2")
    out = tmp_path / "annotated.png"
    r = subprocess.run([py, str(ROOT / "detect.py"), str(SAMPLES / name), "--out", str(out)], capture_output=True, text=True, cwd=str(ROOT), env=env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == truth[name]["lines"][0]
    assert out.is_file()
    r = subprocess.run([py, str(ROOT / "detect.py"), str(SAMPLES / name), "--json", "--out", str(out)], capture_output=True, text=True, cwd=str(ROOT), env=env)
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["text"] == truth[name]["lines"][0]
    assert all({"x", "y", "w", "h", "label", "confidence", "accepted"} <= set(d) for d in data["lines"][0]["digits"])


@needs_model
def test_cli_exit_codes(tmp_path):
    py = sys.executable
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="2")
    blank = tmp_path / "blank.png"
    pp.save_image(str(blank), np.full((100, 100, 3), 255, np.uint8))
    r = subprocess.run([py, str(ROOT / "detect.py"), str(blank), "--out", str(tmp_path / "o.png")], capture_output=True, text=True, cwd=str(ROOT), env=env)
    assert r.returncode == 2
    r = subprocess.run([py, str(ROOT / "detect.py"), str(tmp_path / "missing.png")], capture_output=True, text=True, cwd=str(ROOT), env=env)
    assert r.returncode == 1 and "not found" in r.stderr
