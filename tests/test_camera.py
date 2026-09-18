"""Tests for camera.py helpers (no camera hardware needed)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import camera  # noqa: E402

MODEL = ROOT / "models" / "mnist_cnn.onnx"
SAMPLE = ROOT / "samples" / "clean_00.png"


def test_stabilizer_majority_vote():
    s = camera.Stabilizer(5)
    assert s.stable() == ("", 0.0)
    for t in ["123", "128", "123", "123", "12"]:
        s.push(t)
    text, agreement = s.stable()
    assert text == "123" and agreement == pytest.approx(0.6)
    s.push("999")  # deque drops the oldest "123"
    assert s.stable()[0] == "123"
    s.reset()
    assert s.stable() == ("", 0.0)


def test_roi_rect_is_centred_and_clamped():
    x0, y0, x1, y1 = camera.roi_rect((720, 1280, 3), 0.5)
    assert (x1 - x0, y1 - y0) == (640, 360)
    assert x0 == 320 and y0 == 180
    assert camera.roi_rect((100, 200), 5.0) == (0, 0, 200, 100)   # clamped to the frame
    x0, y0, x1, y1 = camera.roi_rect((100, 200), 0.0)             # clamped to 20 %
    assert (x1 - x0, y1 - y0) == (40, 20)


def test_default_model_prefers_onnx(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    assert camera.default_model().endswith("mnist_cnn.keras")
    (tmp_path / "models" / "mnist_cnn.keras").write_bytes(b"x")
    (tmp_path / "models" / "mnist_cnn.onnx").write_bytes(b"x")
    assert camera.default_model().endswith("mnist_cnn.onnx")


@pytest.mark.skipif(not (MODEL.is_file() and SAMPLE.is_file()), reason="run export_onnx.py and make_samples.py first")
def test_headless_run_on_still_image():
    import json

    truth = json.loads((ROOT / "samples" / "ground_truth.json").read_text(encoding="utf-8"))["clean_00.png"]["lines"][0]
    r = subprocess.run([sys.executable, str(ROOT / "camera.py"), "--source", str(SAMPLE), "--model", str(MODEL), "--no-window", "--frames", "3", "--roi", "1.0"],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    assert f"final reading: {truth!r}" in r.stdout


@pytest.mark.skipif(not MODEL.is_file(), reason="run export_onnx.py first")
def test_detect_frame_offsets_boxes_into_frame_coordinates():
    import cv2

    from predictor import load_predictor
    from preprocess import PreprocessParams

    frame = np.full((720, 1280, 3), 245, np.uint8)
    cv2.putText(frame, "58", (500, 420), cv2.FONT_HERSHEY_SIMPLEX, 4, (20, 20, 20), 10, cv2.LINE_AA)
    result = camera.detect_frame(frame, load_predictor(str(MODEL)), PreprocessParams(max_side=640), roi=0.7, min_conf=0.0)
    assert result["text"] == "58"
    x0, y0, x1, y1 = result["roi"]
    for d in result["lines"][0]["digits"]:
        assert x0 <= d["x"] < x1 and y0 <= d["y"] < y1
