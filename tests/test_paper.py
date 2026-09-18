"""Tests for paper.py: finding the sheet and detecting digits only on it."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paper as pp  # noqa: E402

MODEL = ROOT / "models" / "mnist_cnn.onnx"


def scene(angle: float = 12.0, text: str = "4071", clutter: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Dark wooden desk, a white sheet rotated by `angle` degrees with digits on
    it, plus distracting bright/dark clutter outside the sheet. Returns the
    frame and the sheet's 4 corners (TL, TR, BR, BL) in frame coordinates."""
    frame = np.full((720, 1280, 3), (40, 70, 110), np.uint8)             # brown desk (BGR)
    noise = np.random.default_rng(0).normal(0, 6, frame.shape).astype(np.float32)
    frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if clutter:
        cv2.putText(frame, "88", (40, 700), cv2.FONT_HERSHEY_SIMPLEX, 3, (10, 10, 10), 8)     # dark digits on the desk: must be ignored
        cv2.circle(frame, (1180, 120), 60, (250, 250, 250), -1)                              # white mug: bright but small
        cv2.rectangle(frame, (0, 0), (1280, 40), (200, 60, 200), -1)                         # coloured strip
    sheet = np.full((420, 600, 3), (238, 240, 242), np.uint8)
    cv2.putText(sheet, text, (60, 280), cv2.FONT_HERSHEY_SIMPLEX, 5, (25, 25, 25), 14, cv2.LINE_AA)
    h, w = sheet.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    centre = np.float32([640, 380])
    rad = np.deg2rad(angle)
    rot = np.float32([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    dst = (src - np.float32([w / 2, h / 2])) @ rot.T + centre
    m = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    warped = cv2.warpPerspective(sheet, m, (1280, 720))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), m, (1280, 720))
    frame[mask > 0] = warped[mask > 0]
    return frame, dst.astype(np.float32)


def test_finds_rotated_sheet_and_ignores_clutter():
    frame, corners = scene()
    found = pp.find_paper(frame)
    assert found is not None and not found.fills_frame
    assert 0.2 < found.area_fraction < 0.5
    # every detected corner is within a few pixels of a true corner
    for c in found.quad:
        assert np.min(np.linalg.norm(corners - c, axis=1)) < 12
    # the warped view is upright: about 600x420 and mostly paper-coloured
    h, w = found.warped.shape[:2]
    assert abs(w - 600) < 30 and abs(h - 420) < 30
    assert found.warped.mean() > 180


def test_no_paper_in_dark_scene():
    frame = np.full((480, 640, 3), (30, 40, 50), np.uint8)
    assert pp.find_paper(frame) is None


def test_full_frame_sheet_is_passthrough():
    frame = np.full((300, 500, 3), 240, np.uint8)
    cv2.putText(frame, "12", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (20, 20, 20), 10)
    found = pp.find_paper(frame)
    assert found is not None and found.fills_frame
    assert found.warped is frame
    assert found.map_box(10, 20, 30, 40) == {"x": 10, "y": 20, "w": 30, "h": 40}


def test_map_box_returns_to_frame_coordinates():
    frame, _ = scene(angle=0.0, clutter=False)
    found = pp.find_paper(frame)
    assert found is not None and not found.fills_frame
    box = found.map_box(0, 0, found.warped.shape[1] - 1, found.warped.shape[0] - 1)
    assert abs(box["x"] - 340) < 12 and abs(box["y"] - 170) < 12   # sheet top-left in the frame
    assert "quad" in box and len(box["quad"]) == 4


@pytest.mark.skipif(not MODEL.is_file(), reason="run export_onnx.py first")
def test_digits_detected_only_on_the_paper():
    from detect import run_pipeline
    from predictor import load_predictor
    from preprocess import PreprocessParams

    frame, _ = scene(angle=12.0, text="4071")
    result, sheet, _seg = run_pipeline(frame, PreprocessParams(max_side=1000), load_predictor(str(MODEL)), min_conf=0.0, use_paper=True)
    assert sheet is not None and result["paper"]["found"] is True and result["paper"]["fills_frame"] is False
    assert result["text"] == "4071", result["text"]           # the "88" on the desk is not read
    for d in result["lines"][0]["digits"]:
        assert 330 < d["x"] < 950 and 150 < d["y"] < 600   # boxes were mapped back onto the sheet in the frame
        assert "quad" in d
