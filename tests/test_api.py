"""API tests for server.py (FastAPI TestClient, no network).

    python -m pytest tests/test_api.py -q

Skipped until models/mnist_cnn.onnx (python export_onnx.py) and
samples/ground_truth.json (python make_samples.py) exist.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL = ROOT / "models" / "mnist_cnn.onnx"
TRUTH = ROOT / "samples" / "ground_truth.json"
pytestmark = pytest.mark.skipif(not (MODEL.is_file() and TRUTH.is_file()), reason="run export_onnx.py and make_samples.py first")


@pytest.fixture(scope="module")
def client():
    os.environ["MODEL_PATH"] = str(MODEL)
    from fastapi.testclient import TestClient

    import server

    with TestClient(server.app) as c:
        yield c


@pytest.fixture(scope="module")
def truth():
    return json.loads(TRUTH.read_text(encoding="utf-8"))


def sample(truth, category):
    name = next(n for n, g in truth.items() if g["category"] == category)
    return name, (ROOT / "samples" / name).read_bytes(), truth[name]["lines"]


def test_health(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["backend"] == "onnx"


def test_info(client):
    r = client.get("/api/v1/info")
    assert r.status_code == 200
    body = r.json()
    assert body["model"]["backend"] == "onnx"
    assert "preprocessing_defaults" in body and body["limits"]["api_key_required"] is False


def test_index_and_docs(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_detect_clean_sample(client, truth):
    name, data, lines = sample(truth, "clean")
    r = client.post("/api/v1/detect", params={"annotate": "true"}, files={"image": (name, data, "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == lines[0]
    assert body["digit_count"] == len(lines[0])
    assert body["lines"][0]["digits"][0]["accepted"] is True
    assert body["annotated_png_base64"].startswith("iVBOR")  # PNG signature in base64
    assert body["model"]["backend"] == "onnx"


def test_detect_multiline_sample(client, truth):
    name, data, lines = sample(truth, "multiline")
    r = client.post("/api/v1/detect", files={"image": (name, data, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["text"].split("\n") == lines
    assert body["annotated_png_base64"] is None


def test_annotated_endpoint(client, truth):
    name, data, lines = sample(truth, "clean")
    r = client.post("/api/v1/detect/annotated", files={"image": (name, data, "image/png")})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert r.headers["x-detected-text"] == lines[0]


def test_min_conf_marks_uncertain_digits(client, truth):
    name, data, lines = sample(truth, "clean")
    r = client.post("/api/v1/detect", params={"min_conf": 1.0}, files={"image": (name, data, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert set(body["text"]) <= {"?"} and len(body["text"]) == len(lines[0])
    assert all(d["accepted"] is False for d in body["lines"][0]["digits"])


def test_blank_image_gives_empty_result(client):
    import cv2
    import numpy as np

    ok, buf = cv2.imencode(".png", np.full((120, 200, 3), 255, np.uint8))
    r = client.post("/api/v1/detect", files={"image": ("blank.png", buf.tobytes(), "image/png")})
    assert r.status_code == 200
    assert r.json()["digit_count"] == 0 and r.json()["text"] == ""


def test_bad_upload(client):
    r = client.post("/api/v1/detect", files={"image": ("x.png", b"definitely not an image", "image/png")})
    assert r.status_code == 400
    r = client.post("/api/v1/detect", files={"image": ("x.png", b"", "image/png")})
    assert r.status_code == 400


def test_too_large(client, truth, monkeypatch):
    import server

    name, data, _ = sample(truth, "clean")
    monkeypatch.setitem(server.CONFIG, "max_upload_bytes", 1024)
    r = client.post("/api/v1/detect", files={"image": (name, data, "image/png")})
    assert r.status_code == 413


def test_api_key(client, truth, monkeypatch):
    import server

    name, data, lines = sample(truth, "clean")
    monkeypatch.setitem(server.CONFIG, "api_key", "s3cret")
    assert client.post("/api/v1/detect", files={"image": (name, data, "image/png")}).status_code == 401
    r = client.post("/api/v1/detect", files={"image": (name, data, "image/png")}, headers={"X-API-Key": "s3cret"})
    assert r.status_code == 200 and r.json()["text"] == lines[0]
    assert client.get("/healthz").status_code == 200  # health stays open for the platform's probes
