"""HTTP API for the digit detector (FastAPI + ONNX Runtime, CPU only).

Run locally:
    uvicorn server:app --host 0.0.0.0 --port 7860
Then open:
    http://localhost:7860/          web page: upload an image, see the result
    http://localhost:7860/docs      interactive API documentation (Swagger UI)
    http://localhost:7860/healthz   liveness/readiness probe

Endpoints:
    POST /api/v1/detect             multipart field "image" -> JSON (text, lines, boxes, confidences)
    POST /api/v1/detect/annotated   same input -> annotated PNG
    GET  /api/v1/info               model, limits, default preprocessing parameters

Configuration (environment variables):
    MODEL_PATH     models/mnist_cnn.onnx   .onnx (default, no TensorFlow needed), .tflite or .keras
    MAX_UPLOAD_MB  8                       reject bigger uploads with HTTP 413
    MAX_SIDE       1600                    default working resolution cap
    CORS_ORIGINS   *                       comma-separated origins; empty string disables CORS
    API_KEY        (unset)                 when set, requests must send header X-API-Key
"""
from __future__ import annotations

import base64
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from detect import annotate as draw_annotations
from detect import recognise
from predictor import load_predictor
from preprocess import PreprocessParams, decode_image, segment_image

API_VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"

CONFIG = {
    "model_path": os.environ.get("MODEL_PATH", str(ROOT / "models" / "mnist_cnn.onnx")),
    "max_upload_bytes": int(float(os.environ.get("MAX_UPLOAD_MB", "8")) * 1024 * 1024),
    "max_side": int(os.environ.get("MAX_SIDE", "1600")),
    "cors_origins": [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()],
    "api_key": os.environ.get("API_KEY", ""),
}


class _State:
    predictor = None
    lock = threading.Lock()   # TFLite interpreters are not thread-safe; ONNX is, but the lock costs ~nothing
    started = time.time()
    requests = 0
    digits = 0


STATE = _State()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    STATE.predictor = load_predictor(CONFIG["model_path"])  # fails fast if the model is missing
    yield


app = FastAPI(
    title="Digit Detection API",
    version=API_VERSION,
    description="Detects handwritten or printed digits in an image with OpenCV segmentation and a CNN trained on MNIST. CPU only, no external services.",
    lifespan=lifespan,
)
if CONFIG["cors_origins"]:
    app.add_middleware(CORSMiddleware, allow_origins=CONFIG["cors_origins"], allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class Digit(BaseModel):
    x: int
    y: int
    w: int
    h: int
    label: int = Field(description="predicted digit 0-9")
    confidence: float = Field(description="softmax probability of the label")
    accepted: bool = Field(description="confidence >= min_conf")


class Line(BaseModel):
    line: int
    text: str = Field(description="digits of this text line in reading order; ? marks a digit below min_conf")
    digits: list[Digit]


class RejectedBlob(BaseModel):
    x: int
    y: int
    w: int
    h: int
    reason: str


class DetectResponse(BaseModel):
    text: str = Field(description="all lines joined with newline")
    lines: list[Line]
    digit_count: int
    rejected: list[RejectedBlob]
    image: dict
    model: dict
    timing_ms: dict
    annotated_png_base64: Optional[str] = Field(default=None, description="only when annotate=true")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    if CONFIG["api_key"] and x_api_key != CONFIG["api_key"]:
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key header")


def read_upload(upload: UploadFile) -> np.ndarray:
    limit = CONFIG["max_upload_bytes"]
    data = upload.file.read(limit + 1)
    if not data:
        raise HTTPException(status_code=400, detail="empty upload; send the image as multipart field 'image'")
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"image larger than {limit // (1024 * 1024)} MB")
    try:
        return decode_image(data)
    except ValueError:
        raise HTTPException(status_code=400, detail="cannot decode image; send PNG, JPEG, BMP, WEBP or TIFF")


def build_params(max_side: int, otsu: bool, keep_lines: bool, binary_crops: bool, target_stroke: float) -> PreprocessParams:
    return PreprocessParams(max_side=max_side, use_otsu=otsu, remove_lines=not keep_lines, soft_crops=not binary_crops, target_stroke=target_stroke)


def run_detection(img: np.ndarray, min_conf: float, params: PreprocessParams) -> dict:
    t0 = time.perf_counter()
    seg = segment_image(img, params)
    t_seg = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    if seg.crops:
        with STATE.lock:
            lines = recognise(seg, STATE.predictor, min_conf)
    else:
        lines = []
    t_inf = (time.perf_counter() - t0) * 1000
    STATE.requests += 1
    STATE.digits += len(seg.crops)
    return {
        "text": "\n".join(ln["text"] for ln in lines),
        "lines": lines,
        "digit_count": len(seg.crops),
        "rejected": [r.__dict__ for r in seg.rejected],
        "image": {"width": int(img.shape[1]), "height": int(img.shape[0])},
        "model": {"path": Path(CONFIG["model_path"]).name, "backend": STATE.predictor.backend, "api_version": API_VERSION},
        "timing_ms": {"segmentation": round(t_seg, 1), "inference": round(t_inf, 1)},
    }


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(status_code=500, detail="failed to encode annotated image")
    return buf.tobytes()


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/healthz", tags=["ops"])
def healthz() -> dict:
    return {
        "status": "ok",
        "model": Path(CONFIG["model_path"]).name,
        "backend": STATE.predictor.backend if STATE.predictor else None,
        "api_version": API_VERSION,
        "uptime_s": round(time.time() - STATE.started, 1),
        "requests": STATE.requests,
        "digits": STATE.digits,
    }


@app.get("/api/v1/info", tags=["ops"])
def info() -> dict:
    return {
        "api_version": API_VERSION,
        "model": {"path": Path(CONFIG["model_path"]).name, "backend": STATE.predictor.backend if STATE.predictor else None, "input": "28x28x1 float32, white digit on black"},
        "limits": {"max_upload_mb": CONFIG["max_upload_bytes"] / (1024 * 1024), "max_side_default": CONFIG["max_side"], "api_key_required": bool(CONFIG["api_key"])},
        "preprocessing_defaults": PreprocessParams(max_side=CONFIG["max_side"]).to_dict(),
        "endpoints": {"detect": "POST /api/v1/detect (multipart field 'image')", "annotated": "POST /api/v1/detect/annotated", "docs": "GET /docs"},
    }


@app.post("/api/v1/detect", response_model=DetectResponse, tags=["detection"], dependencies=[Depends(require_api_key)])
def detect(
    image: UploadFile = File(..., description="PNG, JPEG, BMP, WEBP or TIFF image containing digits"),
    min_conf: float = Query(0.5, ge=0.0, le=1.0, description="digits below this confidence are marked ? and accepted=false"),
    annotate: bool = Query(False, description="also return the annotated image as base64 PNG"),
    max_side: int = Query(CONFIG["max_side"], ge=200, le=4000, description="working resolution cap for the longer image side"),
    otsu: bool = Query(False, description="global Otsu threshold instead of adaptive (thick marker strokes)"),
    keep_lines: bool = Query(False, description="do not remove ruled lines / underlines"),
    binary_crops: bool = Query(False, description="feed hard 0/1 masks to the model instead of grayscale ink"),
    target_stroke: float = Query(0.12, ge=0.0, le=0.4, description="stroke width target as a fraction of digit size; 0 disables"),
) -> dict:
    img = read_upload(image)
    result = run_detection(img, min_conf, build_params(max_side, otsu, keep_lines, binary_crops, target_stroke))
    if annotate:
        result["annotated_png_base64"] = base64.b64encode(encode_png(draw_annotations(img, result["lines"]))).decode("ascii")
    return result


@app.post("/api/v1/detect/annotated", tags=["detection"], dependencies=[Depends(require_api_key)],
          responses={200: {"content": {"image/png": {}}, "description": "annotated PNG; detected text in header X-Detected-Text (lines separated by |)"}})
def detect_annotated(
    image: UploadFile = File(...),
    min_conf: float = Query(0.5, ge=0.0, le=1.0),
    max_side: int = Query(CONFIG["max_side"], ge=200, le=4000),
    otsu: bool = Query(False),
    keep_lines: bool = Query(False),
    binary_crops: bool = Query(False),
    target_stroke: float = Query(0.12, ge=0.0, le=0.4),
) -> Response:
    img = read_upload(image)
    result = run_detection(img, min_conf, build_params(max_side, otsu, keep_lines, binary_crops, target_stroke))
    png = encode_png(draw_annotations(img, result["lines"]))
    return Response(content=png, media_type="image/png", headers={"X-Detected-Text": result["text"].replace("\n", "|"), "X-Digit-Count": str(result["digit_count"])})
