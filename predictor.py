"""Model loading and batched inference for .keras, .tflite and .onnx files.

All back-ends share one contract:
    predictor.predict(batch) -> float32 array of shape (N, 10) with softmax probabilities
where batch is float32 (N, 28, 28, 1) in 0..1, white digit on black (see preprocess.crop_to_mnist).
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import warnings
from pathlib import Path

import numpy as np

INPUT_SHAPE = (28, 28, 1)


def prepare_batch(batch: np.ndarray) -> np.ndarray:
    """Coerce (28,28), (28,28,1), (N,28,28) or (N,28,28,1) into float32 (N,28,28,1)."""
    x = np.asarray(batch, dtype=np.float32)
    if x.ndim == 2:
        x = x[None, :, :, None]
    elif x.ndim == 3:
        x = x[None] if x.shape == INPUT_SHAPE else x[..., None]
    if x.ndim != 4 or tuple(x.shape[1:]) != INPUT_SHAPE:
        raise ValueError(f"expected a batch of shape (N, 28, 28, 1), got {x.shape}")
    if x.max() > 1.0 + 1e-6:  # someone passed 0..255 pixels
        x = x / 255.0
    return np.ascontiguousarray(x)


class Predictor:
    """Base class: subclasses set `backend` and implement `_run`."""

    backend = "base"

    def __init__(self, path: str):
        self.path = str(path)

    def _run(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, batch: np.ndarray) -> np.ndarray:
        x = prepare_batch(batch)
        probs = np.asarray(self._run(x), dtype=np.float32)
        return probs.reshape(x.shape[0], 10)

    def predict_labels(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (labels, confidences)."""
        probs = self.predict(batch)
        return probs.argmax(axis=1), probs.max(axis=1)

    def warmup(self) -> None:
        self.predict(np.zeros((1,) + INPUT_SHAPE, np.float32))


class KerasPredictor(Predictor):
    backend = "keras"

    def __init__(self, path: str):
        super().__init__(path)
        import tensorflow as tf
        import keras

        self.model = keras.saving.load_model(self.path, compile=False)
        # A traced graph with a fixed signature: no per-call eager dispatch
        # (about 20 ms per call otherwise) and no retracing for new batch sizes.
        self._infer = tf.function(
            lambda x: self.model(x, training=False),
            input_signature=[tf.TensorSpec([None, 28, 28, 1], tf.float32)],
        )
        self.warmup()

    def _run(self, x: np.ndarray) -> np.ndarray:
        # Direct graph call: far cheaper than model.predict() for small batches.
        return self._infer(x).numpy()


class TFLitePredictor(Predictor):
    backend = "tflite"

    def __init__(self, path: str, num_threads: int | None = None):
        super().__init__(path)
        interpreter_cls = None
        try:
            from ai_edge_litert.interpreter import Interpreter as interpreter_cls  # type: ignore
        except Exception:
            import tensorflow as tf

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                interpreter_cls = tf.lite.Interpreter
        threads = num_threads or os.cpu_count() or 1
        self.interp = interpreter_cls(model_path=self.path, num_threads=threads)
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self._allocated = 0
        self._allocate(1)
        self.warmup()

    def _allocate(self, n: int) -> None:
        self.interp.resize_tensor_input(self.inp["index"], [n, 28, 28, 1])
        self.interp.allocate_tensors()
        self._allocated = n

    def _run(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self._allocated:
            self._allocate(x.shape[0])
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        return np.array(self.interp.get_tensor(self.out["index"]))


class OnnxPredictor(Predictor):
    backend = "onnx"

    def __init__(self, path: str, num_threads: int | None = None):
        super().__init__(path)
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads or os.cpu_count() or 1
        self.session = ort.InferenceSession(self.path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.warmup()

    def _run(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: x})[0]


def compare_predictors(reference: Predictor, others: list[Predictor], x: np.ndarray, y: np.ndarray | None = None, single_runs: int = 200) -> list[dict]:
    """Run every predictor on the same images and report parity with the
    reference plus timing. Returns one dict per predictor (reference first)."""
    import time

    x = prepare_batch(x)
    ref_probs = reference.predict(x)
    rows = []
    for pred in [reference] + list(others):
        t0 = time.perf_counter()
        probs = pred.predict(x)
        batch_ms = (time.perf_counter() - t0) * 1000
        n_single = min(single_runs, len(x))
        t0 = time.perf_counter()
        for i in range(n_single):
            pred.predict(x[i:i + 1])
        single_ms = (time.perf_counter() - t0) * 1000 / max(1, n_single)
        row = {
            "backend": pred.backend,
            "path": pred.path,
            "max_abs_prob_diff": float(np.abs(probs - ref_probs).max()),
            "argmax_agreement": float((probs.argmax(1) == ref_probs.argmax(1)).mean()),
            "batch_ms_per_image": round(batch_ms / len(x), 4),
            "single_ms_per_image": round(single_ms, 3),
        }
        if y is not None:
            row["accuracy"] = float((probs.argmax(1) == np.asarray(y)[: len(x)]).mean())
        rows.append(row)
    return rows


def print_comparison(rows: list[dict]) -> None:
    print(f"{'backend':8s} {'accuracy':>9s} {'max|dp|':>9s} {'argmax=':>8s} {'batch ms/img':>13s} {'single ms/img':>14s}")
    for r in rows:
        acc = f"{r['accuracy']:.4f}" if "accuracy" in r else "   -  "
        print(f"{r['backend']:8s} {acc:>9s} {r['max_abs_prob_diff']:9.2e} {r['argmax_agreement']:8.4f} {r['batch_ms_per_image']:13.3f} {r['single_ms_per_image']:14.3f}")


def load_predictor(path: str) -> Predictor:
    """Pick the back-end from the file extension: .keras, .tflite or .onnx."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"model file not found: {path} (run train.py first)")
    suffix = p.suffix.lower()
    if suffix == ".keras":
        return KerasPredictor(str(p))
    if suffix == ".tflite":
        return TFLitePredictor(str(p))
    if suffix == ".onnx":
        return OnnxPredictor(str(p))
    raise ValueError(f"unsupported model type {suffix!r}; expected .keras, .tflite or .onnx")
