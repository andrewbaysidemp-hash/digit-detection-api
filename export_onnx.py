"""Export the trained Keras model to ONNX and verify it with ONNX Runtime.

Usage:
    python -m pip install -r requirements-optional.txt     # tf2onnx, onnx, onnxruntime
    python export_onnx.py                                  # models/mnist_cnn.keras -> models/mnist_cnn.onnx
    python detect.py photo.png --model models/mnist_cnn.onnx

How it works: Keras 3 exports a TensorFlow SavedModel (model.export), tf2onnx
converts that directory, and the result is checked against the Keras model on
MNIST test images.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export the MNIST CNN to ONNX and check parity.")
    ap.add_argument("--model", default=str(Path("models") / "mnist_cnn.keras"))
    ap.add_argument("--out", default=str(Path("models") / "mnist_cnn.onnx"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--data", default=None, help="local mnist.npz for the parity check (default: Keras cache)")
    ap.add_argument("--samples", type=int, default=500)
    args = ap.parse_args(argv)

    if not Path(args.model).is_file():
        print(f"error: {args.model} not found (run train.py first)", file=sys.stderr)
        return 1
    try:
        import onnxruntime  # noqa: F401
        import tf2onnx  # noqa: F401
    except ImportError as exc:
        print(f"optional dependency missing ({exc.name}). Install with:\n  python -m pip install -r requirements-optional.txt", file=sys.stderr)
        return 3

    import keras

    model = keras.saving.load_model(args.model, compile=False)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        export_dir = Path(tmp) / "saved_model"
        model.export(str(export_dir), verbose=False)
        cmd = [sys.executable, "-m", "tf2onnx.convert", "--saved-model", str(export_dir), "--output", str(out), "--opset", str(args.opset)]
        print("running:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr[-3000:], file=sys.stderr)
            print("error: tf2onnx conversion failed", file=sys.stderr)
            return 1
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")

    from predictor import KerasPredictor, OnnxPredictor, compare_predictors, print_comparison
    from train import load_mnist

    (_, _), (x_test, y_test) = load_mnist(args.data)
    x = (x_test[: args.samples].astype("float32") / 255.0)[..., None]
    rows = compare_predictors(KerasPredictor(args.model), [OnnxPredictor(str(out))], x, y_test[: args.samples])
    print_comparison(rows)
    ok = rows[1]["argmax_agreement"] >= 0.99
    print("parity OK" if ok else "warning: ONNX predictions differ from Keras")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
