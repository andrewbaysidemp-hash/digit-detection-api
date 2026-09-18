"""Export the trained Keras model to TensorFlow Lite (LiteRT) and verify it.

Usage:
    python export_tflite.py                       # models/mnist_cnn.keras -> models/mnist_cnn.tflite
    python export_tflite.py --samples 1000        # compare on more MNIST test images
    python detect.py photo.png --model models/mnist_cnn.tflite

The converter is built into TensorFlow. For inference, predictor.py uses the
ai_edge_litert package when installed (pip install -r requirements-optional.txt)
and otherwise falls back to the deprecated tf.lite.Interpreter.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export the MNIST CNN to TensorFlow Lite and check parity.")
    ap.add_argument("--model", default=str(Path("models") / "mnist_cnn.keras"))
    ap.add_argument("--out", default=str(Path("models") / "mnist_cnn.tflite"))
    ap.add_argument("--data", default=None, help="local mnist.npz for the parity check (default: Keras cache)")
    ap.add_argument("--samples", type=int, default=500, help="MNIST test images used for the parity check")
    args = ap.parse_args(argv)

    if not Path(args.model).is_file():
        print(f"error: {args.model} not found (run train.py first)", file=sys.stderr)
        return 1
    import tensorflow as tf
    import keras

    model = keras.saving.load_model(args.model, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_bytes = converter.convert()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(tflite_bytes)
    print(f"wrote {out} ({len(tflite_bytes) / 1024:.0f} KB, float32)")

    from predictor import KerasPredictor, TFLitePredictor, compare_predictors, print_comparison
    from train import load_mnist

    (_, _), (x_test, y_test) = load_mnist(args.data)
    x = (x_test[: args.samples].astype("float32") / 255.0)[..., None]
    rows = compare_predictors(KerasPredictor(args.model), [TFLitePredictor(str(out))], x, y_test[: args.samples])
    print_comparison(rows)
    ok = rows[1]["argmax_agreement"] >= 0.99
    print("parity OK" if ok else "warning: TFLite predictions differ from Keras")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
