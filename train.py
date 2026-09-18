"""Train a small CNN on MNIST on the CPU and save it as models/mnist_cnn.keras.

Usage:
    python train.py                          # 8 epochs, a few minutes on a 4-core CPU
    python train.py --epochs 3               # quicker, slightly lower accuracy
    python train.py --data path/to/mnist.npz # fully offline with a local copy of the dataset
    python train.py --limit 5000 --epochs 1  # 20-second smoke test

Outputs:
    models/mnist_cnn.keras          the trained model (Keras 3 format)
    models/training_report.json     accuracy, timing, parameter count
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_OUT = Path("models") / "mnist_cnn.keras"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Train the MNIST digit CNN (CPU only).")
    ap.add_argument("--epochs", type=int, default=6, help="training epochs (default 6, about 75 s each on a 4-core CPU)")
    ap.add_argument("--batch-size", type=int, default=128, help="mini-batch size (default 128)")
    ap.add_argument("--lr", type=float, default=1e-3, help="initial Adam learning rate (default 0.001)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help=f"where to save the model (default {DEFAULT_OUT})")
    ap.add_argument("--data", default=None, help="path to a local mnist.npz for offline training (default: Keras cache, downloaded once)")
    ap.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    ap.add_argument("--no-augment", action="store_true", help="disable rotation/zoom/shift augmentation")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for TensorFlow (0 = automatic)")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N training images (smoke tests)")
    ap.add_argument("--val-split", type=float, default=0.05, help="fraction of training data held out for validation (default 0.05)")
    return ap.parse_args(argv)


def configure_tf(threads: int, seed: int):
    """Import TensorFlow/Keras, set threads BEFORE any op runs, fix the seeds."""
    import tensorflow as tf
    import keras

    if threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(max(1, min(2, threads)))
    keras.utils.set_random_seed(seed)
    return tf, keras


def load_mnist(path=None):
    """Return ((x_train, y_train), (x_test, y_test)) as uint8 arrays."""
    if path:
        with np.load(path) as d:
            return (d["x_train"], d["y_train"]), (d["x_test"], d["y_test"])
    from keras.datasets import mnist

    return mnist.load_data()


def build_model(keras):
    """~188k parameters, ~5.5M multiply-adds per image: fast on CPU, >99% on MNIST.

    Deliberately no BatchNormalization: on this Windows/TF 2.21 CPU build the
    BatchNorm inference path (moving statistics) produced a constant class for
    every input even though training-mode accuracy was fine. Plain conv + ReLU
    + dropout has no train/inference discrepancy and converts cleanly to
    TFLite and ONNX."""
    L = keras.layers
    inputs = keras.Input(shape=(28, 28, 1), name="image")
    x = L.Conv2D(32, 3, activation="relu", name="conv1")(inputs)  # 26x26x32
    x = L.MaxPooling2D(2, name="pool1")(x)                        # 13x13x32
    x = L.Conv2D(64, 3, activation="relu", name="conv2")(x)       # 11x11x64
    x = L.Conv2D(64, 3, activation="relu", name="conv3")(x)       # 9x9x64
    x = L.MaxPooling2D(2, name="pool2")(x)                        # 4x4x64
    x = L.Flatten(name="flatten")(x)
    x = L.Dropout(0.4, name="drop1")(x)
    x = L.Dense(128, activation="relu", name="fc1")(x)
    x = L.Dropout(0.3, name="drop2")(x)
    outputs = L.Dense(10, activation="softmax", name="probabilities")(x)
    return keras.Model(inputs, outputs, name="mnist_cnn")


def augment_images(x: np.ndarray, rng: np.random.Generator, max_angle: float = 10.0, max_zoom: float = 0.10, max_shift: float = 2.0) -> np.ndarray:
    """Random rotation (+-max_angle degrees), zoom (+-max_zoom) and shift
    (+-max_shift px) applied with OpenCV, one affine warp per image, black fill
    so the background stays 0 like MNIST.

    This mimics how crops segmented from photos differ from clean MNIST tiles.
    It is done in NumPy once per epoch (about 1-2 s for 57k images) because
    Keras' random image layers more than doubled the CPU step time in our
    measurements. A richer variant (random binarisation, dilation and erosion
    of the tiles) was tried and did not improve accuracy on segmented crops."""
    import cv2

    n = len(x)
    out = np.empty_like(x)
    angles = rng.uniform(-max_angle, max_angle, n)
    scales = rng.uniform(1.0 - max_zoom, 1.0 + max_zoom, n)
    shifts = rng.uniform(-max_shift, max_shift, (n, 2))
    for i in range(n):
        m = cv2.getRotationMatrix2D((13.5, 13.5), float(angles[i]), float(scales[i]))
        m[:, 2] += shifts[i]
        out[i, :, :, 0] = cv2.warpAffine(x[i, :, :, 0], m, (28, 28), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        sys.stdout.reconfigure(line_buffering=True)  # epoch lines show up immediately even when piped to a log
    except (AttributeError, ValueError):
        pass
    tf, keras = configure_tf(args.threads, args.seed)
    print(f"TensorFlow {tf.__version__} | Keras {keras.__version__} | CPUs {os.cpu_count()} | GPUs {len(tf.config.list_physical_devices('GPU'))}")

    (x_train, y_train), (x_test, y_test) = load_mnist(args.data)
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test = (x_test.astype("float32") / 255.0)[..., None]

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(x_train))
    x_train, y_train = x_train[perm], y_train[perm]
    if args.limit > 0:
        x_train, y_train = x_train[: args.limit], y_train[: args.limit]
    n_val = int(len(x_train) * args.val_split) if args.val_split > 0 else 0
    x_val, y_val = x_train[:n_val], y_train[:n_val]
    x_tr, y_tr = x_train[n_val:], y_train[n_val:]
    print(f"train {len(x_tr)} | val {len(x_val)} | test {len(x_test)} | augment {not args.no_augment}")

    model = build_model(keras)
    model.compile(optimizer=keras.optimizers.Adam(args.lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # One fit() call per epoch so that a fresh OpenCV-augmented copy of the
    # training set is used each time. Learning rate: constant for the first
    # three epochs, then halved every epoch.
    history: dict[str, list[float]] = {}
    t0 = time.time()
    for epoch in range(args.epochs):
        lr = args.lr * (0.5 ** max(0, epoch - 2))
        model.optimizer.learning_rate.assign(lr)
        x_epoch = x_tr if args.no_augment else augment_images(x_tr, rng)
        print(f"Epoch {epoch + 1}/{args.epochs} (lr {lr:.1e})")
        h = model.fit(
            x_epoch, y_tr,
            validation_data=(x_val, y_val) if n_val else None,
            epochs=1, batch_size=args.batch_size, verbose=2,
        )
        for key, values in h.history.items():
            history.setdefault(key, []).extend(float(v) for v in values)
    train_seconds = time.time() - t0

    test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=512, verbose=0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    reloaded = keras.saving.load_model(str(out), compile=False)
    probs = np.asarray(reloaded(x_test[:16], training=False))
    if probs.shape != (16, 10) or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4):
        print("error: reloaded model does not produce (N, 10) probabilities", file=sys.stderr)
        return 1

    report = {
        "model_path": str(out),
        "test_accuracy": round(float(test_acc), 5),
        "test_loss": round(float(test_loss), 5),
        "val_accuracy": round(float(history.get("val_accuracy", [float("nan")])[-1]), 5),
        "history": history,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "augmentation": not args.no_augment,
        "train_images": int(len(x_tr)),
        "train_seconds": round(train_seconds, 1),
        "seconds_per_epoch": round(train_seconds / max(1, args.epochs), 1),
        "params": int(model.count_params()),
        "tf_version": tf.__version__,
        "keras_version": keras.__version__,
        "seed": args.seed,
    }
    with open(out.parent / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"saved {out} ({out.stat().st_size / 1024:.0f} KB); test accuracy {test_acc:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
