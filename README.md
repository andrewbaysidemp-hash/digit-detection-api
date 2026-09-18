# Offline handwritten-digit detection on CPU (TensorFlow + OpenCV)

[![CI](https://github.com/andrewbaysidemp-hash/digit-detection-api/actions/workflows/ci.yml/badge.svg)](https://github.com/andrewbaysidemp-hash/digit-detection-api/actions/workflows/ci.yml)
[![Container](https://github.com/andrewbaysidemp-hash/digit-detection-api/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/andrewbaysidemp-hash/digit-detection-api/pkgs/container/digit-detection-api)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/andrewbaysidemp-hash/digit-detection-api)

A complete, offline pipeline for Windows 11 that trains a small CNN on MNIST,
saves it, and reads multi-digit numbers out of photos or scans: OpenCV finds
and crops each digit, the CNN classifies the crops, and the digits are put back
together in reading order. Everything runs on the CPU. Python 3.10 to 3.13.
The same pipeline ships as an HTTP API in a container (see section 8b and
[DEPLOY.md](DEPLOY.md)); a prebuilt image is published by CI:

```
docker run --rm -p 7860:7860 ghcr.io/andrewbaysidemp-hash/digit-detection-api:latest
```

```
Num-Mnist/
  train.py                  train the CNN on MNIST, save models/mnist_cnn.keras + training_report.json
  detect.py                 CLI: image in -> number(s) out, annotated image, optional JSON and debug dump
  preprocess.py             OpenCV pipeline: load, threshold, clean, find/merge/split/order digits, 28x28 crops
  predictor.py              loads .keras / .tflite / .onnx models behind one predict() interface
  make_samples.py           builds test images with exact ground truth from real MNIST test digits
  export_tflite.py          optional: convert to TensorFlow Lite (LiteRT) and check parity + speed
  export_onnx.py            optional: convert to ONNX and check parity + speed with ONNX Runtime
  tests/test_pipeline.py    pytest: preprocessing unit tests + end-to-end accuracy per image category
  tests/test_api.py         pytest: HTTP API tests (FastAPI TestClient)
  requirements.txt          pinned core dependencies (CPU only)
  requirements-optional.txt onnxruntime, tf2onnx, onnx, ai-edge-litert
  server.py                 HTTP API (FastAPI): web page, JSON endpoint, annotated PNG, docs, health
  static/index.html         the upload page served at /
  Dockerfile, .dockerignore container for the API (ONNX Runtime, no TensorFlow, ~400 MB)
  requirements-server.txt   runtime dependencies of the container
  render.yaml, deploy/      platform configs + deploy/huggingface/deploy.py; DEPLOY.md explains the free hosting options
  .github/workflows/        CI (tests), container build + publish to GHCR, auto-deploy to a Hugging Face Space
  models/                   created by train.py (and the export scripts)
  samples/                  created by make_samples.py (54 images + ground_truth.json)
```

## 1. Installation on Windows 11

1. Install Python 3.10, 3.11, 3.12 or 3.13 from python.org or the Microsoft
   Store. TensorFlow 2.21 has no wheel for Python 3.14, so `pip install
   tensorflow` fails there with "No matching distribution found". This project
   was built and tested with Python 3.13; the code itself is Python 3.10
   compatible (checked with `vermin`).
2. Open PowerShell in the project folder and create a virtual environment.
   If several Pythons are installed, pick one explicitly with the `py`
   launcher:

   ```powershell
   py -3.13 -m venv venv          # or: py -3.10 -m venv venv
   .\venv\Scripts\Activate.ps1    # cmd.exe: venv\Scripts\activate.bat
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   If PowerShell refuses to run the activation script, run once:
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

   The download is about 350 MB (TensorFlow is the bulk of it) and takes
   2-10 minutes. Expect roughly 1.5 GB on disk (2 GB with the optional
   packages from section 8).
3. Verify:

   ```powershell
   python -c "import tensorflow as tf, cv2, numpy; print(tf.__version__, cv2.__version__, numpy.__version__, tf.config.list_physical_devices('GPU'))"
   ```

   Expected: `2.21.0 4.14.0 2.x.x []` (the empty list is correct: CPU only).
   The first TensorFlow import takes 5-10 s; log lines about oneDNN are
   informational.

Note on OneDrive: the project works inside a OneDrive folder, but OneDrive
tries to sync the 1 GB `venv` folder and the first TensorFlow import can then
take minutes instead of seconds. Either right-click `venv` -> "Free up
space"/exclude it from sync, or put the project in a non-synced folder such as
`C:\dev\Num-Mnist`.

Everything after installation runs offline. The only network access is the
one-time MNIST download (11 MB) by `train.py`; see `--data` below for a fully
offline alternative.

## 2. Quick start

```powershell
python train.py                       # ~6 min on a 4-core CPU, MNIST test accuracy 0.9947
python make_samples.py                # writes samples\*.png + samples\ground_truth.json
python detect.py samples\clean_00.png # prints the number, writes samples\clean_00_detected.png
python detect.py samples\multiline_00.png --verbose
python detect.py my_photo.jpg --debug debug_out --json
python -m pytest tests -q             # unit tests + end-to-end accuracy on the samples
```

What you should see:

```
> python detect.py samples\clean_00.png
31455

> python detect.py samples\multiline_00.png --verbose
818952
8232
258
  line 0  x=   99 y=   85 w=  60 h= 106  -> 8  conf 1.000
  line 0  x=  195 y=   87 w=  74 h= 115  -> 1  conf 0.982
  ...
  line 2  x=  414 y=  498 w= 116 h= 116  -> 8  conf 1.000
  backend keras | {'segmentation_ms': 195.3, 'model_load_ms': 6959.8, 'inference_ms': 11.4, 'digits': 13}
  annotated image: samples\multiline_00_detected.png
```

(`model_load_ms` includes importing TensorFlow, which is most of the 7 s;
the model itself loads in 0.7 s.)

Measured on the test machine (Windows 11, 4 CPU cores, no GPU):

| item | value |
| --- | --- |
| MNIST test accuracy | 0.9947 (test loss 0.016) |
| training time (6 epochs, 57,000 images) | 357 s, about 60 s per epoch |
| model size | 188,234 parameters; 2,256 KB `.keras`, 740 KB `.tflite`/`.onnx` |
| exact-number accuracy on the 54 generated samples | 52/54 = 96.3 % (266/268 = 99.25 % per digit) |
| segmentation, one image (500-1700 px wide) | 20-200 ms |
| inference per crop, Keras graph call | 0.18 ms batched, 2.9 ms as a single-image call |
| inference per crop, TensorFlow Lite | 0.26 ms batched, 0.45 ms single |
| inference per crop, ONNX Runtime | 0.23 ms batched, 0.24 ms single |

Per-category exact-number accuracy on the samples (6 images each):

| clean | noisy+JPEG | light on dark | multi-line | TrueType fonts | Hershey fonts | rotated 3-5 deg | underline | ruled paper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6/6 | 6/6 | 6/6 | 5/6 | 6/6 | 5/6 | 6/6 | 6/6 | 6/6 |

The two misses are single-digit confusions: an MNIST test "2" written like a
mirrored C (read as 7) and a bold Hershey "6" read as 8. Segmentation found
the correct number of digits and lines on all 54 images.

## 3. Training (`train.py`)

```
python train.py [--epochs 6] [--batch-size 128] [--lr 0.001] [--no-augment]
                [--data path\to\mnist.npz] [--out models\mnist_cnn.keras]
                [--seed 42] [--threads 0] [--limit N] [--val-split 0.05]
```

| layer | output | parameters |
| --- | --- | --- |
| Input `image` | 28x28x1 float32, 0..1, white digit on black | |
| Conv2D 32, 3x3, ReLU | 26x26x32 | 320 |
| MaxPooling 2x2 | 13x13x32 | |
| Conv2D 64, 3x3, ReLU | 11x11x64 | 18,496 |
| Conv2D 64, 3x3, ReLU | 9x9x64 | 36,928 |
| MaxPooling 2x2 | 4x4x64 | |
| Flatten + Dropout 0.4 | 1024 | |
| Dense 128, ReLU + Dropout 0.3 | 128 | 131,200 |
| Dense 10, softmax `probabilities` | 10 | 1,290 |

About 5.5 million multiply-adds per image, so a crop classifies in well under a
millisecond on a CPU. There is deliberately no BatchNormalization (see
"Common mistakes").

Training details:

- Data: `keras.datasets.mnist.load_data()`, cached in
  `%USERPROFILE%\.keras\datasets\mnist.npz` after the first run. For a machine
  with no internet, copy that file over and pass `--data C:\path\mnist.npz`.
- Augmentation: each epoch trains on a freshly warped copy of the training set
  (OpenCV `warpAffine`: rotation +-10 degrees, zoom +-10 %, shift +-2 px,
  black fill). This closes most of the gap between clean MNIST tiles and crops
  segmented from photos. It costs about one second per epoch. `--no-augment`
  disables it.
- Optimizer: Adam, learning rate 0.001 for three epochs, then halved every
  epoch. 5 % of the training set is held out for validation.
- Output: `models/mnist_cnn.keras` (Keras 3 native format) and
  `models/training_report.json` (accuracy, timings, per-epoch history). The
  script reloads the saved file and checks it predicts (N, 10) probabilities.
- `--limit 3000 --epochs 1` is a 30-second smoke test of the whole script.
- `--threads N` pins TensorFlow's intra-op threads (default: automatic).

## 4. Detection (`detect.py`)

```
python detect.py IMAGE [--model models\mnist_cnn.keras] [--min-conf 0.5]
                 [--out annotated.png] [--debug DIR] [--json] [--verbose]
                 [preprocessing options, see --help]
```

- Prints one line of text per detected text line, digits in reading order. A
  digit whose softmax confidence is below `--min-conf` is printed as `?` (it is
  never silently dropped).
- Always writes an annotated copy: `<image>_detected.png` next to the input, or
  `--out PATH`, or inside `--debug DIR`. Green box = accepted digit with its
  label and confidence, orange = below `--min-conf`, thin red (only with
  `--debug`) = rejected blob.
- `--json` prints instead:

  ```json
  {
    "image": "samples/clean_00.png", "model": "models/mnist_cnn.keras", "backend": "keras",
    "text": "4859",
    "lines": [{"line": 0, "text": "4859",
               "digits": [{"x": 61, "y": 55, "w": 70, "h": 91, "label": 4, "confidence": 0.9998, "accepted": true}, "..."]}],
    "rejected": [{"x": 0, "y": 0, "w": 0, "h": 0, "reason": "speck"}],
    "annotated": "samples/clean_00_detected.png",
    "timing_ms": {"segmentation_ms": 41.0, "model_load_ms": 6959.8, "inference_ms": 8.1, "digits": 5}
  }
  ```

- `--debug DIR` writes `01_gray.png`, `02_binary.png` (ink mask after line
  removal), `03_boxes.png` (kept boxes in green with reading-order index and
  line number, rejected blobs in red with the reason) and every model input
  as `crop_NN_lineL.png` (28x28 enlarged 8x). Look at these first whenever a
  result is wrong: they tell you immediately whether segmentation or
  classification failed.
- Exit codes: 0 digits found, 2 no digits found, 1 error (missing/undecodable
  image, missing model).
- `--model models\mnist_cnn.tflite` or `.onnx` switches back-end (section 8).

## 5. The preprocessing pipeline, step by step

All of this is in `preprocess.py`; every number below is a field of
`PreprocessParams` and most are exposed as `detect.py` flags.

1. **Load** (`load_image`). `np.fromfile` + `cv2.imdecode` instead of
   `cv2.imread`, because `imread`/`imwrite` silently fail on Windows paths with
   non-ASCII characters (OneDrive folders, user names, `ü`, `图`). Transparent
   PNGs are flattened onto white, 16-bit images are scaled to 8-bit.
2. **Grayscale** (`to_gray`). Colour carries no digit information; one channel
   is 3x less work.
3. **Downscale** (`downscale`, `max_side=1600`). Phone photos are 3000-4000 px
   wide; nothing below is needed at that resolution, and capping the size makes
   every later threshold behave the same at any input resolution. Boxes are
   scaled back to original coordinates at the end.
4. **Denoise** (Gaussian blur, `blur_ksize=5`). Removes sensor noise and JPEG
   blocking so they do not become thousands of tiny contours.
5. **Threshold with automatic polarity** (`to_binary`). Adaptive Gaussian
   threshold, offset `adaptive_c=10`. Adaptive thresholding compares each
   pixel with its local neighbourhood, so a shadow across the page or a bright
   lamp on one side does not matter. Polarity: whichever side of the Otsu
   threshold holds the majority of pixels is the background, so dark-on-light
   paper and light-on-dark boards both come out as ink = 255 on 0. The window
   size is chosen in two passes (`adaptive_block=0` means auto): a first pass
   with window = shorter side / 8 finds the digits, and if 2.5x their height
   is larger than that window, the image is re-thresholded with the larger
   window. Without this, a window smaller than about twice the digit height
   hollows out thick strokes (the centre of a fat stroke is darker than a
   local mean that is mostly ink) and a bold 1 turns into an outline the model
   reads as 8. If the result is more than half ink (textured background), it
   falls back to a global Otsu threshold; `--otsu` forces that, `--block N`
   fixes the window.
6. **Ruled-line and underline removal** (`remove_horizontal_lines`). Long
   horizontal runs of ink (longer than 2.5 digit heights) are detected with a
   morphological opening and erased; a vertical closing restricted to that band
   re-connects digit strokes that crossed the line. Lines are detected on a
   separate mask thresholded against the paper level (`line_contrast=25`),
   because the adaptive mask breaks a faint line right next to dark digits.
   `--keep-lines` disables the step.
7. **Closing** (3x3 ellipse, `close_ksize=3`). Bridges one-pixel breaks in
   strokes (pen skips, aliasing) so a digit stays one blob.
8. **Contours -> components** (`find_components`). `cv2.findContours` with
   `RETR_EXTERNAL`: only outer contours, so the holes of 0, 6, 8, 9 do not
   become extra "digits". Each component keeps its own pixel mask (filled
   contour AND ink), so the holes are preserved in the crop.
9. **Filtering** (`_prefilter`, `_reject_lines`, `_reject_small`). Rejected
   with a reason code you can see in the debug image: `speck` (< 8 px tall or
   < 40 px area), `border` (wider/taller than 90 % of the image: page edges,
   scanner lid), `line` (aspect > 4, flat and longer than a digit), `vline`
   (tall thin margin lines), `tall` (> 3 digit heights), `small` (shorter
   than 35 % of the digit height after merging: dots, commas, dashes, tails).
   The reference digit height `H_ref` is the ink-weighted median height of the
   digit-shaped blobs, so a digit that shattered into pieces does not drag the
   estimate down.
10. **Fragment merging** (`merge_fragments`). Re-attaches the detached top bar
    of a 5 or 7, a stroke broken by a pen skip or by a removed ruled line. Two
    blobs merge when one is shorter than 85 % of `H_ref`, they overlap
    horizontally by at least 25 % of the narrower one (or the short one is
    right next to the other), their vertical gap is at most 60 % of `H_ref`,
    the union is still digit-shaped (at most 1.35 `H_ref` tall, aspect at most
    1.5), and a flat dash-like piece sits in the upper half of its partner
    (below a digit it is an underline, not part of it). Each blob joins only
    its best partner, so a fragment cannot chain two neighbouring digits.
11. **Touching digits** (`split_touching`). A blob wider than 1.5x its height
    cannot be one digit; it is cut at the column with the least ink near each
    expected boundary (digit count estimated from width / 0.8 height). Between
    aspect 1.25 and 1.5 (a wide 0 looks like this) it is only cut where a real
    neck exists (a column with at most 35 % of the median column ink).
    `--no-split` disables it.
12. **Reading order** (`order_boxes`). Boxes are grouped into text lines by
    vertical overlap (>= 50 %), lines sorted top to bottom, digits within a
    line left to right. The `line` index is in the JSON output.
13. **MNIST-style normalisation** (`crop_to_mnist`). This is the step people
    skip and then wonder why a 99 % model reads everything as 8. MNIST digits
    were made by fitting the glyph into a 20x20 box preserving aspect ratio,
    anti-aliasing it, and placing it in a 28x28 field so that its centre of
    mass is at the centre. The crop is treated the same way: (a) the component's
    ink is taken as grayscale darkness relative to the local paper level
    (`soft_component`, normalised so the stroke core is white) rather than as a
    hard 0/1 mask, which keeps MNIST-like soft edges (`--binary-crops` turns
    this off); (b) stroke width is measured with a distance transform and thin
    pen strokes are dilated (very fat marker strokes eroded) to about 12 % of
    the digit size, like MNIST's 2-3 px of 20 (`target_stroke=0.12`); (c) the
    tight crop is resized with `INTER_AREA` so its longer side is 20 px; (d) it
    is placed on a black 28x28 canvas and shifted so that its centre of mass is
    at (13.5, 13.5); (e) converted to float32 in 0..1. The result is
    white-on-black, exactly what the network was trained on.
14. **Classification** (`detect.py`). All crops of an image go through the
    model in one batch (`model(x, training=False)`; `model.predict()` has
    ~0.5 s of overhead per call and is avoided). Labels and confidences are
    joined per line.

## 6. Common mistakes and fixes

1. **Everything is read as 8 (or 1, or 0) although MNIST accuracy is 99 %.**
   The crop does not look like MNIST: black digit on white, or a digit that
   fills the whole 28x28 tile with no margin. The model's input must be white
   ink on black, fitted into 20x20 and centred by mass. `crop_to_mnist` does
   this; if you write your own crop code, do the same.
2. **Digits detected but wrong label on thin pen strokes.** After downscaling
   a 1-2 px stroke becomes a faint grey smear. Thicken strokes before resizing
   (the `target_stroke` step) or write thicker.
3. **The whole image is one giant box, or nothing is found.** Polarity was
   guessed wrong or the page border is in the picture. Check `02_binary.png`:
   ink must be white on black. Crop the photo to the paper, or force
   `--otsu`.
4. **Holes of 0, 6, 8, 9 come out as extra digits.** You used
   `RETR_LIST`/`RETR_TREE` or `RETR_CCOMP`. Use `RETR_EXTERNAL`.
5. **`cv2.findContours` "too many values to unpack".** OpenCV 3 returned three
   values, OpenCV 4 returns two: `contours, hierarchy = cv2.findContours(...)`.
   `requirements.txt` pins OpenCV 4.x for this reason.
6. **`cv2.imread` returns `None` / `cv2.imwrite` writes nothing** on a path
   with accents, CJK characters or some OneDrive folders. Use
   `np.fromfile` + `cv2.imdecode` and `cv2.imencode(...).tofile()` (see
   `load_image` / `save_image`).
7. **A 5 or 7 loses its top bar and is read as 3, 1 or 6.** The bar is a
   separate contour. Merge fragments (step 10) instead of taking raw contours.
8. **Two touching digits become one wrong digit.** Split wide blobs (step 11)
   or ask people to leave a gap; the width heuristic cannot rescue heavily
   overlapping digits.
9. **Digits ordered wrongly / two lines interleaved.** Sorting by `x` only is
   wrong for several lines. Group into lines first by vertical overlap, then
   sort by `x` (step 12).
10. **Ruled paper: every digit is cut in half.** Remove long horizontal lines
    and repair the strokes (step 6); with `--keep-lines` you will see the
    failure.
11. **Underline or a stray dash counted as a digit.** Reject blobs that are
    much shorter than the digit height (`min_rel_height`) and flat blobs
    longer than a digit (`line`).
12. **Adaptive threshold hollows out thick strokes** (outlines instead of
    digits; a bold 1 is read as 8). The window is smaller than about twice
    the digit height. The auto mode re-thresholds with a window scaled to the
    digits (this single fix took the sample accuracy from 89 % to 96 %); if
    you set `--block` yourself, make it at least 2.5x the digit height, or use
    `--otsu`.
13. **Model trains fine but predicts one class for everything at inference.**
    In this environment (Keras 3.15.1 / TensorFlow 2.21 CPU) models with
    `BatchNormalization` did exactly that: 96 % training accuracy, ~11 % test
    accuracy in inference mode, moving statistics looked normal, and disabling
    oneDNN did not help. The shipped architecture has no BatchNorm; if you add
    it and see chance-level test accuracy, that is why.
14. **Training seems frozen / no epoch output when logging to a file.** Python
    buffers stdout when it is piped. `train.py` switches to line buffering; or
    run `python -u train.py`.
15. **First inference takes a second, the rest are instant.** Model loading
    and the first graph trace are one-off costs. Keep one process alive for
    many images, or use the TFLite/ONNX back-ends whose start-up is ~10x
    cheaper.
16. **`model.predict()` in a loop over single crops is slow.** Batch all crops
    of an image into one call, and use the direct call `model(x,
    training=False)` for small batches.
17. **`pip install tensorflow` fails with "No matching distribution".** The
    interpreter is Python 3.14 (no wheel yet) or 32-bit. Use 3.10-3.13, 64-bit.
18. **Accuracy drops after retraining with `--no-augment`.** Clean MNIST alone
    does not cover the rotation/scale variation of segmented crops; keep
    augmentation on.

## 7. Tips to improve accuracy on real-world images

1. Photograph the paper flat, filling the frame, with the phone parallel to
   it; perspective distortion changes digit shapes more than any other factor.
2. Use even light and avoid your own shadow. Adaptive thresholding tolerates
   gradients, not hard shadow edges cutting through digits.
3. Write with a medium pen, clearly separated digits, height at least 40 px in
   the photo. Very thin pencil needs `--target-stroke 0.15`; very fat marker
   needs `--otsu` or a larger `--block`.
4. Crop away everything that is not the number (table edge, fingers, other
   text). Fewer blobs means fewer chances for a wrong reference height.
5. Look at `--debug` output before tuning anything: if `03_boxes.png` is right
   and the label is wrong, the fix is on the model side; if the boxes are
   wrong, it is a preprocessing parameter.
6. Print the confidences (`--verbose`). Real digits score > 0.95; a 0.5-0.7
   usually means a bad crop (merged fragment, cut-off stroke), not an
   ambiguous digit.
7. Fine-tune the model on your own digits: crop 50-200 examples with
   `--debug`, label them, and continue training for a few epochs with a low
   learning rate. Nothing beats in-domain data.
8. Train longer or with stronger augmentation (`--epochs 12`, or widen the
   angle/zoom ranges in `augment_images`) when your writers slant or vary size.
9. Add a "not a digit" rejection: MNIST models are forced to output one of ten
   classes even for a smudge. The `--min-conf` threshold is the simple version;
   an 11th "background" class trained on non-digit crops is the robust one.
10. Deskew before segmentation if lines are rotated more than ~5 degrees
    (`cv2.minAreaRect` on all ink pixels, then `cv2.warpAffine`), so line
    grouping and top/bottom heuristics hold.
11. Use test-time augmentation for hard crops: classify the crop plus two
    slightly rotated copies and average the probabilities.
12. For printed digits, the Hershey and TrueType sample categories show the
    model handles fonts well, but a font-only fine-tune (render digits with
    Pillow in many fonts) pushes it to near 100 %.

## 8. Optional: faster CPU inference with TensorFlow Lite or ONNX Runtime

The Keras model is already fast per digit; what the alternative runtimes buy
you is start-up time (TensorFlow import + model load is 5-10 s, ONNX Runtime
or LiteRT start in well under a second) and a much smaller dependency for
deployment.

```powershell
python -m pip install -r requirements-optional.txt   # onnxruntime, tf2onnx, onnx, ai-edge-litert
python export_tflite.py      # -> models\mnist_cnn.tflite, prints parity + timing vs Keras
python export_onnx.py        # -> models\mnist_cnn.onnx,   prints parity + timing vs Keras
python detect.py samples\clean_00.png --model models\mnist_cnn.tflite
python detect.py samples\clean_00.png --model models\mnist_cnn.onnx
```

Measured on the test machine (same 500 MNIST test images):

| back-end | start-up (import + load) | per crop, batch of 500 | per crop, single call | max prob. diff vs Keras | argmax agreement |
| --- | --- | --- | --- | --- | --- |
| Keras `.keras` (graph call) | ~7 s (TensorFlow import 6 s + 0.7 s) | 0.18 ms | 2.9 ms | 0 | 100 % |
| TensorFlow Lite `.tflite` (ai-edge-litert) | 0.03 s + package import | 0.26 ms | 0.45 ms | 5e-7 | 100 % |
| ONNX Runtime `.onnx` | 0.2 s + package import | 0.23 ms | 0.24 ms | 4e-7 | 100 % |

All three give 100 % on those 500 images. `detect.py --model models\mnist_cnn.onnx`
still imports TensorFlow only if you use the `.keras` model; with `.tflite`
or `.onnx` the start-up cost is the package import alone.

Notes:

- `export_tflite.py` uses `tf.lite.TFLiteConverter.from_keras_model`, which is
  still built into TensorFlow 2.21. For inference `predictor.py` prefers the
  `ai_edge_litert` interpreter (the maintained LiteRT package) and falls back
  to `tf.lite.Interpreter`, which prints a deprecation warning.
- `export_onnx.py` calls `model.export()` (Keras 3 SavedModel) and then
  `python -m tf2onnx.convert --saved-model ... --opset 17`. tf2onnx 1.17 works
  with TensorFlow 2.21 / Keras 3 for this model; the input keeps the name
  `image` and the batch dimension is dynamic.
- `detect.py` and `predictor.load_predictor` choose the back-end from the file
  extension. Whatever back-end you use, the preprocessing is identical.
- Post-training int8 quantisation is possible with the TFLite converter (add a
  representative dataset); at this model size it saves memory, not time, on a
  desktop CPU.

## 8b. HTTP API and free deployment

`server.py` wraps the same pipeline in a FastAPI service: the upload page at
`/`, `POST /api/v1/detect` (JSON), `POST /api/v1/detect/annotated` (PNG),
`/docs` and `/healthz`, all on one origin. It uses the ONNX model, so the
container has no TensorFlow, starts in under a second and needs less than
300 MB of RAM.

```powershell
python -m pip install fastapi "uvicorn[standard]" python-multipart
uvicorn server:app --host 0.0.0.0 --port 7860        # http://localhost:7860
curl.exe -X POST "http://localhost:7860/api/v1/detect" -F "image=@samples\clean_00.png"
python -m pytest tests\test_api.py -q
```

[DEPLOY.md](DEPLOY.md) walks through hosting it for free: Render (no card,
custom domain, one-click button above), Koyeb (no card), Google Cloud Run
(autoscaling within the free tier, needs a billing account) and Hugging Face
Spaces (Docker Spaces now need a PRO subscription), plus API usage examples,
the `API_KEY` option and capacity notes. CI publishes a ready-to-run image to
`ghcr.io/andrewbaysidemp-hash/digit-detection-api:latest`.

## 9. Known limitations

- Digits that overlap heavily (not just touch) are not separated.
- A digit broken into two full-height halves side by side (a 0 drawn as two
  arcs) is read as two 1s.
- Text lines rotated by more than about 5 degrees may be split or merged
  incorrectly; deskew first.
- Two very different digit sizes in one image (a title and body text) bias the
  reference height; process them separately.
- The model knows only digits: letters, symbols and smudges get a digit label
  with (usually) low confidence. Use `--min-conf` or add a rejection class.
- Ruled-line removal handles horizontal lines only, and a stroke that runs
  along the line (the top arc of a 9 sitting exactly on a rule) is erased with
  it; only strokes that cross the line are repaired.

## 10. Troubleshooting installation

- **Python 3.14**: `No matching distribution found for tensorflow`. Install
  3.13 (or 3.10-3.12) and recreate the venv with `py -3.13 -m venv venv`.
- **`DLL load failed while importing _pywrap_tensorflow_internal`**: install
  the "Microsoft Visual C++ Redistributable 2015-2022 x64", and make sure the
  Python is 64-bit.
- **Long path errors while installing**: enable long paths (`gpedit` ->
  Enable Win32 long paths, or registry `LongPathsEnabled=1`) or move the
  project to a short path such as `C:\dev`.
- **protobuf / numpy version conflicts**: create a fresh venv and install only
  from `requirements.txt`; do not mix in `tensorflow-cpu`, `tf-keras` or old
  `protobuf` pins. On Python 3.10 pip resolves NumPy 2.2.x, on 3.13 NumPy
  2.5.x; both are fine.
- **Very slow TensorFlow import (minutes)**: antivirus scanning or OneDrive
  syncing the venv. Exclude the venv folder from both.
- **`ImportError: DLL load failed` for OpenCV**: same VC++ redistributable;
  also uninstall `opencv-python-headless` if both are present.
- **`pytest` not found**: run it as `python -m pytest`.
