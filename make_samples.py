"""Generate synthetic test images with exact ground truth.

Real MNIST *test* digits are pasted onto a paper-like background (so the true
label of every digit is known), plus font-rendered numbers. Categories:

    clean          dark digits on light paper, varied size and spacing
    noisy          clean + sensor noise + blur + JPEG compression
    light_on_dark  light digits on a dark background
    multiline      2-3 lines of digits
    font_ttf       Windows TrueType fonts via Pillow (skipped if fonts are missing)
    font_hershey   OpenCV Hershey fonts
    rotated        clean, rotated by 3-5 degrees
    underline      clean with a thick underline below the number
    ruled          clean on ruled paper (lines may cross the digits)

Usage:
    python make_samples.py                 # writes samples/ and samples/ground_truth.json
    python make_samples.py --per-category 10 --seed 1
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from preprocess import save_image

CATEGORIES = ["clean", "noisy", "light_on_dark", "multiline", "font_ttf", "font_hershey", "rotated", "underline", "ruled"]
TTF_CANDIDATES = ["arial.ttf", "segoeui.ttf", "comic.ttf", "calibri.ttf", "verdana.ttf", "times.ttf"]


def load_mnist_test(path: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    candidates = [Path(path)] if path else [Path.home() / ".keras" / "datasets" / "mnist.npz"]
    for c in candidates:
        if c.is_file():
            with np.load(str(c)) as d:
                return d["x_test"], d["y_test"]
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from keras.datasets import mnist

    (_, _), (x_test, y_test) = mnist.load_data()
    return x_test, y_test


def digit_alpha(tile: np.ndarray, height: int) -> np.ndarray:
    """Trim a 28x28 MNIST tile to its ink and scale it to the given height. Returns float32 alpha 0..1."""
    ys, xs = np.nonzero(tile > 30)
    crop = tile[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = height / float(crop.shape[0])
    width = max(2, int(round(crop.shape[1] * s)))
    big = cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.clip(big.astype(np.float32) / 255.0, 0.0, 1.0)


def paste(canvas: np.ndarray, alpha: np.ndarray, x: int, y: int, ink: tuple[int, int, int]) -> None:
    h, w = alpha.shape
    region = canvas[y:y + h, x:x + w].astype(np.float32)
    a = alpha[:, :, None]
    ink_arr = np.array(ink, np.float32)[None, None, :]
    canvas[y:y + h, x:x + w] = np.clip(region * (1 - a) + ink_arr * a, 0, 255).astype(np.uint8)


def make_paper(h: int, w: int, color: tuple[int, int, int], sigma: float, rng: np.random.Generator) -> np.ndarray:
    paper = np.empty((h, w, 3), np.float32)
    paper[:] = np.array(color, np.float32)
    if sigma > 0:
        paper += rng.normal(0, sigma, (h, w, 1)).astype(np.float32)
    return np.clip(paper, 0, 255).astype(np.uint8)


def pick_digits(x_test, y_test, n: int, rng) -> tuple[list[np.ndarray], str]:
    idx = rng.integers(0, len(x_test), size=n)
    return [x_test[i] for i in idx], "".join(str(int(y_test[i])) for i in idx)


def compose_lines(x_test, y_test, rng, n_lines: int, paper, ink, noise_sigma: float) -> tuple[np.ndarray, list[str]]:
    """Paste 1..n_lines lines of MNIST digits. Returns (image, labels per line)."""
    height = int(rng.integers(60, 150))
    margin = int(height * 0.6)
    line_gap = int(height * rng.uniform(0.5, 0.9))
    rows = []
    for _ in range(n_lines):
        tiles, label = pick_digits(x_test, y_test, int(rng.integers(3, 7)), rng)
        alphas = [digit_alpha(t, int(height * rng.uniform(0.85, 1.15))) for t in tiles]
        gaps = [int(height * rng.uniform(0.2, 0.6)) for _ in alphas]
        rows.append((alphas, gaps, label))
    width = max(sum(a.shape[1] for a, _, _ in [(x, 0, 0) for x in r[0]]) + sum(r[1]) for r in rows) + 2 * margin
    total_h = 2 * margin + n_lines * int(height * 1.15) + (n_lines - 1) * line_gap
    img = make_paper(total_h, width, paper, noise_sigma, rng)
    labels = []
    y = margin
    for alphas, gaps, label in rows:
        x = margin + int(rng.integers(0, max(1, int(height * 0.5)))) if n_lines > 1 else margin
        base = y + int(height * 1.15)
        for a, g in zip(alphas, gaps):
            jitter = int(rng.integers(-int(height * 0.08), int(height * 0.08) + 1))
            top = max(0, min(img.shape[0] - a.shape[0], base - a.shape[0] + jitter))
            if x + a.shape[1] > img.shape[1]:
                break
            paste(img, a, x, top, ink)
            x += a.shape[1] + g
        labels.append(label)
        y = base + line_gap
    return img, labels


def add_noise_blur_jpeg(img: np.ndarray, rng) -> np.ndarray:
    noisy = img.astype(np.float32) + rng.normal(0, 18, img.shape).astype(np.float32)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    noisy = cv2.GaussianBlur(noisy, (3, 3), 0)
    ok, buf = cv2.imencode(".jpg", noisy, [cv2.IMWRITE_JPEG_QUALITY, 35])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def rotate(img: np.ndarray, degrees: float, fill: tuple[int, int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


def render_ttf(text: str, rng, paper, ink) -> np.ndarray | None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None
    fonts_dir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    available = [fonts_dir / f for f in TTF_CANDIDATES if (fonts_dir / f).is_file()]
    if not available:
        return None
    font_path = available[int(rng.integers(0, len(available)))]
    size = int(rng.integers(90, 150))
    font = ImageFont.truetype(str(font_path), size)
    left, top, right, bottom = font.getbbox(text)
    margin = size // 2
    w, h = right - left + 2 * margin, bottom - top + 2 * margin
    pil = Image.new("RGB", (w, h), (paper[2], paper[1], paper[0]))
    ImageDraw.Draw(pil).text((margin - left, margin - top), text, font=font, fill=(ink[2], ink[1], ink[0]))
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    noise = rng.normal(0, 4, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def render_hershey(text: str, rng, paper, ink) -> np.ndarray:
    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX]
    font = fonts[int(rng.integers(0, len(fonts)))]
    scale = float(rng.uniform(3.0, 5.0))
    thickness = int(rng.integers(4, 11))
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    margin = th // 2
    img = make_paper(th + baseline + 2 * margin, tw + 2 * margin, paper, 4, rng)
    cv2.putText(img, text, (margin, margin + th), font, scale, ink, thickness, cv2.LINE_AA)
    return img


def draw_underline(img: np.ndarray, ink, rng) -> None:
    h, w = img.shape[:2]
    y = int(h * 0.82)
    thickness = int(rng.integers(3, 7))
    cv2.line(img, (int(w * 0.08), y), (int(w * 0.92), y + int(rng.integers(-3, 4))), ink, thickness, cv2.LINE_AA)


def draw_rules(img: np.ndarray, spacing: int, rng) -> None:
    h, w = img.shape[:2]
    y = int(rng.integers(0, spacing))
    while y < h:
        cv2.line(img, (0, y), (w, y), (200, 150, 120), 2, cv2.LINE_AA)
        y += spacing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic digit images with ground truth.")
    ap.add_argument("--out", default="samples", help="output directory (default samples)")
    ap.add_argument("--per-category", type=int, default=6, help="images per category (default 6)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mnist", default=None, help="path to mnist.npz (default: Keras cache)")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    x_test, y_test = load_mnist_test(args.mnist)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    truth: dict[str, dict] = {}
    light_paper, dark_ink = (240, 243, 245), (40, 32, 30)
    dark_paper, light_ink = (28, 26, 24), (235, 235, 230)

    for cat in CATEGORIES:
        written = 0
        for i in range(args.per_category):
            name = f"{cat}_{i:02d}.png"
            if cat == "clean":
                img, labels = compose_lines(x_test, y_test, rng, 1, light_paper, dark_ink, 4)
            elif cat == "noisy":
                img, labels = compose_lines(x_test, y_test, rng, 1, light_paper, dark_ink, 6)
                img = add_noise_blur_jpeg(img, rng)
                name = f"{cat}_{i:02d}.jpg"
            elif cat == "light_on_dark":
                img, labels = compose_lines(x_test, y_test, rng, 1, dark_paper, light_ink, 4)
            elif cat == "multiline":
                img, labels = compose_lines(x_test, y_test, rng, int(rng.integers(2, 4)), light_paper, dark_ink, 4)
            elif cat == "font_ttf":
                _, text = pick_digits(x_test, y_test, int(rng.integers(3, 7)), rng)
                img = render_ttf(text, rng, light_paper, dark_ink)
                if img is None:
                    break
                labels = [text]
            elif cat == "font_hershey":
                _, text = pick_digits(x_test, y_test, int(rng.integers(3, 7)), rng)
                img, labels = render_hershey(text, rng, light_paper, dark_ink), [text]
            elif cat == "rotated":
                img, labels = compose_lines(x_test, y_test, rng, 1, light_paper, dark_ink, 4)
                img = rotate(img, float(rng.uniform(3, 5)) * (1 if rng.random() < 0.5 else -1), light_paper)
            elif cat == "underline":
                img, labels = compose_lines(x_test, y_test, rng, 1, light_paper, dark_ink, 4)
                draw_underline(img, dark_ink, rng)
            else:  # ruled
                img, labels = compose_lines(x_test, y_test, rng, 1, light_paper, dark_ink, 4)
                draw_rules(img, int(img.shape[0] * rng.uniform(0.3, 0.45)), rng)
            if cat == "noisy":
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
                buf.tofile(str(out / name))
            else:
                save_image(str(out / name), img)
            truth[name] = {"lines": labels, "category": cat}
            written += 1
        print(f"{cat:14s} {written} images")

    with open(out / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=2)
    print(f"wrote {len(truth)} images and ground_truth.json to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
