"""OpenCV preprocessing and digit segmentation for the MNIST digit detector.

This module deliberately has no TensorFlow dependency, so it imports fast and
can be unit-tested on its own.

Pipeline (each step is a small function you can call individually):

 1. load_image       unicode-safe read (np.fromfile + cv2.imdecode), alpha flattened onto white
 2. to_gray          BGR -> single channel
 3. downscale        cap the longer side (speed, and makes every threshold scale-relative)
 4. denoise          Gaussian blur (sensor noise, JPEG blocking)
 5. to_binary        adaptive (or Otsu) threshold with automatic polarity, so that
                     ink is ALWAYS 255 on a 0 background afterwards
 6. remove_lines     optional: erase long horizontal rules/underlines that touch digits
 7. close            3x3 morphological closing to bridge hairline breaks in strokes
 8. find_components  external contours -> one mask per blob (holes of 0/6/8/9 are kept)
 9. filter           reject specks, page borders, long lines, oversized blobs
10. merge_fragments  re-attach detached parts of one digit (top bar of a 5, broken strokes)
11. split_touching   cut blobs that are too wide to be one digit at the thinnest column
12. order            group into text lines (top to bottom), then left to right
13. crop_to_mnist    stroke-width normalisation, fit into 20x20 keeping aspect ratio,
                     centre of mass moved to the middle of a 28x28 field, float32 0..1
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x, y, w, h


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessParams:
    """Every numeric knob of the pipeline, with defaults tuned for scans or
    phone photos of handwritten/printed digits at least ~20 px tall."""

    max_side: int = 1600          # working resolution: longer side is capped to this
    blur_ksize: int = 5           # Gaussian blur kernel (odd); 0 disables
    use_otsu: bool = False        # force one global Otsu threshold instead of adaptive
    adaptive_block: int = 0       # adaptive threshold window; 0 = auto (shorter side / 8, odd, >= 31)
    adaptive_c: float = 10.0      # how much darker than the local mean a pixel must be to count as ink
    remove_lines: bool = True     # erase long horizontal lines (ruled paper, underlines)
    line_contrast: float = 25.0   # a line must differ from the paper level by this many gray levels
    close_ksize: int = 3          # closing kernel to reconnect hairline breaks; 0 disables
    min_height_px: int = 8        # absolute speck filter (working-resolution pixels)
    min_area_px: int = 40         # absolute speck filter on bounding-box area
    border_frac: float = 0.9      # a blob wider/taller than this fraction of the image is a border
    max_aspect: float = 4.0       # w/h above this (and short) is a line, never a digit
    min_rel_height: float = 0.35  # after merging, blobs shorter than this * H_ref are dropped
    merge_x_overlap: float = 0.25 # fragments merge if horizontal overlap >= this * narrower width
    merge_y_gap: float = 0.6      # ... and the vertical gap between them <= this * H_ref
    merge_max_height: float = 1.35  # a merged blob may not be taller than this * H_ref
    merge_max_aspect: float = 1.5   # ... nor wider than this * its own height
    split_enabled: bool = True    # cut blobs that are too wide to be one digit
    split_aspect: float = 1.5     # w/h above this is always split (two average digits side by side)
    split_neck_aspect: float = 1.25  # between this and split_aspect, split only at a clear neck ...
    split_neck_ink: float = 0.35  # ... i.e. a column with <= this fraction of the median column ink
    line_overlap: float = 0.5     # boxes on one text line share >= this vertical overlap
    target_stroke: float = 0.12   # desired stroke width as a fraction of the digit size (MNIST-like)
    soft_crops: bool = True       # build the 28x28 tensor from grayscale ink (anti-aliased, like MNIST) instead of the 0/1 mask
    fit_size: int = 20            # MNIST: digit fitted into a 20x20 box ...
    out_size: int = 28            # ... centred by mass inside a 28x28 field

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class Component:
    """One connected blob at working resolution. mask is uint8 0/255 with ink = 255."""

    x: int
    y: int
    w: int
    h: int
    mask: np.ndarray
    merged: bool = False  # assembled from fragments: never split it again

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def box(self) -> Box:
        return (self.x, self.y, self.w, self.h)

    @property
    def ink(self) -> int:
        return int(cv2.countNonZero(self.mask))


@dataclass
class DigitCrop:
    """One detected digit. Box is in ORIGINAL image coordinates; tensor is the
    model input: float32, shape (28, 28, 1), white digit on black, 0..1."""

    x: int
    y: int
    w: int
    h: int
    line: int
    tensor: np.ndarray


@dataclass
class Rejected:
    x: int
    y: int
    w: int
    h: int
    reason: str


@dataclass
class Segmentation:
    crops: list[DigitCrop]
    rejected: list[Rejected]
    gray: np.ndarray          # working-resolution grayscale
    binary: np.ndarray        # working-resolution ink mask (ink = 255)
    scale: float              # working / original size ratio (<= 1)
    ref_height: float         # estimated digit height at working resolution
    params: PreprocessParams = field(default_factory=PreprocessParams)


# --------------------------------------------------------------------------- #
# 1-2. Unicode-safe I/O and colour handling
# --------------------------------------------------------------------------- #
def decode_image(data) -> np.ndarray:
    """Decode encoded image bytes (PNG, JPEG, BMP, WEBP, TIFF...) to 8-bit BGR.
    Accepts bytes or a uint8 array. Flattens transparency onto white and
    converts 16-bit images. Raises ValueError when the data is not an image."""
    buf = np.frombuffer(data, dtype=np.uint8) if isinstance(data, (bytes, bytearray, memoryview)) else np.asarray(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED) if buf.size else None
    if img is None:
        raise ValueError("cannot decode image (unsupported or corrupt data)")
    return _to_bgr8(img)


def load_image(path: str) -> np.ndarray:
    """Read an image file as 8-bit BGR. Works with non-ASCII / OneDrive paths
    where cv2.imread silently fails (np.fromfile + cv2.imdecode)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    try:
        return decode_image(np.fromfile(str(p), dtype=np.uint8))
    except ValueError as exc:
        raise ValueError(f"cannot decode image (unsupported or corrupt file): {path}") from exc


def _to_bgr8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint16:
        img = (img / 257.0).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        bgr = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        return (bgr * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return img


def save_image(path: str, img: np.ndarray) -> None:
    """Write an image through cv2.imencode so non-ASCII paths work on Windows."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"cannot encode image with extension {ext}")
    buf.tofile(str(p))


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------------- #
# 3-5. Scale, denoise, threshold with automatic polarity
# --------------------------------------------------------------------------- #
def downscale(gray: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Shrink so the longer side is <= max_side. Returns (image, scale)."""
    h, w = gray.shape[:2]
    longest = max(h, w)
    if max_side <= 0 or longest <= max_side:
        return gray, 1.0
    s = max_side / float(longest)
    small = cv2.resize(gray, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    return small, s


def _auto_block(shape: tuple[int, ...]) -> int:
    block = max(31, min(shape[0], shape[1]) // 8)
    return block if block % 2 == 1 else block + 1


def background_is_light(gray: np.ndarray) -> bool:
    """Majority vote: the background is whichever side of the Otsu threshold
    holds more pixels (ink is always the minority in a sensible photo)."""
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = float(np.count_nonzero(gray > t))
    return bright >= 0.5 * gray.size


def to_binary(gray: np.ndarray, params: Optional[PreprocessParams] = None) -> np.ndarray:
    """Return a uint8 mask with ink = 255 and background = 0 for either polarity."""
    p = params or PreprocessParams()
    g = gray
    if p.blur_ksize >= 3:
        k = p.blur_ksize if p.blur_ksize % 2 == 1 else p.blur_ksize + 1
        g = cv2.GaussianBlur(g, (k, k), 0)
    light_bg = background_is_light(g)
    if p.use_otsu:
        mode = cv2.THRESH_BINARY_INV if light_bg else cv2.THRESH_BINARY
        _, binary = cv2.threshold(g, 0, 255, mode + cv2.THRESH_OTSU)
        return binary
    block = p.adaptive_block if p.adaptive_block >= 3 else _auto_block(g.shape)
    binary = _adaptive(g, block, light_bg, p.adaptive_c)
    if p.adaptive_block < 3:
        # Second pass: a window smaller than about twice the digit height hollows
        # out thick strokes (their centre is darker than the local mean of the
        # window, which is mostly ink). Measure the digits found so far and
        # re-threshold with a window that scales with them.
        comps = [c for c in find_components(binary) if c.h >= 8 and c.w <= 1.5 * c.h]
        if comps:
            href = reference_height(comps)
            wanted = int(2.5 * href)
            if wanted > block:
                block = min(wanted, (min(g.shape[:2]) // 2) | 1)
                binary = _adaptive(g, block, light_bg, p.adaptive_c)
    ink_frac = cv2.countNonZero(binary) / float(binary.size)
    if ink_frac > 0.5:  # adaptive threshold went wrong (textured or bimodal background): fall back to Otsu
        mode = cv2.THRESH_BINARY_INV if light_bg else cv2.THRESH_BINARY
        _, binary = cv2.threshold(g, 0, 255, mode + cv2.THRESH_OTSU)
    return binary


def _adaptive(g: np.ndarray, block: int, light_bg: bool, c: float) -> np.ndarray:
    block = max(3, block)
    if block % 2 == 0:
        block += 1
    if light_bg:
        # ink = pixels darker than the local mean by C
        return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, c)
    # ink = pixels brighter than the local mean by C (negative C moves the threshold up)
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, -c)


# --------------------------------------------------------------------------- #
# 6-8. Line removal, closing, components
# --------------------------------------------------------------------------- #
def line_source_mask(gray: np.ndarray, params: Optional[PreprocessParams] = None) -> np.ndarray:
    """Ink mask measured against the paper level, used only to find ruled lines
    and underlines. Paper level = median gray (the background is the majority
    of pixels); anything at least line_contrast levels away from it on the ink
    side counts. The adaptive threshold cannot be used for this: next to dark
    digits its local mean drops and a faint line stops counting as ink, so the
    line breaks into short pieces. Otsu cannot either: with strong ink present
    it puts the threshold between the digits and everything else and ignores a
    faint line completely."""
    p = params or PreprocessParams()
    g = gray
    if p.blur_ksize >= 3:
        k = p.blur_ksize if p.blur_ksize % 2 == 1 else p.blur_ksize + 1
        g = cv2.GaussianBlur(g, (k, k), 0)
    paper = float(np.median(g))
    if background_is_light(g):
        mask = g < paper - p.line_contrast
    else:
        mask = g > paper + p.line_contrast
    return mask.astype(np.uint8) * 255


def detect_horizontal_lines(source: np.ndarray, min_length: int, max_thickness: int) -> np.ndarray:
    """Mask of thin horizontal lines in an ink mask: runs longer than
    min_length (morphological opening) plus rows that are ink across >= 60% of
    the width. Blobs thicker than max_thickness are not lines (shadows, boxes)."""
    min_length = max(15, int(min_length))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_length, 1))
    lines = cv2.morphologyEx(source, cv2.MORPH_OPEN, kernel)
    row_ink = (source > 0).sum(axis=1)
    full_rows = row_ink >= 0.6 * source.shape[1]
    if full_rows.any():
        lines[full_rows] = np.maximum(lines[full_rows], source[full_rows])
    if cv2.countNonZero(lines) == 0:
        return lines
    n, labels, stats, _c = cv2.connectedComponentsWithStats(lines, connectivity=8)
    keep = np.zeros_like(lines)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_HEIGHT] <= max_thickness:
            keep[labels == i] = 255
    return keep


def remove_horizontal_lines(binary: np.ndarray, min_length: int, source: Optional[np.ndarray] = None, max_thickness: int = 12) -> np.ndarray:
    """Erase ruled lines / underlines from an ink mask and re-connect the digit
    strokes that crossed them.

    1. detect thin long horizontal lines (in `source`, default: binary itself)
    2. subtract that band (grown 2 px above and below) from the image
    3. inside the band only, a vertical closing bridges the gaps left in
       strokes that passed through the line
    Returns the input object unchanged when no line is found."""
    lines = detect_horizontal_lines(binary if source is None else source, min_length, max_thickness)
    if cv2.countNonZero(lines) == 0:
        return binary
    n, _labels, stats, _c = cv2.connectedComponentsWithStats(lines, connectivity=8)
    thickness = int(stats[1:, cv2.CC_STAT_HEIGHT].max()) if n > 1 else 3
    band = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
    cleaned = cv2.bitwise_and(binary, cv2.bitwise_not(band))
    bridge_h = thickness + 9
    bridge_h = bridge_h if bridge_h % 2 == 1 else bridge_h + 1
    bridged = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, bridge_h)))
    repair = cv2.bitwise_and(bridged, band)
    return cv2.bitwise_or(cleaned, repair)


def find_components(binary: np.ndarray) -> list[Component]:
    """External contours -> Component list. The mask of each component is the
    filled contour AND the binary image, so inner holes survive."""
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    comps: list[Component] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        filled = np.zeros((h, w), np.uint8)
        cv2.drawContours(filled, [c - np.array([[x, y]], dtype=c.dtype)], -1, 255, thickness=cv2.FILLED)
        mask = cv2.bitwise_and(filled, binary[y:y + h, x:x + w])
        comps.append(Component(x, y, w, h, mask))
    return comps


# --------------------------------------------------------------------------- #
# 9. Filtering
# --------------------------------------------------------------------------- #
def reference_height(comps: list[Component], floor: int = 12) -> float:
    """Typical digit height: the ink-weighted median height of digit-shaped
    blobs (w/h <= 1.5). Weighting by ink pixels lets whole digits dominate even
    when one digit has shattered into several small fragments."""
    if not comps:
        return float(floor)
    digitlike = [c for c in comps if c.w <= 1.5 * c.h] or list(comps)
    heights = np.array([c.h for c in digitlike], dtype=np.float64)
    weights = np.array([max(1, c.ink) for c in digitlike], dtype=np.float64)
    order = np.argsort(heights)
    cum = np.cumsum(weights[order])
    idx = int(np.searchsorted(cum, 0.5 * cum[-1]))
    return float(max(floor, heights[order][min(idx, len(order) - 1)]))


def _prefilter(comps: list[Component], shape: tuple[int, int], p: PreprocessParams) -> tuple[list[Component], list[Rejected]]:
    H, W = shape
    kept, rejected = [], []
    for c in comps:
        if c.h < p.min_height_px or c.area < p.min_area_px:
            rejected.append(Rejected(*c.box, "speck"))
        elif c.w > p.border_frac * W or c.h > p.border_frac * H:
            rejected.append(Rejected(*c.box, "border"))
        else:
            kept.append(c)
    return kept, rejected


def _reject_lines(comps: list[Component], href: float, p: PreprocessParams) -> tuple[list[Component], list[Rejected]]:
    kept, rejected = [], []
    for c in comps:
        aspect = c.w / float(c.h)
        if aspect > p.max_aspect and c.h < 0.5 * href and c.w > 1.2 * href:
            rejected.append(Rejected(*c.box, "line"))          # underline, ruled line (longer than a digit)
        elif c.h > p.max_aspect * c.w and c.h > 1.5 * href:
            rejected.append(Rejected(*c.box, "vline"))         # margin line, page edge
        elif c.h > 3.0 * href:
            rejected.append(Rejected(*c.box, "tall"))          # blob far larger than the digits
        else:
            kept.append(c)
    return kept, rejected


def _reject_small(comps: list[Component], href: float, p: PreprocessParams) -> tuple[list[Component], list[Rejected]]:
    kept, rejected = [], []
    for c in comps:
        if c.h < p.min_rel_height * href:
            rejected.append(Rejected(*c.box, "small"))         # dots, dashes, tails, leftover noise
        elif c.h < p.min_height_px or c.area < p.min_area_px:
            rejected.append(Rejected(*c.box, "speck"))
        else:
            kept.append(c)
    return kept, rejected


# --------------------------------------------------------------------------- #
# 10. Fragment merging
# --------------------------------------------------------------------------- #
def _merge_group(group: list[Component]) -> Component:
    x0 = min(c.x for c in group)
    y0 = min(c.y for c in group)
    x1 = max(c.x + c.w for c in group)
    y1 = max(c.y + c.h for c in group)
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for c in group:
        region = mask[c.y - y0:c.y - y0 + c.h, c.x - x0:c.x - x0 + c.w]
        np.maximum(region, c.mask, out=region)
    return Component(x0, y0, x1 - x0, y1 - y0, mask, merged=True)


def merge_fragments(comps: list[Component], href: float, p: PreprocessParams) -> list[Component]:
    """Re-attach pieces of one digit (detached top bar of a 5/7, a stroke broken
    by a pen skip or a removed line). Two blobs are candidates when:
      * at least one is shorter than 0.85 * H_ref (not a whole digit),
      * they overlap horizontally by >= merge_x_overlap of the narrower one, or
        the short one is right next to the other (gap <= 0.1 * H_ref),
      * their vertical gap is <= merge_y_gap * H_ref,
      * the union is still digit-shaped (<= merge_max_height * H_ref tall and
        <= merge_max_aspect wide-to-tall).
    Each blob is united with its single best-scoring partner (union-find), so a
    fragment cannot chain two neighbouring digits together."""
    n = len(comps)
    if n < 2:
        return list(comps)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    best: dict[int, tuple[float, int]] = {}
    for i in range(n):
        a = comps[i]
        for j in range(i + 1, n):
            b = comps[j]
            short = min(a.h, b.h)
            if short >= 0.85 * href:
                continue
            overlap = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)  # negative = horizontal gap
            narrow = max(1, min(a.w, b.w))
            adjacent = short < 0.5 * href and -overlap <= 0.1 * href
            if overlap < p.merge_x_overlap * narrow and not adjacent:
                continue
            vgap = max(a.y, b.y) - min(a.y + a.h, b.y + b.h)      # negative = vertical overlap
            if vgap > p.merge_y_gap * href:
                continue
            # A flat piece (dash-like) may only be the TOP bar of a 5 or 7: it must sit
            # in the upper half of its partner. Below a digit it is an underline/dash.
            flat, tall = (a, b) if a.w > 2.5 * a.h else ((b, a) if b.w > 2.5 * b.h else (None, None))
            if flat is not None and flat.y + flat.h > tall.y + 0.5 * tall.h:
                continue
            mh = max(a.y + a.h, b.y + b.h) - min(a.y, b.y)
            mw = max(a.x + a.w, b.x + b.w) - min(a.x, b.x)
            if mh > p.merge_max_height * href or mw > p.merge_max_aspect * mh:
                continue
            score = overlap / narrow - max(0, vgap) / href
            for k, other in ((i, j), (j, i)):
                if k not in best or score > best[k][0]:
                    best[k] = (score, other)
    for k, (_score, other) in best.items():
        rk, ro = find(k), find(other)
        if rk != ro:
            parent[ro] = rk

    groups: dict[int, list[Component]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(comps[i])
    return [g[0] if len(g) == 1 else _merge_group(g) for g in groups.values()]


# --------------------------------------------------------------------------- #
# 11. Touching digits
# --------------------------------------------------------------------------- #
def split_touching(comp: Component, href: float, p: PreprocessParams) -> list[Component]:
    """A blob much wider than tall cannot be one digit. Guess the digit count
    from its width and cut at the column with the least ink near each expected
    boundary. Above split_aspect the blob is always split; between
    split_neck_aspect and split_aspect (a wide 0 looks like this) it is split
    only where a clear neck exists."""
    aspect = comp.w / float(comp.h)
    if not p.split_enabled or comp.merged or comp.h < 0.6 * href or aspect <= p.split_neck_aspect:
        return [comp]
    n = int(round(comp.w / (0.8 * comp.h)))
    n = max(2, min(n, 8))
    col_ink = (comp.mask > 0).sum(axis=0)
    inked = col_ink[col_ink > 0]
    median_ink = float(np.median(inked)) if inked.size else 1.0
    require_neck = aspect <= p.split_aspect
    cuts: list[int] = []
    for k in range(1, n):
        centre = k * comp.w / n
        lo = int(max(1, centre - 0.15 * comp.w))
        hi = int(min(comp.w - 1, centre + 0.15 * comp.w))
        if hi <= lo:
            continue
        cut = lo + int(np.argmin(col_ink[lo:hi]))
        if require_neck and col_ink[cut] > p.split_neck_ink * median_ink:
            continue
        cuts.append(cut)
    if not cuts:
        return [comp]
    pieces: list[Component] = []
    prev = 0
    for cut in sorted(set(cuts)) + [comp.w]:
        sub = comp.mask[:, prev:cut]
        ys, xs = np.nonzero(sub)
        if xs.size:
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            pieces.append(Component(comp.x + prev + x0, comp.y + y0, x1 - x0, y1 - y0, np.ascontiguousarray(sub[y0:y1, x0:x1])))
        prev = cut
    return pieces if len(pieces) >= 2 else [comp]


# --------------------------------------------------------------------------- #
# 12. Reading order
# --------------------------------------------------------------------------- #
def order_boxes(boxes: list[Box], params: Optional[PreprocessParams] = None) -> list[tuple[int, Box]]:
    """Group boxes into text lines by vertical overlap, sort lines top-to-bottom
    and boxes left-to-right. Returns (line_index, box) pairs in reading order."""
    return [(line, boxes[i]) for line, i in order_indices(boxes, params)]


def order_indices(boxes: list[Box], params: Optional[PreprocessParams] = None) -> list[tuple[int, int]]:
    """Same as order_boxes but returns (line_index, index_into_boxes) pairs."""
    p = params or PreprocessParams()
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][1] + boxes[i][3] / 2.0)
    lines: list[dict] = []
    for i in order:
        b = boxes[i]
        y0, y1 = b[1], b[1] + b[3]
        placed = False
        for ln in lines:
            ov = min(ln["y1"], y1) - max(ln["y0"], y0)
            if ov > 0 and ov >= p.line_overlap * min(b[3], ln["y1"] - ln["y0"]):
                ln["members"].append(i)
                ln["y0"] = min(ln["y0"], y0)
                ln["y1"] = max(ln["y1"], y1)
                placed = True
                break
        if not placed:
            lines.append({"y0": y0, "y1": y1, "members": [i]})
    lines.sort(key=lambda ln: ln["y0"])
    out: list[tuple[int, int]] = []
    for li, ln in enumerate(lines):
        for i in sorted(ln["members"], key=lambda i: boxes[i][0]):
            out.append((li, i))
    return out


# --------------------------------------------------------------------------- #
# 13. MNIST-style normalisation of one blob
# --------------------------------------------------------------------------- #
def _tight(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return np.ascontiguousarray(mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1])


def estimate_stroke_width(mask: np.ndarray) -> float:
    """Mean distance-to-background over ink pixels is about (s + 2) / 4 for a
    stroke of width s, so s is about 4 * mean - 2."""
    ink = mask > 0
    if not ink.any():
        return 1.0
    # pad with background so that a tight crop of a 1 px stroke still has a zero pixel to measure against
    padded = cv2.copyMakeBorder(ink.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
    mean = float(dist[ink].mean())
    if not np.isfinite(mean):
        return 1.0
    return max(1.0, 4.0 * mean - 2.0)


def normalize_stroke(mask: np.ndarray, target_frac: float) -> np.ndarray:
    """Thicken thin pen strokes (or thin very fat marker strokes) so that the
    stroke is roughly target_frac of the digit size, as in MNIST (~2.5 px of 20)."""
    if target_frac <= 0:
        return mask
    size = max(mask.shape)
    target = max(1.0, target_frac * size)
    sw = estimate_stroke_width(mask)
    if sw < target - 0.5:
        k = int(round(target - sw)) + 1
        k = k if k % 2 == 1 else k + 1
        k = min(k, max(3, (size // 6) | 1))
        padded = cv2.copyMakeBorder(mask, k, k, k, k, cv2.BORDER_CONSTANT, value=0)
        return cv2.dilate(padded, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if sw > 2.0 * target:
        k = int(round(sw - 1.5 * target)) + 1
        k = k if k % 2 == 1 else k + 1
        k = min(k, max(3, (size // 10) | 1))
        eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        if cv2.countNonZero(eroded) >= 0.3 * cv2.countNonZero(mask):
            return eroded
    return mask


def crop_to_mnist(mask: np.ndarray, params: Optional[PreprocessParams] = None) -> np.ndarray:
    """Turn a binary blob (ink = nonzero) into a float32 (28, 28) MNIST-style
    image: aspect-preserving fit into 20x20 with anti-aliasing, then the centre
    of mass is moved to the centre of the 28x28 field. Values are 0..1."""
    p = params or PreprocessParams()
    out = p.out_size
    if mask.dtype == bool:
        src = mask.astype(np.uint8) * 255
    elif mask.dtype != np.uint8:
        src = np.clip(mask * (255.0 if mask.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    else:
        src = mask
    m = _tight(src)
    if m is None:
        return np.zeros((out, out), np.float32)
    m = normalize_stroke(m, p.target_stroke)
    m = _tight(m)
    if m is None:
        return np.zeros((out, out), np.float32)
    h, w = m.shape
    scale = p.fit_size / float(max(h, w))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    small = cv2.resize(m, (nw, nh), interpolation=interp)
    canvas = np.zeros((out, out), np.uint8)
    ox, oy = (out - nw) // 2, (out - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = small
    moments = cv2.moments(canvas, binaryImage=False)
    if moments["m00"] > 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        target = (out - 1) / 2.0
        dx = int(round(target - cx))
        dy = int(round(target - cy))
        ys, xs = np.nonzero(canvas)
        dx = int(np.clip(dx, -xs.min(), out - 1 - xs.max()))
        dy = int(np.clip(dy, -ys.min(), out - 1 - ys.max()))
        if dx or dy:
            shift = np.float32([[1, 0, dx], [0, 1, dy]])
            canvas = cv2.warpAffine(canvas, shift, (out, out), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return canvas.astype(np.float32) / 255.0


def soft_component(gray: np.ndarray, comp: Component, light_bg: bool, paper_global: float) -> np.ndarray:
    """Grayscale ink of one component, uint8 0..255 with the stroke core at 255.

    MNIST digits are anti-aliased grayscale, while a thresholded mask is hard
    0/255. Taking the actual ink darkness (paper level minus pixel value,
    normalised so that the 90th percentile inside the stroke is full white)
    keeps the soft stroke edges and gives crops that look much more like the
    training data. Pixels outside the (slightly grown) component mask are
    zeroed so neighbouring digits cannot leak in."""
    region = gray[comp.y:comp.y + comp.h, comp.x:comp.x + comp.w].astype(np.float32)
    near = cv2.dilate(comp.mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    background = region[near == 0]
    paper = float(np.median(background)) if background.size >= 20 else paper_global
    ink = (paper - region) if light_bg else (region - paper)
    inside = ink[comp.mask > 0]
    if inside.size == 0:
        return comp.mask
    core = float(np.percentile(inside, 90))
    if core <= 1.0:
        return comp.mask
    soft = np.clip(ink / core, 0.0, 1.0)
    soft[near == 0] = 0.0
    return (soft * 255.0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def find_digit_components(binary: np.ndarray, params: Optional[PreprocessParams] = None, line_source: Optional[np.ndarray] = None) -> tuple[list[Component], list[Rejected], float, np.ndarray]:
    """Steps 6-11 on an ink mask. Returns (components, rejected, ref_height, binary_used).
    line_source: optional global-threshold mask used to find ruled lines."""
    p = params or PreprocessParams()
    H, W = binary.shape[:2]
    comps = find_components(binary)

    if p.remove_lines:
        # Anything longer than 2.5 digit heights is a line, never a stroke. When no
        # clean digit is visible (all glued to ruled lines) fall back to a quarter
        # of the image width. Lines thicker than 20% of a digit are not lines.
        digitlike = [c for c in comps if c.w <= 1.5 * c.h and c.h >= 20 and c.w < p.border_frac * W and c.h < p.border_frac * H]
        if len(digitlike) >= 2:
            href0 = reference_height(digitlike)
            min_len, max_thick = int(2.5 * href0), max(6, int(0.2 * href0))
        else:
            min_len, max_thick = max(30, W // 4), max(6, H // 25)
        cleaned = remove_horizontal_lines(binary, min_len, line_source, max_thick)
        if cleaned is not binary:
            binary = cleaned
            comps = find_components(binary)

    kept, rejected = _prefilter(comps, (H, W), p)
    href = reference_height(kept)
    kept, rej = _reject_lines(kept, href, p)
    rejected.extend(rej)
    merged = merge_fragments(kept, href, p)
    href = reference_height(merged)
    pieces: list[Component] = []
    for c in merged:
        pieces.extend(split_touching(c, href, p))
    final, rej = _reject_small(pieces, href, p)
    rejected.extend(rej)
    return final, rejected, href, binary


def find_digit_boxes(binary: np.ndarray, params: Optional[PreprocessParams] = None) -> list[Box]:
    """Convenience: filtered, merged, split digit boxes (unordered) from an ink mask."""
    comps, _rej, _href, _bin = find_digit_components(binary, params)
    return [c.box for c in comps]


def segment_image(image: np.ndarray, params: Optional[PreprocessParams] = None, debug_dir: Optional[str] = None) -> Segmentation:
    """Run the whole pipeline on a BGR or grayscale image."""
    p = params or PreprocessParams()
    gray_full = to_gray(image)
    gray, scale = downscale(gray_full, p.max_side)
    binary = to_binary(gray, p)
    if p.close_ksize >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.close_ksize, p.close_ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    line_source = line_source_mask(gray, p) if p.remove_lines else None
    comps, rejected, href, binary = find_digit_components(binary, p, line_source)

    ordered = order_indices([c.box for c in comps], p)
    inv = 1.0 / scale
    if p.soft_crops:
        soft_gray = cv2.GaussianBlur(gray, (3, 3), 0)
        light_bg = background_is_light(soft_gray)
        paper_global = float(np.median(soft_gray))
    crops: list[DigitCrop] = []
    for line_idx, idx in ordered:
        c = comps[idx]
        source = soft_component(soft_gray, c, light_bg, paper_global) if p.soft_crops else c.mask
        crops.append(DigitCrop(
            x=int(round(c.x * inv)), y=int(round(c.y * inv)),
            w=int(round(c.w * inv)), h=int(round(c.h * inv)),
            line=line_idx, tensor=crop_to_mnist(source, p)[:, :, None]))
    rejected_full = [Rejected(int(round(r.x * inv)), int(round(r.y * inv)), int(round(r.w * inv)), int(round(r.h * inv)), r.reason) for r in rejected]

    seg = Segmentation(crops, rejected_full, gray, binary, scale, href, p)
    if debug_dir:
        dump_debug(seg, comps, rejected, debug_dir)
    return seg


def segment_digits(image: np.ndarray, params: Optional[PreprocessParams] = None, debug_dir: Optional[str] = None) -> list[DigitCrop]:
    """Digits of an image in reading order (see segment_image for details)."""
    return segment_image(image, params, debug_dir).crops


def dump_debug(seg: Segmentation, comps: list[Component], rejected: list[Rejected], debug_dir: str) -> None:
    """Write gray / binary / boxes images plus every 28x28 crop (8x enlarged)."""
    d = Path(debug_dir)
    d.mkdir(parents=True, exist_ok=True)
    save_image(str(d / "01_gray.png"), seg.gray)
    save_image(str(d / "02_binary.png"), seg.binary)
    vis = cv2.cvtColor(seg.gray, cv2.COLOR_GRAY2BGR)
    for r in rejected:
        cv2.rectangle(vis, (r.x, r.y), (r.x + r.w, r.y + r.h), (0, 0, 255), 1)
        cv2.putText(vis, r.reason, (r.x, max(10, r.y - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
    ordered = order_boxes([c.box for c in comps], seg.params)
    for i, (line_idx, (x, y, w, h)) in enumerate(ordered):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(vis, f"{i}:L{line_idx}", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 160, 0), 1, cv2.LINE_AA)
    save_image(str(d / "03_boxes.png"), vis)
    for i, crop in enumerate(seg.crops):
        tile = (crop.tensor[:, :, 0] * 255).astype(np.uint8)
        tile = cv2.resize(tile, (224, 224), interpolation=cv2.INTER_NEAREST)
        save_image(str(d / f"crop_{i:02d}_line{crop.line}.png"), tile)


def add_preprocess_args(parser) -> None:
    """Attach the tunable thresholds to an argparse parser (used by detect.py)."""
    d = PreprocessParams()
    g = parser.add_argument_group("preprocessing")
    g.add_argument("--max-side", type=int, default=d.max_side, help=f"working resolution cap for the longer side (default {d.max_side})")
    g.add_argument("--blur", type=int, default=d.blur_ksize, help=f"Gaussian blur kernel, 0 to disable (default {d.blur_ksize})")
    g.add_argument("--otsu", action="store_true", help="use one global Otsu threshold instead of adaptive thresholding")
    g.add_argument("--block", type=int, default=d.adaptive_block, help="adaptive threshold window size, 0 = auto")
    g.add_argument("--c", type=float, default=d.adaptive_c, help=f"adaptive threshold constant C (default {d.adaptive_c})")
    g.add_argument("--keep-lines", action="store_true", help="do not erase long horizontal lines")
    g.add_argument("--no-close", action="store_true", help="skip the 3x3 closing step")
    g.add_argument("--min-rel-height", type=float, default=d.min_rel_height, help=f"drop blobs shorter than this * digit height (default {d.min_rel_height})")
    g.add_argument("--merge-y-gap", type=float, default=d.merge_y_gap, help=f"max vertical gap for fragment merging, * digit height (default {d.merge_y_gap})")
    g.add_argument("--no-split", action="store_true", help="never split wide blobs into several digits")
    g.add_argument("--split-aspect", type=float, default=d.split_aspect, help=f"w/h above which a blob is always split into digits (default {d.split_aspect})")
    g.add_argument("--line-contrast", type=float, default=d.line_contrast, help=f"gray-level contrast a ruled line needs against the paper (default {d.line_contrast})")
    g.add_argument("--target-stroke", type=float, default=d.target_stroke, help=f"stroke width target as fraction of digit size, 0 disables (default {d.target_stroke})")
    g.add_argument("--binary-crops", action="store_true", help="feed hard 0/1 masks to the model instead of grayscale ink")


def params_from_args(args) -> PreprocessParams:
    return PreprocessParams(
        max_side=args.max_side, blur_ksize=args.blur, use_otsu=args.otsu,
        adaptive_block=args.block, adaptive_c=args.c, remove_lines=not args.keep_lines,
        close_ksize=0 if args.no_close else 3, min_rel_height=args.min_rel_height,
        merge_y_gap=args.merge_y_gap, split_enabled=not args.no_split,
        split_aspect=args.split_aspect, target_stroke=args.target_stroke,
        line_contrast=args.line_contrast, soft_crops=not args.binary_crops)
