"""Find the sheet of paper in a photo or camera frame so that digits are only
searched for on the paper, never on the desk, hands or background.

    paper = find_paper(image_bgr)          # None when no sheet-like region exists
    digits = segment_digits(paper.warped)   # top-down view of the sheet only
    paper.map_box(x, y, w, h)               # -> box in original image coordinates

How it works (OpenCV only):
 1. downscale, convert to HSV
 2. paper mask = bright (V above the Otsu threshold) AND unsaturated (S small):
    white/grey paper qualifies, skin, wood, coloured objects do not
 3. morphological closing fills the ink and small gaps, opening drops specks
 4. the largest external contour that covers at least min_area_frac of the
    frame is the sheet; its filled outline is the paper mask (ink holes and
    hands cutting in from the edge are handled by the outline, not the pixels)
 5. the contour's convex hull is approximated to a quadrilateral; if that
    fails, its minimum-area rectangle is used
 6. a perspective warp produces an upright top-down view; pixels outside the
    paper outline are filled with the paper's median colour so the digit
    pipeline sees a plain sheet
When the bright region covers (almost) the whole frame (a scan, a crop, a
white desk), fills_frame is True and the image is used as is.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np


@dataclass
class PaperParams:
    max_side: int = 640           # working size of the paper search
    min_area_frac: float = 0.04   # a sheet must cover at least this fraction of the frame
    max_saturation: int = 90      # HSV saturation (0-255) above which a pixel is a coloured object
    min_value: int = 0            # brightness threshold, 0 = automatic (Otsu)
    close_frac: float = 0.02      # closing kernel as a fraction of the working size
    approx_eps: float = 0.03      # polygon approximation tolerance (fraction of the perimeter)
    edge_margin: int = 6          # pixels of the paper edge blanked after warping (shadows, frayed edges)
    fills_frame_frac: float = 0.9 # above this coverage the sheet is the whole image

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Paper:
    quad: np.ndarray          # (4, 2) float32 corners in original coordinates: TL, TR, BR, BL
    warped: np.ndarray        # BGR top-down view; non-paper pixels filled with the paper colour
    matrix: np.ndarray        # 3x3 perspective transform original -> warped
    inverse: np.ndarray       # 3x3 warped -> original
    area_fraction: float      # sheet area / image area
    fills_frame: bool         # True when the image itself is the sheet (no warp applied)

    def map_box(self, x: int, y: int, w: int, h: int) -> dict:
        """Box in warped coordinates -> axis-aligned box in original coordinates
        (plus the exact projected quadrilateral as 'quad')."""
        if self.fills_frame:
            return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
        pts = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).reshape(-1, 1, 2)
        q = cv2.perspectiveTransform(pts, self.inverse).reshape(4, 2)
        xs, ys = q[:, 0], q[:, 1]
        return {
            "x": int(round(float(xs.min()))), "y": int(round(float(ys.min()))),
            "w": int(round(float(xs.max() - xs.min()))), "h": int(round(float(ys.max() - ys.min()))),
            "quad": [[int(round(float(px))), int(round(float(py)))] for px, py in q],
        }

    def info(self) -> dict:
        return {
            "found": True,
            "fills_frame": self.fills_frame,
            "area_fraction": round(self.area_fraction, 3),
            "quad": [[int(round(float(px))), int(round(float(py)))] for px, py in self.quad],
            "width": int(self.warped.shape[1]),
            "height": int(self.warped.shape[0]),
        }


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Return the 4 points as TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def paper_mask(small_bgr: np.ndarray, p: PaperParams) -> np.ndarray:
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    if p.min_value > 0:
        v_thr = p.min_value
    else:
        v_thr, _ = cv2.threshold(cv2.GaussianBlur(val, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = ((val >= v_thr) & (sat <= p.max_saturation)).astype(np.uint8) * 255
    k = max(3, int(p.close_frac * max(small_bgr.shape[:2])) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    k2 = max(3, (k // 2) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2)))
    return mask


def find_paper(image: np.ndarray, params: Optional[PaperParams] = None) -> Optional[Paper]:
    """Locate the sheet of paper in a BGR image. Returns None if there is no
    bright sheet-like region."""
    p = params or PaperParams()
    H, W = image.shape[:2]
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    scale = min(1.0, p.max_side / float(max(H, W)))
    small = cv2.resize(image, (max(1, int(round(W * scale))), max(1, int(round(H * scale)))), interpolation=cv2.INTER_AREA) if scale < 1.0 else image
    mask = paper_mask(small, p)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area_frac = cv2.contourArea(contour) / float(mask.shape[0] * mask.shape[1])
    if area_frac < p.min_area_frac:
        return None
    inv = 1.0 / scale
    contour_full = (contour.reshape(-1, 2).astype(np.float32) * inv)

    if area_frac >= p.fills_frame_frac:
        quad = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
        eye = np.eye(3, dtype=np.float64)
        return Paper(quad, image, eye, eye, area_frac, True)

    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, p.approx_eps * peri, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        quad = order_corners(approx.reshape(4, 2).astype(np.float32) * inv)
    else:
        quad = order_corners(cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32) * inv)
    quad[:, 0] = np.clip(quad[:, 0], 0, W - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, H - 1)

    tl, tr, br, bl = quad
    out_w = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    out_h = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if out_w < 20 or out_h < 20:
        return None
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(image, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # paper outline mask (filled contour, not the hull, so a hand cutting into the sheet is excluded)
    outline = np.zeros((H, W), np.uint8)
    cv2.fillPoly(outline, [np.round(contour_full).astype(np.int32)], 255)
    outline_w = cv2.warpPerspective(outline, matrix, (out_w, out_h), flags=cv2.INTER_NEAREST)
    if p.edge_margin > 0:
        k = 2 * p.edge_margin + 1
        outline_w = cv2.erode(outline_w, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    inside = outline_w > 0
    if inside.any():
        fill = np.median(warped[inside].reshape(-1, 3), axis=0).astype(np.uint8)
        warped = warped.copy()
        warped[~inside] = fill
    return Paper(quad, warped, matrix, np.linalg.inv(matrix), area_frac, False)


def draw_paper(image: np.ndarray, paper: Optional[Paper], color=(255, 160, 0), thickness: int = 2) -> np.ndarray:
    """Outline the detected sheet (no-op when the sheet is the whole image)."""
    if paper is None or paper.fills_frame:
        return image
    pts = np.round(paper.quad).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [pts], True, color, thickness, cv2.LINE_AA)
    return image
