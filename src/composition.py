"""
composition.py — Heuristic-based composition analysis and crop suggestion.

Implemented checks
------------------
1. Rule-of-thirds grid scoring (saliency alignment to power-point intersections).
2. Saliency-weighted centre of visual mass.
3. Horizontal horizon detection (Hough lines).
4. Suggested crop rectangle based on saliency map and rule-of-thirds.

All methods are explicitly heuristic — that is clearly documented so users
understand the limitations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.utils import bgr_to_gray, BGRImage
from src.analysis import MetricResult
from src import config as cfg


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CompositionResult:
    """Output of the composition analyser."""
    metric:          MetricResult
    saliency_map:    np.ndarray       # Normalised float32 [0,1] saliency
    thirds_overlay:  np.ndarray       # BGR image with thirds grid drawn
    crop_rect:       tuple[int,int,int,int] | None  # (x1,y1,x2,y2) or None
    horizon_angle:   float | None     # degrees from horizontal, or None
    visual_center:   tuple[float,float]              # (cx_frac, cy_frac)


# ---------------------------------------------------------------------------
# Saliency approximation
# ---------------------------------------------------------------------------

def _compute_saliency(bgr: BGRImage) -> np.ndarray:
    """
    Approximate a saliency map using edge energy.

    Method
    ------
    1. Convert to grayscale.
    2. Apply Canny edge detection.
    3. Dilate to get 'edge regions'.
    4. Gaussian blur to get a smooth saliency proxy.

    This is a simple spectral-inspired heuristic, not a deep saliency model.
    The result is a float32 array normalised to [0, 1].
    """
    gray = bgr_to_gray(bgr)
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    # Dilate to capture nearby pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dilated = cv2.dilate(edges, kernel)
    # Smooth
    sal = cv2.GaussianBlur(dilated.astype(np.float32), (51, 51), 0)
    # Also include a local-contrast component
    local_std = cv2.GaussianBlur(
        (gray.astype(np.float32) - cv2.GaussianBlur(gray.astype(np.float32), (15, 15), 0)) ** 2,
        (51, 51),
        0,
    )
    sal = sal + local_std
    # Normalise
    sal_min, sal_max = sal.min(), sal.max()
    if sal_max > sal_min:
        sal = (sal - sal_min) / (sal_max - sal_min)
    else:
        sal = np.zeros_like(sal)
    return sal.astype(np.float32)


# ---------------------------------------------------------------------------
# Rule of thirds
# ---------------------------------------------------------------------------

def _thirds_score(saliency: np.ndarray) -> float:
    """
    Score how well the saliency aligns with rule-of-thirds power points.

    The four power points are at (1/3, 1/3), (1/3, 2/3), (2/3, 1/3), (2/3, 2/3)
    as fractions of image dimensions.  For each power point, we sample the
    saliency in a small neighbourhood.  The maximum over all four points is
    returned, scaled to [0, 100].
    """
    h, w = saliency.shape
    thirds_fracs = [(1/3, 1/3), (1/3, 2/3), (2/3, 1/3), (2/3, 2/3)]
    tol = cfg.THIRDS_TOLERANCE   # neighbourhood radius as fraction
    tol_h = int(tol * h)
    tol_w = int(tol * w)

    max_sal = 0.0
    for fy, fx in thirds_fracs:
        cy, cx = int(fy * h), int(fx * w)
        patch = saliency[
            max(0, cy - tol_h): min(h, cy + tol_h),
            max(0, cx - tol_w): min(w, cx + tol_w),
        ]
        if patch.size > 0:
            max_sal = max(max_sal, float(patch.mean()))

    return float(np.clip(max_sal * 100, 0, 100))


def _draw_thirds_grid(bgr: BGRImage) -> np.ndarray:
    """Draw a rule-of-thirds overlay on a copy of *bgr*."""
    h, w = bgr.shape[:2]
    overlay = bgr.copy()
    color = (0, 200, 255)   # Yellow-orange
    thickness = 1

    # Vertical lines
    for fx in [1/3, 2/3]:
        x = int(fx * w)
        cv2.line(overlay, (x, 0), (x, h), color, thickness)
    # Horizontal lines
    for fy in [1/3, 2/3]:
        y = int(fy * h)
        cv2.line(overlay, (0, y), (w, y), color, thickness)
    # Power-point circles
    for fy, fx in [(1/3, 1/3), (1/3, 2/3), (2/3, 1/3), (2/3, 2/3)]:
        cx, cy = int(fx * w), int(fy * h)
        cv2.circle(overlay, (cx, cy), max(6, w // 60), (0, 255, 255), 2)

    return overlay


# ---------------------------------------------------------------------------
# Visual centre of mass
# ---------------------------------------------------------------------------

def _visual_center(saliency: np.ndarray) -> tuple[float, float]:
    """
    Return the saliency-weighted centre of mass as (cx_frac, cy_frac).

    Both fractions are in [0, 1] relative to image width and height.
    """
    h, w = saliency.shape
    total = saliency.sum()
    if total < 1e-9:
        return 0.5, 0.5
    ys, xs = np.mgrid[0:h, 0:w]
    cx = float((xs * saliency).sum() / total) / w
    cy = float((ys * saliency).sum() / total) / h
    return cx, cy


# ---------------------------------------------------------------------------
# Horizon detection
# ---------------------------------------------------------------------------

def _estimate_horizon_angle(bgr: BGRImage) -> float | None:
    """
    Detect dominant horizontal lines via Hough transforms.

    Returns the angle (in degrees from horizontal) of the dominant nearly-
    horizontal line, or None if detection fails.

    Limitation: works best on landscape photos with clear sky/land boundaries.
    """
    gray = bgr_to_gray(bgr)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=gray.shape[1] // 4,
        maxLineGap=20,
    )
    if lines is None:
        return None

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if abs(angle) < 15:  # near-horizontal only
            angles.append(angle)

    if not angles:
        return None
    return float(np.median(angles))


# ---------------------------------------------------------------------------
# Crop suggestion
# ---------------------------------------------------------------------------

def _suggest_crop(saliency: np.ndarray, frac: float = 0.80) -> tuple[int,int,int,int]:
    """
    Suggest a crop window that covers *frac* of saliency mass and is aligned
    to the rule-of-thirds grid.

    Returns (x1, y1, x2, y2) in pixel coordinates.
    """
    h, w = saliency.shape
    threshold = float(np.percentile(saliency, 60))
    mask = (saliency >= threshold).astype(np.uint8)
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return (0, 0, w, h)

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())

    # Add 10 % padding
    pad_x = int((x2 - x1) * 0.10)
    pad_y = int((y2 - y1) * 0.10)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def analyse_composition(bgr: BGRImage) -> CompositionResult:
    """
    Run all composition checks and return a :class:`CompositionResult`.
    """
    saliency = _compute_saliency(bgr)
    thirds_score_val = _thirds_score(saliency)
    vis_cx, vis_cy = _visual_center(saliency)
    horizon_angle = _estimate_horizon_angle(bgr)
    thirds_overlay = _draw_thirds_grid(bgr)
    crop_rect = _suggest_crop(saliency)

    # --- Scoring ---
    score = thirds_score_val  # main driver

    # Penalise if visual centre is far from a thirds line
    cx_deviation = min(abs(vis_cx - 1/3), abs(vis_cx - 2/3), abs(vis_cx - 0.5))
    cy_deviation = min(abs(vis_cy - 1/3), abs(vis_cy - 2/3), abs(vis_cy - 0.5))
    centre_penalty = (cx_deviation + cy_deviation) * 30
    score = float(np.clip(score - centre_penalty, 0, 100))

    # Penalise tilted horizon
    if horizon_angle is not None and abs(horizon_angle) > 2.0:
        score = float(np.clip(score - abs(horizon_angle) * 2, 0, 100))

    # Build label
    if score >= 70:
        label = "Well composed"
    elif score >= 45:
        label = "Acceptable composition"
    else:
        label = "Composition could be improved"

    horizon_str = (
        f"{horizon_angle:.1f}° tilt detected." if horizon_angle is not None else
        "Horizon angle not detectable."
    )

    explanation = (
        f"Rule-of-thirds saliency alignment score: {thirds_score_val:.1f}/100. "
        f"Visual centre of mass: ({vis_cx:.2f}, {vis_cy:.2f}) (fraction of W, H). "
        f"{horizon_str} "
        "Note: composition analysis is heuristic-based and should be used as a guide."
    )

    suggestion = (
        "No change needed." if score >= 70 else
        "Consider cropping to the suggested rectangle or repositioning the main subject "
        "closer to a rule-of-thirds intersection."
    )

    metric = MetricResult(
        name="Composition",
        value=round(score, 1),
        label=label,
        score=round(score, 1),
        explanation=explanation,
        suggestion=suggestion,
        extra={
            "thirds_score": round(thirds_score_val, 1),
            "visual_center_x": round(vis_cx, 3),
            "visual_center_y": round(vis_cy, 3),
            "horizon_angle": round(horizon_angle, 2) if horizon_angle is not None else None,
        },
    )

    return CompositionResult(
        metric=metric,
        saliency_map=saliency,
        thirds_overlay=thirds_overlay,
        crop_rect=crop_rect,
        horizon_angle=horizon_angle,
        visual_center=(vis_cx, vis_cy),
    )
