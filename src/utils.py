"""
utils.py — General-purpose helpers for the Intelligent Photo Editing Assistant.
"""

from __future__ import annotations

import io
import time
from functools import wraps
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
BGRImage = np.ndarray   # uint8, shape (H, W, 3) in BGR colour order
RGBImage = np.ndarray   # uint8, shape (H, W, 3) in RGB colour order
GrayImage = np.ndarray  # uint8, shape (H, W)


# ---------------------------------------------------------------------------
# Colour-space conversions
# ---------------------------------------------------------------------------

def pil_to_bgr(pil_img: Image.Image) -> BGRImage:
    """Convert a PIL Image (RGB) to an OpenCV BGR numpy array."""
    rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: BGRImage) -> Image.Image:
    """Convert an OpenCV BGR array to a PIL Image (RGB)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def bgr_to_gray(bgr: BGRImage) -> GrayImage:
    """Return a single-channel grayscale array from a BGR image."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def bgr_to_rgb(bgr: BGRImage) -> RGBImage:
    """Swap B and R channels."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def bgr_to_hsv(bgr: BGRImage) -> np.ndarray:
    """Convert BGR to HSV (H: 0-179, S: 0-255, V: 0-255)."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def bgr_to_lab(bgr: BGRImage) -> np.ndarray:
    """Convert BGR to CIE L*a*b* colour space."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)


# ---------------------------------------------------------------------------
# Image byte helpers
# ---------------------------------------------------------------------------

def pil_to_bytes(pil_img: Image.Image, fmt: str = "PNG") -> bytes:
    """Serialise a PIL image to an in-memory byte string."""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()


def bytes_to_pil(data: bytes) -> Image.Image:
    """Deserialise bytes back to a PIL Image."""
    return Image.open(io.BytesIO(data))


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


def safe_divide(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Division that returns *fallback* instead of raising ZeroDivisionError."""
    return numerator / denominator if denominator != 0 else fallback


def score_from_range(
    value: float,
    lo: float,
    hi: float,
    max_score: float = 100.0,
) -> float:
    """
    Return a score in [0, max_score].

    Gives *max_score* when value sits inside [lo, hi], and linearly
    interpolates to 0 when value is at the boundary of the allowed range
    (mirrored on both sides).  Values far outside the range score 0.

    Parameters
    ----------
    value:     measured metric
    lo, hi:    ideal (inclusive) range
    max_score: ceiling score (default 100)
    """
    mid = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    if half <= 0:
        return max_score if value == lo else 0.0
    distance = abs(value - mid)
    ratio = clamp(1.0 - (distance - half) / half, 0.0, 1.0) if distance > half else 1.0
    return ratio * max_score


# ---------------------------------------------------------------------------
# Timing decorator
# ---------------------------------------------------------------------------

def timed(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that returns (result, elapsed_seconds) instead of just result."""
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, float]:
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        return result, time.perf_counter() - t0
    return wrapper


# ---------------------------------------------------------------------------
# Safe resize
# ---------------------------------------------------------------------------

def safe_resize(bgr: BGRImage, max_dim: int) -> BGRImage:
    """
    Downscale *bgr* so that the longest side is at most *max_dim* pixels.
    Upscaling is never performed.  Aspect ratio is preserved.
    """
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return bgr
    scale = max_dim / longest
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
