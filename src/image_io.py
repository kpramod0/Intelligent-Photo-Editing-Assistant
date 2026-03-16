"""
image_io.py — Image upload, validation, metadata extraction, and safe resizing.

Responsibilities
----------------
* Accept an uploaded file object (from Streamlit or a file path).
* Validate format (whitelist) and integrity (can it be decoded?).
* Extract EXIF / basic image metadata if available.
* Return a consistently-typed BGR NumPy array alongside the original PIL image.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ExifTags, UnidentifiedImageError

from src.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, MAX_IMAGE_DIMENSION
from src.utils import pil_to_bgr, safe_resize, BGRImage


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class ImageBundle:
    """
    Holds every representation of an image that the application needs.

    Attributes
    ----------
    original_pil   : The untouched PIL Image (RGB, may be large).
    original_bgr   : The (possibly resized) BGR NumPy array for CV processing.
    display_pil    : Same as original_bgr but back as PIL for display.
    filename       : Original filename or "unknown".
    metadata       : Dictionary of extracted metadata fields.
    was_resized    : True if the image was downscaled to fit MAX_IMAGE_DIMENSION.
    original_size  : (width, height) before any resizing.
    processing_size: (width, height) after resizing.
    """
    original_pil:    Image.Image
    original_bgr:    BGRImage
    display_pil:     Image.Image
    filename:        str = "unknown"
    metadata:        dict = field(default_factory=dict)
    was_resized:     bool = False
    original_size:   tuple[int, int] = (0, 0)
    processing_size: tuple[int, int] = (0, 0)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

class ImageValidationError(ValueError):
    """Raised when uploaded image fails validation."""


def _check_extension(filename: str) -> None:
    """Raise ImageValidationError if the file extension is not whitelisted."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )


def _check_file_size(data: bytes, filename: str) -> None:
    """Raise ImageValidationError if the byte payload exceeds the size limit."""
    size_mb = len(data) / (1024 ** 2)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ImageValidationError(
            f"File '{filename}' is {size_mb:.1f} MB, which exceeds the "
            f"{MAX_FILE_SIZE_MB} MB limit."
        )


def _decode_pil(data: bytes, filename: str) -> Image.Image:
    """
    Attempt to open image bytes with Pillow.

    Raises ImageValidationError if the bytes do not represent a valid image.
    """
    try:
        pil_img = Image.open(io.BytesIO(data))
        pil_img.verify()          # raises on corruption
        # Re-open after verify() (verify() closes the internal file pointer)
        pil_img = Image.open(io.BytesIO(data))
        pil_img = pil_img.convert("RGB")  # normalise to RGB; drops palette etc.
        return pil_img
    except UnidentifiedImageError:
        raise ImageValidationError(
            f"'{filename}' could not be decoded. The file appears to be corrupt "
            "or is not a valid image."
        )
    except Exception as exc:
        raise ImageValidationError(
            f"Failed to open '{filename}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def _extract_metadata(pil_img: Image.Image) -> dict:
    """
    Extract basic metadata from a PIL image.

    Falls back gracefully if EXIF is unavailable.
    """
    meta: dict = {
        "Format":       pil_img.format or "Unknown",
        "Mode":         pil_img.mode,
        "Width (px)":   pil_img.width,
        "Height (px)":  pil_img.height,
        "Megapixels":   f"{(pil_img.width * pil_img.height) / 1_000_000:.2f} MP",
        "Aspect ratio": f"{pil_img.width / max(pil_img.height, 1):.3f}",
    }

    # Try EXIF
    try:
        exif_data = pil_img._getexif()  # type: ignore[attr-defined]
        if exif_data:
            for tag_id, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                if tag_name in (
                    "Make", "Model", "Software",
                    "DateTime", "ExposureTime", "FNumber",
                    "ISOSpeedRatings", "FocalLength",
                    "Flash", "WhiteBalance",
                ):
                    meta[tag_name] = str(value)
    except (AttributeError, Exception):
        pass  # No EXIF or unreadable — silently continue

    return meta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_image_from_bytes(data: bytes, filename: str = "upload.jpg") -> ImageBundle:
    """
    Validate and load image bytes into an :class:`ImageBundle`.

    Parameters
    ----------
    data     : Raw bytes from file upload or disk read.
    filename : Original filename (used for extension check and display).

    Returns
    -------
    ImageBundle with all representations populated.

    Raises
    ------
    ImageValidationError on any validation failure.
    """
    _check_extension(filename)
    _check_file_size(data, filename)
    pil_img = _decode_pil(data, filename)

    metadata = _extract_metadata(pil_img)
    original_size = (pil_img.width, pil_img.height)

    # Convert to BGR for OpenCV processing
    bgr = pil_to_bgr(pil_img)

    # Resize if necessary (preserves aspect ratio)
    bgr_resized = safe_resize(bgr, MAX_IMAGE_DIMENSION)
    was_resized = bgr_resized.shape[:2] != bgr.shape[:2]
    processing_size = (bgr_resized.shape[1], bgr_resized.shape[0])

    # Build display PIL from (possibly resized) BGR
    from src.utils import bgr_to_pil  # local import to avoid circular dependency
    display_pil = bgr_to_pil(bgr_resized)

    return ImageBundle(
        original_pil=pil_img,
        original_bgr=bgr_resized,
        display_pil=display_pil,
        filename=filename,
        metadata=metadata,
        was_resized=was_resized,
        original_size=original_size,
        processing_size=processing_size,
    )


def load_image_from_path(path: str | Path) -> ImageBundle:
    """
    Convenience wrapper to load an image from a local file path.

    Parameters
    ----------
    path : Absolute or relative path to the image.
    """
    p = Path(path)
    if not p.exists():
        raise ImageValidationError(f"File not found: {path}")
    data = p.read_bytes()
    return load_image_from_bytes(data, filename=p.name)
