"""
test_analysis.py — Unit tests for the analysis engine.

Run with:
    cd intelligent_photo_editing_assistant
    pytest tests/ -v
"""

from __future__ import annotations

import io
import os
import sys

import numpy as np
import pytest
from PIL import Image

# Add project root so 'src' is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis import (
    analyse_brightness,
    analyse_contrast,
    analyse_histogram,
    analyse_sharpness,
    analyse_noise,
    analyse_white_balance,
    analyse_saturation,
    analyse_dynamic_range,
    analyse_image,
)
from src.image_io import load_image_from_bytes, ImageValidationError


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_bgr(r: int, g: int, b: int, h: int = 100, w: int = 100) -> np.ndarray:
    """Create a solid-colour BGR image."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = b
    arr[:, :, 1] = g
    arr[:, :, 2] = r
    return arr


def _pil_to_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def dark_bgr():
    """Very dark (underexposed) image."""
    return _make_bgr(20, 20, 20)


@pytest.fixture
def bright_bgr():
    """Very bright (overexposed) image."""
    return _make_bgr(240, 240, 240)


@pytest.fixture
def neutral_bgr():
    """Mid-grey neutral image."""
    return _make_bgr(128, 128, 128)


@pytest.fixture
def noisy_bgr():
    """Synthetic noisy image."""
    rng = np.random.default_rng(42)
    base = np.full((100, 100, 3), 128, dtype=np.float32)
    noise = rng.normal(0, 20, base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def blurry_bgr():
    """Heavily blurred image → low Laplacian variance."""
    import cv2
    base = _make_bgr(128, 100, 80)
    return cv2.GaussianBlur(base, (31, 31), 10)


@pytest.fixture
def sharp_bgr():
    """Image with strong edges → high Laplacian variance."""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    # Checkerboard
    for y in range(100):
        for x in range(100):
            v = 255 if (x // 10 + y // 10) % 2 == 0 else 0
            arr[y, x] = [v, v, v]
    return arr


@pytest.fixture
def red_cast_bgr():
    """Image with a dominant red channel → warm cast."""
    return _make_bgr(200, 100, 80)


@pytest.fixture
def grayscale_bgr():
    """Grayscale image encoded as BGR (all channels equal)."""
    return _make_bgr(128, 128, 128)


# ─────────────────────────────────────────────────────────────────────────────
# Image validation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestImageValidation:
    def test_valid_jpeg_accepted(self):
        pil = Image.fromarray(np.full((50, 50, 3), 128, dtype=np.uint8), mode="RGB")
        data = _pil_to_bytes(pil)
        bundle = load_image_from_bytes(data, filename="test.jpg")
        assert bundle.filename == "test.jpg"

    def test_invalid_extension_rejected(self):
        with pytest.raises(ImageValidationError, match="Unsupported file type"):
            load_image_from_bytes(b"fake", filename="image.bmp")

    def test_corrupt_file_rejected(self):
        with pytest.raises(ImageValidationError):
            load_image_from_bytes(b"not an image at all!", filename="corrupt.jpg")

    def test_large_file_rejected(self):
        # Simulate a very large byte payload
        huge_data = b"\x00" * (51 * 1024 * 1024)  # 51 MB
        with pytest.raises(ImageValidationError, match="exceeds the"):
            load_image_from_bytes(huge_data, filename="huge.jpg")


# ─────────────────────────────────────────────────────────────────────────────
# Brightness tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBrightnessAnalysis:
    def test_dark_image_underexposed(self, dark_bgr):
        result = analyse_brightness(dark_bgr)
        assert result.label == "Underexposed"
        assert result.score < 60

    def test_bright_image_overexposed(self, bright_bgr):
        result = analyse_brightness(bright_bgr)
        assert result.label == "Overexposed"
        assert result.score < 100

    def test_neutral_image_balanced(self, neutral_bgr):
        result = analyse_brightness(neutral_bgr)
        assert result.label == "Balanced"
        assert result.score >= 80

    def test_value_is_mean_brightness(self, neutral_bgr):
        result = analyse_brightness(neutral_bgr)
        # A 128,128,128 BGR image has grayscale mean ≈ 128
        assert abs(result.value - 128.0) < 2.0

    def test_score_in_range(self, dark_bgr):
        result = analyse_brightness(dark_bgr)
        assert 0 <= result.score <= 100


# ─────────────────────────────────────────────────────────────────────────────
# Contrast tests
# ─────────────────────────────────────────────────────────────────────────────

class TestContrastAnalysis:
    def test_flat_image_low_contrast(self, neutral_bgr):
        result = analyse_contrast(neutral_bgr)
        assert result.label == "Low contrast"
        assert result.score < 70

    def test_checkerboard_good_contrast(self, sharp_bgr):
        result = analyse_contrast(sharp_bgr)
        assert result.label in ("Good contrast", "High contrast")
        assert result.score > 70


# ─────────────────────────────────────────────────────────────────────────────
# Histogram tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHistogramAnalysis:
    def test_extremely_dark_clips_shadows(self, dark_bgr):
        # All pixels at ~20 — many near 0
        extreme = _make_bgr(0, 0, 0)
        result = analyse_histogram(extreme)
        assert result.label == "Clipping detected"

    def test_neutral_no_clipping(self, neutral_bgr):
        result = analyse_histogram(neutral_bgr)
        assert result.label == "Healthy histogram"
        assert result.score > 80

    def test_extra_contains_histograms(self, neutral_bgr):
        result = analyse_histogram(neutral_bgr)
        assert "hist_gray" in result.extra
        assert "shadow_frac" in result.extra


# ─────────────────────────────────────────────────────────────────────────────
# Sharpness tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSharpnessAnalysis:
    def test_blurry_image_classified_blurry(self, blurry_bgr):
        result = analyse_sharpness(blurry_bgr)
        assert result.label == "Blurry"
        assert result.score < 70

    def test_sharp_checkerboard_classified_sharp(self, sharp_bgr):
        result = analyse_sharpness(sharp_bgr)
        assert result.label in ("Acceptable", "Sharp")
        assert result.score > 60

    def test_value_is_laplacian_variance(self, sharp_bgr):
        import cv2
        from src.utils import bgr_to_gray
        gray = bgr_to_gray(sharp_bgr)
        expected = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        result = analyse_sharpness(sharp_bgr)
        assert abs(result.value - expected) < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# White balance tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWhiteBalanceAnalysis:
    def test_neutral_image_no_cast(self, neutral_bgr):
        result = analyse_white_balance(neutral_bgr)
        assert result.label == "Neutral"
        assert result.score > 90

    def test_red_image_detects_warm_cast(self, red_cast_bgr):
        result = analyse_white_balance(red_cast_bgr)
        assert "warm" in result.label.lower() or "cast" in result.label.lower()
        assert result.score < 90


# ─────────────────────────────────────────────────────────────────────────────
# Noise tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNoiseAnalysis:
    def test_clean_flat_image_low_noise(self, neutral_bgr):
        result = analyse_noise(neutral_bgr)
        assert result.label == "Low noise"
        assert result.score > 85

    def test_noisy_image_detected(self, noisy_bgr):
        result = analyse_noise(noisy_bgr)
        assert result.label in ("Moderate noise", "High noise")
        assert result.score < 95


# ─────────────────────────────────────────────────────────────────────────────
# Full analysis pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestFullAnalysis:
    def test_returns_all_metrics(self, neutral_bgr):
        result = analyse_image(neutral_bgr)
        assert result.brightness is not None
        assert result.contrast is not None
        assert result.histogram is not None
        assert result.sharpness is not None
        assert result.noise is not None
        assert result.white_balance is not None
        assert result.saturation is not None
        assert result.dynamic_range is not None

    def test_as_dict_returns_8_items(self, neutral_bgr):
        result = analyse_image(neutral_bgr)
        d = result.as_dict()
        assert len(d) == 8

    def test_grayscale_image_low_saturation(self, grayscale_bgr):
        result = analyse_image(grayscale_bgr)
        assert result.saturation.label == "Undersaturated"
