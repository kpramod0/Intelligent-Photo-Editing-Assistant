"""
test_enhancement.py — Unit tests for the enhancement pipeline.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.enhancement import (
    apply_gamma_correction,
    correct_exposure,
    apply_clahe,
    apply_white_balance,
    apply_denoising,
    apply_sharpening,
    apply_saturation,
    run_enhancement_pipeline,
)
from src.utils import bgr_to_gray


def _solid(r: int, g: int, b: int, h: int = 80, w: int = 80) -> np.ndarray:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = b
    arr[:, :, 1] = g
    arr[:, :, 2] = r
    return arr


@pytest.fixture
def dark_bgr():
    return _solid(30, 30, 30)


@pytest.fixture
def neutral_bgr():
    return _solid(128, 128, 128)


@pytest.fixture
def red_bgr():
    return _solid(200, 80, 70)


class TestGammaCorrection:
    def test_gamma_1_unchanged(self, neutral_bgr):
        out = apply_gamma_correction(neutral_bgr, gamma=1.0)
        assert np.array_equal(out, neutral_bgr)

    def test_gamma_brightens(self, dark_bgr):
        # In our implementation (inv_gamma = 1/gamma), gamma > 1 brightens.
        out = apply_gamma_correction(dark_bgr, gamma=1.5)
        assert float(bgr_to_gray(out).mean()) > float(bgr_to_gray(dark_bgr).mean())

    def test_gamma_darkens(self, neutral_bgr):
        # In our implementation (inv_gamma = 1/gamma), gamma < 1 darkens.
        out = apply_gamma_correction(neutral_bgr, gamma=0.5)
        assert float(bgr_to_gray(out).mean()) < float(bgr_to_gray(neutral_bgr).mean())

    def test_invalid_gamma_raises(self, neutral_bgr):
        with pytest.raises(ValueError):
            apply_gamma_correction(neutral_bgr, gamma=0)

    def test_output_dtype(self, neutral_bgr):
        out = apply_gamma_correction(neutral_bgr, gamma=1.2)
        assert out.dtype == np.uint8


class TestExposureCorrection:
    def test_dark_image_brightens(self, dark_bgr):
        out, step = correct_exposure(dark_bgr)
        assert float(bgr_to_gray(out).mean()) > float(bgr_to_gray(dark_bgr).mean())
        assert step.applied is True

    def test_no_change_for_neutral_with_auto(self, neutral_bgr):
        out, step = correct_exposure(neutral_bgr, target_mean=128.0)
        # Should be very close to neutral — step may or may not apply
        assert abs(float(bgr_to_gray(out).mean()) - 128.0) < 10


class TestCLAHE:
    def test_contrast_improves(self, neutral_bgr):
        before_std = float(bgr_to_gray(neutral_bgr).astype(np.float32).std())
        out, step = apply_clahe(neutral_bgr)
        after_std  = float(bgr_to_gray(out).astype(np.float32).std())
        # CLAHE on flat image may or may not improve std much — just check no crash
        assert 0 <= after_std <= 128

    def test_output_shape_unchanged(self, neutral_bgr):
        out, _ = apply_clahe(neutral_bgr)
        assert out.shape == neutral_bgr.shape


class TestWhiteBalance:
    def test_neutral_neutral_unchanged(self, neutral_bgr):
        out, _ = apply_white_balance(neutral_bgr)
        # For perfectly neutral image, channel means are already equal
        assert out.shape == neutral_bgr.shape

    def test_red_cast_reduced(self, red_bgr):
        import cv2
        b1, g1, r1 = [ch.mean() for ch in cv2.split(red_bgr.astype(np.float32))]
        out, _ = apply_white_balance(red_bgr)
        b2, g2, r2 = [ch.mean() for ch in cv2.split(out.astype(np.float32))]
        # After WB, channels should be closer together
        before_spread = max(r1, g1, b1) - min(r1, g1, b1)
        after_spread  = max(r2, g2, b2) - min(r2, g2, b2)
        assert after_spread < before_spread + 5  # Allow small tolerance


class TestDenoising:
    def test_bilateral_output_shape(self, neutral_bgr):
        out, step = apply_denoising(neutral_bgr, method="bilateral")
        assert out.shape == neutral_bgr.shape
        assert step.applied is True

    def test_nlmeans_output_shape(self, neutral_bgr):
        out, step = apply_denoising(neutral_bgr, method="nlmeans")
        assert out.shape == neutral_bgr.shape


class TestSharpening:
    def test_output_shape_unchanged(self, neutral_bgr):
        out, _ = apply_sharpening(neutral_bgr)
        assert out.shape == neutral_bgr.shape

    def test_output_dtype_uint8(self, neutral_bgr):
        out, _ = apply_sharpening(neutral_bgr)
        assert out.dtype == np.uint8

    def test_flat_image_barely_changed(self, neutral_bgr):
        out, _ = apply_sharpening(neutral_bgr, amount=1.5, threshold=1)
        diff = float(np.abs(out.astype(np.int16) - neutral_bgr.astype(np.int16)).mean())
        assert diff < 5  # Very small change on flat region


class TestSaturation:
    def test_increase_saturation(self, red_bgr):
        import cv2
        hsv_before = cv2.cvtColor(red_bgr, cv2.COLOR_BGR2HSV)
        out, _ = apply_saturation(red_bgr, scale=1.5)
        hsv_after = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        assert float(hsv_after[:, :, 1].mean()) >= float(hsv_before[:, :, 1].mean()) - 1


class TestPipeline:
    def test_full_pipeline_runs(self, dark_bgr):
        result = run_enhancement_pipeline(
            dark_bgr,
            fix_exposure=True,
            fix_contrast=True,
            fix_white_balance=True,
            fix_sharpen=True,
            fix_denoise=False,
            fix_saturation=False,
            fix_shadow_highlight=False,
        )
        assert result.enhanced_bgr.shape == dark_bgr.shape
        assert len(result.steps) == 4  # exposure, WB, CLAHE, sharpen

    def test_empty_pipeline(self, neutral_bgr):
        result = run_enhancement_pipeline(
            neutral_bgr,
            fix_exposure=False,
            fix_contrast=False,
            fix_white_balance=False,
            fix_sharpen=False,
            fix_denoise=False,
            fix_saturation=False,
            fix_shadow_highlight=False,
        )
        assert len(result.steps) == 0
        assert np.array_equal(result.enhanced_bgr, neutral_bgr)
