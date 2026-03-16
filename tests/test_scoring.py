"""
test_scoring.py — Unit tests for the scoring engine.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis import analyse_image
from src.scoring import compute_score, _band
from src import config as cfg


def _solid(r: int, g: int, b: int, h: int = 80, w: int = 80) -> np.ndarray:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = b
    arr[:, :, 1] = g
    arr[:, :, 2] = r
    return arr


class TestBandLabels:
    def test_excellent(self):
        assert _band(95) == "Excellent"

    def test_good(self):
        assert _band(80) == "Good"

    def test_average(self):
        assert _band(65) == "Average"

    def test_weak(self):
        assert _band(50) == "Weak"

    def test_poor(self):
        assert _band(30) == "Poor"

    def test_boundary_exactly_90(self):
        assert _band(90) == "Excellent"

    def test_boundary_exactly_0(self):
        assert _band(0) == "Poor"


class TestScoreReport:
    def test_total_in_range(self):
        bgr = _solid(128, 128, 128)
        analysis = analyse_image(bgr)
        report = compute_score(analysis)
        assert 0 <= report.total_score <= 100

    def test_weights_sum_100(self):
        total = sum(cfg.SCORE_WEIGHTS.values())
        assert total == 100

    def test_sub_scores_all_present(self):
        bgr = _solid(128, 128, 128)
        analysis = analyse_image(bgr)
        report = compute_score(analysis)
        for key in cfg.SCORE_WEIGHTS:
            assert key in report.sub_scores, f"Missing sub-score for '{key}'"

    def test_dark_image_lower_score(self):
        dark_bgr = _solid(10, 10, 10)
        neutral_bgr = _solid(128, 128, 128)
        dark_analysis = analyse_image(dark_bgr)
        neutral_analysis = analyse_image(neutral_bgr)
        dark_score = compute_score(dark_analysis).total_score
        neutral_score = compute_score(neutral_analysis).total_score
        assert dark_score < neutral_score

    def test_reductions_list_type(self):
        bgr = _solid(10, 10, 10)
        analysis = analyse_image(bgr)
        report = compute_score(analysis)
        assert isinstance(report.reductions, list)

    def test_band_matches_score(self):
        bgr = _solid(128, 128, 128)
        analysis = analyse_image(bgr)
        report = compute_score(analysis)
        expected_band = _band(report.total_score)
        assert report.band == expected_band

    def test_weighted_scores_sum_equals_total(self):
        bgr = _solid(128, 128, 128)
        analysis = analyse_image(bgr)
        report = compute_score(analysis)
        computed_total = round(sum(report.weighted_scores.values()), 1)
        assert abs(computed_total - report.total_score) < 0.5
