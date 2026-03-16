"""
analysis.py — Image quality analysis engine.

Each public function accepts a BGR NumPy array and returns an
:class:`AnalysisResult` for a single metric, or the top-level
:func:`analyse_image` collects all metrics at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from scipy.signal import convolve2d

from src import config as cfg
from src.utils import bgr_to_gray, bgr_to_hsv, bgr_to_lab, BGRImage, GrayImage


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    """
    Result of one image-quality metric.

    Attributes
    ----------
    name        : Human-readable metric name.
    value       : Measured numeric value (unit depends on metric).
    label       : Short classification, e.g. "Underexposed".
    score       : Sub-score in [0, 100] for the scoring engine.
    explanation : Why this label / score was assigned.
    suggestion  : What enhancement could be applied (or "None needed").
    extra       : Optional dict for histogram arrays, etc.
    """
    name:        str
    value:       float
    label:       str
    score:       float
    explanation: str
    suggestion:  str
    extra:       dict = field(default_factory=dict)


@dataclass
class FullAnalysis:
    """Container for every metric and derived values."""
    brightness:    MetricResult
    contrast:      MetricResult
    histogram:     MetricResult
    sharpness:     MetricResult
    noise:         MetricResult
    white_balance: MetricResult
    saturation:    MetricResult
    dynamic_range: MetricResult
    composition:   Optional[MetricResult] = None   # filled in separately

    def as_dict(self) -> dict[str, MetricResult]:
        d = {
            "exposure":      self.brightness,
            "contrast":      self.contrast,
            "histogram":     self.histogram,
            "sharpness":     self.sharpness,
            "noise":         self.noise,
            "white_balance": self.white_balance,
            "saturation":    self.saturation,
            "dynamic_range": self.dynamic_range,
        }
        if self.composition:
            d["composition"] = self.composition
        return d


# ---------------------------------------------------------------------------
# 1. Brightness / Exposure
# ---------------------------------------------------------------------------

def analyse_brightness(bgr: BGRImage) -> MetricResult:
    """
    Compute grayscale mean and classify exposure.

    Method
    ------
    Convert to grayscale (perceptual weighting via cv2.COLOR_BGR2GRAY)
    and take the mean pixel intensity on a 0–255 scale.

    Thresholds (from config)
    ------------------------
    < BRIGHTNESS_LOW   → Underexposed
    > BRIGHTNESS_HIGH  → Overexposed
    else               → Balanced
    """
    gray = bgr_to_gray(bgr)
    mean_val = float(np.mean(gray))
    p10 = float(np.percentile(gray, 10))
    p90 = float(np.percentile(gray, 90))

    if mean_val < cfg.BRIGHTNESS_LOW:
        label = "Underexposed"
        explanation = (
            f"Mean brightness is {mean_val:.1f}/255, which is below the "
            f"underexposure threshold of {cfg.BRIGHTNESS_LOW}. "
            "Shadows will dominate; detail in dark areas may be lost."
        )
        suggestion = "Apply gamma correction (γ < 1) or increase linear brightness."
        score = (mean_val / cfg.BRIGHTNESS_LOW) * 100
    elif mean_val > cfg.BRIGHTNESS_HIGH:
        label = "Overexposed"
        explanation = (
            f"Mean brightness is {mean_val:.1f}/255, above the overexposure "
            f"threshold of {cfg.BRIGHTNESS_HIGH}. Highlights may be clipped."
        )
        suggestion = "Apply gamma correction (γ > 1) or reduce brightness."
        score = ((255 - mean_val) / (255 - cfg.BRIGHTNESS_HIGH)) * 100
    else:
        label = "Balanced"
        explanation = (
            f"Mean brightness is {mean_val:.1f}/255, well within the ideal "
            f"range [{cfg.BRIGHTNESS_LOW}, {cfg.BRIGHTNESS_HIGH}]."
        )
        suggestion = "No exposure correction needed."
        mid = (cfg.BRIGHTNESS_LOW + cfg.BRIGHTNESS_HIGH) / 2
        score = 100 - abs(mean_val - mid) / (mid - cfg.BRIGHTNESS_LOW) * 15

    score = float(np.clip(score, 0, 100))

    return MetricResult(
        name="Brightness / Exposure",
        value=round(mean_val, 2),
        label=label,
        score=round(score, 1),
        explanation=explanation,
        suggestion=suggestion,
        extra={"p10": round(p10, 2), "p90": round(p90, 2)},
    )


# ---------------------------------------------------------------------------
# 2. Contrast
# ---------------------------------------------------------------------------

def analyse_contrast(bgr: BGRImage) -> MetricResult:
    """
    Compute global contrast as the standard deviation of grayscale luminance.

    Standard deviation is a well-established measure of global contrast
    (Hasler & Suesstrunk, 2003).  A low stddev means most pixels crowd
    around a single intensity — the key symptom of a flat, low-contrast image.
    """
    gray = bgr_to_gray(bgr)
    std_val = float(np.std(gray.astype(np.float32)))

    if std_val < cfg.CONTRAST_LOW:
        label = "Low contrast"
        explanation = (
            f"Grayscale standard deviation is {std_val:.1f}, below the "
            f"low-contrast threshold of {cfg.CONTRAST_LOW}. The image looks flat."
        )
        suggestion = "Apply CLAHE or global histogram stretching."
        score = (std_val / cfg.CONTRAST_LOW) * 70
    elif std_val > cfg.CONTRAST_HIGH:
        label = "High contrast"
        explanation = (
            f"Grayscale std dev is {std_val:.1f}, which is quite high. "
            "The image has stark tonal differences but may clip shadows/highlights."
        )
        suggestion = "Consider highlight/shadow recovery or mild gamma roll-off."
        score = 85.0
    else:
        label = "Good contrast"
        explanation = (
            f"Grayscale std dev is {std_val:.1f} — inside the ideal range "
            f"[{cfg.CONTRAST_LOW}, {cfg.CONTRAST_HIGH}]."
        )
        suggestion = "No contrast enhancement needed."
        score = 100.0

    score = float(np.clip(score, 0, 100))

    return MetricResult(
        name="Contrast",
        value=round(std_val, 2),
        label=label,
        score=round(score, 1),
        explanation=explanation,
        suggestion=suggestion,
    )


# ---------------------------------------------------------------------------
# 3. Histogram Health
# ---------------------------------------------------------------------------

def analyse_histogram(bgr: BGRImage) -> MetricResult:
    """
    Check shadow and highlight clipping in the luminance histogram.

    Clipping is estimated by the fraction of pixels at or near the
    extreme bins (0–7 for shadow, 248–255 for highlights).
    """
    gray = bgr_to_gray(bgr)
    total_pixels = gray.size

    shadow_mask = gray <= 7
    highlight_mask = gray >= 248
    shadow_frac = float(shadow_mask.sum() / total_pixels)
    highlight_frac = float(highlight_mask.sum() / total_pixels)

    # Compute per-channel histograms for extra data
    hist_b = cv2.calcHist([bgr], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([bgr], [1], None, [256], [0, 256]).flatten()
    hist_r = cv2.calcHist([bgr], [2], None, [256], [0, 256]).flatten()
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

    # Build label and score
    issues = []
    if shadow_frac > cfg.CLIP_FRACTION_THRESHOLD:
        issues.append(f"shadow clipping ({shadow_frac*100:.1f}% of pixels)")
    if highlight_frac > cfg.CLIP_FRACTION_THRESHOLD:
        issues.append(f"highlight clipping ({highlight_frac*100:.1f}% of pixels)")

    if issues:
        label = "Clipping detected"
        explanation = (
            f"Histogram clipping found: {' and '.join(issues)}. "
            "Detail in these regions may be irrecoverably lost."
        )
        suggestion = (
            "Try shadow/highlight recovery or exposure adjustment before clipping occurs."
        )
        clip_penalty = (shadow_frac + highlight_frac) / (2 * cfg.CLIP_FRACTION_THRESHOLD)
        score = float(np.clip(100 - clip_penalty * 40, 30, 100))
    else:
        label = "Healthy histogram"
        explanation = (
            f"Shadow clipping: {shadow_frac*100:.2f}%, "
            f"highlight clipping: {highlight_frac*100:.2f}% — both within tolerance."
        )
        suggestion = "No histogram correction needed."
        score = 100.0

    return MetricResult(
        name="Histogram Health",
        value=round((shadow_frac + highlight_frac) * 100, 2),
        label=label,
        score=round(score, 1),
        explanation=explanation,
        suggestion=suggestion,
        extra={
            "hist_gray": hist_gray,
            "hist_b": hist_b,
            "hist_g": hist_g,
            "hist_r": hist_r,
            "shadow_frac": shadow_frac,
            "highlight_frac": highlight_frac,
        },
    )


# ---------------------------------------------------------------------------
# 4. Sharpness / Blur
# ---------------------------------------------------------------------------

def analyse_sharpness(bgr: BGRImage) -> MetricResult:
    """
    Estimate sharpness using the variance of the Laplacian operator.

    The Laplacian is a second-order derivative filter that responds strongly
    to edges.  In a blurry image edges are smoothed away, so the variance
    of the Laplacian response is low.  This is a classical and computationally
    cheap focus measure (Pech-Pacheco et al., 2000).

    Thresholds
    ----------
    Chosen empirically on typical consumer-camera images:
      < SHARPNESS_BLURRY  → Blurry
      < SHARPNESS_SHARP   → Acceptable
      else                → Sharp
    """
    gray = bgr_to_gray(bgr)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if lap_var < cfg.SHARPNESS_BLURRY:
        label = "Blurry"
        explanation = (
            f"Laplacian variance is {lap_var:.1f}, below the blur threshold "
            f"of {cfg.SHARPNESS_BLURRY}. Motion blur or defocus is likely."
        )
        suggestion = "Apply unsharp masking or kernel-based sharpening."
        score = (lap_var / cfg.SHARPNESS_BLURRY) * 60
    elif lap_var < cfg.SHARPNESS_SHARP:
        label = "Acceptable"
        explanation = (
            f"Laplacian variance is {lap_var:.1f}; the image is somewhat sharp "
            "but could benefit from mild sharpening."
        )
        suggestion = "Light sharpening may improve perceived crispness."
        score = 60 + (lap_var - cfg.SHARPNESS_BLURRY) / (cfg.SHARPNESS_SHARP - cfg.SHARPNESS_BLURRY) * 30
    else:
        label = "Sharp"
        explanation = (
            f"Laplacian variance is {lap_var:.1f}, above the sharp threshold "
            f"of {cfg.SHARPNESS_SHARP}. The image is well-focused."
        )
        suggestion = "No sharpening required."
        score = min(100.0, 90 + (lap_var - cfg.SHARPNESS_SHARP) / cfg.SHARPNESS_SHARP * 10)

    score = float(np.clip(score, 0, 100))

    return MetricResult(
        name="Sharpness / Blur",
        value=round(lap_var, 2),
        label=label,
        score=round(score, 1),
        explanation=explanation,
        suggestion=suggestion,
    )


# ---------------------------------------------------------------------------
# 5. Noise Estimation
# ---------------------------------------------------------------------------

def _estimate_noise_sigma(gray: GrayImage) -> float:
    """
    Estimate noise level using a high-frequency residual method.

    The image is smoothed with a 3×3 blur and the residual (original minus
    blurred) is taken.  The standard deviation of this residual is a proxy
    for the RMS noise amplitude.  This is a lightweight but reasonable
    estimator used in many image quality tools.
    """
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
    residual = gray.astype(np.float32) - blurred
    return float(np.std(residual))


def analyse_noise(bgr: BGRImage) -> MetricResult:
    """Estimate noise level from high-frequency residuals."""
    gray = bgr_to_gray(bgr)
    sigma = _estimate_noise_sigma(gray)

    if sigma < cfg.NOISE_LOW:
        label = "Low noise"
        explanation = (
            f"Estimated noise σ ≈ {sigma:.2f} — the image appears clean."
        )
        suggestion = "No denoising required."
        score = 100.0
    elif sigma < cfg.NOISE_HIGH:
        label = "Moderate noise"
        explanation = (
            f"Estimated noise σ ≈ {sigma:.2f}. Mild noise is present. "
            "This could stem from a high ISO or low-light capture."
        )
        suggestion = "Apply bilateral filter or fastNlMeansDenoising."
        score = 100 - (sigma - cfg.NOISE_LOW) / (cfg.NOISE_HIGH - cfg.NOISE_LOW) * 40
    else:
        label = "High noise"
        explanation = (
            f"Estimated noise σ ≈ {sigma:.2f} — significant noise detected. "
            "Denoising is strongly recommended."
        )
        suggestion = "Use fastNlMeansDenoising with a moderate filter strength."
        score = max(0, 60 - (sigma - cfg.NOISE_HIGH) * 3)

    score = float(np.clip(score, 0, 100))

    return MetricResult(
        name="Noise Level",
        value=round(sigma, 3),
        label=label,
        score=round(score, 1),
        explanation=explanation,
        suggestion=suggestion,
    )


# ---------------------------------------------------------------------------
# 6. White Balance / Colour Cast
# ---------------------------------------------------------------------------

def analyse_white_balance(bgr: BGRImage) -> MetricResult:
    """
    Detect colour cast using the grey-world assumption.

    The grey-world assumption states that for a 'neutral' image the mean of
    each RGB channel should be equal.  A significant deviation of one channel
    from the global mean indicates a colour cast.

    Limitations
    -----------
    The grey-world assumption fails for images with a dominant single colour
    (e.g. a green forest photo).  The result should be treated as an
    indicator, not a guarantee.
    """
    # Work in RGB for intuitive channel naming
    b, g, r = cv2.split(bgr.astype(np.float32))
    mean_b, mean_g, mean_r = float(b.mean()), float(g.mean()), float(r.mean())
    overall_mean = (mean_r + mean_g + mean_b) / 3.0

    dev_r = mean_r - overall_mean
    dev_g = mean_g - overall_mean
    dev_b = mean_b - overall_mean
    max_dev = max(abs(dev_r), abs(dev_g), abs(dev_b))

    # Classify cast direction
    cast_parts = []
    if dev_r > cfg.WB_DEVIATION_THRESHOLD:
        cast_parts.append("warm (red/orange)")
    if dev_b > cfg.WB_DEVIATION_THRESHOLD:
        cast_parts.append("cool (blue)")
    if dev_g > cfg.WB_DEVIATION_THRESHOLD:
        cast_parts.append("green")
    if dev_r < -cfg.WB_DEVIATION_THRESHOLD:
        cast_parts.append("cyan")
    if dev_b < -cfg.WB_DEVIATION_THRESHOLD:
        cast_parts.append("yellow/warm")
    if dev_g < -cfg.WB_DEVIATION_THRESHOLD:
        cast_parts.append("magenta")

    if cast_parts:
        label = f"Cast: {', '.join(cast_parts)}"
        explanation = (
            f"Grey-world analysis: R={mean_r:.1f}, G={mean_g:.1f}, B={mean_b:.1f} "
            f"(overall mean={overall_mean:.1f}). "
            f"Maximum channel deviation: {max_dev:.1f} — "
            f"indicates a {', '.join(cast_parts)} cast."
        )
        suggestion = "Apply grey-world white balance correction to equalise channel means."
        score = max(0, 100 - (max_dev / cfg.WB_DEVIATION_THRESHOLD) * 25)
    else:
        label = "Neutral"
        explanation = (
            f"Grey-world analysis: R={mean_r:.1f}, G={mean_g:.1f}, B={mean_b:.1f} "
            f"(overall mean={overall_mean:.1f}). "
            "Channels are balanced — no strong cast detected."
        )
        suggestion = "No white balance correction needed."
        score = 100.0

    score = float(np.clip(score, 0, 100))

    return MetricResult(
        name="White Balance",
        value=round(max_dev, 2),
        label=label,
        score=round(score, 1),
        explanation=explanation,
        suggestion=suggestion,
        extra={
            "mean_r": round(mean_r, 2),
            "mean_g": round(mean_g, 2),
            "mean_b": round(mean_b, 2),
        },
    )


# ---------------------------------------------------------------------------
# 7. Saturation
# ---------------------------------------------------------------------------

def analyse_saturation(bgr: BGRImage) -> MetricResult:
    """
    Measure colour saturation via the S channel of HSV.

    HSV saturation (0–255) indicates the vividness of hues.  Low saturation
    makes images look washed-out or grey; high saturation makes them look
    garish.
    """
    hsv = bgr_to_hsv(bgr)
    s_channel = hsv[:, :, 1].astype(np.float32)
    mean_sat = float(s_channel.mean())

    if mean_sat < cfg.SATURATION_LOW:
        label = "Undersaturated"
        explanation = (
            f"Mean HSV saturation is {mean_sat:.1f}/255 — below {cfg.SATURATION_LOW}. "
            "Colours appear washed-out or grey."
        )
        suggestion = "Increase saturation in HSV or use a vibrance boost."
        score = (mean_sat / cfg.SATURATION_LOW) * 70
    elif mean_sat > cfg.SATURATION_HIGH:
        label = "Oversaturated"
        explanation = (
            f"Mean HSV saturation is {mean_sat:.1f}/255 — above {cfg.SATURATION_HIGH}. "
            "Colours may look unnatural or garish."
        )
        suggestion = "Reduce saturation slightly for a more natural look."
        score = 75.0
    else:
        label = "Balanced"
        explanation = (
            f"Mean HSV saturation is {mean_sat:.1f}/255 — within the ideal "
            f"range [{cfg.SATURATION_LOW}, {cfg.SATURATION_HIGH}]."
        )
        suggestion = "No saturation change needed."
        score = 100.0

    score = float(np.clip(score, 0, 100))

    return MetricResult(
        name="Saturation",
        value=round(mean_sat, 2),
        label=label,
        score=round(score, 1),
        explanation=explanation,
        suggestion=suggestion,
    )


# ---------------------------------------------------------------------------
# 8. Dynamic Range
# ---------------------------------------------------------------------------

def analyse_dynamic_range(bgr: BGRImage) -> MetricResult:
    """
    Estimate the effective dynamic range of the image.

    Method
    ------
    * Build the grayscale histogram.
    * Count the fraction of non-zero bins (bins with ≥ 1 pixel).
    * A higher fraction indicates a wider tonal spread — better dynamic range.

    Additionally, compute the 1st–99th percentile luminance spread as an
    alternative / complementary measure.
    """
    gray = bgr_to_gray(bgr)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    non_zero_ratio = float(np.count_nonzero(hist) / 256)
    p1  = float(np.percentile(gray, 1))
    p99 = float(np.percentile(gray, 99))
    percentile_spread = p99 - p1

    if non_zero_ratio < cfg.DYNAMIC_RANGE_LOW:
        label = "Narrow dynamic range"
        explanation = (
            f"Only {non_zero_ratio*100:.1f}% of histogram bins are occupied "
            f"(threshold: {cfg.DYNAMIC_RANGE_LOW*100:.0f}%). "
            "The tonal range is compressed — the image may look flat or banded."
        )
        suggestion = "Histogram stretching or CLAHE can expand tonal range."
        score = (non_zero_ratio / cfg.DYNAMIC_RANGE_LOW) * 70
    else:
        label = "Wide dynamic range"
        explanation = (
            f"{non_zero_ratio*100:.1f}% of histogram bins are occupied "
            f"and the 1st–99th percentile luminance spread is {percentile_spread:.1f}. "
            "The image captures a good range of tones."
        )
        suggestion = "No dynamic range intervention required."
        score = min(100.0, 80 + non_zero_ratio * 20)

    score = float(np.clip(score, 0, 100))

    return MetricResult(
        name="Dynamic Range",
        value=round(non_zero_ratio * 100, 1),
        label=label,
        score=round(score, 1),
        explanation=explanation,
        suggestion=suggestion,
        extra={"percentile_spread": round(percentile_spread, 1), "p1": p1, "p99": p99},
    )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def analyse_image(bgr: BGRImage) -> FullAnalysis:
    """
    Run all analysis metrics on a BGR image.

    Returns a :class:`FullAnalysis` containing one :class:`MetricResult`
    per metric.  Composition analysis is *not* run here — it lives in
    ``composition.py`` and is attached separately.
    """
    return FullAnalysis(
        brightness=analyse_brightness(bgr),
        contrast=analyse_contrast(bgr),
        histogram=analyse_histogram(bgr),
        sharpness=analyse_sharpness(bgr),
        noise=analyse_noise(bgr),
        white_balance=analyse_white_balance(bgr),
        saturation=analyse_saturation(bgr),
        dynamic_range=analyse_dynamic_range(bgr),
    )
