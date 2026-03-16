"""
enhancement.py — Image enhancement pipeline.

Each function accepts a BGR NumPy array, applies one correction method,
and returns the enhanced BGR array together with a human-readable log entry
describing what was done and why.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from src import config as cfg
from src.utils import bgr_to_gray, bgr_to_hsv, BGRImage


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class EnhancementStep:
    """Records one applied enhancement."""
    name:        str
    applied:     bool
    reason:      str
    before_stat: float | None = None
    after_stat:  float | None = None
    unit:        str = ""


@dataclass
class EnhancementResult:
    """Container returned by the enhancement pipeline."""
    enhanced_bgr: BGRImage
    steps:        list[EnhancementStep]


# ---------------------------------------------------------------------------
# 1. Exposure/Gamma correction
# ---------------------------------------------------------------------------

def apply_gamma_correction(bgr: BGRImage, gamma: float) -> BGRImage:
    """
    Apply gamma correction using a lookup table (LUT) for efficiency.

    A gamma < 1.0 brightens; gamma > 1.0 darkens.
    The output pixel value is: out = (in / 255)^(1/gamma) * 255 .

    Parameters
    ----------
    bgr   : Input BGR image.
    gamma : Desired gamma value (positive float, not zero).
    """
    if gamma <= 0:
        raise ValueError("gamma must be a positive number.")
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(bgr, table)


def correct_exposure(
    bgr: BGRImage,
    target_mean: float = 128.0,
    gamma_override: float | None = None,
) -> tuple[BGRImage, EnhancementStep]:
    """
    Adaptively correct exposure via gamma.

    If *gamma_override* is given it is used directly; otherwise gamma is
    estimated so that the current grayscale mean maps to *target_mean*
    using the relationship:  target = mean^(1/gamma)  →  gamma = log(mean)/log(target).
    """
    gray = bgr_to_gray(bgr)
    before_mean = float(np.mean(gray))

    if gamma_override is not None:
        gamma = gamma_override
    else:
        # Avoid log(0) — clamp mean to [1, 254]
        clamped_mean = float(np.clip(before_mean, 1, 254))
        clamped_target = float(np.clip(target_mean, 1, 254))
        try:
            gamma = np.log(clamped_mean / 255.0) / np.log(clamped_target / 255.0)
            gamma = float(np.clip(gamma, 0.2, 5.0))
        except (ValueError, ZeroDivisionError):
            gamma = 1.0

    out = apply_gamma_correction(bgr, gamma)
    after_mean = float(np.mean(bgr_to_gray(out)))

    applied = abs(gamma - 1.0) > 0.05
    return out, EnhancementStep(
        name="Exposure / Gamma Correction",
        applied=applied,
        reason=f"Gamma={gamma:.3f} applied to shift mean brightness from "
               f"{before_mean:.1f} → {after_mean:.1f}.",
        before_stat=round(before_mean, 1),
        after_stat=round(after_mean, 1),
        unit="mean brightness (0–255)",
    )


# ---------------------------------------------------------------------------
# 2. Contrast — CLAHE
# ---------------------------------------------------------------------------

def apply_clahe(
    bgr: BGRImage,
    clip_limit: float = cfg.DEFAULT_CLAHE_CLIP,
    tile_size: int = cfg.DEFAULT_CLAHE_TILE,
) -> tuple[BGRImage, EnhancementStep]:
    """
    Apply Contrast Limited Adaptive Histogram Equalisation (CLAHE) to the
    L channel of the LAB colour space, then convert back.

    CLAHE operates per-tile and clips the histogram to prevent noise
    amplification — an improvement over plain global HE.
    """
    gray = bgr_to_gray(bgr)
    before_std = float(np.std(gray.astype(np.float32)))

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size),
    )
    l_eq = clahe.apply(l_ch)
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    out = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    after_std = float(np.std(bgr_to_gray(out).astype(np.float32)))

    return out, EnhancementStep(
        name="CLAHE Contrast Enhancement",
        applied=True,
        reason=(
            f"CLAHE (clip={clip_limit}, tile={tile_size}×{tile_size}) applied "
            "to L-channel of LAB image."
        ),
        before_stat=round(before_std, 2),
        after_stat=round(after_std, 2),
        unit="grayscale std-dev",
    )


# ---------------------------------------------------------------------------
# 3. Histogram stretching (global)
# ---------------------------------------------------------------------------

def apply_histogram_stretching(bgr: BGRImage) -> tuple[BGRImage, EnhancementStep]:
    """
    Linear contrast stretching using the 1st and 99th percentile values.

    Clips 1 % of the darkest and 1 % of the brightest pixels to avoid
    outlier influence.
    """
    gray = bgr_to_gray(bgr)
    p1  = int(np.percentile(gray, 1))
    p99 = int(np.percentile(gray, 99))

    channels = cv2.split(bgr)
    stretched = []
    for ch in channels:
        stretched.append(np.clip(
            (ch.astype(np.float32) - p1) / max(p99 - p1, 1) * 255,
            0, 255,
        ).astype(np.uint8))
    out = cv2.merge(stretched)

    return out, EnhancementStep(
        name="Histogram Stretching",
        applied=True,
        reason=(
            f"Linear stretch from [{p1}, {p99}] to [0, 255] on each channel."
        ),
        before_stat=float(p1),
        after_stat=float(p99),
        unit="1st–99th percentile range",
    )


# ---------------------------------------------------------------------------
# 4. White balance — grey-world correction
# ---------------------------------------------------------------------------

def apply_white_balance(bgr: BGRImage) -> tuple[BGRImage, EnhancementStep]:
    """
    Grey-world white balance correction.

    Scale each BGR channel so that its mean equals the overall pixel mean.
    This is a simple but effective first-order colour cast removal technique.

    Limitation: fails on images with a dominant single hue.
    """
    b, g, r = cv2.split(bgr.astype(np.float32))
    overall_mean = (b.mean() + g.mean() + r.mean()) / 3.0
    b_scale = overall_mean / max(b.mean(), 1e-6)
    g_scale = overall_mean / max(g.mean(), 1e-6)
    r_scale = overall_mean / max(r.mean(), 1e-6)

    b_c = np.clip(b * b_scale, 0, 255).astype(np.uint8)
    g_c = np.clip(g * g_scale, 0, 255).astype(np.uint8)
    r_c = np.clip(r * r_scale, 0, 255).astype(np.uint8)
    out = cv2.merge([b_c, g_c, r_c])

    return out, EnhancementStep(
        name="Grey-World White Balance",
        applied=True,
        reason=(
            f"Channel scales: R×{r_scale:.3f}, G×{g_scale:.3f}, B×{b_scale:.3f}. "
            "Each channel mean equalised to the overall mean."
        ),
        before_stat=round(overall_mean, 1),
        after_stat=None,
        unit="mean brightness (0–255)",
    )


# ---------------------------------------------------------------------------
# 5. Denoising
# ---------------------------------------------------------------------------

def apply_denoising(
    bgr: BGRImage,
    method: str = "bilateral",
    h: int = cfg.DEFAULT_DENOISE_H,
) -> tuple[BGRImage, EnhancementStep]:
    """
    Reduce noise using either bilateral filtering or Non-Local Means.

    Parameters
    ----------
    method : "bilateral" (faster, preserves edges) or "nlmeans" (stronger).
    h      : Filter strength for both methods.
    """
    if method == "nlmeans":
        out = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 7, 21)
        method_desc = f"Non-Local Means (h={h})"
    else:  # bilateral
        out = cv2.bilateralFilter(bgr, d=9, sigmaColor=h * 7, sigmaSpace=h * 7)
        method_desc = f"Bilateral filter (d=9, sigma={h*7})"

    return out, EnhancementStep(
        name="Denoising",
        applied=True,
        reason=f"{method_desc} applied to suppress high-frequency noise.",
        before_stat=None,
        after_stat=None,
        unit="",
    )


# ---------------------------------------------------------------------------
# 6. Sharpening — Unsharp Masking
# ---------------------------------------------------------------------------

def apply_sharpening(
    bgr: BGRImage,
    amount: float = cfg.DEFAULT_SHARPEN_AMOUNT,
    radius: int = 1,
    threshold: int = 3,
) -> tuple[BGRImage, EnhancementStep]:
    """
    Unsharp Masking (USM) sharpening.

    Method
    ------
    1. Blur the image with a Gaussian kernel (radius).
    2. Compute the residual: residual = original − blurred.
    3. Add a scaled residual back: sharp = original + amount × residual.
    4. Only add residual where |residual| > threshold (avoids noise amplification).

    Parameters
    ----------
    amount    : Sharpening strength multiplier (typically 1.0–2.5).
    radius    : Gaussian blur kernel radius (pixels).
    threshold : Minimum change to apply sharpening (avoids noise on flat regions).
    """
    kernel_size = 2 * radius + 1
    blurred = cv2.GaussianBlur(bgr.astype(np.float32), (kernel_size, kernel_size), 0)
    residual = bgr.astype(np.float32) - blurred
    residual[np.abs(residual) < threshold] = 0
    out = np.clip(bgr.astype(np.float32) + amount * residual, 0, 255).astype(np.uint8)

    return out, EnhancementStep(
        name="Unsharp Masking Sharpening",
        applied=True,
        reason=(
            f"USM: amount={amount}, radius={radius}px, threshold={threshold}. "
            "Edges enhanced without amplifying flat-area noise."
        ),
        before_stat=None,
        after_stat=None,
        unit="",
    )


# ---------------------------------------------------------------------------
# 7. Saturation adjustment
# ---------------------------------------------------------------------------

def apply_saturation(
    bgr: BGRImage,
    scale: float = 1.3,
) -> tuple[BGRImage, EnhancementStep]:
    """
    Scale the S channel in HSV to adjust colour saturation.

    A scale > 1 increases saturation; scale < 1 desaturates.
    Values are clipped to [0, 255] to prevent overflow.
    """
    hsv = bgr_to_hsv(bgr).astype(np.float32)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    before_mean = float(s_ch.mean())
    s_ch = np.clip(s_ch * scale, 0, 255)
    out_hsv = cv2.merge([h_ch, s_ch, v_ch]).astype(np.uint8)
    out = cv2.cvtColor(out_hsv, cv2.COLOR_HSV2BGR)
    after_mean = float(cv2.split(bgr_to_hsv(out).astype(np.float32))[1].mean())

    return out, EnhancementStep(
        name="Saturation Adjustment",
        applied=True,
        reason=f"S-channel scaled by {scale:.2f}: {before_mean:.1f} → {after_mean:.1f}.",
        before_stat=round(before_mean, 1),
        after_stat=round(after_mean, 1),
        unit="mean HSV saturation (0–255)",
    )


# ---------------------------------------------------------------------------
# 8. Highlight / Shadow recovery (simple tone-mapping approach)
# ---------------------------------------------------------------------------

def apply_shadow_highlight_recovery(
    bgr: BGRImage,
    shadow_lift: float = 0.15,
    highlight_compress: float = 0.85,
) -> tuple[BGRImage, EnhancementStep]:
    """
    Perform simple shadow lift and highlight compression using a tone curve.

    The idea is to remap the [0, 255] range with a sigmoid-like or piecewise
    linear tone curve that lifts very dark tones (aids shadow detail) and
    compresses very bright tones (preserves highlight detail).

    Parameters
    ----------
    shadow_lift          : Fractional lift for the 0 value (0 = no lift).
    highlight_compress   : Fractional cap for the 255 value (1 = no compression).
    """
    # Build a small LUT
    table = np.arange(256, dtype=np.float32) / 255.0
    # Linear remap: [0,1] → [shadow_lift, highlight_compress]
    table = table * (highlight_compress - shadow_lift) + shadow_lift
    table = np.clip(table * 255, 0, 255).astype(np.uint8)
    out = cv2.LUT(bgr, table)

    return out, EnhancementStep(
        name="Shadow/Highlight Recovery",
        applied=True,
        reason=(
            f"Tone curve: shadows lifted to {shadow_lift*255:.0f}, "
            f"highlights compressed to {highlight_compress*255:.0f}."
        ),
        before_stat=None,
        after_stat=None,
        unit="",
    )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_enhancement_pipeline(
    bgr: BGRImage,
    *,
    fix_exposure:         bool = True,
    fix_contrast:         bool = True,
    fix_white_balance:    bool = True,
    fix_denoise:          bool = False,
    fix_sharpen:          bool = True,
    fix_saturation:       bool = False,
    fix_shadow_highlight: bool = False,
    # Manual overrides
    gamma_override:       float | None = None,
    denoise_method:       str = "bilateral",
    denoise_strength:     int = cfg.DEFAULT_DENOISE_H,
    clahe_clip:           float = cfg.DEFAULT_CLAHE_CLIP,
    clahe_tile:           int = cfg.DEFAULT_CLAHE_TILE,
    sharpen_amount:       float = cfg.DEFAULT_SHARPEN_AMOUNT,
    saturation_scale:     float = 1.3,
    shadow_lift:          float = 0.15,
    highlight_compress:   float = 0.85,
) -> EnhancementResult:
    """
    Run selected enhancements sequentially and return the result.

    The pipeline order matters:
    1. Exposure (gamma) first — correct overall brightness.
    2. White balance — remove colour cast on corrected brightness.
    3. Contrast (CLAHE) — expand tonal range after WB.
    4. Denoising — remove noise before sharpening.
    5. Sharpening — enhance detail after denoising.
    6. Saturation — colour vibrancy last.
    7. Shadow/Highlight — optional final tone adjustment.
    """
    current = bgr.copy()
    steps: list[EnhancementStep] = []

    if fix_exposure:
        current, step = correct_exposure(current, gamma_override=gamma_override)
        steps.append(step)

    if fix_white_balance:
        current, step = apply_white_balance(current)
        steps.append(step)

    if fix_contrast:
        current, step = apply_clahe(current, clip_limit=clahe_clip, tile_size=clahe_tile)
        steps.append(step)

    if fix_denoise:
        current, step = apply_denoising(current, method=denoise_method, h=denoise_strength)
        steps.append(step)

    if fix_sharpen:
        current, step = apply_sharpening(current, amount=sharpen_amount)
        steps.append(step)

    if fix_saturation:
        current, step = apply_saturation(current, scale=saturation_scale)
        steps.append(step)

    if fix_shadow_highlight:
        current, step = apply_shadow_highlight_recovery(
            current, shadow_lift=shadow_lift, highlight_compress=highlight_compress
        )
        steps.append(step)

    return EnhancementResult(enhanced_bgr=current, steps=steps)
