"""
config.py — Central configuration for the Intelligent Photo Editing Assistant.

All tuneable thresholds, weights, and constants live here so that they can
be reviewed and adjusted without touching business logic.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")
MAX_IMAGE_DIMENSION: int = 4096   # pixels; larger images are resized down
MAX_FILE_SIZE_MB: float = 50.0    # megabytes

# ---------------------------------------------------------------------------
# Analysis thresholds
# ---------------------------------------------------------------------------

# Brightness (0–255 grayscale mean)
BRIGHTNESS_LOW: float = 60.0
BRIGHTNESS_HIGH: float = 195.0

# Contrast (grayscale standard deviation)
CONTRAST_LOW: float = 30.0
CONTRAST_HIGH: float = 80.0

# Sharpness (variance of Laplacian)
SHARPNESS_BLURRY: float = 80.0
SHARPNESS_SHARP: float = 300.0

# Noise (estimated sigma, 0–255 scale)
NOISE_LOW: float = 3.0
NOISE_HIGH: float = 15.0

# Saturation (HSV S-channel mean, 0–255 scale)
SATURATION_LOW: float = 40.0
SATURATION_HIGH: float = 180.0

# Dynamic range — fraction of histogram bins that must be non-zero
DYNAMIC_RANGE_LOW: float = 0.40

# Clipping fraction threshold (fraction of pixels in shadow/highlight bins)
CLIP_FRACTION_THRESHOLD: float = 0.02   # 2 %

# White balance — maximum acceptable channel deviation (grey-world)
WB_DEVIATION_THRESHOLD: float = 15.0   # abs difference from mean, 0–255

# ---------------------------------------------------------------------------
# Scoring weights (must sum to 100)
# ---------------------------------------------------------------------------
SCORE_WEIGHTS: dict[str, int] = {
    "exposure":       15,
    "contrast":       10,
    "histogram":      10,
    "sharpness":      15,
    "noise":          10,
    "white_balance":  10,
    "saturation":      5,
    "dynamic_range":  10,
    "composition":    15,
}

# Score interpretation bands
SCORE_BANDS: list[tuple[int, str]] = [
    (90, "Excellent"),
    (75, "Good"),
    (60, "Average"),
    (40, "Weak"),
    (0,  "Poor"),
]

# ---------------------------------------------------------------------------
# Enhancement defaults
# ---------------------------------------------------------------------------
DEFAULT_GAMMA: float = 1.0
DEFAULT_CLAHE_CLIP: float = 2.0
DEFAULT_CLAHE_TILE: int = 8
DEFAULT_DENOISE_H: int = 10          # fastNlMeans filter strength
DEFAULT_SHARPEN_AMOUNT: float = 1.5  # unsharp-mask amount
DEFAULT_SATURATION_SCALE: float = 1.0

# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------
THIRDS_TOLERANCE: float = 0.12   # fraction of image dimension
