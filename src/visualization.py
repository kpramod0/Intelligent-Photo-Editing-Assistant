"""
visualization.py — Chart and image overlay generation for the Streamlit UI.

All functions return Matplotlib figures or annotated NumPy images ready for
display with st.pyplot() or st.image().
"""

from __future__ import annotations

from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from src.utils import bgr_to_rgb, bgr_to_gray, BGRImage
from src.analysis import FullAnalysis
from src.scoring import ScoreReport


# ---------------------------------------------------------------------------
# Histogram plots
# ---------------------------------------------------------------------------

def plot_histogram(bgr: BGRImage, title: str = "Colour Histogram") -> plt.Figure:
    """
    Plot BGR channel histograms and a luminance histogram on one figure.

    Returns a Matplotlib Figure (caller must close it to free memory).
    """
    fig, (ax_lum, ax_rgb) = plt.subplots(
        1, 2, figsize=(10, 3.5), facecolor="#1a1a2e"
    )

    for ax in (ax_lum, ax_rgb):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    gray = bgr_to_gray(bgr)
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    ax_lum.fill_between(range(256), hist_gray, color="#e0e0e0", alpha=0.7)
    ax_lum.set_title("Luminance Histogram", color="white", fontsize=9)
    ax_lum.set_xlabel("Intensity (0–255)", color="white", fontsize=8)
    ax_lum.set_ylabel("Pixel Count", color="white", fontsize=8)
    ax_lum.set_xlim(0, 255)
    # Shadow / highlight shading
    ax_lum.axvspan(0, 7, alpha=0.3, color="blue", label="Shadow clip zone")
    ax_lum.axvspan(248, 255, alpha=0.3, color="red", label="Highlight clip zone")
    ax_lum.legend(fontsize=7, labelcolor="white", facecolor="#1a1a2e", edgecolor="#555")

    COLORS = [("B", "#4fc3f7"), ("G", "#81c784"), ("R", "#ef9a9a")]
    for idx, (ch_name, color) in enumerate(COLORS):
        hist = cv2.calcHist([bgr], [idx], None, [256], [0, 256]).flatten()
        ax_rgb.plot(hist, color=color, linewidth=1, label=ch_name, alpha=0.85)

    ax_rgb.set_title("RGB Channel Histograms", color="white", fontsize=9)
    ax_rgb.set_xlabel("Intensity (0–255)", color="white", fontsize=8)
    ax_rgb.set_ylabel("Pixel Count", color="white", fontsize=8)
    ax_rgb.set_xlim(0, 255)
    ax_rgb.legend(fontsize=8, labelcolor="white", facecolor="#1a1a2e", edgecolor="#555")

    fig.suptitle(title, color="white", fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Before / After comparison
# ---------------------------------------------------------------------------

def plot_before_after(
    original_bgr: BGRImage,
    enhanced_bgr: BGRImage,
    title: str = "Before / After",
) -> plt.Figure:
    """
    Side-by-side comparison of original and enhanced images.
    """
    orig_rgb = bgr_to_rgb(original_bgr)
    enh_rgb  = bgr_to_rgb(enhanced_bgr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="#1a1a2e")
    ax1.imshow(orig_rgb)
    ax1.set_title("ORIGINAL", color="white", fontsize=13, fontweight="bold", pad=8)
    ax1.axis("off")

    ax2.imshow(enh_rgb)
    ax2.set_title("ENHANCED", color="#00e5ff", fontsize=13, fontweight="bold", pad=8)
    ax2.axis("off")

    fig.suptitle(title, color="white", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Score radar / bar chart
# ---------------------------------------------------------------------------

def plot_score_bars(score_report: ScoreReport) -> plt.Figure:
    """
    Horizontal bar chart of per-metric sub-scores.
    """
    LABELS = {
        "exposure":      "Brightness / Exposure",
        "contrast":      "Contrast",
        "histogram":     "Histogram Health",
        "sharpness":     "Sharpness",
        "noise":         "Noise Level",
        "white_balance": "White Balance",
        "saturation":    "Saturation",
        "dynamic_range": "Dynamic Range",
        "composition":   "Composition",
    }

    keys   = list(score_report.sub_scores.keys())
    values = [score_report.sub_scores[k] for k in keys]
    labels = [LABELS.get(k, k) for k in keys]

    # Colour by score
    colors = ["#4caf50" if v >= 75 else ("#ff9800" if v >= 50 else "#f44336") for v in values]

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")
    bars = ax.barh(labels, values, color=colors, height=0.55, edgecolor="none")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Sub-Score (0–100)", color="white", fontsize=9)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}", va="center", color="white", fontsize=8)
    ax.set_title("Per-Metric Quality Scores", color="white", fontsize=11)
    ax.axvline(75, color="#4caf50", linestyle="--", linewidth=0.8, alpha=0.6, label="Good (75)")
    ax.axvline(50, color="#ff9800", linestyle="--", linewidth=0.8, alpha=0.6, label="Weak (50)")
    ax.legend(fontsize=7, labelcolor="white", facecolor="#1a1a2e", edgecolor="#555")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Saliency overlay
# ---------------------------------------------------------------------------

def overlay_saliency(bgr: BGRImage, saliency: np.ndarray) -> np.ndarray:
    """
    Return a BGR image with the saliency map overlaid as a heatmap.

    Parameters
    ----------
    bgr      : Input BGR image.
    saliency : Float32 saliency map in [0, 1], same spatial size as bgr.
    """
    sal_uint8 = (saliency * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(sal_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(bgr, 0.6, heatmap, 0.4, 0)
    return overlay


# ---------------------------------------------------------------------------
# Crop suggestion overlay
# ---------------------------------------------------------------------------

def draw_crop_rect(bgr: BGRImage, crop_rect: tuple[int,int,int,int]) -> np.ndarray:
    """
    Draw the suggested crop rectangle on a copy of *bgr*.

    Parameters
    ----------
    crop_rect : (x1, y1, x2, y2) pixel coordinates.
    """
    out = bgr.copy()
    x1, y1, x2, y2 = crop_rect
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 100), 3)
    # Dim outside
    mask = np.zeros(out.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    dimmed = (out.astype(np.float32) * 0.45).astype(np.uint8)
    out = np.where(mask[:, :, None] == 255, out, dimmed)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 100), 3)
    cv2.putText(out, "Suggested Crop", (x1 + 4, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
    return out


# ---------------------------------------------------------------------------
# Metric gauge (single value progress-like plot)
# ---------------------------------------------------------------------------

def plot_gauge(value: float, label: str, max_val: float = 100.0) -> plt.Figure:
    """
    Simple horizontal gauge bar for a single metric score.
    """
    fig, ax = plt.subplots(figsize=(4, 0.7), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, 1)
    clr = "#4caf50" if value >= 75 else ("#ff9800" if value >= 50 else "#f44336")
    ax.barh(0, value, height=0.6, color=clr, edgecolor="none")
    ax.barh(0, max_val, height=0.6, color="#333", edgecolor="none", zorder=0)
    ax.text(max_val / 2, 0, f"{label}: {value:.0f}", va="center", ha="center",
            color="white", fontsize=8, fontweight="bold", zorder=5)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    return fig
