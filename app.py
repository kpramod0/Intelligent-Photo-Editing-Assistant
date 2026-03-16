"""
app.py — Intelligent Photo Editing Assistant
Main Streamlit application entry point.

Tabs
----
1. 📤 Upload
2. 🔍 Analysis
3. ✨ Enhancements
4. 🔀 Comparison
5. 📋 Audit Report
6. ℹ️ About
"""

from __future__ import annotations

import time
import io
import sys
import os

# Ensure 'src' is importable when running from project root
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ── Backend imports ──────────────────────────────────────────────────────────
from src.image_io import load_image_from_bytes, ImageValidationError, ImageBundle
from src.analysis import analyse_image, FullAnalysis
from src.composition import analyse_composition, CompositionResult
from src.enhancement import run_enhancement_pipeline, EnhancementResult
from src.scoring import compute_score, ScoreReport
from src.reporting import generate_markdown_report, generate_audit_report
from src.visualization import (
    plot_histogram,
    plot_before_after,
    plot_score_bars,
    overlay_saliency,
    draw_crop_rect,
)
from src.utils import bgr_to_pil, bgr_to_rgb, pil_to_bytes
from src import config as cfg
import matplotlib.pyplot as plt

# ── Helper ───────────────────────────────────────────────────────────────────
def plt_close(fig):
    """Close a matplotlib figure to free memory."""
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intelligent Photo Editing Assistant",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29, #1a1a3e, #24243e);
    min-height: 100vh;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12122a 0%, #1a1a3e 100%);
    border-right: 1px solid #333;
}

.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00e5ff, #7c4dff, #e91e8c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
}

.sub-title {
    color: #b0bec5;
    font-size: 0.85rem;
    margin-bottom: 1.5rem;
}

/* ── Score card ── */
.score-card {
    background: linear-gradient(135deg, #1e1e3f, #252550);
    border: 1px solid #3a3a6a;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    margin: 0.3rem 0;
}

.score-number {
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1;
}

.score-label {
    font-size: 0.78rem;
    color: #90caf9;
    margin-top: 0.15rem;
}

/* ── Metric card ── */
.metric-card {
    background: #1a1a3a;
    border-left: 4px solid #7c4dff;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
}

.metric-card.good  { border-left-color: #4caf50; }
.metric-card.warn  { border-left-color: #ff9800; }
.metric-card.bad   { border-left-color: #f44336; }

/* ── Section header ── */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #00e5ff;
    border-bottom: 1px solid #333;
    padding-bottom: 0.3rem;
    margin: 1rem 0 0.8rem 0;
}

/* ── Tab colours ── */
.stTabs [data-baseweb="tab"] {
    background: #1a1a3a;
    color: #ccc;
    border-radius: 8px 8px 0 0;
    padding: 0.4rem 1rem;
    margin-right: 4px;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c4dff, #00e5ff) !important;
    color: white !important;
}

/* ── Buttons ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #7c4dff, #00e5ff);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: opacity 0.2s;
}

.stDownloadButton > button:hover { opacity: 0.85; }

/* ── Progress bars ── */
.stProgress > div > div { background: linear-gradient(90deg, #7c4dff, #00e5ff); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state keys
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "bundle":      None,
        "analysis":    None,
        "composition": None,
        "enhancement": None,
        "score":       None,
        "md_report":   None,
        "audit_report": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    st.markdown("### 🛠️ Enhancement Controls")
    fix_exposure      = st.checkbox("Exposure Correction",     value=True)
    fix_contrast      = st.checkbox("CLAHE Contrast",          value=True)
    fix_wb            = st.checkbox("White Balance",           value=True)
    fix_sharpen       = st.checkbox("Sharpening",              value=True)
    fix_denoise       = st.checkbox("Denoising",               value=False)
    fix_saturation    = st.checkbox("Saturation Tuning",       value=False)
    fix_shadow_hl     = st.checkbox("Shadow/Highlight Recovery", value=False)

    st.divider()
    st.markdown("### 🎚️ Fine-tune")
    gamma_val     = st.slider("Gamma override (1.0 = auto)", 0.2, 3.0, 1.0, 0.05,
                               help="Set to 1.0 for automatic adaptive gamma.")
    clahe_clip    = st.slider("CLAHE clip limit",  1.0, 8.0, cfg.DEFAULT_CLAHE_CLIP, 0.5)
    clahe_tile    = st.select_slider("CLAHE tile size", [4, 8, 16, 32], cfg.DEFAULT_CLAHE_TILE)
    sharpen_amt   = st.slider("Sharpening amount", 0.5, 3.0, cfg.DEFAULT_SHARPEN_AMOUNT, 0.1)
    sat_scale     = st.slider("Saturation scale",  0.5, 2.0, 1.3, 0.05)
    denoise_meth  = st.selectbox("Denoise method", ["bilateral", "nlmeans"])

    st.divider()
    if st.session_state.bundle:
        b: ImageBundle = st.session_state.bundle
        st.markdown("### 📁 File Info")
        st.caption(f"**{b.filename}**")
        st.caption(f"Size: {b.original_size[0]}×{b.original_size[1]} px")
        if b.was_resized:
            st.caption(f"⚠️ Resized to {b.processing_size[0]}×{b.processing_size[1]} for processing.")


# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📷 Intelligent Photo Editing Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Computational Photography · Image Quality Analysis · Auto-Enhancement · Academic Project</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_upload, tab_analysis, tab_enhance, tab_compare, tab_audit, tab_about = st.tabs([
    "📤 Upload",
    "🔍 Analysis",
    "✨ Enhancements",
    "🔀 Comparison",
    "📋 Audit Report",
    "ℹ️ About",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Upload
# ══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown('<p class="section-header">Upload Your Photo</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drag & drop or click to upload",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG. Max 50 MB.",
        key="file_uploader",
    )

    if uploaded is not None:
        # Check if we already have this file loaded to avoid clearing state on every rerun
        current_bundle: ImageBundle | None = st.session_state.bundle
        if current_bundle is None or current_bundle.filename != uploaded.name:
            data = uploaded.read()
            with st.spinner("🔍 Validating and loading image…"):
                try:
                    bundle = load_image_from_bytes(data, filename=uploaded.name)
                    st.session_state.bundle      = bundle
                    st.session_state.analysis    = None
                    st.session_state.composition = None
                    st.session_state.enhancement = None
                    st.session_state.score       = None
                    st.session_state.md_report   = None
                    st.session_state.audit_report = None
                    st.success(f"✅ Loaded **{bundle.filename}** successfully!")
                except ImageValidationError as e:
                    st.error(f"❌ Validation failed: {e}")
                    st.session_state.bundle = None

    if st.session_state.bundle:
        bundle: ImageBundle = st.session_state.bundle
        col_img, col_meta = st.columns([2, 1])

        with col_img:
            st.image(bundle.display_pil, caption="Original Image", use_container_width=True)

        with col_meta:
            st.markdown("#### 📷 Image Metadata")
            for key, val in bundle.metadata.items():
                st.markdown(f"**{key}:** `{val}`")

        st.divider()
        st.markdown("#### 🚀 Run Analysis Pipeline")
        if st.button("▶ Analyse Image", type="primary", use_container_width=True):
            progress = st.progress(0, text="Starting analysis…")
            bgr = bundle.original_bgr

            # Analysis
            progress.progress(15, text="Analysing brightness, contrast, histogram…")
            analysis = analyse_image(bgr)
            time.sleep(0.05)

            progress.progress(40, text="Analysing sharpness, noise, white balance…")
            time.sleep(0.05)

            progress.progress(55, text="Analysing composition…")
            composition = analyse_composition(bgr)
            analysis.composition = composition.metric
            time.sleep(0.05)

            progress.progress(70, text="Computing quality score…")
            score = compute_score(analysis)
            time.sleep(0.05)

            progress.progress(85, text="Generating reports…")
            md_report = generate_markdown_report(
                bundle.filename, analysis, score, None, composition, bundle.metadata
            )
            audit = generate_audit_report(bundle.filename, analysis, score, None)
            time.sleep(0.05)

            progress.progress(100, text="Done!")

            st.session_state.analysis    = analysis
            st.session_state.composition = composition
            st.session_state.score       = score
            st.session_state.md_report   = md_report
            st.session_state.audit_report = audit

            st.success("✅ Analysis complete! Navigate to the **Analysis** tab to view results.")
    else:
        st.info("👆 Please upload an image to get started.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    if not st.session_state.analysis:
        st.info("📤 Upload an image and click **Analyse Image** first.")
    else:
        analysis: FullAnalysis           = st.session_state.analysis
        score_report: ScoreReport        = st.session_state.score
        composition: CompositionResult   = st.session_state.composition
        bundle: ImageBundle              = st.session_state.bundle
        bgr = bundle.original_bgr

        # ── Score cards row ───────────────────────────────────────────────
        st.markdown('<p class="section-header">Quality Score</p>', unsafe_allow_html=True)

        band_color = {
            "Excellent": "#4caf50", "Good": "#8bc34a", "Average": "#ff9800",
            "Weak": "#ff5722", "Poor": "#f44336",
        }.get(score_report.band, "#9e9e9e")

        c1, c2, c3 = st.columns([1, 1.5, 1])
        with c2:
            st.markdown(
                f'<div class="score-card">'
                f'<div class="score-number" style="color:{band_color}">{score_report.total_score:.1f}</div>'
                f'<div style="font-size:1.1rem;color:{band_color};font-weight:600">{score_report.band}</div>'
                f'<div class="score-label">Overall Quality Score (out of 100)</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Score bar chart ───────────────────────────────────────────────
        col_bars, col_hist = st.columns([1, 1])
        with col_bars:
            st.markdown("**Per-Metric Scores**")
            fig_bars = plot_score_bars(score_report)
            st.pyplot(fig_bars, use_container_width=True)
            plt_close(fig_bars)

        with col_hist:
            st.markdown("**Colour Histogram**")
            fig_hist = plot_histogram(bgr)
            st.pyplot(fig_hist, use_container_width=True)
            plt_close(fig_hist)

        st.divider()

        # ── Metric details ────────────────────────────────────────────────
        st.markdown('<p class="section-header">Detailed Metric Analysis</p>', unsafe_allow_html=True)

        metrics_list = list(analysis.as_dict().items())

        for row_start in range(0, len(metrics_list), 2):
            row_metrics = metrics_list[row_start: row_start + 2]
            cols = st.columns(len(row_metrics))
            for col, (key, metric) in zip(cols, row_metrics):
                with col:
                    if metric.score >= 75:
                        css_cls, icon = "good", "🟢"
                    elif metric.score >= 50:
                        css_cls, icon = "warn", "🟡"
                    else:
                        css_cls, icon = "bad",  "🔴"

                    st.markdown(
                        f'<div class="metric-card {css_cls}">'
                        f'<strong>{icon} {metric.name}</strong><br>'
                        f'<small>Value: <code>{metric.value}</code> &nbsp;|&nbsp; Score: <strong>{metric.score:.0f}/100</strong><br>'
                        f'<em>{metric.label}</em></small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    with st.expander("Details", expanded=False):
                        st.markdown(f"**Explanation:** {metric.explanation}")
                        st.markdown(f"**Suggestion:** `{metric.suggestion}`")
                        if metric.extra:
                            st.json({k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                     for k, v in metric.extra.items()
                                     if not isinstance(v, np.ndarray)})

        # ── Composition visuals ───────────────────────────────────────────
        st.divider()
        st.markdown('<p class="section-header">Composition Analysis</p>', unsafe_allow_html=True)
        cc1, cc2, cc3 = st.columns(3)

        with cc1:
            st.image(bgr_to_rgb(composition.thirds_overlay),
                     caption="Rule-of-Thirds Grid", use_container_width=True)
        with cc2:
            sal_overlay = overlay_saliency(bgr, composition.saliency_map)
            st.image(bgr_to_rgb(sal_overlay),
                     caption="Saliency Map Overlay", use_container_width=True)
        with cc3:
            if composition.crop_rect:
                crop_img = draw_crop_rect(bgr, composition.crop_rect)
                st.image(bgr_to_rgb(crop_img),
                         caption="Suggested Crop Area", use_container_width=True)

        if composition.metric:
            m = composition.metric
            flag = "🟢" if m.score >= 70 else ("🟡" if m.score >= 45 else "🔴")
            st.markdown(f"**{flag} Composition Score:** {m.score:.1f}/100 — *{m.label}*")
            st.caption(m.explanation)
            if composition.horizon_angle is not None:
                st.caption(f"📐 Detected horizon tilt: **{composition.horizon_angle:.1f}°**")

        # ── What reduced score ────────────────────────────────────────────
        if score_report.reductions:
            st.divider()
            st.markdown('<p class="section-header">⬇️ What Reduced Your Score?</p>', unsafe_allow_html=True)
            for r in score_report.reductions:
                st.markdown(f"- {r}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Enhancements
# ══════════════════════════════════════════════════════════════════════════════
with tab_enhance:
    if not st.session_state.analysis:
        st.info("📤 Run the analysis first (Upload tab).")
    else:
        bundle: ImageBundle = st.session_state.bundle
        bgr = bundle.original_bgr

        st.markdown('<p class="section-header">Apply Enhancements</p>', unsafe_allow_html=True)
        st.caption(
            "Toggle corrections in the **sidebar** and click **Apply Enhancements** below. "
            "The pipeline runs: Exposure → White Balance → Contrast → Denoise → Sharpen → Saturation → Shadow/Highlight"
        )

        if st.button("✨ Apply Enhancements", type="primary", use_container_width=True):
            gamma_override = None if abs(gamma_val - 1.0) < 0.05 else gamma_val
            with st.spinner("Applying enhancement pipeline…"):
                t0 = time.perf_counter()
                result = run_enhancement_pipeline(
                    bgr,
                    fix_exposure=fix_exposure,
                    fix_contrast=fix_contrast,
                    fix_white_balance=fix_wb,
                    fix_sharpen=fix_sharpen,
                    fix_denoise=fix_denoise,
                    fix_saturation=fix_saturation,
                    fix_shadow_highlight=fix_shadow_hl,
                    gamma_override=gamma_override,
                    clahe_clip=clahe_clip,
                    clahe_tile=clahe_tile,
                    sharpen_amount=sharpen_amt,
                    saturation_scale=sat_scale,
                    denoise_method=denoise_meth,
                    denoise_strength=cfg.DEFAULT_DENOISE_H,
                )
                elapsed = time.perf_counter() - t0
            st.session_state.enhancement = result
            # Regenerate reports with enhancement info
            score = st.session_state.score
            analysis_st = st.session_state.analysis
            composition_st = st.session_state.composition
            st.session_state.md_report = generate_markdown_report(
                bundle.filename, analysis_st, score,
                result, composition_st, bundle.metadata
            )
            st.session_state.audit_report = generate_audit_report(
                bundle.filename, analysis_st, score, result
            )
            st.success(f"✅ Enhancement pipeline completed in {elapsed:.2f}s — {len(result.steps)} step(s) applied.")

        enh: EnhancementResult | None = st.session_state.enhancement
        if enh:
            st.divider()
            col_orig, col_enh = st.columns(2)
            with col_orig:
                st.image(bgr_to_rgb(bgr), caption="Original", use_container_width=True)
            with col_enh:
                st.image(bgr_to_rgb(enh.enhanced_bgr), caption="Enhanced", use_container_width=True)

            st.divider()
            st.markdown("#### 🛠️ Enhancement Steps Applied")
            for step in enh.steps:
                icon = "✅" if step.applied else "⏭️"
                before_after = ""
                if step.before_stat is not None and step.after_stat is not None:
                    before_after = f" &nbsp;|&nbsp; `{step.unit}`: **{step.before_stat} → {step.after_stat}**"
                st.markdown(f"{icon} **{step.name}**{before_after}<br><small>{step.reason}</small>",
                            unsafe_allow_html=True)

            # Download button
            st.divider()
            enh_pil = bgr_to_pil(enh.enhanced_bgr)
            img_bytes = pil_to_bytes(enh_pil, "PNG")
            st.download_button(
                "⬇️ Download Enhanced Image (PNG)",
                data=img_bytes,
                file_name=f"enhanced_{bundle.filename.split('.')[0]}.png",
                mime="image/png",
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    if not st.session_state.enhancement:
        st.info("✨ Apply enhancements first (Enhancements tab).")
    else:
        bundle: ImageBundle     = st.session_state.bundle
        enh: EnhancementResult  = st.session_state.enhancement
        bgr = bundle.original_bgr
        enh_bgr = enh.enhanced_bgr

        st.markdown('<p class="section-header">Before / After Comparison</p>', unsafe_allow_html=True)

        fig_cmp = plot_before_after(bgr, enh_bgr)
        st.pyplot(fig_cmp, use_container_width=True)
        plt_close(fig_cmp)

        st.divider()
        st.markdown("#### 📊 Histogram Comparison")
        h_c1, h_c2 = st.columns(2)
        with h_c1:
            fig_h1 = plot_histogram(bgr, title="Original — Histogram")
            st.pyplot(fig_h1, use_container_width=True)
            plt_close(fig_h1)
        with h_c2:
            fig_h2 = plot_histogram(enh_bgr, title="Enhanced — Histogram")
            st.pyplot(fig_h2, use_container_width=True)
            plt_close(fig_h2)

        # Quantitative delta
        st.divider()
        st.markdown("#### 📈 Quantitative Changes")

        from src.analysis import analyse_brightness, analyse_contrast, analyse_sharpness, analyse_noise

        orig_br  = analyse_brightness(bgr)
        enh_br   = analyse_brightness(enh_bgr)
        orig_ct  = analyse_contrast(bgr)
        enh_ct   = analyse_contrast(enh_bgr)
        orig_sh  = analyse_sharpness(bgr)
        enh_sh   = analyse_sharpness(enh_bgr)
        orig_no  = analyse_noise(bgr)
        enh_no   = analyse_noise(enh_bgr)

        import pandas as pd
        df = pd.DataFrame({
            "Metric":    ["Brightness (mean)", "Contrast (std)", "Sharpness (Lap. var)", "Noise (σ)"],
            "Original":  [orig_br.value, orig_ct.value, orig_sh.value, orig_no.value],
            "Enhanced":  [enh_br.value,  enh_ct.value,  enh_sh.value,  enh_no.value],
        })
        df["Change"] = df["Enhanced"] - df["Original"]
        df["Change %"] = ((df["Change"] / (df["Original"].abs() + 1e-9)) * 100).round(1).astype(str) + "%"
        st.dataframe(df.style.format({"Original": "{:.2f}", "Enhanced": "{:.2f}", "Change": "{:+.2f}"}),
                     use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Audit Report
# ══════════════════════════════════════════════════════════════════════════════
with tab_audit:
    if not st.session_state.audit_report:
        st.info("📤 Run an analysis first.")
    else:
        st.markdown('<p class="section-header">End-to-End Audit Report</p>', unsafe_allow_html=True)

        audit_str: str = st.session_state.audit_report
        md_str: str    = st.session_state.md_report

        with st.expander("📋 Full Audit Report (text)", expanded=True):
            st.code(audit_str, language="markdown")

        st.divider()
        with st.expander("📝 Full Markdown Analysis Report", expanded=False):
            st.markdown(md_str)

        st.divider()
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "⬇️ Download Audit Report (.txt)",
                data=audit_str.encode("utf-8"),
                file_name="audit_report.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with col_dl2:
            st.download_button(
                "⬇️ Download Analysis Report (.md)",
                data=md_str.encode("utf-8"),
                file_name="analysis_report.md",
                mime="text/markdown",
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — About
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
## 📷 Intelligent Photo Editing Assistant

A **Computational Photography** mini-project that automatically analyses
and enhances photographs using classical image processing algorithms.

---

### 🎯 What This Project Does

Upload any JPG or PNG photograph. The system will:

1. **Validate** the file (format, integrity, size).
2. **Analyse** the image across 9 quality dimensions.
3. **Score** the photo on a transparent 100-point scale.
4. **Enhance** it with configurable correction algorithms.
5. **Compare** before and after with side-by-side histograms.
6. **Export** the enhanced image and a full audit report.

---

### 🔬 Analysis Metrics

| # | Metric | Method |
|---|--------|--------|
| 1 | Brightness / Exposure | Grayscale mean pixel intensity |
| 2 | Contrast | Standard deviation of luminance |
| 3 | Histogram Health | Shadow/highlight clipping fraction |
| 4 | Sharpness | Variance of Laplacian (Pech-Pacheco 2000) |
| 5 | Noise Level | High-frequency residual std-dev |
| 6 | White Balance | Grey-world assumption (channel deviation) |
| 7 | Saturation | HSV S-channel mean |
| 8 | Dynamic Range | Fraction of occupied histogram bins |
| 9 | Composition | Edge saliency + rule-of-thirds alignment |

---

### ✨ Enhancement Methods

| Method | Algorithm |
|--------|-----------|
| Exposure Correction | Adaptive gamma via LUT |
| Contrast Enhancement | CLAHE on LAB L-channel |
| White Balance | Grey-world channel scaling |
| Denoising | Bilateral filter / fastNlMeans |
| Sharpening | Unsharp masking (USM) |
| Saturation Tuning | HSV S-channel scaling |
| Shadow/Highlight | Piecewise linear tone curve |

---

### 🏗️ Architecture

```
app.py (Streamlit UI — 6 tabs)
└── src/
    ├── config.py        — Thresholds, weights, defaults
    ├── utils.py         — Colour-space & numeric helpers
    ├── image_io.py      — Upload, validate, metadata
    ├── analysis.py      — 8 metric analysis functions
    ├── composition.py   — Saliency, thirds, crop suggestion
    ├── enhancement.py   — 7 enhancement algorithms
    ├── scoring.py       — Weighted quality score
    ├── reporting.py     — Markdown + audit report generator
    └── visualization.py — Matplotlib plots & overlays
```

---

### ⚠️ Known Limitations

- All thresholds are **empirical** and may not suit all photographic styles.
- Grey-world WB fails on **single-hue dominant** images.
- Composition analysis is **heuristic** (not perceptual deep learning).
- Sharpness via Laplacian can be confused by **heavy texture**.
- The project does **not** use neural networks or GPU acceleration.

---

### 🔮 Future Enhancements

- Deep-learning saliency model (e.g. U²-Net) for better composition.
- Semantic segmentation for background-aware processing.
- GPU acceleration via OpenCV CUDA or PyTorch.
- HDR tone mapping for high-dynamic-range photography.
- PDF export of the full report (via ReportLab or fpdf2).
- Batch processing mode for multiple images.

---

### 📚 Tech Stack

`Python 3.11` · `Streamlit` · `OpenCV` · `NumPy` · `Pillow` ·
`Matplotlib` · `scikit-image` · `pandas` · `scipy`

---

*Built for Computational Photography Academic Project, 2024–25*
    """)


