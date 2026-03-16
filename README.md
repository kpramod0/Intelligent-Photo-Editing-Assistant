# 📷 Intelligent Photo Editing Assistant

> A **Computational Photography** mini-project built with Python + Streamlit that automatically analyses, scores, and enhances photographs using classical image processing algorithms.

---

## 🌟 Features

| Feature | Description |
|---------|-------------|
| 📤 **Upload & Validate** | JPG/JPEG/PNG with format, integrity, and size checks |
| 🔍 **9-Metric Analysis** | Exposure, Contrast, Histogram, Sharpness, Noise, White Balance, Saturation, Dynamic Range, Composition |
| 📊 **Quality Scoring** | Transparent weighted score out of 100 with band labels |
| ✨ **7 Enhancements** | Gamma, CLAHE, WB, Bilateral/NLM denoising, Unsharp Masking, Saturation, Tone Curve |
| 🔀 **Before/After** | Side-by-side comparison with histogram delta table |
| 📋 **Audit Report** | Full functional, algorithmic, and academic suitability audit |
| ⬇️ **Export** | Download enhanced image (PNG) + analysis report (MD) + audit report (TXT) |

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit ≥ 1.32
- **Computer Vision:** OpenCV ≥ 4.9
- **Image I/O:** Pillow ≥ 10.2
- **Numerics:** NumPy ≥ 1.26, SciPy ≥ 1.12
- **Visualisation:** Matplotlib ≥ 3.8
- **Data:** pandas ≥ 2.2
- **Testing:** pytest ≥ 8.1
- **Python:** 3.11+

---

## 📦 Installation

```bash
# 1. Clone / navigate to project folder
cd intelligent_photo_editing_assistant

# 2. Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

```bash
# From inside the intelligent_photo_editing_assistant/ directory:
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 🗂️ Project Structure

```
intelligent_photo_editing_assistant/
│
├── app.py                  ← Streamlit app (6 tabs)
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── audit_report.md         ← Standalone end-to-end audit
│
├── sample_images/          ← Test images (add your own)
├── reports/                ← Auto-generated reports saved here
│
├── tests/
│   ├── test_analysis.py    ← 30+ tests for analysis engine
│   ├── test_enhancement.py ← Tests for enhancement pipeline
│   └── test_scoring.py     ← Tests for scoring engine
│
└── src/
    ├── __init__.py
    ├── config.py           ← All thresholds, weights, constants
    ├── utils.py            ← Colour-space & numeric helpers
    ├── image_io.py         ← Upload, validate, metadata
    ├── analysis.py         ← 8 quality analysis functions
    ├── composition.py      ← Saliency, thirds, crop suggestion
    ├── enhancement.py      ← 7 enhancement algorithms
    ├── scoring.py          ← Weighted quality score engine
    ├── reporting.py        ← Markdown + audit report generation
    └── visualization.py    ← Histograms, overlays, bar charts
```

---

## 🔬 Module Explanations

### `src/config.py`
Central configuration file. All tuneable thresholds and scoring weights live here. Change this file to adjust what counts as "underexposed" or the relative importance of each metric without touching business logic.

### `src/analysis.py`
Contains one function per metric. Each returns a `MetricResult` dataclass with `value`, `label`, `score`, `explanation`, and `suggestion` fields.

| Function | Method | Reference |
|----------|--------|-----------|
| `analyse_brightness()` | Grayscale mean | Classic |
| `analyse_contrast()` | Grayscale std-dev | Hasler & Suesstrunk 2003 |
| `analyse_histogram()` | Shadow/highlight clipping | - |
| `analyse_sharpness()` | Variance of Laplacian | Pech-Pacheco et al. 2000 |
| `analyse_noise()` | Residual high-freq std-dev | - |
| `analyse_white_balance()` | Grey-world assumption | Buchsbaum 1980 |
| `analyse_saturation()` | HSV S-channel mean | - |
| `analyse_dynamic_range()` | Histogram bin occupancy | - |

### `src/composition.py`
Heuristic composition analysis:
- **Saliency map** via Canny edges + local contrast energy
- **Rule-of-thirds** scoring by sampling saliency at power-point intersections
- **Visual centre of mass** — saliency-weighted centroid
- **Horizon angle** via Hough line transform
- **Suggested crop** based on saliency distribution

### `src/enhancement.py`
Pipeline functions in order of application:
1. `correct_exposure()` — Adaptive gamma LUT
2. `apply_white_balance()` — Grey-world channel scaling
3. `apply_clahe()` — CLAHE on LAB L-channel
4. `apply_denoising()` — Bilateral / fastNlMeansDenoising
5. `apply_sharpening()` — Unsharp masking
6. `apply_saturation()` — HSV S-channel scale
7. `apply_shadow_highlight_recovery()` — Piecewise tone curve

### `src/scoring.py`
Weighted average of sub-scores using the weights in `config.SCORE_WEIGHTS` (summing to 100). Produces a `ScoreReport` with sub-scores, weighted contributions, quality band, and human-readable explanations.

---

## 🧪 Running Tests

```bash
cd intelligent_photo_editing_assistant
pytest tests/ -v
```

Expected: **30+ passing tests** covering analysis accuracy, enhancement behaviour, scoring correctness, and image validation.

---

## ⚠️ Limitations

1. **Empirical thresholds** — brightness/contrast cutoffs suit typical consumer photos; artistic or HDR images may be misclassified.
2. **Grey-world WB** — fails on scenes dominated by a single hue (green forest, blue ocean).
3. **Heuristic composition** — edge saliency is not the same as human aesthetic perception.
4. **Sharpness metric** — Laplacian variance can be high in textured images even if soft in smooth regions.
5. **No GPU acceleration** — processing large images may take a few seconds on CPU.
6. **No deep learning** — all methods are classical; results are explainable but less powerful than neural models.

---

## 🔮 Future Enhancements

- **Deep saliency model** (U²-Net) for composition
- **Semantic segmentation** for background-aware enhancement
- **HDR tone mapping** support
- **PDF report export** (fpdf2 / ReportLab)
- **Batch processing** mode
- **Mobile-friendly** Streamlit layout
- **Colour grading presets** (cinematic, warm, cool)
- **Face detection** for portrait-specific enhancement

---

## 📡 Demo Flow (2 Minutes)

1. Launch app: `streamlit run app.py`
2. **Upload tab** → Upload a test photo → Click **Analyse Image**
3. **Analysis tab** → Review score, histogram, per-metric cards
4. **Enhancements tab** → Click **Apply Enhancements** → View corrected image
5. **Comparison tab** → Review before/after and delta table
6. **Audit tab** → Show audit report → Download report

---

## 🎓 Academic Project Info

- **Course:** Computational Photography
- **Year:** 2024–25
- **Language:** Python 3.11
- **Lines of Code:** ~2,000 (excluding tests + docs)

---

*Built with ❤️ using OpenCV, Streamlit, and classical image processing.*
