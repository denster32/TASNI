#!/usr/bin/env python3
"""
Compile TASNI paper from Markdown to PDF using weasyprint.
Includes figures from output/figures directory.
"""

import base64
from pathlib import Path

import markdown
from weasyprint import CSS, HTML

# Paper directory
PAPER_DIR = Path(__file__).parent
FIGURES_DIR = PAPER_DIR.parent / "output" / "figures"

# CSS for academic paper styling
CSS_STYLE = """
@page {
    size: letter;
    margin: 0.75in;
    @bottom-center {
        content: counter(page);
    }
}

body {
    font-family: 'Times New Roman', Times, serif;
    font-size: 10pt;
    line-height: 1.4;
    text-align: justify;
    max-width: 100%;
}

h1 {
    font-size: 14pt;
    font-weight: bold;
    text-align: center;
    margin-top: 0;
    margin-bottom: 18pt;
}

h2 {
    font-size: 12pt;
    font-weight: bold;
    margin-top: 14pt;
    margin-bottom: 8pt;
}

h3 {
    font-size: 10pt;
    font-weight: bold;
    margin-top: 10pt;
    margin-bottom: 6pt;
}

h4 {
    font-size: 10pt;
    font-style: italic;
    margin-top: 8pt;
    margin-bottom: 4pt;
}

p {
    margin-bottom: 8pt;
    text-indent: 0.2in;
}

p:first-of-type, h2 + p, h3 + p, h4 + p, table + p, ul + p, ol + p, blockquote + p, figure + p, .figure + p {
    text-indent: 0;
}

table {
    border-collapse: collapse;
    margin: 10pt auto;
    font-size: 9pt;
    width: auto;
}

th, td {
    border: 1px solid #333;
    padding: 3pt 6pt;
    text-align: left;
}

th {
    background-color: #f0f0f0;
    font-weight: bold;
}

blockquote {
    margin: 10pt 20pt;
    font-style: italic;
    border-left: 2pt solid #ccc;
    padding-left: 10pt;
}

code {
    font-family: 'Courier New', monospace;
    font-size: 8pt;
    background-color: #f5f5f5;
    padding: 1pt 2pt;
}

pre {
    font-family: 'Courier New', monospace;
    font-size: 8pt;
    background-color: #f5f5f5;
    padding: 6pt;
    overflow-x: auto;
    margin: 10pt 0;
}

ul, ol {
    margin-left: 20pt;
    margin-bottom: 8pt;
}

li {
    margin-bottom: 3pt;
}

hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 18pt 0;
}

.title-block {
    text-align: center;
    margin-bottom: 18pt;
}

.abstract {
    margin: 18pt 30pt;
    font-size: 9pt;
}

.abstract h2 {
    text-align: center;
}

.keywords {
    font-size: 9pt;
    margin-top: 10pt;
}

strong {
    font-weight: bold;
}

em {
    font-style: italic;
}

/* Figure styling */
.figure {
    margin: 16pt auto;
    text-align: center;
    page-break-inside: avoid;
}

.figure img {
    max-width: 100%;
    height: auto;
}

.figure-caption {
    font-size: 9pt;
    margin-top: 6pt;
    text-align: justify;
    padding: 0 20pt;
}

.figure-label {
    font-weight: bold;
}

/* Two-column figure */
.figure-wide {
    width: 100%;
}

.figure-wide img {
    width: 100%;
}
"""


def get_image_data_uri(image_path):
    """Convert image to base64 data URI for embedding in HTML."""
    if not image_path.exists():
        return None
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    suffix = image_path.suffix.lower()
    if suffix == ".png":
        mime = "image/png"
    elif suffix in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif suffix == ".pdf":
        return None  # Can't embed PDF in HTML
    else:
        mime = "application/octet-stream"
    return f"data:{mime};base64,{data}"


def create_figure_html(fig_num, image_path, caption, label=None):
    """Create HTML for a figure with caption."""
    data_uri = get_image_data_uri(image_path)
    if data_uri is None:
        return f"<p><em>[Figure {fig_num} not found: {image_path}]</em></p>"

    label_text = label or f"Figure {fig_num}"
    return f"""
<div class="figure">
    <img src="{data_uri}" alt="{label_text}">
    <div class="figure-caption">
        <span class="figure-label">{label_text}.</span> {caption}
    </div>
</div>
"""


def read_markdown(filename):
    """Read a markdown file."""
    filepath = PAPER_DIR / filename
    if filepath.exists():
        return filepath.read_text()
    return ""


def combine_paper():
    """Combine all paper sections into a single markdown document with figures."""

    # Title block
    title_block = """
<div class="title-block">

# The Thermal Anomaly Search for Non-communicating Intelligence (TASNI): Discovery of Four Fading Thermal Orphans in the AllWISE Catalog

**First Author**<sup>1</sup>, **Second Author**<sup>2</sup>, **Third Author**<sup>3</sup>

<sup>1</sup>Department of Astronomy, University
<sup>2</sup>Institute for Astrophysics
<sup>3</sup>Space Science Center

</div>
"""

    # Abstract
    abstract_content = """
<div class="abstract">

## Abstract

We present the **Thermal Anomaly Search for Non-communicating Intelligence (TASNI)**, a systematic pipeline to identify mid-infrared sources in the AllWISE catalog that lack counterparts at optical, near-infrared, and radio wavelengths. From 747 million AllWISE sources, our multi-wavelength veto strategy isolates 4,137 "thermal anomalies"—objects detectable only in the mid-infrared with thermal colors (W1−W2 > 0.5 mag). The 100 highest-scoring candidates ("golden sample") have mean effective temperature T_eff = 265 ± 36 K, mean W1−W2 color of 1.99 ± 0.36 mag, and mean proper motion μ = 216 ± 149 mas/yr, consistent with extremely cold brown dwarfs at distances of 20–100 pc. Analysis of 10-year NEOWISE light curves reveals that 45% of the golden sample is photometrically stable, 50% shows variability consistent with brown dwarf atmospheres, and 5% exhibits systematic fading. We identify four "fading thermal orphans"—sources with unprecedented combinations of extreme W1−W2 colors (1.53–3.37 mag), room-temperature emission (T_eff = 251–293 K), high proper motions (55–359 mas/yr), and monotonic fading at rates of 18–53 mmag/yr over the decade-long baseline. None of these four sources appear in SIMBAD or any astronomical catalog. The most likely interpretation is that these objects are extremely cold Y-type brown dwarfs, possibly young objects undergoing rapid cooling. While our search was partly motivated by the technosignature hypothesis, all observed properties are consistent with natural astrophysical sources. Spectroscopic follow-up is urgently needed to confirm the nature of these unusual objects.

**Keywords:** brown dwarfs — infrared: stars — stars: low-mass — surveys — techniques: photometric

</div>
"""

    # Figure definitions with captions
    figures = {
        1: {
            "file": "fig1_allsky_galactic.png",
            "caption": "All-sky distribution of the 100 golden thermal anomaly candidates in Galactic coordinates (Mollweide projection). Sources are color-coded by effective temperature (K). The distribution is concentrated at high Galactic latitudes (|b| > 30°), minimizing contamination from the Galactic plane. The mean Galactic latitude is |b| = 43.3°.",
        },
        2: {
            "file": "fig2_color_magnitude.png",
            "caption": "Color-magnitude diagram showing W1−W2 color versus W1 magnitude for the golden sample. Points are color-coded by effective temperature. The dashed horizontal line marks W1−W2 = 2.0 mag, the approximate boundary for Y dwarf classification. Sources with W1−W2 > 2.0 mag (33% of the sample) occupy the Y dwarf color regime. The reddest source, J143046.35−025927.8, has W1−W2 = 3.37 mag.",
        },
        3: {
            "file": "fig3_distributions.png",
            "caption": "Distributions of key parameters for the golden sample. Left: Effective temperature histogram showing 87% of sources have T_eff < 300 K (dashed line), with a mean of 265 ± 36 K. Right: Proper motion distribution showing 69% have μ > 100 mas/yr, indicating nearby distances of 20–100 pc.",
        },
        4: {
            "file": "fig4_variability.png",
            "caption": "NEOWISE variability analysis results. Left: Distribution of W1 RMS variability amplitude. Center: Variability classification showing 45% stable (NORMAL), 50% variable (VARIABLE), and 5% systematically fading (FADING). Right: Linear trend slopes for all sources, with the four fading thermal orphans highlighted as outliers with significant negative trends.",
        },
        5: {
            "file": "fig5_fading_lightcurves.png",
            "caption": "Ten-year NEOWISE light curves for the four fading thermal orphans. Each panel shows W1 (blue) and W2 (orange) photometry from 2014–2024. Solid lines indicate linear fits to the fading trends. All four sources show monotonic dimming in both bands, with fade rates ranging from 17.9 to 52.6 mmag/yr. The fastest fader (J231029.40−060547.3) has dimmed by ~0.5 mag over the decade.",
        },
        6: {
            "file": "fig6_pipeline_flowchart.png",
            "caption": "TASNI pipeline flowchart showing the multi-wavelength veto strategy. Starting from 747 million AllWISE sources, successive vetoes against Gaia (optical), 2MASS (near-IR), Pan-STARRS/Legacy (deep optical), and NVSS (radio) isolate 4,137 thermal anomalies. The 100 highest-scoring candidates form the golden sample, of which 4 are identified as fading thermal orphans.",
        },
    }

    # Create figure HTML
    figure_html = {}
    for fig_num, fig_info in figures.items():
        fig_path = FIGURES_DIR / fig_info["file"]
        figure_html[fig_num] = create_figure_html(fig_num, fig_path, fig_info["caption"])

    # Build the paper with figures inserted at appropriate locations
    paper_content = f"""
{title_block}

{abstract_content}

---

## 1. Introduction

### 1.1 Motivation: Searching for Thermal Anomalies

The identification of unusual astrophysical sources has historically driven major discoveries, from quasars to gamma-ray bursts. In the modern era of large-area sky surveys, systematic searches for anomalous objects—those that defy easy classification—offer a promising avenue for discovering new phenomena.

One class of potentially anomalous sources comprises objects that emit primarily in the thermal infrared while remaining undetected at other wavelengths. Such "thermal orphans" could arise from several physical mechanisms:

1. **Extremely cold brown dwarfs**: Objects with T_eff ≲ 300 K emit predominantly at wavelengths λ > 10 μm, with negligible optical flux. The coldest known brown dwarf, WISE J085510.83−071442.5, has T_eff ≈ 250 K and is detectable only in the mid-infrared.

2. **Dust-obscured sources**: Objects embedded in optically thick dust shells re-radiate absorbed energy as thermal emission.

3. **Technosignatures**: Theoretical considerations suggest that advanced technological civilizations might be detectable through their waste heat. A structure intercepting stellar luminosity would re-radiate at temperatures T ~ 300 K for Sun-like stars at 1 AU separation.

### 1.2 The Y Dwarf Population

Brown dwarfs are substellar objects with masses below the hydrogen-burning limit (~0.075 M_☉). The Y dwarf spectral class, defined by T_eff ≲ 500 K, represents the coldest end of the brown dwarf sequence. Approximately 30 Y dwarfs are currently known, identified primarily through WISE color selection.

### 1.3 This Work

We present the Thermal Anomaly Search for Non-communicating Intelligence (TASNI), a systematic pipeline to identify mid-infrared sources lacking counterparts across the electromagnetic spectrum.

{figure_html[6]}

---

## 2. Methods

### 2.1 Data Sources

Our parent sample is drawn from the **AllWISE Source Catalog**, containing 747,634,026 sources. For temporal analysis, we utilize the **NEOWISE Reactivation Single-Exposure Source Table**, providing multi-epoch photometry from 2013 to present.

To identify sources lacking counterparts, we cross-match against:
- **Gaia DR3**: 1.8 billion sources with G < 21 mag
- **2MASS**: 470 million sources with J, H, K_s photometry
- **Pan-STARRS DR1**: 3 billion detections in grizy bands
- **Legacy Survey DR10**: Deep optical imaging to g ≈ 24.7 mag
- **NVSS**: 1.4 GHz radio survey

### 2.2 Source Selection Pipeline

| Selection Stage | Sources | Reduction |
|-----------------|---------|-----------|
| AllWISE Catalog | 747,634,026 | — |
| No Gaia DR3 | 406,387,755 | 46% |
| Thermal (W1−W2 > 0.5) | ~1,000,000 | — |
| No 2MASS | 62,856 | 94% |
| No Pan-STARRS/Legacy | 39,151 | 38% |
| No NVSS radio | 4,137 | 89% |
| **Golden targets** | **100** | — |
| **Fading sources** | **4** | 4% |

### 2.3 Variability Analysis

We retrieved NEOWISE multi-epoch photometry for all 100 golden targets, yielding 38,700 individual measurements over a 9.2-year baseline (2013.9–2024.5). Sources are classified as:
- **NORMAL**: Stable emission (χ²_ν < 3)
- **VARIABLE**: Significant variability (χ²_ν > 3)
- **FADING**: Systematic dimming (dm/dt > 15 mmag/yr, p < 0.01)

---

## 3. Results

### 3.1 Golden Sample Properties

| Parameter | Mean ± Std | Range |
|-----------|------------|-------|
| W1 (mag) | 14.25 ± 0.89 | 10.79–16.48 |
| W2 (mag) | 12.26 ± 0.78 | 8.75–14.35 |
| W1−W2 (mag) | 1.99 ± 0.36 | 1.53–3.67 |
| T_eff (K) | 265 ± 36 | 205–466 |
| μ (mas/yr) | 216 ± 149 | 0–663 |

{figure_html[1]}

{figure_html[2]}

### 3.2 Temperature and Kinematic Distributions

**Key finding: 87% of the golden sample (87/100 sources) have T_eff < 300 K—cooler than typical room temperature on Earth.**

The proper motion distribution implies distances of 10–100 pc for most sources, consistent with the local brown dwarf population.

{figure_html[3]}

### 3.3 Variability Results

| Classification | Count | Fraction |
|----------------|-------|----------|
| NORMAL | 45 | 45% |
| VARIABLE | 50 | 50% |
| FADING | 5 | 5% |

The 50% variable fraction is consistent with known brown dwarf populations.

{figure_html[4]}

### 3.4 Discovery of Fading Thermal Orphans

**The most significant result is the identification of four sources exhibiting systematic fading over the 10-year NEOWISE baseline.**

| Designation | W1−W2 (mag) | T_eff (K) | PM (mas/yr) | Fade Rate (mmag/yr) | SIMBAD |
|-------------|-------------|-----------|-------------|---------------------|--------|
| J143046.35−025927.8 | **3.37** | 293 | 55 | 25.5 | Unclassified |
| J231029.40−060547.3 | 1.75 | 258 | 165 | **52.6** | Unclassified |
| J193547.43+601201.5 | 1.53 | 251 | **306** | 22.9 | Unclassified |
| J060501.01−545944.5 | 2.00 | 253 | **359** | 17.9 | Unclassified |

{figure_html[5]}

**Key characteristics:**
- **Extreme W1−W2 colors**: All four have W1−W2 > 1.5 mag
- **Room temperature emission**: T_eff range 251–293 K
- **High proper motion**: Three have μ > 150 mas/yr, implying distances of 18–40 pc
- **No prior identification**: Not catalogued in SIMBAD or VizieR

---

## 4. Discussion

### 4.1 Y Dwarf Interpretation

The most parsimonious explanation is that these objects are **extremely cold Y-type brown dwarfs**:

| Evidence | Observation | Y Dwarf Expectation |
|----------|-------------|---------------------|
| Colors | W1−W2 = 1.53–3.37 mag | Y dwarfs: W1−W2 > 1.5 mag ✓ |
| Temperature | T_eff = 250–293 K | Y dwarfs: T < 400 K ✓ |
| Proper motion | μ = 55–359 mas/yr | Nearby objects ✓ |
| Optical invisibility | No Gaia/PS1/Legacy | Expected for T < 300 K ✓ |

**Possible mechanisms for fading:**
1. **Secular cooling**: Young objects (< 100 Myr) cooling rapidly
2. **Atmospheric variability**: Evolving cloud properties
3. **Unresolved binarity**: Orbital geometry changes

### 4.2 Constraints on Non-Natural Origins

While our search was motivated partly by the technosignature hypothesis, evidence argues against artificial origins:

- High proper motions → distances of 20–100 pc → luminosities of ~10⁻⁶ L_☉
- All properties consistent with brown dwarf populations
- Fading naturally explained by cooling

**Conclusion: The TASNI sample is well-explained by natural astrophysical sources.**

### 4.3 Future Observations

**Priority 1: Near-Infrared Spectroscopy**
- Keck/NIRES or VLT/KMOS (1–2.5 μm)
- Confirm Y dwarf via CH₄, H₂O, NH₃ features

**Priority 2: Parallax Measurements**
- Ground-based astrometry for model-independent distances

**Priority 3: JWST/MIRI Spectroscopy**
- 5–28 μm probes SED peak for 250 K objects

---

## 5. Conclusions

1. **Pipeline Results**: From 747 million AllWISE sources → 4,137 thermal anomalies → 100 golden targets

2. **Golden Sample**: Mean T_eff = 265 K, mean W1−W2 = 1.99 mag, 87% cooler than room temperature

3. **Discovery**: Four fading thermal orphans with unprecedented properties—none in any astronomical catalog

4. **Interpretation**: Most likely extremely cold Y-type brown dwarfs

5. **No evidence for artificial origins**: All properties consistent with natural sources

> **The four fading thermal orphans represent a potentially new class of ultracool dwarf or a previously uncharacterized variability phenomenon. Spectroscopic confirmation is urgently needed.**

---

## Acknowledgments

This publication makes use of data from WISE/NEOWISE, Gaia, SIMBAD, and VizieR.

---

## References

Baraffe, I., et al. 2003, A&A, 402, 701 •
Cushing, M. C., et al. 2011, ApJ, 743, 50 •
Dyson, F. J. 1960, Science, 131, 1667 •
Kirkpatrick, J. D., et al. 2012, ApJ, 753, 156 •
Kirkpatrick, J. D., et al. 2021, ApJS, 253, 7 •
Luhman, K. L. 2014, ApJL, 786, L18 •
Mainzer, A., et al. 2014, ApJ, 792, 30 •
Meisner, A. M., et al. 2020, ApJ, 899, 123 •
Wright, E. L., et al. 2010, AJ, 140, 1868 •
Wright, J. T., et al. 2014, ApJ, 792, 26
"""

    return paper_content


def main():
    print("Compiling TASNI paper with figures to PDF...")

    # Combine markdown
    paper_md = combine_paper()

    # Convert to HTML
    md = markdown.Markdown(extensions=["tables", "fenced_code"])
    html_content = md.convert(paper_md)

    # Full HTML document
    html_doc = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>TASNI Paper</title>
</head>
<body>
{html_content}
</body>
</html>
"""

    # Save HTML for inspection
    html_path = PAPER_DIR / "tasni_paper.html"
    html_path.write_text(html_doc)
    print(f"HTML saved to: {html_path}")

    # Convert to PDF
    pdf_path = PAPER_DIR / "tasni_paper.pdf"
    HTML(string=html_doc).write_pdf(pdf_path, stylesheets=[CSS(string=CSS_STYLE)])
    print(f"PDF saved to: {pdf_path}")

    # Get file size
    size_kb = pdf_path.stat().st_size / 1024
    print(f"PDF size: {size_kb:.1f} KB")

    # Count pages (approximate)
    print("Figures included: 6")


if __name__ == "__main__":
    main()
