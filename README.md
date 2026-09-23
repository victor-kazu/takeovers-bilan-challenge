# French Tax Filing (*Liasse Fiscale*) Extraction Pipeline

An automated, zero-marginal-cost financial extraction pipeline designed to parse up to 12 standardized French GAAP metrics from annual tax filings (*liasses fiscales*). The engine handles both standard (*Régime Réel Normal* / 2050 series) and simplified (*Régime Réel Simplifié* / 2033 series) filings, outputting normalized values, dynamic currency units (`EUR` vs `kEUR`), 1-indexed page locations, and normalized bounding box coordinates `[x0, y0, x1, y1]`.

* **Video Walkthrough (3 min)**: `https://www.loom.com/share/c17ec271253e4c1586a764aeb9e07434`

---

## How to Run

### 1. Requirements
Ensure Python 3.9+ is installed, then install the local processing dependencies:
```bash
pip install pymupdf pillow opencv-python numpy scikit-learn
```

### 2. Environment Setup
Copy the environment template:
```bash
cp .env.example .env
```
*(No paid cloud API keys or vision provider credentials are required; the pipeline executes entirely on-device).*

### 3. Execution
Run the pipeline from the repository root:
```bash
python extract_bilan.py
```
The script will process all 15 scoped filings across the 5 target companies and write the compliant output to `results.json`.

---

## The Trade-Off: What Was Chosen, Cost, Runtime, and Accuracy

### Measured Run Metrics
* **Cost**: **0.00 EUR / page**. Derived directly from local execution: zero API tokens or paid inference endpoints were used.
* **Throughput**: **0.024 seconds / page** (~9.97 seconds across all 415 pages in the 15 scoped documents).
* **Coverage Floor**: **5 to 11 fields extracted per filing** across the corpus without missing-document crashes or blank runs.

### Architectural Decisions

| Technique Chosen | Alternative Considered | Trade-off Rationale |
|---|---|---|
| **Global Text-Cached Search** | Rigid Page Header Classification | Scanned or clipped headers caused page classifiers to miss statements entirely. Pre-caching page text strings allows scanning every page for unique Cerfa line codes in sub-milliseconds without runtime penalties. |
| **Adaptive Cone of Vision** | Affine OpenCV Deskewing (`cv2.warpAffine`) | Physical image deskewing introduced interpolation blur and micro-rotation artifacts on clean PDFs. The "Cone of Vision" expands the row search window linearly (±0.012 near the label to ±0.040 · Δx at the margin), capturing tilted rows on skewed scans (SIREN `820561470`) without pulling false positives on straight pages. |
| **Dynamic K-Means with O(1) Bypass** | Static Horizontal Coordinate Slicing | Hardcoded bounding coordinates break when columns like "Amortissement" or "Brut" are omitted. Unsupervised clustering identifies actual column centroids dynamically, while an O(1) bypass skips clustering when token counts match expected columns. |
| **Form-Agnostic Dual Mapping** | Standard Regime (2050) Only | Entities frequently transition between standard and simplified regimes (e.g., SIREN `445070311`). Supporting both 2050 series (`CL`, `DL`, `FL`) and 2033 series (`090`, `142`, `210`) unlocked high extraction yields across filing regimes. |

### What I Would Do Differently With One Week
1. **Targeted Offline Multimodal Fallback**: Route low-coverage documents (filings with ≤ 5 fields) to a small, quantized open-weights vision-language model (e.g., Qwen2-VL-7B or Florence-2) running locally on CPU/GPU to parse irregular annex layouts while maintaining zero API cost.
2. **Multi-Period Temporal Reconciliation**: Use the prior-year (N-1) column from year T to cross-verify the current-year (N) column of filing T-1 for the same SIREN, programmatically fixing isolated OCR digit-confusion errors.

---

## What Was Cut, What Is Broken, and Why

In financial extraction, omitting an unverified figure is preferable to reporting a fabricated one:

1. **Multi-Line Cost of Goods Sold (`PL_COGS_FRGAAP`) on Simplified Schedules**:
   * *What happened*: COGS is constructed from purchases plus change in inventory rather than reported on a single line.
   * *Why cut*: When inventory change rows are omitted or lumped into subcontracting, synthesizing partial sums leads to distorted margins. Partial multi-line additions were dropped in favor of strict anchor matches.
2. **Unstructured Headcount Guessing (`META_AVG_WORKFORCE_FRGAAP`)**:
   * *What happened*: Earlier iterations matched 3-digit simplified codes (`376`) against random dates, postal codes, and note references, reporting impossible headcounts (e.g., 4,376 or 376,460 employees).
   * *Resolution*: Implemented token-boundary validation and a strict sanity ceiling (0 ≤ workforce ≤ 500). Unstructured mentions in narrative text notes were left out.
3. **Calendar Years Mistaken for Financial Results**:
   * *Known edge case*: In SIREN `328024377` (`2021-12-17` and `2022-12-13`), table column headers (`2021` / `2022`) near the top margin (y ≈ 0.02) can occasionally register as financial results when the line code `GP` is degraded.
4. **Non-Cerfa Auditor Annexes**:
   * *Known limitation*: Filings like SIREN `328024377` (2020) and `401009741` (2025) feature auditor summaries that depart from Cerfa grids. Global searches were bounded to prevent pulling non-reconciling figures into primary balance sheet slots.

---

## How AI Tools Were Used

In compliance with the challenge guidelines:

* **Coordinate Projection Formulation**: AI assisted in scaffolding the coordinate normalization pipeline (`polygon_to_norm`), mapping raw 300-DPI pixel bounding polygons from the OCR payloads into normalized `[x0, y0, x1, y1]` fractional coordinates with a top-left origin `[0.0, 1.0]`.
* **Cerfa Box Code Taxonomy & Regime Routing**: AI accelerated domain research by indexing corresponding Cerfa line codes across standard (*Régime Réel Normal* / Forms 2050–2053) and simplified (*Régime Réel Simplifié* / Forms 2033-A–B) tax regimes (e.g., mapping Revenue to both `FL` and `210`, and Total Assets to both `CL` and `090`).
* **Geometric Skew Modeling & "Cone of Vision"**: When early image-level OpenCV affine transformations (`cv2.warpAffine` / `cv2.getRotationMatrix2D`) caused blur and micro-rotation artifacts on straight pages, AI helped design an adaptive geometric "Cone of Vision." This search corridor maintains a narrow vertical tolerance (±0.012) near the anchor label and expands dynamically (±0.040 · Δx) across the page to swallow tilted rows without pulling in adjacent vertical text lines.
* **Accounting Edge-Case & String Normalization**: AI assisted in crafting parsing rules to handle non-standard accounting syntax:
  * Extracting French parenthetical negative accounting numbers (e.g., `(1 000)` → `-1000.0`).
  * Stripping leading table hyphens and dashes so positive values (such as Cash and Revenue) were not wrongly classified as negative.
  * Designing multi-token clustering heuristics to merge horizontally fragmented digits (e.g., `"1476"` and `"746"` into `1476746`) while enforcing horizontal distance and cluster-width constraints (< 0.16) to prevent runaway multi-column concatenations.
* **Runtime Optimization & K-Means Bypass**: AI was used to profile and eliminate computational bottlenecks:
  * Diagnosed a slowdown caused by running K-Means across hundreds of irrelevant pages and helped implement pre-computed raw string/token caches to achieve O(1) page skipping.
  * Designed an in-memory centroid cache per page.
  * Added an O(1) K-Means bypass that sorts numbers directly whenever token counts match expected table columns, dropping total execution time from over 240 seconds to under 10 seconds (~0.024s/page).
* **Output Validation & Schema Compliance**: AI was used to audit intermediate `results.json` files against `results.schema.json` and `financial_fields.json`, verifying coordinate bounds, mandatory field keys, numerical data types, and unit constraints (`count` for workforce, `EUR`/`kEUR` for monetary fields). Ground-truth bounding boxes were cross-checked using `tools/bbox_viewer.py`.
