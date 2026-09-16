# French Tax Filing (*Liasse Fiscale*) Extraction Pipeline

An automated, zero-marginal-cost financial extraction pipeline designed to parse up to 12 standardized French GAAP metrics from annual tax filings (*liasses fiscales*). The engine handles both standard (*Régime Réel Normal* / 2050 series) and simplified (*Régime Réel Simplifié* / 2033 series) filings, outputting normalized values, dynamic currency units (`EUR` vs `kEUR`), 1-indexed page locations, and normalized bounding box coordinates `[x0, y0, x1, y1]`.

* **Video Walkthrough (3 min)**: `[Insert your Loom / Screen Recording link here]`

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
* **Coordinate Projection Formulation**: AI was used to derive the normalized coordinate transformation formulas (`polygon_to_norm`), mapping 300-DPI pixel space to `[0, 1]` bounding boxes with top-left origins.
* **Cerfa Box Code Taxonomy**: AI accelerated domain research by indexing corresponding Cerfa line codes across standard (2050–2053) and simplified (2033-A/B) schedules.
* **Accounting Edge Case Handling**: AI helped identify regressions involving parenthetical negative accounting numbers (e.g., `(1 000)` → `-1000.0`) and leading-hyphen table formatting.
* **Ground-Truth Box Auditing**: Bounding box outputs were inspected against PDF page layouts using `tools/bbox_viewer.py` to confirm alignment with current Exercise (N) columns.
