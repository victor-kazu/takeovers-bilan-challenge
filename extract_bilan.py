from __future__ import annotations

import glob
import json
import math
import os
import re
import time
import unicodedata
import pymupdf

TARGET_DOCUMENTS = [
    {"siren": "820561470", "doc_id": "6493e4372f502414800f8164", "pdf": "data/820561470/bilans/pdf/bilan_2023-06-05_6493e4372f502414800f8164.pdf"},
    {"siren": "820561470", "doc_id": "6543d3fd08093cdace058668", "pdf": "data/820561470/bilans/pdf/bilan_2023-06-13_6543d3fd08093cdace058668.pdf"},
    {"siren": "820561470", "doc_id": "67458f18cea78a70070fa226", "pdf": "data/820561470/bilans/pdf/bilan_2024-01-15_67458f18cea78a70070fa226.pdf"},
    {"siren": "328024377", "doc_id": "63e8ebbb54febda17c19ee7c", "pdf": "data/328024377/bilans/pdf/bilan_2020-12-24_63e8ebbb54febda17c19ee7c.pdf"},
    {"siren": "328024377", "doc_id": "63e8ebbb54febda17c19ee7d", "pdf": "data/328024377/bilans/pdf/bilan_2021-12-17_63e8ebbb54febda17c19ee7d.pdf"},
    {"siren": "328024377", "doc_id": "63e8ebbb54febda17c19ee7e", "pdf": "data/328024377/bilans/pdf/bilan_2022-12-13_63e8ebbb54febda17c19ee7e.pdf"},
    {"siren": "445070311", "doc_id": "63e2481c916269756a09542b", "pdf": "data/445070311/bilans/pdf/bilan_2022-02-14_63e2481c916269756a09542b.pdf"},
    {"siren": "445070311", "doc_id": "65a4095d5fd178b16b09b860", "pdf": "data/445070311/bilans/pdf/bilan_2023-11-21_65a4095d5fd178b16b09b860.pdf"},
    {"siren": "445070311", "doc_id": "6860f28ca0138eae340c7453", "pdf": "data/445070311/bilans/pdf/bilan_2025-05-15_6860f28ca0138eae340c7453.pdf"},
    {"siren": "504304205", "doc_id": "63e13943526e1f30cd100db5", "pdf": "data/504304205/bilans/pdf/bilan_2017-05-31_63e13943526e1f30cd100db5.pdf"},
    {"siren": "504304205", "doc_id": "63e13943526e1f30cd100db6", "pdf": "data/504304205/bilans/pdf/bilan_2018-10-24_63e13943526e1f30cd100db6.pdf"},
    {"siren": "504304205", "doc_id": "66cd893cedec9b09d50191e8", "pdf": "data/504304205/bilans/pdf/bilan_2024-08-06_66cd893cedec9b09d50191e8.pdf"},
    {"siren": "401009741", "doc_id": "63e881158be6eb9f9d1ff975", "pdf": "data/401009741/bilans/pdf/bilan_2022-11-30_63e881158be6eb9f9d1ff975.pdf"},
    {"siren": "401009741", "doc_id": "65784e5da67d84faf4042736", "pdf": "data/401009741/bilans/pdf/bilan_2023-11-20_65784e5da67d84faf4042736.pdf"},
    {"siren": "401009741", "doc_id": "68f0a715f28d8aaf48046416", "pdf": "data/401009741/bilans/pdf/bilan_2025-10-03_68f0a715f28d8aaf48046416.pdf"},
]

def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()

def load_ocr_pages(siren: str, doc_id: str) -> list[dict]:
    ocr_dir = os.path.join("data", siren, "bilans", "ocr", doc_id)
    files = sorted(glob.glob(os.path.join(ocr_dir, "page_*.json")))
    pages = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            pages.append(json.load(fp))
    return pages

def detect_document_unit(pages: list[dict]) -> str:
    keur_patterns = [
        r"\bk€\b", r"\bkeur\b", r"en milliers", r"milliers d['’]?euros",
        r"montants.*en k", r"exprimes? en k",
    ]
    for page in pages:
        for item in page.get("ocr", []):
            norm = strip_accents(item.get("text", ""))
            for pattern in keur_patterns:
                if re.search(pattern, norm):
                    return "kEUR"
    return "EUR"

def classify_pages(pages: list[dict]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for page_idx, page in enumerate(pages, start=1):
        lines = [strip_accents(item.get("text", "")) for item in page.get("ocr", [])]
        combined = " ".join(lines)
        if "2050" not in mapping and ("2050" in combined or ("bilan" in combined and "actif" in combined and "passif" not in combined[:120])):
            mapping["2050"] = page_idx
        if "2051" not in mapping and ("2051" in combined or ("bilan" in combined and "passif" in combined)):
            mapping["2051"] = page_idx
        if "2052" not in mapping and ("2052" in combined or ("compte de resultat" in combined and "charges" in combined)):
            mapping["2052"] = page_idx
    return mapping

def polygon_to_norm(polygon: list[list[float]], page_w_pt: float, page_h_pt: float) -> list[float]:
    scale = 300.0 / 72.0
    w_px, h_px = page_w_pt * scale, page_h_pt * scale
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return [
        round(max(0.0, min(xs) / w_px), 4),
        round(max(0.0, min(ys) / h_px), 4),
        round(min(1.0, max(xs) / w_px), 4),
        round(min(1.0, max(ys) / h_px), 4),
    ]

def clean_number(text: str) -> float | None:
    cleaned = text.replace(" ", "").replace("\u00a0", "").replace(",", ".")
    m = re.search(r"[-−]?\d+(\.\d+)?", cleaned)
    if not m:
        return None
    try:
        val = float(m.group(0).replace("−", "-"))
        return val
    except ValueError:
        return None

def merge_row_number_tokens(items_on_row: list[dict], page_w_pt: float, page_h_pt: float) -> list[dict]:
    """Merge only tightly-spaced digit fragments that belong to the same number."""
    sorted_items = sorted(
        [it for it in items_on_row if clean_number(it.get("text", "")) is not None],
        key=lambda x: polygon_to_norm(x["polygon"], page_w_pt, page_h_pt)[0]
    )

    if not sorted_items:
        return []

    merged = []
    current_cluster = [sorted_items[0]]

    for it in sorted_items[1:]:
        prev_box = polygon_to_norm(current_cluster[-1]["polygon"], page_w_pt, page_h_pt)
        curr_box = polygon_to_norm(it["polygon"], page_w_pt, page_h_pt)
        
        # Horizontal gap must be tiny (< 0.012 page width) and vertically aligned
        gap = curr_box[0] - prev_box[2]
        v_diff = abs(((curr_box[1] + curr_box[3]) / 2) - ((prev_box[1] + prev_box[3]) / 2))
        
        # Only merge if gap is very tight (within the same column cell)
        if 0 <= gap < 0.012 and v_diff < 0.010:
            current_cluster.append(it)
        else:
            merged.append(current_cluster)
            current_cluster = [it]
    merged.append(current_cluster)

    results = []
    for cluster in merged:
        all_boxes = [polygon_to_norm(c["polygon"], page_w_pt, page_h_pt) for c in cluster]
        combined_box = [
            min(b[0] for b in all_boxes),
            min(b[1] for b in all_boxes),
            max(b[2] for b in all_boxes),
            max(b[3] for b in all_boxes),
        ]
        combined_text = " ".join(c.get("text", "").strip() for c in cluster)
        val = clean_number(combined_text)
        if val is not None:
            results.append({
                "value": val,
                "bbox": combined_box,
                "snippet": combined_text,
                "confidence": min(float(c.get("score", 0.95)) for c in cluster)
            })
    return results

def extract_field_value(
    page_data: dict,
    page_w_pt: float,
    page_h_pt: float,
    code: str,
    label_keywords: list[str],
    col_idx: int = 0
) -> dict | None:
    ocr_items = page_data.get("ocr", [])

    # 1. Fallback / direct: Search by row label first if it's a major total line
    label_item = None
    for item in ocr_items:
        norm = strip_accents(item.get("text", ""))
        if any(kw in norm for kw in label_keywords):
            bbox = polygon_to_norm(item["polygon"], page_w_pt, page_h_pt)
            if bbox[0] < 0.48 and bbox[1] > 0.30:  # Must be in lower table section
                label_item = (item, bbox)
                break

    if label_item:
        _, l_bbox = label_item
        y_center = (l_bbox[1] + l_bbox[3]) / 2.0

        row_items = [
            it for it in ocr_items
            if abs(((polygon_to_norm(it["polygon"], page_w_pt, page_h_pt)[1] + polygon_to_norm(it["polygon"], page_w_pt, page_h_pt)[3]) / 2.0) - y_center) <= 0.015
            and polygon_to_norm(it["polygon"], page_w_pt, page_h_pt)[0] > 0.45
        ]
        merged_candidates = merge_row_number_tokens(row_items, page_w_pt, page_h_pt)
        merged_candidates = [c for c in merged_candidates if not (abs(c["value"]) in [1.0, 2.0, 3.0, 4.0] and len(c["snippet"].strip()) <= 2)]

        if merged_candidates:
            merged_candidates.sort(key=lambda x: x["bbox"][0])
            pick = merged_candidates[min(col_idx, len(merged_candidates) - 1)]
            return pick

    # 2. Search by 2-letter box code
    code_items = []
    for item in ocr_items:
        text = item.get("text", "").strip().upper()
        if re.search(rf"\b{re.escape(code)}\b", text):
            bbox = polygon_to_norm(item["polygon"], page_w_pt, page_h_pt)
            code_items.append((item, bbox))

    if code_items:
        valid_codes = [c for c in code_items if 0.15 <= c[1][1] <= 0.95]
        target_code, c_bbox = valid_codes[-1] if valid_codes else code_items[-1]
        c_y = (c_bbox[1] + c_bbox[3]) / 2.0
        c_x = c_bbox[2]

        row_items = [
            it for it in ocr_items
            if abs(((polygon_to_norm(it["polygon"], page_w_pt, page_h_pt)[1] + polygon_to_norm(it["polygon"], page_w_pt, page_h_pt)[3]) / 2.0) - c_y) <= 0.018
            and polygon_to_norm(it["polygon"], page_w_pt, page_h_pt)[0] >= c_x - 0.05
        ]
        merged_candidates = merge_row_number_tokens(row_items, page_w_pt, page_h_pt)
        merged_candidates = [c for c in merged_candidates if not (abs(c["value"]) in [1.0, 2.0, 3.0, 4.0] and len(c["snippet"].strip()) <= 2)]

        if merged_candidates:
            merged_candidates.sort(key=lambda x: x["bbox"][0])
            pick = merged_candidates[min(col_idx, len(merged_candidates) - 1)]
            return pick

    return None
def process_filing(doc_info: dict) -> dict:
    siren = doc_info["siren"]
    doc_id = doc_info["doc_id"]
    pdf_path = doc_info["pdf"]

    pages = load_ocr_pages(siren, doc_id)
    unit = detect_document_unit(pages)
    page_map = classify_pages(pages)
    doc_fitz = pymupdf.open(pdf_path)
    fields = []

    actif_res = None
    passif_total_res = None

    # 1. Total Assets: Liasse 2050 (Actif Net)
    if "2050" in page_map:
        p_num = page_map["2050"]
        p_fitz = doc_fitz[p_num - 1]
        actif_res = extract_field_value(
            pages[p_num - 1], p_fitz.rect.width, p_fitz.rect.height,
            code="CL", label_keywords=["total general", "total ( i a vi )", "total (i a vi)"],
            col_idx=2
        )

    # 2. Total Passif (TOTAL GENERAL on Form 2051) for self-check reconciliation
    if "2051" in page_map:
        p_num_passif = page_map["2051"]
        p_fitz_passif = doc_fitz[p_num_passif - 1]
        passif_total_res = extract_field_value(
            pages[p_num_passif - 1], p_fitz_passif.rect.width, p_fitz_passif.rect.height,
            code="DQ", label_keywords=["total general", "total (i a iv)", "total ( i a iv )"],
            col_idx=0
        )

    # Reconciliation Anchor Check (Actif == Passif)
    if actif_res and passif_total_res:
        a_val = actif_res["value"]
        p_val = passif_total_res["value"]
        # Allow up to 1.5% relative error for minor OCR noise
        rel_diff = abs(a_val - p_val) / max(a_val, p_val) if max(a_val, p_val) > 0 else 0
        if rel_diff <= 0.015:
            actif_res["confidence"] = 1.0  # Confirmed by accounting reconciliation
        elif a_val <= 50 or (p_val > a_val and a_val < 100000 and p_val > 100000):
            # If Actif caught an anomalous small line, fallback to reconciled Passif total
            actif_res = passif_total_res

    if actif_res and actif_res["value"] > 50:
        fields.append({
            "field_key": "BS_TOTAL_ASSETS_FRGAAP",
            "value": actif_res["value"],
            "unit": unit,
            "page": page_map["2050"],
            "bbox": actif_res["bbox"],
            "snippet": actif_res["snippet"],
            "confidence": actif_res["confidence"],
        })

    # 3. Total Equity: Box code 'DL' or label 'total capitaux propres' on Form 2051
    if "2051" in page_map:
        p_num = page_map["2051"]
        p_fitz = doc_fitz[p_num - 1]
        res = extract_field_value(
            pages[p_num - 1], p_fitz.rect.width, p_fitz.rect.height,
            code="DL", label_keywords=["total capitaux propres", "total i"],
            col_idx=0
        )
        if res and res["value"] > 50:
            fields.append({
                "field_key": "BS_TOTAL_EQUITY_FRGAAP",
                "value": res["value"],
                "unit": unit,
                "page": p_num,
                "bbox": res["bbox"],
                "snippet": res["snippet"],
                "confidence": res["confidence"],
            })

    # 4. Share Capital: Box code 'DA' or label 'capital social' on Form 2051
    if "2051" in page_map:
        p_num = page_map["2051"]
        p_fitz = doc_fitz[p_num - 1]
        res = extract_field_value(
            pages[p_num - 1], p_fitz.rect.width, p_fitz.rect.height,
            code="DA", label_keywords=["capital social", "capital individuel"],
            col_idx=0
        )
        if res and res["value"] > 50:
            fields.append({
                "field_key": "BS_CAPITAL_EQUITY_FRGAAP",
                "value": res["value"],
                "unit": unit,
                "page": p_num,
                "bbox": res["bbox"],
                "snippet": res["snippet"],
                "confidence": res["confidence"],
            })

    return {
        "pdf": pdf_path,
        "siren": siren,
        "fiscal_year_end": None,
        "fields": fields,
    }

def run_full_pipeline():
    start_time = time.time()
    total_pages = 0
    all_docs_output = []

    for doc in TARGET_DOCUMENTS:
        pages = load_ocr_pages(doc["siren"], doc["doc_id"])
        total_pages += len(pages)
        res = process_filing(doc)
        all_docs_output.append(res)
        print(f"SIREN {res['siren']} | Matched {len(res['fields'])} fields in {os.path.basename(res['pdf'])}")

    elapsed = time.time() - start_time
    sec_per_page = round(elapsed / total_pages, 4) if total_pages else 0.0

    output = {
        "documents": all_docs_output,
        "run": {
            "cost_eur_per_page": 0.0,
            "seconds_per_page": sec_per_page,
            "pages_processed": total_pages,
            "model": "provided OCR + French tax liasse code & geometric regex parser",
            "notes": "Cost is 0 EUR because extraction relies entirely on the provided OCR and local geometric parsing. High precision prioritized on key balance sheet metrics."
        }
    }

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nWrote results.json successfully! Processed {total_pages} pages in {elapsed:.2f}s ({sec_per_page}s/page).")

if __name__ == "__main__":
    run_full_pipeline()