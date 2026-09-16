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
        if "2050" not in mapping and ("2050" in combined or ("bilan" in combined and "actif" in combined and "passif" not in combined[:150])):
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
        # Exclude small single digits that represent table column headings like (1), (2), (3)
        if abs(val) in [1.0, 2.0, 3.0, 4.0] and len(text.strip()) <= 3:
            return None
        return val
    except ValueError:
        return None

def extract_field_value(
    page_data: dict,
    page_w_pt: float,
    page_h_pt: float,
    code: str,
    label_keywords: list[str],
    col_idx: int = 0
) -> dict | None:
    ocr_items = page_data.get("ocr", [])
    
    # 1. First preference: look for standard 2-letter box code
    code_items = []
    for item in ocr_items:
        text = item.get("text", "").strip().upper()
        if re.search(rf"\b{re.escape(code)}\b", text):
            bbox = polygon_to_norm(item["polygon"], page_w_pt, page_h_pt)
            code_items.append((item, bbox))
            
    if code_items:
        valid_codes = [c for c in code_items if 0.1 <= c[1][1] <= 0.95]
        target_code, c_bbox = valid_codes[-1] if valid_codes else code_items[-1]
        c_y = (c_bbox[1] + c_bbox[3]) / 2.0
        c_x = c_bbox[2]

        candidates = []
        for item in ocr_items:
            bbox = polygon_to_norm(item["polygon"], page_w_pt, page_h_pt)
            num = clean_number(item.get("text", ""))
            if num is None:
                continue
            item_y = (bbox[1] + bbox[3]) / 2.0
            item_x = bbox[0]
            if item_x >= c_x - 0.05 and abs(item_y - c_y) <= 0.025:
                candidates.append((item, bbox, num, item_x))

        if candidates:
            candidates.sort(key=lambda x: x[3])
            pick = candidates[min(col_idx, len(candidates) - 1)]
            return {
                "value": pick[2],
                "bbox": pick[1],
                "snippet": pick[0].get("text", ""),
                "confidence": float(pick[0].get("score", 0.95)),
            }

    # 2. Fallback: Horizontal row alignment from label keyword
    label_item = None
    for item in ocr_items:
        norm = strip_accents(item.get("text", ""))
        if any(kw in norm for kw in label_keywords):
            bbox = polygon_to_norm(item["polygon"], page_w_pt, page_h_pt)
            if bbox[0] < 0.48:
                label_item = (item, bbox)
                break

    if not label_item:
        return None

    _, l_bbox = label_item
    y_center = (l_bbox[1] + l_bbox[3]) / 2.0

    row_candidates = []
    for item in ocr_items:
        bbox = polygon_to_norm(item["polygon"], page_w_pt, page_h_pt)
        num = clean_number(item.get("text", ""))
        if num is None:
            continue
        item_y = (bbox[1] + bbox[3]) / 2.0
        if abs(item_y - y_center) <= 0.020 and bbox[0] > 0.45:
            row_candidates.append((item, bbox, num, bbox[0]))

    if not row_candidates:
        return None

    row_candidates.sort(key=lambda x: x[3])
    pick = row_candidates[min(col_idx, len(row_candidates) - 1)]
    return {
        "value": pick[2],
        "bbox": pick[1],
        "snippet": pick[0].get("text", ""),
        "confidence": float(pick[0].get("score", 0.90)),
    }

def process_filing(doc_info: dict) -> dict:
    siren = doc_info["siren"]
    doc_id = doc_info["doc_id"]
    pdf_path = doc_info["pdf"]

    pages = load_ocr_pages(siren, doc_id)
    unit = detect_document_unit(pages)
    page_map = classify_pages(pages)
    doc_fitz = pymupdf.open(pdf_path)
    fields = []

    # 1. Total Assets: Box code 'CL' or label 'total general' on Liasse 2050 (col_idx=2: Net)
    if "2050" in page_map:
        p_num = page_map["2050"]
        p_fitz = doc_fitz[p_num - 1]
        res = extract_field_value(
            pages[p_num - 1], p_fitz.rect.width, p_fitz.rect.height,
            code="CL", label_keywords=["total general", "total ( i a vi )", "total (i a vi)"],
            col_idx=2
        )
        if res:
            fields.append({
                "field_key": "BS_TOTAL_ASSETS_FRGAAP",
                "value": res["value"],
                "unit": unit,
                "page": p_num,
                "bbox": res["bbox"],
                "snippet": res["snippet"],
                "confidence": res["confidence"],
            })

    # 2. Total Equity: Box code 'DL' or label 'total capitaux propres' on Liasse 2051
    if "2051" in page_map:
        p_num = page_map["2051"]
        p_fitz = doc_fitz[p_num - 1]
        res = extract_field_value(
            pages[p_num - 1], p_fitz.rect.width, p_fitz.rect.height,
            code="DL", label_keywords=["total capitaux propres", "total i"],
            col_idx=0
        )
        if res:
            fields.append({
                "field_key": "BS_TOTAL_EQUITY_FRGAAP",
                "value": res["value"],
                "unit": unit,
                "page": p_num,
                "bbox": res["bbox"],
                "snippet": res["snippet"],
                "confidence": res["confidence"],
            })

    # 3. Share Capital: Box code 'DA' or label 'capital social' on Liasse 2051
    if "2051" in page_map:
        p_num = page_map["2051"]
        p_fitz = doc_fitz[p_num - 1]
        res = extract_field_value(
            pages[p_num - 1], p_fitz.rect.width, p_fitz.rect.height,
            code="DA", label_keywords=["capital social", "capital individuel"],
            col_idx=0
        )
        if res:
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