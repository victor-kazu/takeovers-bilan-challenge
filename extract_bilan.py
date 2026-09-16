from __future__ import annotations

import glob
import json
import math
import os
import re
import time
import unicodedata

import pymupdf
import cv2
import numpy as np
from sklearn.cluster import KMeans

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

def get_page_skew_angle(page_fitz) -> float:
    pix = page_fitz.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if pix.n >= 3 else img
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    angles = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 300:
            [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
            angle_deg = math.degrees(math.atan2(float(vy[0]), float(vx[0])))
            if abs(angle_deg) < 15:
                angles.append(angle_deg)
    return float(np.median(angles)) if angles else 0.0

def deskew_ocr_page(page_data: dict, angle: float, page_w_pt: float, page_h_pt: float):
    if abs(angle) < 0.5:
        return 
    scale = 300.0 / 72.0
    cx, cy = (page_w_pt * scale) / 2.0, (page_h_pt * scale) / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    for item in page_data.get("ocr", []):
        pts = np.array([item["polygon"]], dtype=np.float32)
        item["polygon"] = cv2.transform(pts, M)[0].tolist()

def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)).lower()

def load_ocr_pages(siren: str, doc_id: str) -> list[dict]:
    files = sorted(glob.glob(os.path.join("data", siren, "bilans", "ocr", doc_id, "page_*.json")))
    return [json.load(open(f, "r", encoding="utf-8")) for f in files]

def detect_document_unit(pages: list[dict]) -> str:
    keur_patterns = [r"\bk€\b", r"\bkeur\b", r"en milliers", r"milliers d['’]?euros", r"montants.*en k", r"exprimes? en k"]
    for page in pages:
        for item in page.get("ocr", []):
            norm = strip_accents(item.get("text", ""))
            if any(re.search(pat, norm) for pat in keur_patterns): return "kEUR"
    return "EUR"

def classify_pages(pages: list[dict]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for page_idx, page in enumerate(pages, start=1):
        combined = " ".join([strip_accents(item.get("text", "")) for item in page.get("ocr", [])])
        
        # Standard Forms (Régime Normal)
        if "2050" not in mapping and ("2050" in combined or ("bilan" in combined and "actif" in combined and "passif" not in combined[:120])): mapping["2050"] = page_idx
        if "2051" not in mapping and ("2051" in combined or ("bilan" in combined and "passif" in combined)): mapping["2051"] = page_idx
        if "2052" not in mapping and ("2052" in combined or ("compte de resultat" in combined and "charges d'exploitation" in combined)): mapping["2052"] = page_idx
        if "2053" not in mapping and ("2053" in combined or ("compte de resultat" in combined and "suite" in combined)): mapping["2053"] = page_idx
        
        # Simplified Forms (Régime Simplifié)
        if "2033-A" not in mapping and ("2033-a" in combined or "2033 a" in combined or "10956" in combined or ("bilan simplifie" in combined)): mapping["2033-A"] = page_idx
        if "2033-B" not in mapping and ("2033-b" in combined or "2033 b" in combined or "10957" in combined or ("compte de resultat simplifie" in combined)): mapping["2033-B"] = page_idx
        
        if "2058-C" not in mapping and ("2058-c" in combined or "2058 c" in combined): mapping["2058-C"] = page_idx
    return mapping

def polygon_to_norm(polygon: list[list[float]], page_w_pt: float, page_h_pt: float) -> list[float]:
    scale = 300.0 / 72.0
    w_px, h_px = page_w_pt * scale, page_h_pt * scale
    xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
    return [round(max(0.0, min(xs) / w_px), 4), round(max(0.0, min(ys) / h_px), 4), round(min(1.0, max(xs) / w_px), 4), round(min(1.0, max(ys) / h_px), 4)]

def clean_number(text: str) -> float | None:
    m = re.search(r"[-−]?\d+(\.\d+)?", text.replace(" ", "").replace("\u00a0", "").replace(",", "."))
    try: return float(m.group(0).replace("−", "-")) if m else None
    except ValueError: return None

def build_column_clusters(ocr_items: list[dict], page_w_pt: float, page_h_pt: float, expected_cols: int) -> list[float]:
    x_centers = [[polygon_to_norm(item["polygon"], page_w_pt, page_h_pt)[0]] for item in ocr_items if clean_number(item.get("text", "")) is not None and polygon_to_norm(item["polygon"], page_w_pt, page_h_pt)[0] > 0.40]
    if len(x_centers) < expected_cols: return []
    kmeans = KMeans(n_clusters=expected_cols, random_state=42, n_init=10)
    kmeans.fit(x_centers)
    return sorted(kmeans.cluster_centers_.flatten())

def merge_row_number_tokens(items_on_row: list[dict], page_w_pt: float, page_h_pt: float) -> list[dict]:
    sorted_items = sorted([it for it in items_on_row if clean_number(it.get("text", "")) is not None], key=lambda x: polygon_to_norm(x["polygon"], page_w_pt, page_h_pt)[0])
    if not sorted_items: return []
    merged, current_cluster = [], [sorted_items[0]]
    for it in sorted_items[1:]:
        prev_box, curr_box = polygon_to_norm(current_cluster[-1]["polygon"], page_w_pt, page_h_pt), polygon_to_norm(it["polygon"], page_w_pt, page_h_pt)
        if 0 <= curr_box[0] - prev_box[2] < 0.012 and abs(((curr_box[1] + curr_box[3]) / 2) - ((prev_box[1] + prev_box[3]) / 2)) < 0.010: current_cluster.append(it)
        else:
            merged.append(current_cluster)
            current_cluster = [it]
    merged.append(current_cluster)

    results = []
    for cluster in merged:
        all_boxes = [polygon_to_norm(c["polygon"], page_w_pt, page_h_pt) for c in cluster]
        combined_text = " ".join(c.get("text", "").strip() for c in cluster)
        val = clean_number(combined_text)
        if val is not None:
            results.append({"value": val, "bbox": [min(b[0] for b in all_boxes), min(b[1] for b in all_boxes), max(b[2] for b in all_boxes), max(b[3] for b in all_boxes)], "snippet": combined_text, "confidence": min(float(c.get("score", 0.95)) for c in cluster)})
    return results

def extract_field_value(page_data: dict, page_w_pt: float, page_h_pt: float, codes: list[str], label_keywords: list[str], col_idx: int = 0, min_y: float = 0.05, max_y: float = 0.95, expected_cols: int = 2) -> dict | None:
    ocr_items = page_data.get("ocr", [])
    centroids = build_column_clusters(ocr_items, page_w_pt, page_h_pt, expected_cols)

    # Box code search (checks multiple possible codes)
    code_items = []
    for code in codes:
        code_items.extend([(it, polygon_to_norm(it["polygon"], page_w_pt, page_h_pt)) for it in ocr_items if re.search(rf"\b{re.escape(code)}\b", it.get("text", "").strip().upper()) and min_y <= polygon_to_norm(it["polygon"], page_w_pt, page_h_pt)[1] <= max_y])
    
    target_y, target_x_min = None, 0.0
    if code_items:
        _, c_bbox = code_items[-1]
        target_y, target_x_min = (c_bbox[1] + c_bbox[3]) / 2.0, c_bbox[2] - 0.05
    else:
        label_item = next(((it, polygon_to_norm(it["polygon"], page_w_pt, page_h_pt)) for it in ocr_items if any(kw in strip_accents(it.get("text", "")) for kw in label_keywords) and polygon_to_norm(it["polygon"], page_w_pt, page_h_pt)[0] < 0.50 and min_y <= polygon_to_norm(it["polygon"], page_w_pt, page_h_pt)[1] <= max_y), None)
        if label_item:
            _, l_bbox = label_item
            target_y, target_x_min = (l_bbox[1] + l_bbox[3]) / 2.0, 0.40
            
    if target_y is None: return None

    row_items = [it for it in ocr_items if abs(((polygon_to_norm(it["polygon"], page_w_pt, page_h_pt)[1] + polygon_to_norm(it["polygon"], page_w_pt, page_h_pt)[3]) / 2.0) - target_y) <= 0.020 and polygon_to_norm(it["polygon"], page_w_pt, page_h_pt)[0] >= target_x_min]
    merged = [c for c in merge_row_number_tokens(row_items, page_w_pt, page_h_pt) if not (abs(c["value"]) in [1.0, 2.0, 3.0, 4.0] and len(c["snippet"].strip()) <= 2)]
    
    if not merged: return None
    merged.sort(key=lambda x: x["bbox"][0])
    
    if centroids and col_idx < len(centroids):
        best_cand, min_dist = None, float('inf')
        for cand in merged:
            if abs(cand["bbox"][0] - centroids[col_idx]) < min_dist and abs(cand["bbox"][0] - centroids[col_idx]) < 0.08:
                min_dist, best_cand = abs(cand["bbox"][0] - centroids[col_idx]), cand
        if best_cand: return best_cand
    return merged[min(col_idx, len(merged) - 1)]

def process_filing(doc_info: dict) -> dict:
    siren, doc_id, pdf_path = doc_info["siren"], doc_info["doc_id"], doc_info["pdf"]
    pages = load_ocr_pages(siren, doc_id)
    unit = detect_document_unit(pages)
    page_map = classify_pages(pages)
    doc_fitz = pymupdf.open(pdf_path)
    fields = []
    deskewed_pages = set()

    def prep_page(p_num: int):
        p_fitz = doc_fitz[p_num - 1]
        w, h = p_fitz.rect.width, p_fitz.rect.height
        if p_num not in deskewed_pages:
            deskew_ocr_page(pages[p_num - 1], get_page_skew_angle(p_fitz), w, h)
            deskewed_pages.add(p_num)
        return w, h

    def add_field(field_key: str, res: dict | None, p_num: int, field_unit: str = unit):
        if res and res.get("value") is not None:
            fields.append({"field_key": field_key, "value": res["value"], "unit": field_unit, "page": p_num, "bbox": res["bbox"], "snippet": res["snippet"], "confidence": res["confidence"]})

    # --- BALANCE SHEET (Actif / Passif) ---
    # Standard Form 2050 (Actif) OR Simplified Form 2033-A (Actif + Passif)
    if "2050" in page_map or "2033-A" in page_map:
        p_num = page_map.get("2050", page_map.get("2033-A"))
        w, h = prep_page(p_num)
        
        # 2050 uses CL (col 2), 2033-A uses 090 (col 0/1 depending on the layout, usually the last col)
        col = 2 if "2050" in page_map else 1
        res_assets = extract_field_value(pages[p_num - 1], w, h, codes=["CL", "090"], label_keywords=["total general", "total de l'actif", "total (i a vi)"], col_idx=col, min_y=0.30, expected_cols=4 if "2050" in page_map else 2)
        if res_assets and res_assets["value"] > 50: add_field("BS_TOTAL_ASSETS_FRGAAP", res_assets, p_num)

        res_cash = extract_field_value(pages[p_num - 1], w, h, codes=["CF", "086"], label_keywords=["disponibilites"], col_idx=col, expected_cols=4 if "2050" in page_map else 2)
        add_field("BS_CASH_CURRENT_ASSET_FRGAAP", res_cash, p_num)

        # If it's a 2033-A, Passif is on the SAME page
        if "2033-A" in page_map:
            res_eq = extract_field_value(pages[p_num - 1], w, h, codes=["142"], label_keywords=["total des capitaux propres"], col_idx=0, min_y=0.40, expected_cols=2)
            if res_eq and res_eq["value"] > 50: add_field("BS_TOTAL_EQUITY_FRGAAP", res_eq, p_num)
            
            res_cap = extract_field_value(pages[p_num - 1], w, h, codes=["120"], label_keywords=["capital social"], col_idx=0, expected_cols=2)
            if res_cap and res_cap["value"] > 50: add_field("BS_CAPITAL_EQUITY_FRGAAP", res_cap, p_num)

    # Standard Form 2051 (Passif)
    if "2051" in page_map:
        p_num = page_map["2051"]
        w, h = prep_page(p_num)
        res_eq = extract_field_value(pages[p_num - 1], w, h, codes=["DL"], label_keywords=["total capitaux propres", "total i"], col_idx=0, min_y=0.20, expected_cols=2)
        if res_eq and res_eq["value"] > 50: add_field("BS_TOTAL_EQUITY_FRGAAP", res_eq, p_num)
        res_cap = extract_field_value(pages[p_num - 1], w, h, codes=["DA"], label_keywords=["capital social", "capital individuel"], col_idx=0, max_y=0.35, expected_cols=2)
        if res_cap and res_cap["value"] > 50: add_field("BS_CAPITAL_EQUITY_FRGAAP", res_cap, p_num)

    # --- PROFIT & LOSS ---
    # Standard Form 2052 OR Simplified Form 2033-B
    if "2052" in page_map or "2033-B" in page_map:
        p_num = page_map.get("2052", page_map.get("2033-B"))
        w, h = prep_page(p_num)

        add_field("PL_REVENUE_FRGAAP", extract_field_value(pages[p_num - 1], w, h, codes=["FL", "210"], label_keywords=["chiffre d'affaires net"], col_idx=0, expected_cols=2), p_num)
        
        # External Services (FW on 2052, roughly 242 on 2033-B)
        add_field("PL_EXT_SERVICES_COSTS_FRGAAP", extract_field_value(pages[p_num - 1], w, h, codes=["FW", "242"], label_keywords=["autres achats et charges externes", "consommations de l'exercice"], col_idx=0, expected_cols=2), p_num)
        
        # Personnel
        res_sal = extract_field_value(pages[p_num - 1], w, h, codes=["FY", "250"], label_keywords=["salaires et traitements", "charges de personnel"], col_idx=0, expected_cols=2)
        res_soc = extract_field_value(pages[p_num - 1], w, h, codes=["FZ"], label_keywords=["charges sociales"], col_idx=0, expected_cols=2) if "2052" in page_map else None
        
        if res_sal and res_soc:
            add_field("PL_PERSONNEL_COSTS_FRGAAP", {"value": res_sal["value"] + res_soc["value"], "bbox": [min(res_sal["bbox"][0], res_soc["bbox"][0]), min(res_sal["bbox"][1], res_soc["bbox"][1]), max(res_sal["bbox"][2], res_soc["bbox"][2]), max(res_sal["bbox"][3], res_soc["bbox"][3])], "snippet": f"{res_sal['snippet']} + {res_soc['snippet']}", "confidence": round(min(res_sal["confidence"], res_soc["confidence"]), 4)}, p_num)
        elif res_sal: add_field("PL_PERSONNEL_COSTS_FRGAAP", res_sal, p_num)

        add_field("PL_DEPRECIATION_AMORTIZATION_FRGAAP", extract_field_value(pages[p_num - 1], w, h, codes=["GA", "254"], label_keywords=["dotations aux amortissements"], col_idx=0, expected_cols=2), p_num)

    # Standard Form 2053 OR simplified falls under 2033-B
    if "2053" in page_map or "2033-B" in page_map:
        p_num = page_map.get("2053", page_map.get("2033-B"))
        w, h = prep_page(p_num)
        add_field("PL_FINANCIAL_RESULTS_FRGAAP", extract_field_value(pages[p_num - 1], w, h, codes=["GP", "284"], label_keywords=["resultat financier"], col_idx=0, expected_cols=2), p_num)
        add_field("PL_INCOME_TAX_FRGAAP", extract_field_value(pages[p_num - 1], w, h, codes=["HK", "306"], label_keywords=["impots sur les benefices"], col_idx=0, expected_cols=2), p_num)

    return {"pdf": pdf_path, "siren": siren, "fiscal_year_end": None, "fields": fields}

def run_full_pipeline():
    start_time = time.time()
    total_pages = 0
    all_docs_output = []

    for doc in TARGET_DOCUMENTS:
        pages = load_ocr_pages(doc["siren"], doc["doc_id"])
        total_pages += len(pages)
        res = process_filing(doc)
        all_docs_output.append(res)
        print(f"SIREN {res['siren']} | Matched {len(res['fields']):2d} fields in {os.path.basename(res['pdf'])}")

    elapsed = time.time() - start_time
    sec_per_page = round(elapsed / total_pages, 4) if total_pages else 0.0

    output = {
        "documents": all_docs_output,
        "run": {
            "cost_eur_per_page": 0.0,
            "seconds_per_page": sec_per_page,
            "pages_processed": total_pages,
            "model": "Provided OCR + Form-Agnostic CV Pipeline (Régime Normal + Simplifié)",
            "notes": "Handles standard forms (2050-2053) and simplified forms (2033-A/B)."
        }
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nWrote results.json successfully! Processed {total_pages} pages in {elapsed:.2f}s ({sec_per_page}s/page).")

if __name__ == "__main__":
    run_full_pipeline()