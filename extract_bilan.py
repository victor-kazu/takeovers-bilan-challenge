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
        if "2052" not in mapping and ("2052" in combined or ("compte de resultat" in combined and "charges d'exploitation" in combined)):
            mapping["2052"] = page_idx
        if "2053" not in mapping and ("2053" in combined or ("compte de resultat" in combined and "suite" in combined)):
            mapping["2053"] = page_idx
        if "2058-C" not in mapping and ("2058-c" in combined or "2058 c" in combined or "renseignements divers" in combined):
            mapping["2058-C"] = page_idx
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
        gap = curr_box[0] - prev_box[2]
        v_diff = abs(((curr_box[1] + curr_box[3]) / 2) - ((prev_box[1] + prev_box[3]) / 2))

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
    col_idx: int = 0,
    min_y: float = 0.10,
    max_y: float = 0.95
) -> dict | None:
    ocr_items = page_data.get("ocr", [])

    # 1. Search by 2-letter box code
    code_items = []
    for item in ocr_items:
        text = item.get("text", "").strip().upper()
        if re.search(rf"\b{re.escape(code)}\b", text):
            bbox = polygon_to_norm(item["polygon"], page_w_pt, page_h_pt)
            if min_y <= bbox[1] <= max_y:
                code_items.append((item, bbox))

    if code_items:
        target_code, c_bbox = code_items[-1]
        c_y = (c_bbox[1] + c_bbox[3]) / 2.0
        c_x = c_bbox[2]

        row_items = [
            it for it in ocr_items
            if abs(((polygon_to_norm(it["polygon"], page_w_pt, page_h_pt)[1] + polygon_to_norm(it["polygon"], page_w_pt, page_h_pt)[3]) / 2.0) - c_y) <= 0.020
            and polygon_to_norm(it["polygon"], page_w_pt, page_h_pt)[0] >= c_x - 0.05
        ]
        merged = merge_row_number_tokens(row_items, page_w_pt, page_h_pt)
        merged = [c for c in merged if not (abs(c["value"]) in [1.0, 2.0, 3.0, 4.0] and len(c["snippet"].strip()) <= 2)]
        if merged:
            merged.sort(key=lambda x: x["bbox"][0])
            return merged[min(col_idx, len(merged) - 1)]

    # 2. Search by row label
    label_item = None
    for item in ocr_items:
        norm = strip_accents(item.get("text", ""))
        if any(kw in norm for kw in label_keywords):
            bbox = polygon_to_norm(item["polygon"], page_w_pt, page_h_pt)
            if bbox[0] < 0.50 and min_y <= bbox[1] <= max_y:
                label_item = (item, bbox)
                break

    if label_item:
        _, l_bbox = label_item
        y_center = (l_bbox[1] + l_bbox[3]) / 2.0

        row_items = [
            it for it in ocr_items
            if abs(((polygon_to_norm(it["polygon"], page_w_pt, page_h_pt)[1] + polygon_to_norm(it["polygon"], page_w_pt, page_h_pt)[3]) / 2.0) - y_center) <= 0.018
            and polygon_to_norm(it["polygon"], page_w_pt, page_h_pt)[0] > 0.45
        ]
        merged = merge_row_number_tokens(row_items, page_w_pt, page_h_pt)
        merged = [c for c in merged if not (abs(c["value"]) in [1.0, 2.0, 3.0, 4.0] and len(c["snippet"].strip()) <= 2)]
        if merged:
            merged.sort(key=lambda x: x["bbox"][0])
            return merged[min(col_idx, len(merged) - 1)]

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

    # Helper to push a field
    def add_field(field_key: str, res: dict | None, p_num: int, field_unit: str = unit):
        if res and res["value"] is not None:
            fields.append({
                "field_key": field_key,
                "value": res["value"],
                "unit": field_unit,
                "page": p_num,
                "bbox": res["bbox"],
                "snippet": res["snippet"],
                "confidence": res["confidence"]
            })

    # --- 1. Form 2050 (Actif) ---
    if "2050" in page_map:
        p_num = page_map["2050"]
        p_fitz = doc_fitz[p_num - 1]
        w, h = p_fitz.rect.width, p_fitz.rect.height

        # BS_TOTAL_ASSETS_FRGAAP (Code CL)
        res_assets = extract_field_value(pages[p_num - 1], w, h, code="CL", label_keywords=["total general", "total (i a vi)"], col_idx=2, min_y=0.40)
        
        # Self-Check reconciliation with Passif (Code DQ)
        if "2051" in page_map:
            p_passif = page_map["2051"]
            p_fitz_passif = doc_fitz[p_passif - 1]
            res_passif_total = extract_field_value(pages[p_passif - 1], p_fitz_passif.rect.width, p_fitz_passif.rect.height, code="DQ", label_keywords=["total general", "total (i a iv)"], col_idx=0, min_y=0.60)
            if res_assets and res_passif_total:
                diff = abs(res_assets["value"] - res_passif_total["value"]) / max(res_assets["value"], res_passif_total["value"])
                if diff <= 0.015:
                    res_assets["confidence"] = 1.0

        if res_assets and res_assets["value"] > 50:
            add_field("BS_TOTAL_ASSETS_FRGAAP", res_assets, p_num)

        # BS_CASH_CURRENT_ASSET_FRGAAP (Disponibilités, Code CF)
        res_cash = extract_field_value(pages[p_num - 1], w, h, code="CF", label_keywords=["disponibilites"], col_idx=2)
        add_field("BS_CASH_CURRENT_ASSET_FRGAAP", res_cash, p_num)

    # --- 2. Form 2051 (Passif) ---
    if "2051" in page_map:
        p_num = page_map["2051"]
        p_fitz = doc_fitz[p_num - 1]
        w, h = p_fitz.rect.width, p_fitz.rect.height

        # BS_TOTAL_EQUITY_FRGAAP (Code DL)
        res_eq = extract_field_value(pages[p_num - 1], w, h, code="DL", label_keywords=["total capitaux propres", "total i"], col_idx=0, min_y=0.25)
        if res_eq and res_eq["value"] > 50:
            add_field("BS_TOTAL_EQUITY_FRGAAP", res_eq, p_num)

        # BS_CAPITAL_EQUITY_FRGAAP (Code DA)
        res_cap = extract_field_value(pages[p_num - 1], w, h, code="DA", label_keywords=["capital social", "capital individuel"], col_idx=0, max_y=0.25)
        if res_cap and res_cap["value"] > 50:
            add_field("BS_CAPITAL_EQUITY_FRGAAP", res_cap, p_num)

    # --- 3. Form 2052 (Compte de résultat - 1ère partie) ---
    if "2052" in page_map:
        p_num = page_map["2052"]
        p_fitz = doc_fitz[p_num - 1]
        w, h = p_fitz.rect.width, p_fitz.rect.height

        # PL_REVENUE_FRGAAP (Code FL)
        res_rev = extract_field_value(pages[p_num - 1], w, h, code="FL", label_keywords=["chiffre d'affaires net", "total i"], col_idx=0)
        add_field("PL_REVENUE_FRGAAP", res_rev, p_num)

        # PL_EXT_SERVICES_COSTS_FRGAAP (Code FW)
        res_ext = extract_field_value(pages[p_num - 1], w, h, code="FW", label_keywords=["autres achats et charges externes"], col_idx=0)
        add_field("PL_EXT_SERVICES_COSTS_FRGAAP", res_ext, p_num)

        # PL_PERSONNEL_COSTS_FRGAAP (Salaires FY + Charges FZ)
        res_sal = extract_field_value(pages[p_num - 1], w, h, code="FY", label_keywords=["salaires et traitements"], col_idx=0)
        res_soc = extract_field_value(pages[p_num - 1], w, h, code="FZ", label_keywords=["charges sociales"], col_idx=0)
        if res_sal and res_soc:
            total_pers = res_sal["value"] + res_soc["value"]
            bbox_pers = [
                min(res_sal["bbox"][0], res_soc["bbox"][0]),
                min(res_sal["bbox"][1], res_soc["bbox"][1]),
                max(res_sal["bbox"][2], res_soc["bbox"][2]),
                max(res_sal["bbox"][3], res_soc["bbox"][3]),
            ]
            add_field("PL_PERSONNEL_COSTS_FRGAAP", {
                "value": total_pers,
                "bbox": bbox_pers,
                "snippet": f"{res_sal['snippet']} + {res_soc['snippet']}",
                "confidence": round(min(res_sal["confidence"], res_soc["confidence"]), 4)
            }, p_num)
        elif res_sal:
            add_field("PL_PERSONNEL_COSTS_FRGAAP", res_sal, p_num)

        # PL_DEPRECIATION_AMORTIZATION_FRGAAP (Code GA)
        res_dot = extract_field_value(pages[p_num - 1], w, h, code="GA", label_keywords=["dotations aux amortissements", "dotations d'exploitation"], col_idx=0)
        add_field("PL_DEPRECIATION_AMORTIZATION_FRGAAP", res_dot, p_num)

        # PL_COGS_FRGAAP (Achats FS + Variation stocks FT)
        res_fs = extract_field_value(pages[p_num - 1], w, h, code="FS", label_keywords=["achats de marchandises"], col_idx=0)
        res_ft = extract_field_value(pages[p_num - 1], w, h, code="FT", label_keywords=["variation de stock"], col_idx=0)
        if res_fs and res_ft:
            add_field("PL_COGS_FRGAAP", {
                "value": res_fs["value"] + res_ft["value"],
                "bbox": [
                    min(res_fs["bbox"][0], res_ft["bbox"][0]),
                    min(res_fs["bbox"][1], res_ft["bbox"][1]),
                    max(res_fs["bbox"][2], res_ft["bbox"][2]),
                    max(res_fs["bbox"][3], res_ft["bbox"][3]),
                ],
                "snippet": f"{res_fs['snippet']} + {res_ft['snippet']}",
                "confidence": round(min(res_fs["confidence"], res_ft["confidence"]), 4)
            }, p_num)
        elif res_fs:
            add_field("PL_COGS_FRGAAP", res_fs, p_num)

    # --- 4. Form 2053 (Compte de résultat - 2nde partie) ---
    if "2053" in page_map:
        p_num = page_map["2053"]
        p_fitz = doc_fitz[p_num - 1]
        w, h = p_fitz.rect.width, p_fitz.rect.height

        # PL_FINANCIAL_RESULTS_FRGAAP (Résultat financier, Code GP)
        res_fin = extract_field_value(pages[p_num - 1], w, h, code="GP", label_keywords=["resultat financier", "total v - vi"], col_idx=0)
        add_field("PL_FINANCIAL_RESULTS_FRGAAP", res_fin, p_num)

        # PL_INCOME_TAX_FRGAAP (Impôts sur les bénéfices, Code HK)
        res_tax = extract_field_value(pages[p_num - 1], w, h, code="HK", label_keywords=["impots sur les benefices"], col_idx=0)
        add_field("PL_INCOME_TAX_FRGAAP", res_tax, p_num)

    # --- 5. Form 2058-C (Divers) ---
    if "2058-C" in page_map:
        p_num = page_map["2058-C"]
        p_fitz = doc_fitz[p_num - 1]
        w, h = p_fitz.rect.width, p_fitz.rect.height

        # META_AVG_WORKFORCE_FRGAAP (Code YP or text 'effectif')
        res_wf = extract_field_value(pages[p_num - 1], w, h, code="YP", label_keywords=["effectif moyen du personnel", "effectif moyen"], col_idx=0)
        if res_wf:
            add_field("META_AVG_WORKFORCE_FRGAAP", res_wf, p_num, field_unit="count")

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
        print(f"SIREN {res['siren']} | Matched {len(res['fields']):2d} fields in {os.path.basename(res['pdf'])}")

    elapsed = time.time() - start_time
    sec_per_page = round(elapsed / total_pages, 4) if total_pages else 0.0

    output = {
        "documents": all_docs_output,
        "run": {
            "cost_eur_per_page": 0.0,
            "seconds_per_page": sec_per_page,
            "pages_processed": total_pages,
            "model": "provided OCR + French tax liasse code & geometric regex parser",
            "notes": "Cost is 0 EUR because extraction relies entirely on the provided OCR and local geometric parsing. High precision prioritized across all 12 financial fields."
        }
    }

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nWrote results.json successfully! Processed {total_pages} pages in {elapsed:.2f}s ({sec_per_page}s/page).")

if __name__ == "__main__":
    run_full_pipeline()