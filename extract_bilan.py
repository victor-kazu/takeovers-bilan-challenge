from __future__ import annotations

import glob
import json
import os
import re
import time
import unicodedata
import warnings

import pymupdf
import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster")

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
    """Remove accents and converts everything to lowercase"""

    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)).lower()

def load_ocr_pages(siren: str, doc_id: str) -> list[dict]:
    """Load OCR JSON files for a given SIREN and document ID.
    Returns a list of dictionaries, each representing a page's OCR data."""

    files = sorted(glob.glob(os.path.join("data", siren, "bilans", "ocr", doc_id, "page_*.json"))) #Sort the OCR JSON files by page number 
    return [json.load(open(f, "r", encoding="utf-8")) for f in files]                              

def detect_document_unit(pages: list[dict]) -> str:
    """Detects if the document uses kEUR (thousands of euros) or EUR (euros) based on OCR text patterns."""

    keur_patterns = [r"\bk€\b", r"\bkeur\b", r"en milliers", r"milliers d['’]?euros", r"montants.*en k", r"exprimes? en k"]
    for page in pages:
        for item in page.get("ocr", []):
            if any(re.search(pat, strip_accents(item.get("text", ""))) for pat in keur_patterns): 
                return "kEUR"
    return "EUR"

def polygon_to_norm(polygon: list[list[float]], page_w_pt: float, page_h_pt: float) -> list[float]:
    """Convert a polygon (list of points) to normalized bounding box coordinates [x_min, y_min, x_max, y_max] in the range [0, 1]."""

    scale = 300.0 / 72.0    #300 DPI OCR pixel coordinates by 72 DPI native PDF points
    w_px, h_px = page_w_pt * scale, page_h_pt * scale   #Converts PDF points to OCR pixel coordinates
    xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]   #Extract x and y coordinates from the polygon points
    return [round(max(0.0, min(xs) / w_px), 4), round(max(0.0, min(ys) / h_px), 4), round(min(1.0, max(xs) / w_px), 4), round(min(1.0, max(ys) / h_px), 4)]

def clean_number(text: str) -> float | None:
    """Clean and extract a numeric value from a text string. 
    Returns None if the text is not a valid number or contains unwanted patterns such as:
    dates, percentages, and tax form identifiers."""

    trimmed = text.strip()
    if re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", trimmed):  #Dates
        return None
    if "%" in trimmed or "quote-part" in trimmed.lower():  #Percentages
        return None
    if re.search(r"205\d-[a-z]|dgfip", trimmed, re.IGNORECASE):  #Tax form identifiers
        return None

    clean_text = trimmed.replace(" ", "").replace("\u00a0", "").replace(",", ".") #Converts commas to dots for decimal representation
    is_neg = False
    if re.search(r"^\(.*\)$", clean_text):  #Negative numbers in parentheses
        is_neg = True
        clean_text = clean_text.replace("(", "").replace(")", "") 
    elif clean_text.startswith("-") and not re.search(r"^-\d", clean_text): 
        clean_text = clean_text.lstrip("-")

    m = re.search(r"[-−]?\d+(\.\d+)?", clean_text) 
    if not m:
        return None
    try: 
        val = float(m.group(0).replace("−", "-")) #Cast string to float 
        if abs(val) >= 1e11: #Discard values that are too large to be realistic for financial statements
            return None
        return -val if is_neg else val
    except ValueError:
        return None

def build_column_clusters(ocr_items: list[dict], page_w_pt: float, page_h_pt: float, expected_cols: int) -> list[float]:
    """Builds column clusters using KMeans based on the x-coordinates of OCR items that contain valid numbers.
    Returns a sorted list of cluster centers representing the x-coordinates of the columns."""


    x_centers = [[polygon_to_norm(it["polygon"], page_w_pt, page_h_pt)[0]] for it in ocr_items if clean_number(it.get("text", "")) is not None and polygon_to_norm(it["polygon"], page_w_pt, page_h_pt)[0] > 0.40] #Filtering out items that are too far left (likely labels) and only keeping those with valid numbers
    unique_pts = sorted(list(set(x[0] for x in x_centers)))
    if len(unique_pts) <= expected_cols: #If the number of unique x-coordinates is less than or equal to the expected number of columns, return the unique x-coordinates as the cluster centers. 
        return unique_pts                #Fast path to avoid unnecessary KMeans computation.
    kmeans = KMeans(n_clusters=expected_cols, random_state=42, n_init=1)
    kmeans.fit(x_centers)
    return sorted(kmeans.cluster_centers_.flatten())

def merge_row_number_tokens(items_on_row: list[dict], page_w_pt: float, page_h_pt: float) -> list[dict]:
    """Merges OCR items on the same row that are likely part of the same numeric value based on proximity and vertical alignment.
    Returns a list of merged items with combined text, bounding boxes, and cleaned numeric values."""

    sorted_items = sorted([it for it in items_on_row if clean_number(it.get("text", "")) is not None], key=lambda x: polygon_to_norm(x["polygon"], page_w_pt, page_h_pt)[0]) #Sort items by their x-coordinate (left to right) and filter out those without valid numeric values
    if not sorted_items: 
        return []
    
    merged, current_cluster = [], [sorted_items[0]]
    for it in sorted_items[1:]:
        prev_box = polygon_to_norm(current_cluster[-1]["polygon"], page_w_pt, page_h_pt)
        curr_box = polygon_to_norm(it["polygon"], page_w_pt, page_h_pt)
        
        gap = curr_box[0] - prev_box[2] #Horizontal gap between the current item and the previous item
        v_diff = abs(((curr_box[1] + curr_box[3]) / 2) - ((prev_box[1] + prev_box[3]) / 2)) #Vertical difference between the centers of the current and previous items
        cluster_w = curr_box[2] - polygon_to_norm(current_cluster[0]["polygon"], page_w_pt, page_h_pt)[0] #Width of the current cluster from the first item to the current item
        
        if 0 <= gap < 0.015 and v_diff < 0.012 and cluster_w < 0.16:
            current_cluster.append(it)
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
            results.append({
                "value": val, 
                "bbox": [min(b[0] for b in all_boxes), min(b[1] for b in all_boxes), max(b[2] for b in all_boxes), max(b[3] for b in all_boxes)], 
                "snippet": combined_text, 
                "confidence": min(float(c.get("score", 0.95)) for c in cluster)
            })
    return results

def is_valid_code_match(raw_token: str, code: str) -> bool:
    """Checks if a raw OCR token matches a given code, considering various formats such as parentheses, brackets, and word boundaries."""

    token = raw_token.strip().upper()
    if len(code) == 2:
        return bool(re.search(rf"\b{re.escape(code)}\b", token)) #Match 2-letter codes as whole words only e.g. CL, DL, FL
    return bool(re.search(rf"(\({re.escape(code)}\)|\[{re.escape(code)}\]|\b{re.escape(code)}\b)", token)) #Match 3-letter codes in parentheses, brackets, or as whole words e.g. (090), [210]

def extract_field_value(page_data: dict, page_w_pt: float, page_h_pt: float, codes: list[str], label_keywords: list[str], col_idx: int = 0, expected_cols: int = 2, centroids: list[float] = []) -> dict | None:
    """Extracts a field value from a page's OCR data based on specified codes and label keywords.
    Returns a dictionary containing the extracted value, bounding box, snippet, page number, and confidence"""
    
    ocr_items = page_data.get("ocr", [])
    target_y, target_x_min = None, 0.0
    matched_code = None
    
    # 1. Box Code Search
    for it in reversed(ocr_items):
        raw_text = it.get("text", "")
        if "dgfip" in raw_text.lower() or "n° 205" in raw_text.lower(): #ignore Direction Générale des Finances Publiques references
            continue
        for code in codes: #Check if the OCR text matches any of the specified codes
            if is_valid_code_match(raw_text, code):
                bbox = polygon_to_norm(it["polygon"], page_w_pt, page_h_pt)
                target_y = (bbox[1] + bbox[3]) / 2.0
                target_x_min = bbox[2] - 0.05
                matched_code = code
                break
        if target_y is not None:
            break
            
    # 2. Row Label Fallback
    if target_y is None: #If no Cerfa code matches, falls back to label matching e.g. "Total General"
        for it in ocr_items:
            norm_text = strip_accents(it.get("text", ""))
            if "dgfip" in norm_text or "n° 205" in norm_text:
                continue
            bbox = polygon_to_norm(it["polygon"], page_w_pt, page_h_pt)
            if any(kw in norm_text for kw in label_keywords) and bbox[0] < 0.55:
                target_y = (bbox[1] + bbox[3]) / 2.0
                target_x_min = 0.40
                break

    if target_y is None:
        return None

    row_items = []
    for it in ocr_items:
        bbox = polygon_to_norm(it["polygon"], page_w_pt, page_h_pt)
        item_y = (bbox[1] + bbox[3]) / 2.0
        item_x = bbox[0]
        
        token_clean = re.sub(r'[^A-Z0-9]', '', it.get("text", "").upper())
        if matched_code and token_clean == matched_code:
            continue
            
        dynamic_y_tolerance = 0.015 + (max(0, item_x - target_x_min) * 0.040) #Dynamic tolerance increases for items further to the right, allowing for slight vertical misalignments in multi-column layouts
        if abs(item_y - target_y) <= dynamic_y_tolerance and item_x >= target_x_min:
            row_items.append(it)

    merged = [c for c in merge_row_number_tokens(row_items, page_w_pt, page_h_pt) if not (abs(c["value"]) in [1.0, 2.0, 3.0, 4.0] and len(c["snippet"].strip()) <= 2)]
    if not merged:
        return None
    merged.sort(key=lambda x: x["bbox"][0])
    
    if centroids: #If column centroids are provided, attempt to match the extracted values to the closest centroid based on x-coordinate proximity. This helps in selecting the correct column value in multi-column layouts.
        target_centroid = centroids[min(col_idx, len(centroids) - 1)]
        best_cand, min_dist = None, float('inf')
        for cand in merged:
            dist = abs(cand["bbox"][0] - target_centroid)
            if dist < min_dist and dist < 0.10:
                min_dist, best_cand = dist, cand
        if best_cand:
            return best_cand
        
    return merged[min(col_idx, len(merged) - 1)]

def process_filing(doc_info: dict) -> dict:
    """Processes a single filing document, extracting relevant financial fields from its OCR data and PDF representation.
    Returns a dictionary containing the PDF path, SIREN, fiscal year end (if detected), and a list of extracted fields with their values, units, page numbers, bounding boxes, snippets, and confidence scores."""

    siren, doc_id, pdf_path = doc_info["siren"], doc_info["doc_id"], doc_info["pdf"]
    pages = load_ocr_pages(siren, doc_id)
    unit = detect_document_unit(pages)
    doc_fitz = pymupdf.open(pdf_path)
    fields = []
    
    page_caches = []
    for p in pages:
        raw_text = " ".join(it.get("text", "") for it in p.get("ocr", []))
        page_caches.append({
            "raw_upper": raw_text.upper(),
            "raw_lower": strip_accents(raw_text)
        })

    centroid_cache = {}

    def find_best_field_globally(codes: list[str], label_keywords: list[str], col_idx: int = 0, expected_cols: int = 2) -> dict | None: 
        """Searches across all pages of the document to find the best matching field value based on specified codes and label keywords.
        Returns the best matching field value with its associated metadata (page number, bounding box, snippet)"""
        best_res = None
        for p_idx, page in enumerate(pages):
            cache = page_caches[p_idx]
            has_code = any(is_valid_code_match(cache["raw_upper"], code) for code in codes) 
            has_label = any(kw in cache["raw_lower"] for kw in label_keywords)
            if not has_code and not has_label: #If neither the specified codes nor label keywords are found on the page, skip to the next page to optimize processing time.
                continue
                
            w, h = doc_fitz[p_idx].rect.width, doc_fitz[p_idx].rect.height
            cache_key = (p_idx, expected_cols)
            if cache_key not in centroid_cache:
                centroid_cache[cache_key] = build_column_clusters(page.get("ocr", []), w, h, expected_cols)
                
            res = extract_field_value(page, w, h, codes, label_keywords, col_idx, expected_cols, centroids=centroid_cache[cache_key]) 
            if res and abs(res["value"]) > 10:
                res["page"] = p_idx + 1
                if best_res is None or res["confidence"] > best_res["confidence"]:
                    best_res = res
        return best_res

    def add_field(field_key: str, res: dict | None, field_unit: str = unit): 
        """Adds a field to the extracted fields list if a valid result is found.
        Performs additional validation checks based on the field type to ensure data quality and consistency."""

        if res and res.get("value") is not None:
            val = res["value"]
            if field_key in ["BS_TOTAL_ASSETS_FRGAAP", "BS_CASH_CURRENT_ASSET_FRGAAP", "PL_REVENUE_FRGAAP", "BS_CAPITAL_EQUITY_FRGAAP"]: #Thesefields cannot be negative
                val = abs(val)
                if val <= 0:
                    return
            if field_key == "BS_CAPITAL_EQUITY_FRGAAP" and int(val) in [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2050, 2051, 2052, 2053, 2058, 2059]: #Avoid false positives where the capital equity field is mistakenly extracted as a tax form identifier
                return
            if field_key == "META_AVG_WORKFORCE_FRGAAP":
                if val > 1000 or val <= 0 or val == 376 or len(res.get("snippet", "").strip()) <= 3:
                    return
            fields.append({
                "field_key": field_key, 
                "value": val, 
                "unit": field_unit, 
                "page": res["page"], 
                "bbox": res["bbox"], 
                "snippet": res["snippet"], 
                "confidence": res["confidence"]
            })

    # Balance Sheet Fields
    add_field("BS_TOTAL_ASSETS_FRGAAP", find_best_field_globally(codes=["CL", "090"], label_keywords=["total general", "total de l'actif", "total (i a vi)", "total i a vi"], col_idx=2, expected_cols=4))
    add_field("BS_TOTAL_EQUITY_FRGAAP", find_best_field_globally(codes=["DL", "142"], label_keywords=["total capitaux propres", "total des capitaux propres", "total i"], col_idx=0, expected_cols=2))
    add_field("BS_CASH_CURRENT_ASSET_FRGAAP", find_best_field_globally(codes=["CF", "086"], label_keywords=["disponibilites", "caisse"], col_idx=2, expected_cols=4))
    add_field("BS_CAPITAL_EQUITY_FRGAAP", find_best_field_globally(codes=["DA", "120"], label_keywords=["capital social", "capital individuel", "capital souscrit"], col_idx=0, expected_cols=2))

    # Income Statement (P&L) Fields
    add_field("PL_REVENUE_FRGAAP", find_best_field_globally(codes=["FL", "210"], label_keywords=["chiffre d'affaires net", "montant net du chiffre d'affaires", "total des produits d'exploitation"], col_idx=0, expected_cols=2))
    add_field("PL_EXT_SERVICES_COSTS_FRGAAP", find_best_field_globally(codes=["FW", "242"], label_keywords=["autres achats et charges externes", "consommations de l'exercice", "achats et charges externes"], col_idx=0, expected_cols=2))
    add_field("PL_DEPRECIATION_AMORTIZATION_FRGAAP", find_best_field_globally(codes=["GA", "254"], label_keywords=["dotations aux amortissements", "dotations d'exploitation"], col_idx=0, expected_cols=2))
    add_field("PL_FINANCIAL_RESULTS_FRGAAP", find_best_field_globally(codes=["GP", "284"], label_keywords=["resultat financier", "resultat de l'exercice", "total v - vi"], col_idx=0, expected_cols=2))
    add_field("PL_INCOME_TAX_FRGAAP", find_best_field_globally(codes=["HK", "306"], label_keywords=["impots sur les benefices", "impot sur les benefices", "impots sur le benefice"], col_idx=0, expected_cols=2))
    add_field("META_AVG_WORKFORCE_FRGAAP", find_best_field_globally(codes=["YP", "376"], label_keywords=["effectif moyen du personnel", "effectif moyen"], col_idx=0, expected_cols=2), field_unit="count")

    # Composite Personnel Costs
    res_sal = find_best_field_globally(codes=["FY", "250"], label_keywords=["salaires et traitements", "charges de personnel", "frais de personnel"], col_idx=0, expected_cols=2)
    res_soc = find_best_field_globally(codes=["FZ", "252"], label_keywords=["charges sociales", "cotisations sociales"], col_idx=0, expected_cols=2)
    if res_sal and res_soc and res_sal["page"] == res_soc["page"]:
        add_field("PL_PERSONNEL_COSTS_FRGAAP", {
            "value": res_sal["value"] + res_soc["value"], 
            "bbox": [min(res_sal["bbox"][0], res_soc["bbox"][0]), min(res_sal["bbox"][1], res_soc["bbox"][1]), max(res_sal["bbox"][2], res_soc["bbox"][2]), max(res_sal["bbox"][3], res_soc["bbox"][3])], 
            "snippet": f"{res_sal['snippet']} + {res_soc['snippet']}", 
            "page": res_sal["page"], 
            "confidence": round(min(res_sal["confidence"], res_soc["confidence"]), 4)
        })
    elif res_sal: 
        add_field("PL_PERSONNEL_COSTS_FRGAAP", res_sal)

    # Composite COGS
    res_fs = find_best_field_globally(codes=["FS", "212"], label_keywords=["achats de marchandises"], col_idx=0, expected_cols=2)
    res_ft = find_best_field_globally(codes=["FT", "214"], label_keywords=["variation de stock"], col_idx=0, expected_cols=2)
    if res_fs and res_ft and res_fs["page"] == res_ft["page"]:
        add_field("PL_COGS_FRGAAP", {
            "value": res_fs["value"] + res_ft["value"], 
            "bbox": [min(res_fs["bbox"][0], res_ft["bbox"][0]), min(res_fs["bbox"][1], res_ft["bbox"][1]), max(res_fs["bbox"][2], res_ft["bbox"][2]), max(res_fs["bbox"][3], res_ft["bbox"][3])], 
            "snippet": f"{res_fs['snippet']} + {res_ft['snippet']}", 
            "page": res_fs["page"], 
            "confidence": round(min(res_fs["confidence"], res_ft["confidence"]), 4)
        })
    elif res_fs: 
        add_field("PL_COGS_FRGAAP", res_fs)

    return {"pdf": pdf_path, "siren": siren, "fiscal_year_end": None, "fields": fields}

def run_full_pipeline():
    """Runs the full extraction pipeline for all target documents, processing each document's OCR data and extracting relevant financial fields.
    Outputs the results to a JSON file and prints a summary of the processing time and pages processed"""
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
            "model": "Unconstrained Multi-Statement Router + K-Means + Accounting Normalization",
            "notes": "Cost is 0 EUR. Full document multi-page search allows capturing statements across standard and simplified packages. Sanitizes digit merging, strips date and percentage artifacts, normalizes OCR table bullets, and eliminates false positives."
        }
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nWrote results.json successfully! Processed {total_pages} pages in {elapsed:.2f}s ({sec_per_page}s/page).")

if __name__ == "__main__":
    run_full_pipeline()