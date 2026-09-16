from __future__ import annotations

import glob
import json
import os
import re
import unicodedata

# 15 target filings scoped in BRIEF.md[cite: 3]
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
    """Normalize text to lowercase ASCII to handle OCR diacritics and casing[cite: 6]."""
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def load_ocr_pages(siren: str, doc_id: str) -> list[dict]:
    """Read all OCR page JSON files in sequential order[cite: 3, 6]."""
    ocr_dir = os.path.join("data", siren, "bilans", "ocr", doc_id)
    files = sorted(glob.glob(os.path.join(ocr_dir, "page_*.json")))
    pages = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            pages.append(json.load(fp))
    return pages


def detect_document_unit(pages: list[dict]) -> str:
    """Detect whether a document reports in 'kEUR' or standard 'EUR'[cite: 1, 2, 3]."""
    keur_patterns = [
        r"\bk€\b",
        r"\bkeur\b",
        r"en milliers",
        r"milliers d['’]?euros",
        r"montants.*en k",
        r"exprimes? en k",
    ]
    for page in pages:
        for item in page.get("ocr", []):
            norm_text = strip_accents(item.get("text", ""))
            for pattern in keur_patterns:
                if re.search(pattern, norm_text):
                    return "kEUR"
    return "EUR"


def classify_pages(pages: list[dict]) -> dict[str, int]:
    """Map standard French tax forms (liasses) to their 1-indexed page number[cite: 1, 3]."""
    mapping: dict[str, int] = {}

    for page_idx, page in enumerate(pages, start=1):
        lines = [strip_accents(item.get("text", "")) for item in page.get("ocr", [])]
        combined = " ".join(lines)

        # Liasse 2050: Bilan - Actif[cite: 1]
        if "2050" not in mapping:
            if "2050" in combined or ("bilan" in combined and "actif" in combined and "passif" not in combined[:150]):
                mapping["2050"] = page_idx

        # Liasse 2051: Bilan - Passif[cite: 1]
        if "2051" not in mapping:
            if "2051" in combined or ("bilan" in combined and "passif" in combined):
                mapping["2051"] = page_idx

        # Liasse 2052: Compte de résultat (Part 1)[cite: 1]
        if "2052" not in mapping:
            if "2052" in combined or ("compte de resultat" in combined and "charges d'exploitation" in combined):
                mapping["2052"] = page_idx

        # Liasse 2053: Compte de résultat, suite (Part 2)[cite: 1]
        if "2053" not in mapping:
            if "2053" in combined or ("compte de resultat" in combined and "suite" in combined):
                mapping["2053"] = page_idx

        # Liasse 2058-C: Renseignements divers (Workforce)[cite: 1]
        if "2058-C" not in mapping:
            if "2058-c" in combined or "2058 c" in combined or "renseignements divers" in combined:
                mapping["2058-C"] = page_idx

    return mapping


if __name__ == "__main__":
    print(f"{'SIREN':<12} | {'Doc ID':<26} | {'Unit':<6} | {'Classified Pages (Form: Page)'}")
    print("-" * 80)
    for doc in TARGET_DOCUMENTS:
        pages = load_ocr_pages(doc["siren"], doc["doc_id"])
        unit = detect_document_unit(pages)
        classified = classify_pages(pages)
        pages_summary = ", ".join(f"{form}: p.{p}" for form, p in classified.items())
        print(f"{doc['siren']:<12} | {doc['doc_id']:<26} | {unit:<6} | {pages_summary}")