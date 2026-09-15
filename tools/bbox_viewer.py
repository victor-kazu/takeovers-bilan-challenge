#!/usr/bin/env python3
"""Draw OCR boxes, and your own boxes, on top of a page of a PDF.

Grounding is the point of both data challenges: every value you report carries the
page and the box it came from. This is the only piece of code we give you, because
checking a box by eye is tedious and has nothing to do with what we are assessing.

    pip install pymupdf pillow

Look at a page with the shipped OCR drawn over it:

    python tools/bbox_viewer.py \
        --pdf  data/<siren>/actes/pdf/<file>.pdf \
        --page 2 \
        --ocr  data/<siren>/actes/ocr/<doc_id> \
        -o page2.png

Find where a phrase sits, and get the normalized box you should report:

    python tools/bbox_viewer.py --pdf <pdf> --page 2 --ocr <ocr_dir> --grep "capital social"

Check a box you are about to submit (red), against the OCR (grey):

    python tools/bbox_viewer.py --pdf <pdf> --page 2 --ocr <ocr_dir> \
        --bbox 0.11,0.32,0.87,0.36 -o check.png

Coordinates
-----------
What we ship in the OCR JSON: ``ocr[].polygon`` is four [x, y] points in PIXELS at
300 dpi, and ``page`` is 1-indexed.

What you must submit: ``[x0, y0, x1, y1]`` as fractions of page width and height,
between 0 and 1, origin top-left, page 1-indexed.

Converting between them needs the page size, which is in the PDF in points:

    px_per_point = 300 / 72
    x_norm = x_px / (page_width_points * px_per_point)

``--grep`` prints boxes already converted, so you can check your own conversion
against it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

DPI_OF_OCR = 300
POINTS_PER_INCH = 72


def load_ocr_page(ocr_path: str, page: int) -> dict | None:
    """Accept either a directory of page_NNN.json files, or one such file."""
    if not os.path.exists(ocr_path):
        print(f"no OCR at {ocr_path}", file=sys.stderr)
        print("  Not every document in this corpus has OCR. You can still render the "
              "page and read it yourself, or run your own OCR over it.", file=sys.stderr)
        return None
    if os.path.isdir(ocr_path):
        hit = os.path.join(ocr_path, f"page_{page:03d}.json")
        if not os.path.exists(hit):
            available = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ocr_path, "page_*.json")))
            print(f"no OCR for page {page} in {ocr_path}", file=sys.stderr)
            if available:
                print(f"  pages present: {available[0]} … {available[-1]}", file=sys.stderr)
            return None
        return json.load(open(hit, encoding="utf-8"))
    return json.load(open(ocr_path, encoding="utf-8"))


def polygon_to_norm(polygon, page_w_pt: float, page_h_pt: float):
    """[[x,y] x4] in 300-dpi pixels  ->  [x0,y0,x1,y1] normalized 0-1."""
    scale = DPI_OF_OCR / POINTS_PER_INCH
    w_px, h_px = page_w_pt * scale, page_h_pt * scale
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return [min(xs) / w_px, min(ys) / h_px, max(xs) / w_px, max(ys) / h_px]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--page", type=int, required=True, help="1-indexed")
    ap.add_argument("--ocr", help="directory of page_NNN.json, or a single page json")
    ap.add_argument("--bbox", action="append", default=[],
                    help="your own box as x0,y0,x1,y1 normalized 0-1. Repeatable.")
    ap.add_argument("--grep", help="print the normalized box of every OCR line containing this text")
    ap.add_argument("--dpi", type=int, default=150, help="render resolution of the output image")
    ap.add_argument("-o", "--out", help="write a PNG here")
    args = ap.parse_args()

    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz  # older PyMuPDF releases
        except ImportError:
            print("needs PyMuPDF:  pip install pymupdf pillow", file=sys.stderr)
            return 2

    doc = fitz.open(args.pdf)
    if not 1 <= args.page <= len(doc):
        print(f"page {args.page} out of range, this PDF has {len(doc)}", file=sys.stderr)
        return 2
    page = doc[args.page - 1]
    w_pt, h_pt = page.rect.width, page.rect.height

    if args.ocr and os.path.isdir(args.ocr):
        n_ocr = len(glob.glob(os.path.join(args.ocr, "page_*.json")))
        if n_ocr and n_ocr != len(doc):
            print(f"note: {n_ocr} OCR pages for a {len(doc)}-page PDF — "
                  f"some pages of this document have no OCR.", file=sys.stderr)

    ocr = load_ocr_page(args.ocr, args.page) if args.ocr else None
    lines = (ocr or {}).get("ocr") or []

    if args.grep:
        needle = args.grep.lower()
        found = 0
        for line in lines:
            text = (line.get("text") or "").strip()
            if needle in text.lower():
                box = polygon_to_norm(line["polygon"], w_pt, h_pt)
                print(f'  [{", ".join(f"{v:.4f}" for v in box)}]  {text}')
                found += 1
        if not found:
            print(f"  '{args.grep}' not found in the OCR of page {args.page}")
        if not args.out:
            return 0

    if not args.out:
        if args.ocr is None:
            print(f"page {args.page}: {w_pt:.0f} x {h_pt:.0f} points, {len(doc)} pages in this PDF")
            print("pass --ocr <dir> to load the OCR, and -o out.png to render it")
        else:
            print(f"page {args.page}: {w_pt:.0f} x {h_pt:.0f} points, {len(lines)} OCR lines")
            print("pass -o out.png to render it")
        return 0

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("needs Pillow:  pip install pillow", file=sys.stderr)
        return 2

    pix = page.get_pixmap(dpi=args.dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    def to_px(box):
        return [box[0] * img.width, box[1] * img.height, box[2] * img.width, box[3] * img.height]

    for line in lines:
        poly = line.get("polygon")
        if not poly:
            continue
        draw.rectangle(to_px(polygon_to_norm(poly, w_pt, h_pt)), outline=(90, 110, 130, 200), width=1)

    for i, raw in enumerate(args.bbox):
        try:
            box = [float(v) for v in raw.replace(" ", "").split(",")]
            assert len(box) == 4
        except Exception:
            print(f"--bbox {raw!r} is not x0,y0,x1,y1", file=sys.stderr)
            return 2
        if max(box) > 1.0:
            print(f"--bbox {raw}: values above 1.0 — these must be normalized 0-1", file=sys.stderr)
        draw.rectangle(to_px(box), outline=(200, 30, 30, 255), width=3)
        draw.text((box[0] * img.width + 4, box[1] * img.height + 4), str(i + 1), fill=(200, 30, 30, 255))

    Image.alpha_composite(img, overlay).convert("RGB").save(args.out)
    print(f"wrote {args.out}  ({len(lines)} OCR boxes, {len(args.bbox)} of yours)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
