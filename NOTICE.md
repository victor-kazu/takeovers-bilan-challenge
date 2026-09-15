# Data provenance

Every PDF and every piece of metadata in `challenges/*/data/` comes from the **Registre
National des Entreprises (RNE)**, the French national companies register, operated by the
**INPI** (Institut National de la Propriété Industrielle).

- Source: <https://data.inpi.fr> · <https://registre-national-entreprises.inpi.fr>
- These are **public filings**. Deposited actes and annual accounts are published by the
  registry and available to anyone; INPI distributes them as open data under the
  *Licence Ouverte / Open Licence* (Etalab).
- Anyone can create a free INPI account and retrieve the same documents, plus any others
  they want.

Nothing here has been altered in content. Two mechanical changes were made:

1. **Large scanned PDFs were re-rendered at 150 dpi** to keep the repository a reasonable
   size. Only documents that were already pure image scans with no text layer were
   touched, and page count and page geometry are unchanged.
2. **OCR was reformatted** from the internal storage format into one JSON file per page.
   The content is unmodified.

The OCR JSON under `data/*/ocr/` is Takeovers' own pipeline output. It is provided as a
convenience for these challenges. It contains errors, as OCR does.

## Personal data

These filings name real people — company officers and shareholders — because that is
what the public register records about a company. Please treat the documents as what they
are: public corporate records, provided for the purpose of this hiring exercise. Do not
redistribute them, and do not use them for anything else.

## The code

`tools/bbox_viewer.py` and the challenge briefs and schemas are © Takeovers SAS, provided
to candidates for the purpose of completing these challenges.
