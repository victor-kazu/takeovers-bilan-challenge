# Challenge: reading French annual accounts

**Track:** Data / ML Engineer · **Time:** 6–8 hours · **Window:** 7 days

Pick this one *or* the [actes challenge](../actes/BRIEF.md). Not both.

---

## The situation

A French company files its annual accounts — its *bilan* — with the registry every
year. The filing is a standardised tax form, the *liasse fiscale*: a fixed set of
numbered pages (2050, 2051, 2052 …) with fixed line codes. In principle it is the most
structured document in French corporate life.

In practice it arrives as a PDF that may be a clean digital export, a photocopy scanned
at an angle, or a form filled in by an accountant who used the boxes creatively. The
numbers may be in euros or in thousands of euros, said once in small print. Companies
use different accounting software that lays the same form out differently.

Turning that into a table you can compare across companies and years is the first thing
Takeovers had to build, and we are still improving it.

## What we ask of you

**Build a pipeline that extracts 12 financial fields from the filings in `data/`, and
report each value with the place in the document you read it from.**

Then tell us what it cost — in euros per page, in seconds per page, and in accuracy —
and defend the trade-off you chose.

### The data

The repository has one shared corpus at `../../data/`, used by both data challenges. It
holds twenty companies, each with their actes and their annual filings:

```
data/<siren>/bilans/pdf/bilan_<deposit_date>_<doc_id>.pdf
data/<siren>/bilans/meta/                       the registry's index entry
data/<siren>/bilans/ocr/<doc_id>/page_001.json  our OCR, one file per page
data/<siren>/actes/…                            not needed for this challenge
```

**Your scope is these 15 documents**, from 5 of those companies:

| siren | deposited | file |
|---|---|---|
| `820561470` | 2023-06-05 | `bilan_2023-06-05_6493e4372f502414800f8164.pdf` |
| `820561470` | 2023-06-13 | `bilan_2023-06-13_6543d3fd08093cdace058668.pdf` |
| `820561470` | 2024-01-15 | `bilan_2024-01-15_67458f18cea78a70070fa226.pdf` |
| `328024377` | 2020-12-24 | `bilan_2020-12-24_63e8ebbb54febda17c19ee7c.pdf` |
| `328024377` | 2021-12-17 | `bilan_2021-12-17_63e8ebbb54febda17c19ee7d.pdf` |
| `328024377` | 2022-12-13 | `bilan_2022-12-13_63e8ebbb54febda17c19ee7e.pdf` |
| `445070311` | 2022-02-14 | `bilan_2022-02-14_63e2481c916269756a09542b.pdf` |
| `445070311` | 2023-11-21 | `bilan_2023-11-21_65a4095d5fd178b16b09b860.pdf` |
| `445070311` | 2025-05-15 | `bilan_2025-05-15_6860f28ca0138eae340c7453.pdf` |
| `504304205` | 2017-05-31 | `bilan_2017-05-31_63e13943526e1f30cd100db5.pdf` |
| `504304205` | 2018-10-24 | `bilan_2018-10-24_63e13943526e1f30cd100db6.pdf` |
| `504304205` | 2024-08-06 | `bilan_2024-08-06_66cd893cedec9b09d50191e8.pdf` |
| `401009741` | 2022-11-30 | `bilan_2022-11-30_63e881158be6eb9f9d1ff975.pdf` |
| `401009741` | 2023-11-20 | `bilan_2023-11-20_65784e5da67d84faf4042736.pdf` |
| `401009741` | 2025-10-03 | `bilan_2025-10-03_68f0a715f28d8aaf48046416.pdf` |

The five were chosen to be unlike each other on purpose. Measured over the pages we ship:

| siren | what makes it awkward |
|---|---|
| `820561470` | a crooked scan — mean skew 0.7°, with rotated pages |
| `328024377` | **reports in thousands of euros**, said in passing on a handful of lines |
| `401009741` | a dense liasse grid — our table detector finds ~12 tables per page |
| `445070311` | far sparser — ~6 tables per page, and the thinnest text |
| `504304205` | the wordiest — around 100 text lines per page |

If your pipeline only works on the first company you open, you will find out here.

Three filings per company is also deliberate: each restates the previous exercise in its
own N-1 column, so in most cases you can check a figure against the year before without
us telling you the answers. The deposit dates are not evenly spaced — some filed late, or
twice in a year.

**OCR coverage across the wider corpus is uneven**, though every document in your scope
above has it. Ignore the other companies; they belong to the other challenge.

### Using our OCR, or not

We give you our OCR because we would rather see how you turn text into structured data
than watch you install an OCR engine. **Using it is a perfectly good choice, and a
defensible one to write in your README.** Running your own OCR, or sending page images
to a vision model, is also a good choice. They cost different amounts and they fail in
different ways — which is the point of the cost question.

The OCR JSON per page looks like this:

```json
{
  "page": 4,
  "skew_angle": 0.31,
  "layout": [ { "label": "table", "bbox": [x1,y1,x2,y2], "cells": [ … ] } ],
  "ocr":    [ { "polygon": [[x,y],[x,y],[x,y],[x,y]], "text": "…", "score": 0.98 } ]
}
```

`layout` is our table detector — sometimes it finds the liasse grid cleanly, sometimes
it finds nothing useful. `ocr` is the flat list of text lines and is always populated.
**All coordinates in the OCR are pixels at 300 dpi**, and `page` is 1-indexed.

### The fields

The 12 fields are defined in [`schema/financial_fields.json`](schema/financial_fields.json)
with an English and a French label, which statement they belong to, and which page of the
liasse they usually sit on.

They span all three main pages of the form, so no single template will get you all of
them. Two are deliberately awkward: cost of goods sold is not printed as a line and has
to be built, and average workforce is not a monetary value at all.

Note `BS_TOTAL_ASSETS_FRGAAP`: it must reconcile against the other side of the balance
sheet. That is a check you can run yourself.

### What you deliver

A `results.json` at the root of your repository, matching
[`schema/results.schema.json`](schema/results.schema.json). Every value carries:

- `value` and `unit` — `EUR` or `kEUR`. **A pipeline that ignores the units question is
  wrong by a factor of 1000 on some of these filings.**
- `page`, 1-indexed
- `bbox` — `[x0, y0, x1, y1]` **normalized 0–1** against page width and height, origin
  top-left

The OCR ships in 300-dpi pixels and you must submit normalized fractions, so there is a
conversion to do. `tools/bbox_viewer.py --grep` prints boxes already converted, if you
want to check yours against ours.

Plus a `run` block with your measured cost and time per page.

And a `README.md` covering:

- how to run it
- **the trade-off**: what you chose, what it cost per page in euros and in seconds, what
  it bought you in accuracy, and what you would do differently with a week
- **how you used AI tools** — see the [repository README](../../README.md#ai-tools)
- what you cut, and why

And a **`.env.example`** naming every environment variable your code reads — API keys,
tokens, model names — with no values in it, so we know what to set to run your pipeline.
This matters more here than usual: your cost-per-page claim only means something if we
can see which provider and model produced it. Never commit a real key or a `.env` file.
See [Submitting](../../README.md#submitting).

## What we are actually reading

We do not publish an answer key, and we are not withholding one to be difficult: for
work like this the answer is often genuinely contested, and being asked to reach
defensible conclusions in a domain you do not know is the job.

So we read: whether the numbers are right where we can check them; whether the boxes
point at the right place on the page; whether your cost claim is derived or guessed;
whether your README tells us what is broken. **A pipeline that does 6 fields well and
says so beats one that reports all 12 with three of them silently wrong.**

There is more here than fits in 6–8 hours. That is on purpose. Choosing what to leave
out, and saying that you left it out, is part of the answer.

## Ground rules

Everything in the [repository README](../../README.md) applies: AI tools are allowed and
must be declared, external sources are fair game, and submission is a PR to your own
repository.
