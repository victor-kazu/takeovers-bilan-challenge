# Challenge: reconstructing who owned a company

**Track:** Data / ML Engineer · **Time:** 6–8 hours · **Window:** 7 days

Pick this one *or* the [bilan challenge](../bilan/BRIEF.md). Not both.

---

## The situation

When a French company changes — raises capital, admits a shareholder, transfers shares —
it deposits the minutes of that decision with the registry. These are the *actes*: the
statutes at incorporation, then a couple of decades of general-meeting minutes and
presidential decisions, scanned and filed in no particular order.

Nowhere in that pile is there a statement of who owns the company. There is only a
sequence of changes, each written in the legal French of the year it happened, each
assuming you have read the ones before it. The ownership history exists only as
something you reconstruct.

That reconstruction is the core of what Takeovers sells.

## What we ask of you

**Rebuild the capital composition of ARCHEAN TECHNOLOGIES (SIREN 480489707) over its
whole life, from its filings, with every change grounded in the document it came from.**

Two artefacts, in one `results.json`:

1. **`events[]`** — the capital and shareholder movements you can find, using the codes
   we define.
2. **`capital_timeline[]`** — the state of the cap table after each of those events: the
   capital, the shares, and who held them.

The timeline is the one that matters. Events are how you justify it.

## Scope

Only **capital composition**. These documents also record auditors, presidents, address
changes, name changes and object changes — all out of scope, ignore them.

[`schema/event_codes.json`](schema/event_codes.json) defines the five codes we score:

| code | what it is |
|---|---|
| `CAPITAL_INCREASE` | the nominal share capital goes up |
| `CAPITAL_DECREASE` | the nominal share capital goes down |
| `SHAREHOLDER_ENTRY` | someone joins the cap table |
| `SHAREHOLDER_END` | someone leaves it entirely |
| `SHAREHOLDER_SHARE_TRANSFER` | shares move from one party to another |

Each has a description and its expected payload fields in that file. `CAPITAL_DUAL_CLASS`
is listed too but is **not scored** — emit it if you spot it, ignore it at no cost.

A warning worth taking seriously: a share transfer and a shareholder entry can describe
the same movement seen from two sides. Deciding what that means for the timeline is part
of the problem, and it is one we have got wrong ourselves more than once.

## The data

The repository has one shared corpus at `../../data/`, used by both data challenges:

```
data/480489707/actes/pdf/    17 actes, 2005 → 2025   ← the subject
data/480489707/actes/meta/   the registry's own index entry per document
data/480489707/actes/ocr/    our OCR, one JSON per page
data/480489707/bilans/…      9 annual filings, same shape
data/<19 other sirens>/…     every company has both actes and bilans
```

Every company folder is the same shape. **OCR coverage is uneven** — most companies have
it for most documents, some have gaps, and a few have none at all. That is what our data
actually looks like: it reflects what has been through our pipeline, nothing more. If you
want a document we have no OCR for, you are free to read it yourself.

The OCR JSON per page:

```json
{
  "page": 4,
  "ocr": [ { "polygon": [[x,y],[x,y],[x,y],[x,y]], "text": "…", "score": 0.98 } ],
  "layout": []
}
```

**Coordinates in the OCR are pixels at 300 dpi** and `page` is 1-indexed. What you submit
is normalized — see below.

### About the other nineteen companies

A few of them are related to ARCHEAN TECHNOLOGIES. Most are not — they are ordinary
French companies with no connection to it whatsoever, and they are here so that the size
of the group is not something you can read off the folder listing. We are not telling you
which is which; working that out is the bonus (below).

For the main task you can ignore all of them — the capital timeline of 480489707 is
reconstructable from its own folder.

## What you deliver

A `results.json` at the root of your repository, matching
[`schema/results.schema.json`](schema/results.schema.json).

Every event carries a `source`:

```json
"source": {
  "inpi_id": "63e9593b8be6eb9f9d257ec0",
  "page": 3,
  "bbox": [0.116, 0.610, 0.920, 0.626],
  "snippet": "…"
}
```

`bbox` is `[x0, y0, x1, y1]` **normalized 0–1**, origin top-left, page 1-indexed. The OCR
gives you 300-dpi pixels, so there is a conversion to do — `tools/bbox_viewer.py --grep`
prints boxes already converted if you want to check yours.

A grounded event is worth more than an ungrounded one. An event you cannot ground is
still worth reporting, with a note saying so.

Dates: `event_date` is when the decision took effect, which is usually **not** the
deposit date in the filename. The gap is sometimes months.

Plus a `README.md` covering how to run it, **how you used AI tools**
(see the [repository README](../../README.md#ai-tools)), what you could not resolve, and
what you would do next.

And a **`.env.example`** naming every environment variable your code reads — API keys,
tokens, model names — with no values in it, so we know what to set to run your pipeline.
Never commit a real key or a `.env` file. See
[Submitting](../../README.md#submitting).

## Bonus: the group

Optional. Do the timeline first.

The twenty companies in `data/` are not a flat list. A few own others; at least one is
named in another's filings without owning anything; most have nothing to do with any of
it. Do not assume the group is large just because the folder is.

Reconstruct what you can, as `group.nodes` and `group.edges` in your `results.json`, with
evidence on each edge.

Three things worth knowing before you start:

- **Not every relationship is recorded in an acte.** Some of them are only visible in
  the annual filings.
- **It does not stop at one hop.** A company that owns ARCHEAN TECHNOLOGIES may itself be
  owned by someone, and that will not be written anywhere in ARCHEAN TECHNOLOGIES' own
  documents. You have to go and look.
- **Some names cannot be resolved to a company at all.** `resolved: false` is a correct
  answer, and a better one than a SIREN you guessed.

## What we are actually reading

We do not publish an answer key. Partly because handing you one would make this a
copying exercise, and partly because we genuinely do not have a settled one — our own
cross-checks flag contradictions in this company's capital chain that no one has
adjudicated yet.

So we read: whether the timeline is internally coherent, whether the boxes point where
you say they do, whether you noticed the places the documents disagree with each other,
and whether your README is honest about what you could not do.

There is far more here than fits in 6–8 hours. That is deliberate. **We are not
expecting a complete answer.** How you decide what to attack first, and how clearly you
say what you left, is a large part of what we are reading.

## Ground rules

Everything in the [repository README](../../README.md) applies: AI tools are allowed and
must be declared, external sources are fair game — including opening your own free INPI
account to pull documents we did not give you — and submission is a PR to your own
repository.
