# PDF → Markdown (local vision-model transcription)

Convert a scanned book PDF into clean Markdown by sending each **full page
image** to a local Ollama vision model (`qwen2.5vl:7b`). Built for an
unattended overnight run on a large book: incremental writes, resumable,
crash-tolerant, with a QA net that flags suspicious pages.

## Requirements

```bash
pip install pymupdf requests
# Ollama running locally with the model pulled:
ollama pull qwen2.5vl:7b
```

Ollama is expected at `http://localhost:11434` (change `OLLAMA_URL` near the
top of `pdf_to_markdown.py` if different).

## Usage

```bash
# Full body run
python pdf_to_markdown.py 48LawsOfPower.pdf power.md

# Verification run: first 3 pages only (do this first — see below)
python pdf_to_markdown.py 48LawsOfPower.pdf power.md --pages -3

# Specific pages / ranges (see --pages below)
python pdf_to_markdown.py 48LawsOfPower.pdf power.md --pages 233-234
python pdf_to_markdown.py 48LawsOfPower.pdf power.md --pages 100,105,230-250
```

### Page numbering

Every page number this tool takes or prints — `--pages`, the manifest, the
`<!-- page N -->` markers, the log — is the **1-indexed physical PDF page**
(the Nth page of the file). It is **not** the printed page label, so
front-matter numbered in Roman numerals does not shift anything: `--pages 1`
is always the very first page of the file.

### `--pages` selector

A permanent, general page selector (replaces the old `--test`). Comma-separated
tokens, each of:

| Token | Meaning |
|-------|---------|
| `N` | single page |
| `A-B` | inclusive range |
| `A-` | from A to the last page |
| `-B` | from page 1 to B |

Examples: `--pages 233-234`, `--pages 100,105,230-250`, `--pages 450-`.
Out-of-range values are clamped. (`--start`/`--end` still work when `--pages`
is absent.)

Outputs, all next to `power.md`:

| File | Purpose |
|------|---------|
| `power.md` | Combined body Markdown, pages separated by `<!-- page N -->` + `---` |
| `power.manifest.json` | Completed/failed page numbers (drives resume) |
| `power.flagged_pages.txt` | QA flags: length-ratio / word-overlap / failures |
| `power.<timestamp>.log` | Per-page progress log |

## Margins (the outer sidebar column)

This book has a second, narrow column on the **outer edge** of each page
(right on odd/recto pages, left on even/verso pages) holding italic sidebar
stories that continue across pages like footnotes. The full-page body pass
tends to ignore them, so there is a separate two-step feature — **extract**,
then **fold** — that leaves the body file untouched.

### 1. Extract — `--margins-only`

Crops the outer column per page and transcribes only that:

```bash
# calibrate on a couple of known pages first, then check the crops
python pdf_to_markdown.py 48LawsOfPower.pdf power.md --margins-only --pages 233-234
# then the whole book
python pdf_to_markdown.py 48LawsOfPower.pdf power.md --margins-only
```

Produces (next to `power.md`, non-destructive to `power.md` itself):

| File | Purpose |
|------|---------|
| `power.margins.md` | Per-page margin transcriptions (`<!-- page N -->` blocks) |
| `power.margins.md.manifest.json` | Independent resume manifest for the margins pass |
| `power.margins_crops/pN.png` | The exact crop sent to the model — **spot-check these** |
| `power.margins.<timestamp>.log` | Per-page margin log |

**Calibrating the crop** (do this on pages 233–234 before the full margins run):
open `power.margins_crops/p233.png` etc. and confirm the whole sidebar column
is inside the crop with no body text bleeding in. Tune these constants at the
top of the script:

```python
MARGIN_WIDTH_FRAC = 0.30   # widen if the column is being clipped
MARGIN_SIDE = "auto"       # "auto" = outer edge by parity; force "left"/"right"
MARGIN_PARITY_OFFSET = 0   # set to 1 if auto picks the wrong side for your file
MARGIN_HEADER_FRAC = 0.06  # top trim (running header)
MARGIN_FOOTER_FRAC = 0.06  # bottom trim (page number)
```

Because the margin alternates edges, `auto` picks right for odd physical pages
and left for even. If your file's front matter makes that land on the wrong
side, set `MARGIN_PARITY_OFFSET = 1` to flip it (verify against the saved crops).

### 2. Fold — `--fold-margins`

Folds the extracted margins into the body as **per-chapter endnotes**. Sidebar
stories aren't page-anchored footnotes — they're a per-chapter stream — so each
chapter's margins are collected and dropped as a `## Sidebar Stories` section at
the **end of that chapter**:

```bash
python pdf_to_markdown.py 48LawsOfPower.pdf power.md --fold-margins
# -> writes power.merged.md  (body + power.margins.md; neither is modified)
```

Chapter boundaries are detected generally, in this order:

1. the **PDF's table of contents / bookmarks** (`doc.get_toc()`) — authoritative;
2. else **top-level `#` headers** in `power.md`;
3. else everything collected into a single `## Sidebar Stories` section at the
   end of the document.

No transcription happens in this mode — it's pure text assembly, so it's cheap
to re-run after you fix a crop and re-extract a few pages.

## The truncation fix (verify this first)

The previous "one sentence per page" problem was **output truncation**: the
encoded page image fills a small default context window, leaving no room for
the transcription. The request now sends an `options` object with a large
context and generous output budget:

```python
NUM_CTX = 8192        # raise this first if pages still come out short
NUM_PREDICT = 4096    # max output tokens (-1 = unlimited)
```

**Before the full run, do a 3-page test (`--pages -3`) and read the log word
counts.** Each dense body-text page should transcribe as *hundreds* of words.
If any page is short, raise `NUM_CTX` (e.g. 12288 or 16384) and re-run.
The QA net also catches this automatically: a truncated page trips the
`length_ratio` flag in `flagged_pages.txt`.

> Note: the author verified the full robustness pipeline (image DPI cap,
> `num_ctx`/`num_predict` in the payload, incremental writes, manifest
> resume, catch-all placeholders, retries with backoff, and both QA flags)
> against a synthetic PDF and a mock Ollama server. The final word-count
> check against the **real** `qwen2.5vl:7b` model must be run on your
> machine, since it needs the GPU + model — that is exactly what `--pages -3`
> is for. The margins extract/fold plumbing (crop + parity, `--pages` parsing,
> and all three chapter-boundary paths) was likewise verified against a
> synthetic two-column PDF with a bookmark TOC; the crop framing itself needs
> your eyes on `power.margins_crops/pN.png`.

## Tunable constants (top of `pdf_to_markdown.py`)

`MODEL`, `OLLAMA_URL`, `NUM_CTX`, `NUM_PREDICT`, `MAX_IMAGE_DIM` (longest
image side in px; oversized images can cause a bare HTTP 400), `RETRY_COUNT`,
`RETRY_BACKOFF`, and the two QA thresholds `QA_LENGTH_RATIO_THRESHOLD` /
`QA_WORD_OVERLAP_THRESHOLD`.

## Robustness (how an overnight run survives)

- **Incremental**: each page is appended and `fsync`ed as it completes — a
  crash at page 300 keeps pages 1–299.
- **Resumable**: the manifest is saved after every page. Re-running the same
  command skips completed pages and continues. (Output is appended, so process
  ranges in ascending order.)
- **Never dies mid-run**: any per-page failure (render, HTTP, unexpected) is
  logged, written as a `> [!WARNING]` placeholder so page numbering stays
  intact, and the loop moves on. Only a killed process or power loss stops it.
- **Retries**: each transcription is retried with linear backoff; HTTP errors
  log Ollama's **actual response body**, not just the status line.

## Launching an unattended overnight run

Use `nohup` (or tmux) so it survives your terminal / SSH session closing, and
save the PID so you can stop it later.

**Option A — nohup + saved PID:**

```bash
cd /path/to/pdf-to-markdown
nohup python3 pdf_to_markdown.py 48LawsOfPower.pdf power.md \
    > power.console.log 2>&1 &
echo $! > power.pid          # save the PID
echo "started PID $(cat power.pid)"

# check progress later
tail -f power.console.log     # or: tail -f power.*.log

# stop it later
kill "$(cat power.pid)"       # graceful; safe to resume afterward
```

**Option B — tmux (detachable session):**

```bash
tmux new -s transcribe
python3 pdf_to_markdown.py 48LawsOfPower.pdf power.md
# detach: Ctrl-b then d      reattach: tmux attach -t transcribe
# stop:   reattach and Ctrl-c, or: tmux kill-session -t transcribe
```

After stopping (or a crash/power loss), just re-run the **same command** —
it picks up from the manifest where it left off.

> ⚠️ **Run only one instance against a given output file.** Concurrent
> instances share and corrupt the manifest and interleave writes into the
> `.md`. One PDF → one output file → one running process.

## Morning spot-check

Open `power.flagged_pages.txt`. Each line is a page plus the reason and metric
value, e.g.:

```
page 3: length_ratio=0.024, word_overlap=0.154
page 88: transcription_failed: HTTP 400 from Ollama: ...
```

Only those pages need a look. To redo a flagged/failed page, remove its number
from the `completed` list in `power.manifest.json` (and delete its
`<!-- page N -->` block from `power.md` to avoid a duplicate), then re-run with
`--start N --end N`.
