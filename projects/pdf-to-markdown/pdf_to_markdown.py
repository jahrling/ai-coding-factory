#!/usr/bin/env python3
"""
pdf_to_markdown.py — Transcribe a scanned book PDF to clean Markdown using a
local Ollama vision model (full-page transcription, one call per page).

Three modes:
  (default)        Body transcription: full page image -> Markdown.
  --margins-only   Sidebar pass: crop the narrow outer margin column and
                   transcribe only that, into a companion *.margins.md.
  --fold-margins   Merge step: fold the margins into the body Markdown as a
                   per-chapter "Sidebar Stories" section, into *.merged.md.

Designed for unattended overnight runs: incremental writes, resumable via a
manifest, a catch-all so a single bad page never kills the run, retries with
backoff, per-page logging, and a QA net that flags suspicious pages by
cross-checking against the PDF's own text layer.

Page numbers everywhere are 1-indexed PHYSICAL PDF pages (the Nth page of the
file), NOT the printed page labels — so front-matter Roman numerals do not
shift anything.

Usage:
    python pdf_to_markdown.py input.pdf output.md
    python pdf_to_markdown.py input.pdf output.md --pages 233-234
    python pdf_to_markdown.py input.pdf output.md --pages 100,105,230-250
    python pdf_to_markdown.py input.pdf output.md --margins-only
    python pdf_to_markdown.py input.pdf output.md --fold-margins

See README.md for how to launch this for an unattended overnight run.
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from datetime import datetime

import requests

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("PyMuPDF is required: pip install pymupdf")


# ----------------------------------------------------------------------------
# Tunable constants
# ----------------------------------------------------------------------------
MODEL = "qwen2.5vl:7b"
OLLAMA_URL = "http://localhost:11434"

# Context window and output budget. The encoded page image consumes most of a
# small default context, which starves the transcription and causes the model
# to emit only a sentence or two. A large num_ctx leaves room for a full page
# of output; num_predict caps (or with -1, uncaps) how many tokens it may emit.
NUM_CTX = 8192          # raise this first if pages still come out short
NUM_PREDICT = 4096      # max output tokens; -1 for unlimited
TEMPERATURE = 0.1       # near-deterministic transcription

# Cap the longest side of the rendered image (page or margin crop). Oversized
# images can make Ollama reject the request with a bare HTTP 400. DPI is
# computed per image.
MAX_IMAGE_DIM = 2000    # pixels

# Retries for transient transcription failures.
RETRY_COUNT = 3         # attempts per page
RETRY_BACKOFF = 3       # seconds; grows linearly per attempt (3s, 6s, ...)
REQUEST_TIMEOUT = 600   # seconds per HTTP call

# QA thresholds for the body pass (cross-check against the PDF text layer).
#   LENGTH_RATIO: flag when transcription word count is below this fraction of
#   the PDF text-layer word count for the page. Directly catches truncation.
#   WORD_OVERLAP: flag when the fraction of PDF-text-layer words that also
#   appear in the transcription falls below this. Catches dropped content,
#   misreads, and hallucination.
QA_LENGTH_RATIO_THRESHOLD = 0.55
QA_WORD_OVERLAP_THRESHOLD = 0.45
QA_MIN_PDF_WORDS = 25   # skip QA on near-empty pages (blank / image-only)

# --- Margins pass (the narrow outer sidebar column) -------------------------
# This layout has a second, narrow column on the OUTER edge of each page
# (right on recto/odd pages, left on verso/even pages) holding italic sidebar
# stories that continue across pages like footnotes. We crop that column and
# transcribe it on its own so the model can't ignore it.
MARGIN_WIDTH_FRAC = 0.30     # fraction of page width the outer column occupies
MARGIN_SIDE = "auto"         # "auto" (outer edge by parity) | "left" | "right"
MARGIN_PARITY_OFFSET = 0     # add to page number before the odd/even test;
                             # set to 1 if auto picks the wrong side
MARGIN_HEADER_FRAC = 0.06    # trim this fraction off the top (running header)
MARGIN_FOOTER_FRAC = 0.06    # trim this fraction off the bottom (page number)
SAVE_MARGIN_CROPS = True     # write each crop PNG for spot-checking the framing
MARGIN_SECTION_TITLE = "Sidebar Stories"   # heading used when folding
NO_MARGIN_SENTINEL = "(no margin content)"

# --- De-hyphenation ---------------------------------------------------------
# Line-break hyphens ("impres-\nsion") are fused back into whole words in every
# transcription and across the page seam when folding. Only a hyphen sitting
# immediately before a newline is treated as an artifact; inline hyphens
# (self-esteem) are never touched. A reference vocabulary built from the PDF's
# own text layer protects genuine compounds: the hyphen is kept when fusing
# would create a non-word but both halves are real words (so "well-\nknown"
# -> "well-known", not "wellknown"). Always on; no CLI flag. With no text
# layer it degrades to a plain broad join.
DEHYPHENATE = True
_VOCAB = frozenset()   # populated per run in main() from the PDF text layer


PROMPT = """You are transcribing ONE page from a scanned book into clean Markdown. \
Read the whole page image and reproduce its text exactly.

Rules:
- Transcribe ALL body text verbatim, keeping the original paragraph breaks.
- Format epigraphs, pull-quotes, and attributed quotations (these are often \
italic and often set off in a sidebar or margin) as Markdown blockquotes using \
`>`. Put the attribution (the author or source) on its own line inside the \
blockquote.
- Use Markdown headers (#, ##, ###) for chapter titles and section headings, \
matching their visual hierarchy.
- Omit running headers, running footers, and page numbers.
- Preserve the natural reading order of the page.
- Do NOT summarize, translate, comment, or add anything that is not on the page. \
Output only the page's Markdown transcription.

Transcribe the page now."""


MARGIN_PROMPT = """This image is the narrow OUTER MARGIN COLUMN of a scanned book \
page. It contains sidebar stories, epigraphs, and attributed quotations set in \
small italic type. Transcribe everything in this column in reading order, top to \
bottom.

Rules:
- Each item usually has a short ALL-CAPS or small-caps TITLE, an italic body, and \
an attribution/source line (author, work, date).
- Format each item as a Markdown blockquote (`>`). Put the title in bold on its \
own line, then the body, then the attribution on its own line.
- The text may begin or end mid-sentence because these stories continue from the \
previous page or onto the next — transcribe EXACTLY what is visible and do NOT \
complete, guess, or invent text.
- Ignore any body text, page numbers, or running headers that bleed in from the \
edge of the crop.
- If the column is empty or has no readable sidebar text, output exactly: {sentinel}

Transcribe the margin column now.""".format(sentinel=NO_MARGIN_SENTINEL)


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
def _zoom_for(longest_pts):
    return MAX_IMAGE_DIM / longest_pts if longest_pts > 0 else 1.0


def render_page_png(page):
    """Render a full page to PNG bytes, longest side ~= MAX_IMAGE_DIM."""
    rect = page.rect
    zoom = _zoom_for(max(rect.width, rect.height))
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return pix.tobytes("png")


def margin_side_for_page(page_num):
    """Which edge the outer sidebar column sits on for this page."""
    if MARGIN_SIDE in ("left", "right"):
        return MARGIN_SIDE
    # auto: outer edge. Recto (odd) pages have the margin on the right.
    return "right" if ((page_num + MARGIN_PARITY_OFFSET) % 2 == 1) else "left"


def render_margin_png(page, page_num):
    """Crop the outer margin column and render it to PNG bytes.

    Returns (png_bytes, side, clip_rect).
    """
    rect = page.rect
    W, H = rect.width, rect.height
    top = H * MARGIN_HEADER_FRAC
    bottom = H * (1.0 - MARGIN_FOOTER_FRAC)
    side = margin_side_for_page(page_num)
    if side == "right":
        x0, x1 = W * (1.0 - MARGIN_WIDTH_FRAC), W
    else:
        x0, x1 = 0.0, W * MARGIN_WIDTH_FRAC
    clip = fitz.Rect(x0, top, x1, bottom)
    zoom = _zoom_for(max(clip.width, clip.height))
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
    return pix.tobytes("png"), side, clip


# ----------------------------------------------------------------------------
# Transcription
# ----------------------------------------------------------------------------
def strip_code_fences(text):
    """Remove leading/trailing ```markdown ... ``` wrappers the model may add."""
    t = text.strip()
    t = re.sub(r"^\s*```[a-zA-Z0-9]*[ \t]*\n", "", t)
    t = re.sub(r"\n?[ \t]*```[ \t]*$", "", t)
    return t.strip()


def transcribe_image(png_bytes, log, prompt=PROMPT):
    """Send one image to Ollama /api/generate and return the transcription.

    Retries transient failures with linear backoff. On an HTTP error, the
    server's actual response body is surfaced in the log so a 400 is
    diagnosable rather than opaque.
    """
    b64 = base64.b64encode(png_bytes).decode("ascii")
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": [b64],
        "stream": False,
        "options": {
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "temperature": TEMPERATURE,
        },
    }
    url = OLLAMA_URL.rstrip("/") + "/api/generate"

    last_err = None
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                body = resp.text[:2000] if resp.text else "(empty body)"
                raise RuntimeError(f"HTTP {resp.status_code} from Ollama: {body}")
            data = resp.json()
            text = data.get("response", "")
            if not text.strip():
                raise RuntimeError("empty 'response' field from Ollama")
            return strip_code_fences(text)
        except Exception as e:  # noqa: BLE001 - transient network/HTTP/JSON errors
            last_err = e
            log(f"    attempt {attempt}/{RETRY_COUNT} failed: {e}")
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_BACKOFF * attempt)

    raise RuntimeError(f"all {RETRY_COUNT} attempts failed: {last_err}")


# ----------------------------------------------------------------------------
# QA net
# ----------------------------------------------------------------------------
WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _words(text):
    return WORD_RE.findall(text.lower())


# ----------------------------------------------------------------------------
# De-hyphenation
# ----------------------------------------------------------------------------
# A word fragment, a hyphen, a single newline (optionally through a blockquote
# "> " continuation marker), then a lowercase continuation. The single newline
# keeps us inside one paragraph — a blank line (paragraph break) never matches,
# so distinct paragraphs and list items are safe. Digits are excluded, so
# number ranges like "1620-\n1705" are left alone.
DEHYPH_RE = re.compile(r"([A-Za-z]+)-[ \t]*\n[ \t]*(?:>[ \t]*)?([a-z][A-Za-z]*)")


def build_vocab(doc):
    """Lowercase word set from the whole PDF text layer, used only to tell a
    soft line-break hyphen from a genuine compound. Content accuracy does not
    matter here — just which words exist somewhere in the book."""
    vocab = set()
    for i in range(doc.page_count):
        for w in WORD_RE.findall(doc.load_page(i).get_text().lower()):
            if w.isalpha() and len(w) >= 2:
                vocab.add(w)
    return frozenset(vocab)


def _fuse_or_keep(left, right):
    """Decide how to resolve a line-break hyphen between `left` and `right`."""
    fused = left + right
    if _VOCAB:
        if fused.lower() in _VOCAB:
            return fused                       # clearly one word -> fuse
        if left.lower() in _VOCAB and right.lower() in _VOCAB:
            return f"{left}-{right}"            # genuine compound -> keep hyphen
    return fused                               # default (and no-vocab): fuse


def dehyphenate(text):
    """Fuse soft line-break hyphens across the whole text. Idempotent."""
    if not DEHYPHENATE or not text:
        return text
    return DEHYPH_RE.sub(lambda m: _fuse_or_keep(m.group(1), m.group(2)), text)


def qa_flags(transcription, pdf_text):
    """Return a list of (reason, metric_value) for a page, empty if it looks ok."""
    flags = []
    pdf_words = _words(pdf_text)
    trans_words = _words(transcription)

    if len(pdf_words) < QA_MIN_PDF_WORDS:
        return flags

    ratio = len(trans_words) / len(pdf_words)
    if ratio < QA_LENGTH_RATIO_THRESHOLD:
        flags.append(("length_ratio", round(ratio, 3)))

    pdf_vocab = set(pdf_words)
    trans_vocab = set(trans_words)
    overlap = len(pdf_vocab & trans_vocab) / len(pdf_vocab)
    if overlap < QA_WORD_OVERLAP_THRESHOLD:
        flags.append(("word_overlap", round(overlap, 3)))

    return flags


# ----------------------------------------------------------------------------
# Persistence: manifest, output, log, flagged pages
# ----------------------------------------------------------------------------
def manifest_path(output_path):
    return output_path + ".manifest.json"


def load_manifest(output_path):
    p = manifest_path(output_path)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "completed": set(data.get("completed", [])),
                "failed": set(data.get("failed", [])),
            }
        except Exception:
            pass
    return {"completed": set(), "failed": set()}


def save_manifest(output_path, manifest):
    """Atomically persist the manifest (temp file + rename + fsync)."""
    p = manifest_path(output_path)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            {
                "completed": sorted(manifest["completed"]),
                "failed": sorted(manifest["failed"]),
            },
            f,
        )
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def append_page(output_path, page_num, body):
    """Append one page block to the output file and fsync it to disk."""
    block = f"<!-- page {page_num} -->\n\n{body.strip()}\n\n---\n\n"
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(block)
        f.flush()
        os.fsync(f.fileno())


def make_logger(log_path):
    def log(msg):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return log


def write_flag(flag_path, page_num, reasons):
    with open(flag_path, "a", encoding="utf-8") as f:
        parts = ", ".join(f"{r}={v}" for r, v in reasons)
        f.write(f"page {page_num}: {parts}\n")


# ----------------------------------------------------------------------------
# Page selection
# ----------------------------------------------------------------------------
def parse_pages(spec, total):
    """Parse a --pages spec into a sorted list of 1-indexed physical pages.

    Accepts comma-separated tokens, each of:
        N        single page
        A-B      inclusive range
        A-       from A to the last page
        -B       from page 1 to B
    Out-of-range values are clamped to [1, total].
    """
    out = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.endswith("-"):
            a, b = int(tok[:-1]), total
        elif tok.startswith("-"):
            a, b = 1, int(tok[1:])
        elif "-" in tok:
            lo, hi = tok.split("-", 1)
            a, b = int(lo), int(hi)
        else:
            a = b = int(tok)
        if a > b:
            a, b = b, a
        for p in range(a, b + 1):
            out.add(p)
    return sorted(p for p in out if 1 <= p <= total)


def selected_pages(args, total):
    if args.pages:
        return parse_pages(args.pages, total)
    start = max(1, args.start) if args.start else 1
    end = min(total, args.end) if args.end else total
    return list(range(start, end + 1))


# ----------------------------------------------------------------------------
# Body pass
# ----------------------------------------------------------------------------
def run_body(doc, pages, output_path, log, flag_path):
    manifest = load_manifest(output_path)
    already = len([p for p in manifest["completed"] if p in set(pages)])
    if already:
        log(f"Resuming: {already} of the selected page(s) already done; skipping them.")

    processed = 0
    for page_num in pages:  # ascending, 1-indexed physical
        if page_num in manifest["completed"]:
            continue
        t0 = time.time()
        try:
            page = doc.load_page(page_num - 1)
            pdf_text = page.get_text()
            png = render_page_png(page)
            transcription = transcribe_image(png, log, PROMPT)
            transcription = dehyphenate(transcription)
            append_page(output_path, page_num, transcription)

            reasons = qa_flags(transcription, pdf_text)
            if reasons:
                write_flag(flag_path, page_num, reasons)
            words = len(_words(transcription))
            dt = time.time() - t0
            flag_note = f"  FLAGGED({reasons})" if reasons else ""
            log(f"page {page_num}: OK  words={words}  {dt:.1f}s{flag_note}")
        except Exception as e:  # noqa: BLE001 - catch-all: never die mid-run
            placeholder = (
                f"> [!WARNING] Transcription failed for page {page_num}: "
                f"{str(e)[:500]}"
            )
            try:
                append_page(output_path, page_num, placeholder)
            except Exception as ee:  # noqa: BLE001
                log(f"page {page_num}: ALSO failed to write placeholder: {ee}")
            manifest["failed"].add(page_num)
            write_flag(flag_path, page_num, [("transcription_failed", str(e)[:120])])
            log(f"page {page_num}: FAILED — {e}")

        manifest["completed"].add(page_num)
        save_manifest(output_path, manifest)
        processed += 1

    log(f"Body done. processed={processed}  completed_total={len(manifest['completed'])}"
        f"  failed_total={len(manifest['failed'])}")
    if manifest["failed"]:
        log(f"Failed pages (placeholders written): {sorted(manifest['failed'])}")


# ----------------------------------------------------------------------------
# Margins pass
# ----------------------------------------------------------------------------
def run_margins(doc, pages, margins_path, crops_dir, log):
    manifest = load_manifest(margins_path)
    already = len([p for p in manifest["completed"] if p in set(pages)])
    if already:
        log(f"Resuming margins: {already} of the selected page(s) already done; skipping.")
    if SAVE_MARGIN_CROPS:
        os.makedirs(crops_dir, exist_ok=True)

    processed = 0
    for page_num in pages:
        if page_num in manifest["completed"]:
            continue
        t0 = time.time()
        try:
            page = doc.load_page(page_num - 1)
            png, side, clip = render_margin_png(page, page_num)
            if SAVE_MARGIN_CROPS:
                with open(os.path.join(crops_dir, f"p{page_num}.png"), "wb") as f:
                    f.write(png)
            transcription = transcribe_image(png, log, MARGIN_PROMPT)
            empty = transcription.strip() == NO_MARGIN_SENTINEL
            if not empty:
                transcription = dehyphenate(transcription)
            append_page(margins_path, page_num, transcription)

            words = len(_words(transcription))
            dt = time.time() - t0
            note = "  (empty)" if empty else f"  words={words}"
            log(f"page {page_num}: margin OK  side={side}{note}  {dt:.1f}s")
        except Exception as e:  # noqa: BLE001 - catch-all
            placeholder = (
                f"> [!WARNING] Margin transcription failed for page {page_num}: "
                f"{str(e)[:500]}"
            )
            try:
                append_page(margins_path, page_num, placeholder)
            except Exception as ee:  # noqa: BLE001
                log(f"page {page_num}: ALSO failed to write placeholder: {ee}")
            manifest["failed"].add(page_num)
            log(f"page {page_num}: margin FAILED — {e}")

        manifest["completed"].add(page_num)
        save_manifest(margins_path, manifest)
        processed += 1

    log(f"Margins done. processed={processed}  "
        f"completed_total={len(manifest['completed'])}  "
        f"failed_total={len(manifest['failed'])}")
    if manifest["failed"]:
        log(f"Failed margin pages: {sorted(manifest['failed'])}")


# ----------------------------------------------------------------------------
# Fold: merge margins into the body as per-chapter endnotes
# ----------------------------------------------------------------------------
PAGE_MARKER_RE = re.compile(r"<!-- page (\d+) -->")


def parse_page_blocks(md_path):
    """Parse a *.md we wrote into an ordered list of (page_num, body_text)."""
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()
    parts = PAGE_MARKER_RE.split(text)
    blocks = []
    for i in range(1, len(parts), 2):
        num = int(parts[i])
        body = parts[i + 1].strip()
        if body.endswith("---"):
            body = body[:-3].rstrip()
        blocks.append((num, body))
    return blocks


def chapter_starts_from_toc(doc):
    """Return sorted 1-indexed page numbers where chapters start, or None.

    Uses the PDF outline/bookmarks. The shallowest outline level is treated as
    the chapter level.
    """
    toc = doc.get_toc(simple=True)  # [[level, title, page], ...], page 1-indexed
    if not toc:
        return None
    min_level = min(lvl for lvl, _title, _pg in toc)
    starts = sorted({pg for lvl, _title, pg in toc if lvl == min_level and pg >= 1})
    return starts or None


def fold_margins(doc, body_path, margins_path, merged_path, log):
    body_blocks = parse_page_blocks(body_path)
    if not body_blocks:
        log(f"FATAL: no page blocks found in {body_path}")
        return

    margins = {}
    if os.path.exists(margins_path):
        for num, txt in parse_page_blocks(margins_path):
            if txt and txt.strip() != NO_MARGIN_SENTINEL:
                margins[num] = txt
    else:
        log(f"WARNING: margins file not found: {margins_path} — writing body only")

    # Determine chapter-start pages.
    starts = chapter_starts_from_toc(doc)
    if starts:
        source = "PDF table of contents"
    else:
        starts = [num for num, body in body_blocks if body.lstrip().startswith("# ")]
        source = "top-level '#' headers in the body"
    if not starts:
        source = "none found — all margins collected at document end"
    start_set = set(starts)
    log(f"Chapter boundaries from: {source}  ({len(start_set)} chapters)")

    out = []
    bucket = []
    started = False

    def flush():
        if not bucket:
            return
        # Assemble the chapter's sidebar fragments. When a fragment ends with a
        # letter+hyphen, the story continues onto the next page's fragment, so
        # join them with a single newline (not a blank line) and let
        # dehyphenate() fuse the split word across the seam; otherwise keep the
        # blank line between distinct items.
        parts = []
        for frag in bucket:
            frag = frag.strip()
            if parts and re.search(r"[A-Za-z]-\s*$", parts[-1]):
                parts[-1] = parts[-1].rstrip() + "\n" + frag
            else:
                parts.append(frag)
        section_body = dehyphenate("\n\n".join(parts))
        out.append(f"## {MARGIN_SECTION_TITLE}\n\n{section_body}\n\n---\n\n")
        bucket.clear()

    for num, body in body_blocks:
        if started and num in start_set:
            flush()  # close out the previous chapter's sidebars
        # Also clean the body on merge, so an existing run made before
        # de-hyphenation existed still comes out clean without re-transcribing.
        out.append(f"<!-- page {num} -->\n\n{dehyphenate(body)}\n\n---\n\n")
        started = True
        if num in margins:
            bucket.append(margins[num])
    flush()

    with open(merged_path, "w", encoding="utf-8") as f:
        f.write("".join(out))
    log(f"Wrote merged Markdown: {merged_path}  "
        f"(pages={len(body_blocks)}, margin-bearing pages={len(margins)})")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Transcribe a scanned PDF to Markdown via a local Ollama "
                    "vision model. Page numbers are 1-indexed PHYSICAL PDF "
                    "pages, not printed page labels."
    )
    ap.add_argument("input_pdf")
    ap.add_argument("output_md")
    ap.add_argument("--pages", metavar="SPEC",
                    help="Pages to process, 1-indexed physical PDF pages. "
                         "Comma-separated N, A-B, A- (to end), or -B (from "
                         "start). E.g. '233-234' or '100,105,230-250'.")
    ap.add_argument("--start", type=int, metavar="N",
                    help="1-indexed first page (used only if --pages absent).")
    ap.add_argument("--end", type=int, metavar="N",
                    help="1-indexed last page (used only if --pages absent).")
    ap.add_argument("--margins-only", action="store_true",
                    help="Transcribe only the outer sidebar column into "
                         "<output>.margins.md (does not touch the body file).")
    ap.add_argument("--fold-margins", action="store_true",
                    help="Merge <output>.margins.md into <output_md> as "
                         "per-chapter 'Sidebar Stories' endnotes, writing "
                         "<output>.merged.md. No transcription is done.")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.margins_only and args.fold_margins:
        sys.exit("--margins-only and --fold-margins are mutually exclusive.")

    output_path = args.output_md
    base = os.path.splitext(output_path)[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    margins_path = base + ".margins.md"
    merged_path = base + ".merged.md"
    crops_dir = base + ".margins_crops"

    if args.fold_margins:
        log_path = f"{base}.fold.{ts}.log"
    elif args.margins_only:
        log_path = f"{base}.margins.{ts}.log"
    else:
        log_path = f"{base}.{ts}.log"
    log = make_logger(log_path)

    if not os.path.exists(args.input_pdf):
        log(f"FATAL: input PDF not found: {args.input_pdf}")
        sys.exit(1)

    doc = fitz.open(args.input_pdf)
    total_pages = doc.page_count

    global _VOCAB
    if DEHYPHENATE:
        _VOCAB = build_vocab(doc)
        log(f"De-hyphenation on; compound-guard vocab={len(_VOCAB)} words "
            f"from the PDF text layer")

    if args.fold_margins:
        log(f"FOLD  body={output_path}  margins={margins_path}  -> {merged_path}")
        fold_margins(doc, output_path, margins_path, merged_path, log)
        doc.close()
        return

    pages = selected_pages(args, total_pages)
    if not pages:
        log("No pages selected; nothing to do.")
        doc.close()
        return

    log(f"Model={MODEL}  Ollama={OLLAMA_URL}")
    log(f"num_ctx={NUM_CTX}  num_predict={NUM_PREDICT}  max_image_dim={MAX_IMAGE_DIM}")
    log(f"PDF={args.input_pdf}  physical_pages={total_pages}")
    log(f"Selected {len(pages)} page(s): {pages[0]}..{pages[-1]} "
        f"(1-indexed physical)  Log={log_path}")

    if args.margins_only:
        log(f"MODE=margins-only  width_frac={MARGIN_WIDTH_FRAC}  side={MARGIN_SIDE}"
            f"  parity_offset={MARGIN_PARITY_OFFSET}  crops_dir={crops_dir}")
        log(f"Output={margins_path}")
        run_margins(doc, pages, margins_path, crops_dir, log)
    else:
        flag_path = base + ".flagged_pages.txt"
        log(f"MODE=body  Output={output_path}  Flags={flag_path}")
        run_body(doc, pages, output_path, log, flag_path)

    doc.close()


if __name__ == "__main__":
    main()
