#!/usr/bin/env python3
"""
pdf_to_markdown.py — Transcribe a scanned book PDF to clean Markdown using a
local Ollama vision model (full-page transcription, one call per page).

Designed for unattended overnight runs: incremental writes, resumable via a
manifest, a catch-all so a single bad page never kills the run, retries with
backoff, per-page logging, and a QA net that flags suspicious pages by
cross-checking against the PDF's own text layer.

Usage:
    python pdf_to_markdown.py input.pdf output.md
    python pdf_to_markdown.py input.pdf output.md --test 3
    python pdf_to_markdown.py input.pdf output.md --start 100 --end 200

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

# Cap the longest side of the rendered page image. Oversized images can make
# Ollama reject the request with a bare HTTP 400. DPI is computed per page.
MAX_IMAGE_DIM = 2000    # pixels

# Retries for transient transcription failures.
RETRY_COUNT = 3         # attempts per page
RETRY_BACKOFF = 3       # seconds; grows linearly per attempt (3s, 6s, ...)
REQUEST_TIMEOUT = 600   # seconds per HTTP call

# QA thresholds (cross-check against the PDF text layer).
#   LENGTH_RATIO: flag when transcription word count is below this fraction of
#   the PDF text-layer word count for the page. Directly catches truncation.
#   WORD_OVERLAP: flag when the fraction of PDF-text-layer words that also
#   appear in the transcription falls below this. Catches dropped content,
#   misreads, and hallucination.
QA_LENGTH_RATIO_THRESHOLD = 0.55
QA_WORD_OVERLAP_THRESHOLD = 0.45
QA_MIN_PDF_WORDS = 25   # skip QA on near-empty pages (blank / image-only)


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


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
def render_page_png(page):
    """Render a PyMuPDF page to PNG bytes, longest side ~= MAX_IMAGE_DIM.

    DPI is computed per page from its point dimensions so pages of different
    sizes all land near the same pixel budget.
    """
    rect = page.rect
    longest_pts = max(rect.width, rect.height)
    if longest_pts <= 0:
        zoom = 1.0
    else:
        zoom = MAX_IMAGE_DIM / longest_pts
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return pix.tobytes("png")


# ----------------------------------------------------------------------------
# Transcription
# ----------------------------------------------------------------------------
CODE_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*\n?|\n?```\s*$")


def strip_code_fences(text):
    """Remove leading/trailing ```markdown ... ``` wrappers the model may add."""
    t = text.strip()
    # Strip a leading fence line (```md / ```markdown / bare ```).
    t = re.sub(r"^\s*```[a-zA-Z0-9]*[ \t]*\n", "", t)
    # Strip a trailing fence.
    t = re.sub(r"\n?[ \t]*```[ \t]*$", "", t)
    return t.strip()


def transcribe_image(png_bytes, log):
    """Send one page image to Ollama /api/generate and return the transcription.

    Retries transient failures with linear backoff. On an HTTP error, the
    server's actual response body is surfaced in the log so a 400 is
    diagnosable rather than opaque.
    """
    b64 = base64.b64encode(png_bytes).decode("ascii")
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
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
                # Surface Ollama's real error body, not just the status line.
                body = resp.text[:2000] if resp.text else "(empty body)"
                raise RuntimeError(
                    f"HTTP {resp.status_code} from Ollama: {body}"
                )
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


def qa_flags(transcription, pdf_text):
    """Return a list of (reason, metric_value) for a page, empty if it looks ok.

    Cross-checks the model's transcription against the PDF's own text layer.
    """
    flags = []
    pdf_words = _words(pdf_text)
    trans_words = _words(transcription)

    if len(pdf_words) < QA_MIN_PDF_WORDS:
        return flags  # too little reference text to judge

    # Length ratio (catches truncation).
    ratio = len(trans_words) / len(pdf_words)
    if ratio < QA_LENGTH_RATIO_THRESHOLD:
        flags.append(("length_ratio", round(ratio, 3)))

    # Word overlap: fraction of PDF vocabulary present in the transcription.
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
# Main
# ----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Transcribe a scanned PDF to Markdown via a local Ollama "
                    "vision model."
    )
    ap.add_argument("input_pdf")
    ap.add_argument("output_md")
    ap.add_argument("--test", type=int, metavar="N",
                    help="Process only the first N pages (verification run).")
    ap.add_argument("--start", type=int, metavar="N",
                    help="1-indexed first page to process (inclusive).")
    ap.add_argument("--end", type=int, metavar="N",
                    help="1-indexed last page to process (inclusive).")
    return ap.parse_args()


def resolve_range(args, total_pages):
    """Return an inclusive 1-indexed (start, end) page range from the args."""
    start = 1
    end = total_pages
    if args.start:
        start = max(1, args.start)
    if args.end:
        end = min(total_pages, args.end)
    if args.test:
        # --test N means the first N pages of the selected start.
        end = min(end, start + args.test - 1)
    return start, end


def main():
    args = parse_args()

    output_path = args.output_md
    base = os.path.splitext(output_path)[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"{base}.{ts}.log"
    flag_path = f"{base}.flagged_pages.txt"

    log = make_logger(log_path)

    if not os.path.exists(args.input_pdf):
        log(f"FATAL: input PDF not found: {args.input_pdf}")
        sys.exit(1)

    doc = fitz.open(args.input_pdf)
    total_pages = doc.page_count
    start, end = resolve_range(args, total_pages)

    manifest = load_manifest(output_path)

    log(f"Model={MODEL}  Ollama={OLLAMA_URL}")
    log(f"num_ctx={NUM_CTX}  num_predict={NUM_PREDICT}  max_image_dim={MAX_IMAGE_DIM}")
    log(f"PDF={args.input_pdf}  pages={total_pages}  range={start}..{end}")
    log(f"Output={output_path}  Log={log_path}  Flags={flag_path}")
    already = len([p for p in manifest["completed"] if start <= p <= end])
    if already:
        log(f"Resuming: {already} page(s) in this range already completed; skipping them.")

    processed = 0
    for page_num in range(start, end + 1):  # 1-indexed
        if page_num in manifest["completed"]:
            continue

        t0 = time.time()
        try:
            page = doc.load_page(page_num - 1)  # fitz is 0-indexed
            pdf_text = page.get_text()

            png = render_page_png(page)
            transcription = transcribe_image(png, log)

            append_page(output_path, page_num, transcription)

            # QA cross-check against the PDF text layer.
            reasons = qa_flags(transcription, pdf_text)
            if reasons:
                write_flag(flag_path, page_num, reasons)

            words = len(_words(transcription))
            dt = time.time() - t0
            flag_note = f"  FLAGGED({reasons})" if reasons else ""
            log(f"page {page_num}: OK  words={words}  {dt:.1f}s{flag_note}")

        except Exception as e:  # noqa: BLE001 - catch-all: never die mid-run
            # Write a placeholder so page numbering stays intact and log it.
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

        # Mark done (success or placeholder) so a rerun doesn't duplicate it.
        manifest["completed"].add(page_num)
        save_manifest(output_path, manifest)
        processed += 1

    log(f"Done. processed={processed}  completed_total={len(manifest['completed'])}"
        f"  failed_total={len(manifest['failed'])}")
    if manifest["failed"]:
        log(f"Failed pages (placeholders written): {sorted(manifest['failed'])}")
    doc.close()


if __name__ == "__main__":
    main()
