"""
Accounts Filing — Bank Statement to Excel extraction backend.

Accepts a bank statement PDF (text-based or scanned), figures out which
columns are Date / Narration / Debit / Credit / Balance regardless of which
bank issued it, and returns the parsed transactions as JSON. The WordPress
front-end turns that JSON into an .xlsx file client-side (SheetJS), so this
service never has to generate or store a spreadsheet.

Design notes:
- Nothing is written to disk except a short-lived temp file for the
  duration of one request; it is deleted in a `finally` block no matter
  what happens.
- Text-based PDFs are parsed with pdfplumber (fast, accurate). Scanned /
  photographed statements fall back to Tesseract OCR page-by-page.
- Column detection is header-driven and bank-agnostic: we look for a
  header row containing recognizable labels (Date, Narration/Particulars,
  Debit/Withdrawal, Credit/Deposit, Balance) via fuzzy matching, record
  each header's x-position, then assign every word on every later line to
  the nearest header column by x-position. This is what lets one parser
  handle SBI/HDFC/ICICI/Axis/Kotak/etc. statements without a per-bank
  template.
"""

import io
import os
import re
import tempfile
import threading
import time
import uuid
from typing import Optional

import pdfplumber
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from pdf2image import convert_from_path
from rapidfuzz import fuzz
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Accounts Filing — Bank Statement Extractor")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "ALLOWED_ORIGINS", "https://accountsfiling.com,https://www.accountsfiling.com"
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

MAX_FILE_BYTES = int(os.environ.get("MAX_FILE_MB", "20")) * 1024 * 1024
MAX_PAGES = int(os.environ.get("MAX_PAGES", "60"))
OCR_DPI = int(os.environ.get("OCR_DPI", "300"))
# Tesseract is built with OpenMP and by default starts one thread per visible
# CPU core. Confirmed real: on a 2-core machine that made each scanned page
# take ~9.3s instead of ~2.0s (4.5x slower — the threads spend their time
# fighting each other), and a hosted instance with a fractional CPU quota but
# many visible cores is hit even harder (10 scanned pages took over 5 minutes
# there, with no result returned). One thread per tesseract process is
# fastest; pages are parallelised across processes instead (OCR_WORKERS).
os.environ["OMP_THREAD_LIMIT"] = "1"
OCR_WORKERS = max(1, int(os.environ.get("OCR_WORKERS", "2")))


# ---------------------------------------------------------------------------
# Column vocabulary — this is the part to extend if a bank's statement
# uses a header word we don't recognize yet.
# ---------------------------------------------------------------------------

COLUMN_KEYWORDS = {
    "date": ["date", "txn date", "transaction date", "value date", "posting date"],
    "narration": [
        "narration", "description", "particulars", "transaction details",
        "remarks", "details", "transaction remarks",
    ],
    "ref": [
        "chq", "ref no", "reference", "cheque no", "chq/ref no", "ref number",
        "instrument no", "instrument number",
    ],
    "debit": ["debit", "withdrawal", "withdrawals", "withdrawal amt", "dr", "withdrawal amount"],
    "credit": ["credit", "deposit", "deposits", "deposit amt", "cr", "deposit amount"],
    "amount": ["amount"],  # some banks use a single Amount + Dr/Cr indicator column
    "balance": ["balance", "closing balance", "running balance", "available balance"],
    # Real IDBI Bank net-banking statement export prints "S.No" as its own
    # leading column before the date. It carries no useful data, but it
    # still needs to be recognized and given its own column boundary —
    # otherwise its values have nowhere to go but bleed into the Date
    # column right next to it (confirmed with a real IDBI statement: the
    # Date cell came out as "1 13/06/2025 09:32:45", with the leading "1"
    # being the S.No value). Never read by parse_page_set — same
    # deliberately-dropped treatment as a non-primary duplicate column.
    "serial": ["s.no", "sr.no", "sr no", "sl.no", "sl no", "serial no", "serial number"],
}

DATE_RE = re.compile(
    r"^(\d{1,2}[-/. ]([A-Za-z]{3,9}|\d{1,2})[-/. ]\d{2,4}|\d{4}-\d{2}-\d{2})$"
)
# Same shape as DATE_RE but without the end anchor — a real IDBI statement
# prints a transaction time right after the date IN THE SAME cell ("Txn
# Date" and "Value Date" have no separate time column, e.g. cell text ends
# up as "13/06/2025 09:32:45"), which DATE_RE's exact-length match rejects
# outright. Used only for the "does this line start a new row" decision in
# parse_page_set — the row's stored date value is left exactly as extracted
# (date + time together), this just recognizes it as a real date so the row
# isn't dropped or wrongly merged into the previous one.
DATE_PREFIX_RE = re.compile(
    r"^(\d{1,2}[-/. ]([A-Za-z]{3,9}|\d{1,2})[-/. ]\d{2,4}|\d{4}-\d{2}-\d{2})"
)
TIME_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")
AMOUNT_RE = re.compile(r"^[\d,]+\.\d{2}$|^[\d,]+$")
# Stricter than AMOUNT_RE: requires the two-decimal-places shape every real
# amount in an Indian bank statement actually has ("3,500.00"). Used only to
# decide whether a dateless line is a genuine transaction row (see
# parse_page_set) — a bare integer like a wrapped invoice number or year
# ("2026") can end up misassigned into a numeric column when a narration
# continuation line's last word overflows past that column's estimated
# left/right edge (a header label is usually narrower than its data column,
# see the "Known limitations" note in README.md), and without this stricter
# check that stray digit string alone was enough to fabricate a bogus row.
REAL_AMOUNT_RE = re.compile(r"^[\d,]+\.\d{2}$")
# A bare small integer — what a genuine S.No/serial-number cell looks like.
SERIAL_RE = re.compile(r"^\d{1,6}$")


# ---------------------------------------------------------------------------
# Word/line extraction
# ---------------------------------------------------------------------------

def _is_border_noise(text):
    """True for a token that is only table-grid-line debris: OCR reads the
    vertical rules of a bordered statement table as stray "|" tokens (confirmed
    real on a scanned HDFC statement, where they landed inside the narration
    and ref cells of most rows). No real statement content is made only of
    these characters."""
    return text != "" and all(ch in "|¦" for ch in text)


def _pages_have_text_layer(pdf_path, password):
    """Per-page True/False for "has a real text layer", via poppler's
    pdftotext (already installed here, since pdf2image needs poppler), or
    None if that can't be determined for any reason (tool missing, encrypted
    file, timeout...) — callers then just use their original, slower path.
    Exists because pdfplumber's own text-extraction pass is surprisingly
    slow on image-only pages (~2s EACH — it decodes the full-page scan just
    to discover there's no text on it): 78s of pure waste before OCR even
    started on a real 43-page scanned statement, vs ~2s for this check."""
    import subprocess

    cmd = ["pdftotext"]
    if password:
        cmd += ["-upw", password]
    cmd += [pdf_path, "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=60)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    pages = out.stdout.decode("utf-8", errors="ignore").split("\f")
    if pages and pages[-1].strip() == "":
        pages = pages[:-1]  # pdftotext ends with a trailing form feed
    return [sum(1 for ch in p if not ch.isspace()) >= 20 for p in pages]


# A narrow Date column makes some banks (a real IndusInd statement) wrap the
# date itself onto two lines — "2026-" on the transaction's line and "07-31"
# on the next line — and when the row sits at the bottom of a page the second
# half can land at the TOP of the next page, right under the repeated table
# header. Neither half is a date on its own, so no row ever starts and the
# statement comes out empty (or nearly). This runs on the extracted words,
# before any line grouping, and glues each date-prefix word to the matching
# fragment directly below it in the same x position (or, for the last row on
# a page, the first matching fragment at the top of the next page).
_SPLIT_DATE_PREFIX_RE = re.compile(
    r"^(\d{4}[-/]|\d{1,2}[-/](\d{1,2}|[A-Za-z]{3,9})[-/])$"
)
SPLIT_DATE_MAX_DROP = 30.0  # fragment must be this close below its prefix


def _join_split_dates(pages_words):
    def _fits(prefix, frag):
        return bool(DATE_RE.match(prefix["text"] + frag["text"]))

    def _find_below(page_words, p, consumed):
        best = None
        for w in page_words:
            if id(w) in consumed or w is p:
                continue
            if abs(w["x0"] - p["x0"]) > 3.0:
                continue
            drop = w["top"] - p["top"]
            if drop <= 0 or drop > SPLIT_DATE_MAX_DROP:
                continue
            if not _fits(p, w):
                continue
            if best is None or drop < best["top"] - p["top"]:
                best = w
        return best

    consumed = set()
    removals = {}  # page index -> set(id(word))
    joined = {}  # id(prefix word) -> fragment word
    for pi, pw in enumerate(pages_words):
        if not pw:
            continue
        prefixes = [w for w in pw if _SPLIT_DATE_PREFIX_RE.match(w["text"])]
        prefixes.sort(key=lambda w: w["top"])
        for p in prefixes:
            frag = _find_below(pw, p, consumed)
            frag_page = pi
            if frag is None and pi + 1 < len(pages_words) and pages_words[pi + 1]:
                # only the LAST prefix on the page may continue on the next
                # page, and only with a fragment that sits above every
                # prefix of that next page (i.e. before its first row)
                if p is prefixes[-1]:
                    nxt = pages_words[pi + 1]
                    next_prefix_tops = [
                        w["top"] for w in nxt if _SPLIT_DATE_PREFIX_RE.match(w["text"])
                    ]
                    limit = min(next_prefix_tops) if next_prefix_tops else None
                    cands = [
                        w
                        for w in nxt
                        if id(w) not in consumed
                        and abs(w["x0"] - p["x0"]) <= 3.0
                        and _fits(p, w)
                        and (limit is None or w["top"] < limit)
                    ]
                    if cands:
                        frag = min(cands, key=lambda w: w["top"])
                        frag_page = pi + 1
            if frag is None:
                continue
            consumed.add(id(frag))
            joined[id(p)] = frag
            removals.setdefault(frag_page, set()).add(id(frag))
    if not joined:
        return pages_words
    out = []
    for pi, pw in enumerate(pages_words):
        if not pw:
            out.append(pw)
            continue
        drop = removals.get(pi, set())
        new = []
        for w in pw:
            if id(w) in drop:
                continue
            if id(w) in joined:
                f = joined[id(w)]
                w = dict(w)
                w["text"] = w["text"] + f["text"]
                w["x1"] = max(w["x1"], w["x0"] + (f["x1"] - f["x0"]) + (w["x1"] - w["x0"]))
            new.append(w)
        out.append(new)
    return out



def extract_pages_words(pdf_path: str, password: Optional[str], progress=None):
    """Return a list of pages, each a list of word dicts:
    {text, x0, x1, top, bottom}. Falls back to OCR for pages with no
    extractable text layer (scanned/photographed statements).
    Raises ValueError("bad_password") / ValueError("too_many_pages") as needed.
    """
    pages_words = []
    needs_ocr_pages = []
    # Fast pre-check (poppler's pdftotext, ~2s for a 43-page file) of which
    # pages carry ANY text layer at all — see _pages_have_text_layer().
    # None means "couldn't tell", which falls back to the old behaviour.
    text_flags = _pages_have_text_layer(pdf_path, password)

    try:
        with pdfplumber.open(pdf_path, password=password or "") as pdf:
            if len(pdf.pages) > MAX_PAGES:
                raise ValueError("too_many_pages")
            for i, page in enumerate(pdf.pages):
                if (
                    text_flags is not None
                    and i < len(text_flags)
                    and not text_flags[i]
                ):
                    # confirmed image-only page: skip pdfplumber entirely
                    pages_words.append(None)
                    needs_ocr_pages.append(i)
                    continue
                words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
                # pdfplumber caches every page it touches (including each
                # page's decoded images) for the life of the PDF object.
                # Confirmed real: on a 43-page, 17.9MB image-only scan that
                # cache alone grew to ~3.4GB — enough to get the process
                # killed on a small hosted instance — versus ~240MB once
                # each page's cache is flushed as soon as it's been read.
                page.flush_cache()
                text_len = sum(len(w["text"]) for w in words)
                if text_len < 20:
                    # essentially no text layer -> scanned page, needs OCR
                    pages_words.append(None)
                    needs_ocr_pages.append(i)
                else:
                    pages_words.append(
                        [
                            {
                                "text": w["text"],
                                "x0": w["x0"],
                                "x1": w["x1"],
                                "top": w["top"],
                                "bottom": w["bottom"],
                            }
                            for w in words
                        ]
                    )
    except Exception as e:
        msg = str(e).lower()
        type_name = type(e).__name__.lower()
        if "password" in msg or "encrypt" in msg or "password" in type_name:
            raise ValueError("bad_password")
        raise

    if needs_ocr_pages:
        if pytesseract is None:
            raise ValueError("ocr_unavailable")
        # Render and OCR ONE page at a time. The original version rendered
        # every page up front with a single convert_from_path() call, which
        # holds every page's full-resolution bitmap in memory at once (~26MB
        # per A4 page at 300 DPI, before PIL overhead) — fine for a 6-page
        # scan, but a real 43-page scanned HDFC statement needed >1GB and was
        # killed by the OS (out-of-memory) before a single word was read, which
        # on a small hosted instance would take the whole service down for
        # every user, not just fail one request.
        ocr_done = [0]

        def _ocr_one_page(i):
            imgs = convert_from_path(
                pdf_path, dpi=OCR_DPI, userpw=password or None,
                first_page=i + 1, last_page=i + 1,
            )
            if not imgs:
                return []
            image = imgs[0]
            data = pytesseract.image_to_data(
                image, output_type=pytesseract.Output.DICT
            )
            del imgs, image
            scale = 72.0 / OCR_DPI  # convert pixel coords back to PDF-point scale
            words = []
            for j, text in enumerate(data["text"]):
                text = text.strip()
                if not text or _is_border_noise(text):
                    continue
                x0 = data["left"][j] * scale
                y0 = data["top"][j] * scale
                words.append(
                    {
                        "text": text,
                        "x0": x0,
                        "x1": x0 + data["width"][j] * scale,
                        "top": y0,
                        "bottom": y0 + data["height"][j] * scale,
                    }
                )
            ocr_done[0] += 1
            if progress:
                progress(f"Reading scanned page {ocr_done[0]} of {len(needs_ocr_pages)}")
            return words

        # At most OCR_WORKERS pages are in memory at once (each worker holds
        # one page's bitmap), so the memory ceiling stays low.
        if OCR_WORKERS > 1 and len(needs_ocr_pages) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=OCR_WORKERS) as pool:
                for i, words in zip(needs_ocr_pages, pool.map(_ocr_one_page, needs_ocr_pages)):
                    pages_words[i] = words
        else:
            for i in needs_ocr_pages:
                pages_words[i] = _ocr_one_page(i)

    return _join_split_dates(pages_words)


def group_into_lines(words, y_tolerance=3.0):
    """Cluster words into lines by vertical position, then sort each line
    left-to-right. Returns list of {"top": float, "words": [word,...]}."""
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines = []
    current = [words_sorted[0]]
    current_top = words_sorted[0]["top"]
    for w in words_sorted[1:]:
        if abs(w["top"] - current_top) <= y_tolerance:
            current.append(w)
            current_top = (current_top + w["top"]) / 2
        else:
            lines.append(current)
            current = [w]
            current_top = w["top"]
    lines.append(current)
    return [
        {"top": sum(w["top"] for w in line) / len(line),
         "words": sorted(line, key=lambda w: w["x0"])}
        for line in lines
    ]


def merge_header_tokens(line_words, x_gap=8.0):
    """Merge adjacent words on the header line into label groups, so a
    two-word header like 'Value Date' becomes one label with one x-range.

    x_gap was originally 18.0, which turned out to be too generous: on a
    header with several narrow columns close together (confirmed with a
    synthetic current-account-style statement carrying an "Instrument No"
    column right before "Narration", only ~13pt apart), it glued the two
    into one "Instrument No Narration" label that didn't fuzzy-match any
    known column type at all — worse than the earlier-known "Cheque No"/
    "Transaction Remarks" merge, this one made the header undetectable and
    the whole statement come back empty rather than the intended clear
    error. Measured across every header sample in tests/, the words within
    one genuine multi-word label (e.g. "Value" + "Date") sit about 2pt
    apart, while even the closest two real, separate columns are 12-13pt
    apart — 8.0 sits safely between those two numbers.
    """
    groups = []
    current = [line_words[0]]
    for w in line_words[1:]:
        if w["x0"] - current[-1]["x1"] <= x_gap:
            current.append(w)
        else:
            groups.append(current)
            current = [w]
    groups.append(current)
    return [
        {
            "label": " ".join(w["text"] for w in g),
            "x0": min(w["x0"] for w in g),
            "x1": max(w["x1"] for w in g),
        }
        for g in groups
    ]


def match_column_type(label: str):
    """Classify a header label into one of COLUMN_KEYWORDS's types.

    Checks for an exact keyword appearing as a whole word/phrase in the
    label FIRST, before ever falling back to fuzzy scoring. This matters
    because plain fuzz.ratio() can be fooled by two keywords for opposite
    meanings that happen to share a lot of characters — confirmed for
    real: "Debit Amount" scores 84.6 against the "credit" type's "deposit
    amount" keyword (they share length and most letters) but only 58.8
    against its own "debit" keyword, so the old fuzzy-only version
    silently swapped every Debit/Credit column on a real ICICI statement
    the user shared — about as bad a bug as this tool can have, since it
    doesn't just misplace a word, it reverses money in vs. money out.
    A whole-word check catches this immediately ("debit" is plainly in
    "debit amount"), and fuzzy matching is kept only as a fallback for
    headers that don't contain an exact keyword at all (genuine
    misspellings/OCR noise/unlisted wording)."""
    label_l = label.lower().strip(" :.")

    exact_hits = []  # (keyword_length, col_type)
    for col_type, keywords in COLUMN_KEYWORDS.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", label_l):
                exact_hits.append((len(kw), col_type))
    if exact_hits:
        # "amount" is deliberately generic (for a single-Amount-column
        # bank that shows direction via a separate Dr/Cr marker instead),
        # which means it's *also* a plain substring of "Debit Amount" /
        # "Credit Amount" / "Withdrawal Amount" / "Deposit Amount" — and
        # being a longer word than "debit"/"credit" themselves, a plain
        # longest-keyword tie-break would wrongly prefer it over the far
        # more specific, correct type. So: any specific type (debit,
        # credit, ...) always outranks "amount"; only fall back to
        # "amount" when nothing more specific matched. Ties between two
        # non-"amount" types (rare) fall back to the longest keyword.
        exact_hits.sort(key=lambda h: (h[1] != "amount", h[0]), reverse=True)
        return exact_hits[0][1]

    best_type, best_score = None, 0
    for col_type, keywords in COLUMN_KEYWORDS.items():
        for kw in keywords:
            score = fuzz.ratio(label_l, kw)
            if score > best_score:
                best_score, best_type = score, col_type
    if best_score >= 72:
        return best_type
    return None


def _mark_primary_columns(typed):
    """Some banks (ICICI is the one confirmed so far) print both a "Value
    Date" and a "Transaction Date" column — both fuzzy-match the single
    "date" keyword type. If both were treated as equally authoritative,
    assign_line_to_columns would collide them into one dict key and
    concatenate two separate dates into a single unparseable string (e.g.
    "01/09/2026 01/09/2026"), which fails DATE_RE and silently produces
    zero extracted rows instead of a clear error — confirmed with a
    synthetic ICICI-style sample in tests/make_bank_samples.py.

    So whenever a type appears more than once in the header, exactly one
    column is marked primary — preferring a "transaction"/"txn"-labelled
    date column (the date the transaction actually happened) over a
    "value"/"posting" date (a settlement date), since that's what
    bookkeeping normally wants — and the rest are marked non-primary so
    their words get their own cell key in assign_line_to_columns instead
    of colliding with, and corrupting, the primary column's value."""
    by_type = {}
    for c in typed:
        by_type.setdefault(c["type"], []).append(c)
    for cols in by_type.values():
        for c in cols:
            c["primary"] = False
        if len(cols) == 1:
            cols[0]["primary"] = True
            continue
        chosen = next(
            (c for c in cols if "transaction" in c["label"].lower() or "txn" in c["label"].lower()),
            None,
        )
        if chosen is None:
            chosen = next(
                (c for c in cols if "value" not in c["label"].lower() and "posting" not in c["label"].lower()),
                None,
            )
        if chosen is None:
            chosen = cols[0]
        chosen["primary"] = True


MAX_HEADER_LINES = 3
# How close together (in PDF points) consecutive lines need to be to be
# treated as parts of the SAME multi-line header block, rather than two
# unrelated lines that each happen to carry a recognizable label. Confirmed
# real with an IDBI Bank statement, whose header is genuinely split across
# three separate visual lines — "Withdrawals Deposits Balance", then
# "S.No Txn Date Value Date Description Cheque No", then "(Dr) (Cr)
# (INR)" — none of which alone contains a date + amount + narration label,
# so the old single-line-only scan fell through to the freeform fallback
# and produced garbage. Measured on that real header: consecutive header
# lines sit ~4pt apart, while the nearest unrelated line above it (the
# statement's own date-range line) sits ~39pt away, and the first real
# transaction row sits ~12pt below the header's last line — 6.0 sits
# safely between the two.
HEADER_LINE_MERGE_GAP = 6.0


def find_header_and_columns(lines):
    """Scan lines for the one that looks like a table header (contains a
    Date-ish label plus at least one amount-ish label), trying not just
    each line alone but also up to MAX_HEADER_LINES consecutive lines
    merged together when they sit close enough vertically to plausibly be
    one multi-line header block (see HEADER_LINE_MERGE_GAP). Returns
    (last_header_line_index, [{type, x0, x1, label, primary}, ...]) or
    (None, None)."""
    for idx in range(len(lines)):
        typed = []
        prev_top = None
        for offset in range(MAX_HEADER_LINES):
            li = idx + offset
            if li >= len(lines):
                break
            if offset > 0 and lines[li]["top"] - prev_top > HEADER_LINE_MERGE_GAP:
                break  # too far from the previous line — not the same header block
            prev_top = lines[li]["top"]
            groups = merge_header_tokens(lines[li]["words"])
            for g in groups:
                t = match_column_type(g["label"])
                if t:
                    typed.append({"type": t, "x0": g["x0"], "x1": g["x1"], "label": g["label"]})
            types_found = {c["type"] for c in typed}
            has_date = "date" in types_found
            has_amount = bool(types_found & {"debit", "credit", "amount", "balance"})
            has_narration = "narration" in types_found
            if has_date and has_amount and has_narration:
                _mark_primary_columns(typed)
                return li, sorted(typed, key=lambda c: c["x0"])
    return None, None


def column_boundaries(columns):
    """Boundary between column i and i+1 is the midpoint of the gap between
    them (col[i].x1 -> col[i+1].x0), not the midpoint of their centers —
    that way a wide column (e.g. Narration) doesn't steal words from a
    narrow neighbour (e.g. Ref No) just because its center sits further
    away. Returns a list of right-edges, one per column (last is +inf)."""
    edges = []
    for i, c in enumerate(columns):
        if i + 1 < len(columns):
            edges.append((c["x1"] + columns[i + 1]["x0"]) / 2)
        else:
            edges.append(float("inf"))
    return edges


_NUMERIC_COLUMN_TYPES = {"debit", "credit", "amount", "balance"}
# The various ways a bank prints "nothing here" in a debit/credit cell.
# "NA" is confirmed real — an actual ICICI statement screenshot the user
# shared prints "NA" (not blank, not "-") for whichever of Debit/Credit
# didn't apply to a row.
_NUMERIC_PLACEHOLDER_VALUES = {"-", "--", "na", "n.a", "n.a.", "nil"}

# A bare Cr/Dr balance-sign marker — confirmed real with a Bank of Baroda
# statement, which prints its running balance as a plain number followed by
# "Cr" (positive) or "Dr" (overdrawn/negative), e.g. "37549.63 Cr" or
# "390052.48 Dr", as two separate words rather than one combined cell. Before
# this was recognised as valid numeric-column content, the leftward-overflow
# guard below walked it backward out of the balance column entirely (since
# it doesn't look like an amount on its own) — straight past Debit/Credit —
# into the non-numeric ref/CHQ.NO. column, polluting nearly every row's ref
# field with a stray "Cr". See clean_amount() for where the actual sign
# conversion happens.
_CR_DR_WORD_RE = re.compile(r"^(?:cr|dr)$", re.IGNORECASE)

# A comma used where a decimal point belongs — confirmed real (3 occurrences
# across a 6-page Bank of Baroda statement), e.g. "37549,63" instead of
# "37549.63", always in the balance column. See _normalize_amount_token().
_COMMA_DECIMAL_RE = re.compile(r"^(\d[\d,]*),(\d{2})$")


def _normalize_amount_token(text):
    """Fix a token whose decimal point was rendered as a comma instead of a
    period. A trailing ",DD" (exactly two digits) with no period anywhere in
    the token can never be a legitimate Indian-style thousands separator —
    that always groups the rightmost segment as exactly THREE digits (e.g.
    "1,23,456"), never two — so seeing exactly two digits after the last
    comma, with no period in the token at all, is unambiguous proof of a
    corrupted decimal point rather than a real thousands comma. Leaves
    everything else (ordinary text, properly formatted amounts, genuine
    comma-grouped integers) untouched."""
    t = text.strip()
    if "." in t:
        return text
    m = _COMMA_DECIMAL_RE.match(t)
    if not m:
        return text
    return f"{m.group(1)}.{m.group(2)}"


def _looks_like_real_amount_word(text):
    """True for something that genuinely belongs in a debit/credit/amount/
    balance cell: a properly formatted amount (comma-separated, two decimal
    places — every real amount in these statements has this shape), one of
    the usual "nil" placeholder marks banks print for an empty cell, or a
    bare Cr/Dr balance-sign marker (see _CR_DR_WORD_RE above)."""
    t = text.strip()
    if t.lower() in _NUMERIC_PLACEHOLDER_VALUES:
        return True
    if _CR_DR_WORD_RE.match(t):
        return True
    return bool(REAL_AMOUNT_RE.match(t.replace(",", "")))


def _looks_like_serial_word(text):
    """True for a bare small integer — a genuine S.No/serial-number cell."""
    return bool(SERIAL_RE.match(text.strip()))


def _looks_like_date_or_time_word(text):
    """True for something that genuinely belongs in a date-type cell: a
    real date, or a time printed alongside it in the same cell (confirmed
    real: a real IDBI statement has no separate time column — "Txn Date"
    and "Value Date" each carry a date and a time as two words on the same
    line, e.g. "13/06/2025" then "09:32:45")."""
    t = text.strip()
    return bool(DATE_PREFIX_RE.match(t)) or bool(TIME_RE.match(t))


def assign_line_to_columns(line_words, columns):
    """Assign each word to a column by its start position against the
    gap-midpoint boundaries, and join same-column words with a space.

    A column marked non-primary (a duplicate-typed column — see
    _mark_primary_columns, e.g. ICICI's Value Date alongside Transaction
    Date) gets its own unique key here instead of the shared type key, so
    its words never land in — and corrupt — the primary column's value.
    parse_page_set only ever reads the fixed set of real field names
    (date/narration/ref/debit/credit/amount/balance), so these extra keys
    are simply never looked at again; that column's data is deliberately
    dropped rather than silently garbling the primary one.

    A word whose x-position nominally lands in a debit/credit/amount/
    balance column but doesn't actually look like an amount is redirected
    back to the nearest column to its left that isn't one of those types
    instead. This matters because a column's boundary is derived from its
    HEADER LABEL's own width, not the real printed width of its data —
    confirmed with a synthetic current-account statement where a normal
    business narration ("...FROM STARLIGHT ENTERPRISES INV 2201") ran
    past the short word "Narration"'s own extent and the tail words
    ("ENTERPRISES INV 2201") landed, purely by x-position, inside the
    neighbouring Debit column, corrupting real transaction data rather
    than just an odd word placement between two text columns. Guarding on
    "does this actually look like an amount" catches that without needing
    real column-grid detection.

    The opposite can also happen at the START of a row: a column's data can
    start further LEFT than its own header label, encroaching on whatever
    sits before it. Confirmed real with the same IDBI statement: its
    leading "S.No" column has no real data width to speak of, and the date
    column right after it prints its actual date value well to the left of
    where the "Txn Date"/"Value Date" labels themselves start — so by
    x-position alone, a serial number can land in the date cell (corrupting
    it — the whole reason S.No is tracked as its own "serial" column at
    all) while the real date, or even wrapped narration text from the
    column after THAT, can land one column too early. A word landing in a
    "serial" cell that isn't a bare small number, or in a "date" cell that
    isn't a date/time, is walked forward instead — the mirror image of the
    guard above."""
    edges = column_boundaries(columns)
    key_for = [
        c["type"] if c.get("primary", True) else f"_dup_{c['type']}_{i}"
        for i, c in enumerate(columns)
    ]
    cells = {k: [] for k in key_for}
    prev_idx = None
    prev_x1 = None
    for w in line_words:
        # Fix a comma-for-decimal-point rendering glitch (see
        # _normalize_amount_token) before any shape check runs, so both the
        # recognition logic below and the stored cell value see the
        # corrected text.
        text = _normalize_amount_token(w["text"])
        idx = 0
        while idx < len(edges) - 1 and w["x0"] >= edges[idx]:
            idx += 1
        while (
            idx > 0
            and columns[idx]["type"] in _NUMERIC_COLUMN_TYPES
            and not _looks_like_real_amount_word(text)
        ):
            idx -= 1
        while idx < len(columns) - 1 and (
            (columns[idx]["type"] == "serial" and not _looks_like_serial_word(text))
            or (columns[idx]["type"] == "date" and not _looks_like_date_or_time_word(text))
        ):
            idx += 1
        # A narration that runs on past its own header label (the header is
        # usually much narrower than the data, see above) can reach into a
        # following text column — confirmed real with an IndusInd statement,
        # whose "...TFS YAMUNA AIRPORT" narration put "AIRPORT" in the Ref No
        # cell. A word that starts clearly LEFT of that column's own header
        # label, and sits one normal word-space after a narration word, is
        # the same sentence continuing, not the start of the ref value.
        if (
            idx > 0
            and prev_idx == idx - 1
            and columns[idx]["type"] == "ref"
            and columns[idx - 1]["type"] == "narration"
            and prev_x1 is not None
            and w["x0"] - prev_x1 <= 6.0
            and w["x0"] < columns[idx]["x0"] - 4.0
        ):
            idx -= 1
        prev_idx, prev_x1 = idx, w["x1"]
        cells[key_for[idx]].append(text)
    return {t: " ".join(v).strip() for t, v in cells.items()}


# A number followed by a Cr/Dr sign marker — e.g. "390052.48 Dr" or
# "37549.63 Cr". Confirmed real with a Bank of Baroda statement; its sign
# marker is occasionally duplicated at a second, slightly-offset vertical
# position (a likely PDF-rendering/duplicate-text-layer artifact specific
# to this bank's export template), so the suffix group tolerates a repeated
# "Cr"/"Dr" ("... Dr Dr") rather than assuming exactly one.
_AMOUNT_WITH_CR_DR_RE = re.compile(
    r"^([\d,]+(?:\.\d{2})?)\s*((?:dr|cr)(?:\s+(?:dr|cr))*)$", re.IGNORECASE
)
# A bare Cr/Dr marker with no digits attached at all — e.g. a continuation
# line that only ever carried the sign word, or the same duplicate-marker
# artifact landing on its own. Resolves to "" (see below) rather than being
# treated as a real value.
_BARE_CR_DR_RE = re.compile(r"^(?:dr|cr)(?:\s+(?:dr|cr))*$", re.IGNORECASE)


def clean_amount(s: str):
    if not s:
        return ""
    s = s.strip()
    if _BARE_CR_DR_RE.match(s):
        # Harmless to drop: the "only fill balance if currently empty" merge
        # guard in parse_page_set treats an empty string exactly like no
        # value was found on this line at all.
        return ""
    m = _AMOUNT_WITH_CR_DR_RE.match(s)
    if m:
        digits, suffix = m.group(1), m.group(2)
        is_dr = "dr" in suffix.lower()
        digits = digits.replace(",", "")
        if not AMOUNT_RE.match(digits):
            return digits  # leave as-is; better to show raw than silently drop data
        return ("-" + digits) if is_dr else digits
    s = s.replace(",", "")
    if s == "" or s.lower() in _NUMERIC_PLACEHOLDER_VALUES:
        return ""
    if not AMOUNT_RE.match(s.replace(",", "")):
        return s  # leave as-is; better to show raw than silently drop data
    return s


MAX_ROW_LINE_GAP = 29.0
# Confirmed real with an IDBI statement: every page ends with a footer
# block ("IDBI Bank Ltd. Regd. Office...", "Page X of Y") separated from
# the last real transaction line by a distinctly larger vertical gap
# (measured ~33-39pt there) than any gap seen between two genuine
# transaction lines on that same statement (a continuation line sits ~6-8pt
# below the line it wraps from; two different transactions' first lines sit
# ~11-18pt apart). Without this, the footer's own lines have no date and
# get silently appended as if they were more wrapped narration/ref text for
# whatever the last real transaction happened to be.
#
# Originally set to 24.0, which sat safely between IDBI's own numbers but
# turned out too tight once a second real statement was tested: a real
# IDFC FIRST Bank statement has a genuine narration-wrap gap of up to
# ~24.1pt between two lines of the SAME wrapped transaction (this bank's
# line spacing runs a bit taller than IDBI's), which tripped the old 24.0
# threshold as if it were the page's footer boundary — silently truncating
# every real transaction after that point on the page, on every affected
# page. Confirmed by direct measurement across the entire real 22-page
# IDFC document: 24.1pt is its true maximum in-table gap, and its footer
# boundary gap is never smaller than ~90pt on any page that still has real
# transaction data (one page that's pure disclaimer/legend text with no
# transactions at all has a smaller ~25.4pt footer-entry gap, but that's
# harmless to miss here — with no transaction ever added on that page,
# there's nothing for a missed break to wrongly merge). 29.0 sits with
# margin on both sides of the two real statements measured so far: above
# both banks' real in-table maximums (18pt IDBI, 24.1pt IDFC) and below
# IDBI's own real footer minimum (33.6pt) — the number to revisit if a
# future real statement's line spacing runs even taller than IDFC's.


# A dateless run is split between the row before it and the row after it at
# its single clearly-largest vertical gap (see _split_dateless_run). How much
# larger than every other gap in the run that gap must be before it is trusted.
RUN_SPLIT_DISTINCT_RATIO = 1.25


def _split_dateless_run(a_top, run_tops, b_top):
    """Decide how many of the dateless lines sitting between two row-starting
    (dated) lines belong to the row ABOVE them (trailing wrap) — the rest
    belong to the row BELOW (leading narration). Returns k: the first k lines
    trail the upper row, the remaining len(run_tops)-k lead the lower row.

    Why this exists: banks lay a wrapped narration out in two different
    ways, and both are confirmed real —
      * TOP-aligned (IDBI, HDFC): the dated line is the narration's FIRST
        line, every wrapped line follows it.
      * CENTRE-aligned (IDFC FIRST, Bank of Baroda): the dated line sits in
        the MIDDLE of a multi-line narration block, so some narration lines
        come BEFORE it. IDFC e.g. prints "IMPS/509307268522/" and
        "ABHIJITRAJENDRA/" above the date line and "ICIC0007304/4765/" /
        "IMPSTransaction" below it.
    Both layouts keep the lines of ONE row packed at a small pitch and leave
    a visibly larger gap BETWEEN two rows' blocks, so the right split is
    simply the run's single largest gap: top-aligned runs have it last
    (k = all lines trail), centre-aligned runs have it somewhere inside.
    If no gap is clearly larger than the rest (uniform spacing), there is no
    evidence for a leading line, so everything trails — the safe default and
    the behaviour this tool always had.

    A previous version compared each line's distance to the last dated line
    against its distance to the next one. That is NOT the same thing and was
    wrong: the last wrapped line of any multi-line top-aligned narration is
    always nearer the NEXT row than the previous dated line, so it silently
    moved narration text onto the wrong transaction — 9 rows of a real IDBI
    statement and 241 of 248 on a real IDFC one. Amounts were never affected
    (which is why a balance-reconciliation check didn't catch it)."""
    m = len(run_tops)
    if m == 0:
        return 0
    tops = [a_top] + list(run_tops) + [b_top]
    gaps = [tops[i + 1] - tops[i] for i in range(len(tops) - 1)]  # len m+1
    k = max(range(len(gaps)), key=lambda i: gaps[i])
    others = [g for i, g in enumerate(gaps) if i != k]
    if not others or gaps[k] < max(others) * RUN_SPLIT_DISTINCT_RATIO + 0.5:
        return m
    return k


def _first_line_start_idx(page_lines, columns):
    """Index of the first line on a page that genuinely starts a transaction:
    its FIRST word is a date. (Only the date cell being a date isn't enough
    in header-less mode: page boilerplate such as "From : 01/06/2026 To :
    30/09/2026" has a date that lands in the date column by x-position.)"""
    for i, line in enumerate(page_lines):
        if DATE_RE.match(line["words"][0]["text"].strip()):
            cell = assign_line_to_columns(line["words"], columns)
            if DATE_PREFIX_RE.match(cell.get("date", "").strip()):
                return i
    return None


# A bare "Page 5 6" / "Page 2 of 9" line — a page-number footer. Confirmed
# real with a Bank of Baroda statement whose footer ("Page N M", then a
# "Customer Care" line) sits only ~15pt below the last wrapped narration
# line, i.e. INSIDE the gap guard, so five rows picked up
# "@ https://www.bankofbaroda.in Customer Care" and "Page 5 6" in narration/ref.
PAGE_FOOTER_RE = re.compile(
    r"^\s*page\s*(?:no\.?)?\s*\d+(?:\s*(?:of|/)?\s*\d+)?\s*$", re.IGNORECASE
)


def _line_text(line):
    return " ".join(w["text"] for w in line["words"]).strip()


def _find_repeat_header_blocks(page_lines, from_idx):
    """(first_idx, last_idx) of every table-header block at/after from_idx.
    A statement can start a SECOND table mid-page (confirmed real: a Bank of
    Baroda file whose current-account table ends and an overdraft-account
    table begins on the same page, each with its own title and header)."""
    blocks = []
    pos = from_idx
    while pos < len(page_lines):
        li, cols = find_header_and_columns(page_lines[pos:])
        if not cols:
            break
        last = pos + li
        first = last
        while (
            first - 1 >= pos
            and last - first < MAX_HEADER_LINES - 1
            and page_lines[first]["top"] - page_lines[first - 1]["top"] <= HEADER_LINE_MERGE_GAP
            and any(
                match_column_type(g["label"])
                for g in merge_header_tokens(page_lines[first - 1]["words"])
            )
        ):
            first -= 1
        blocks.append((first, last))
        pos = last + 1
    return blocks


def parse_page_set(all_lines_by_page, columns_override=None):
    """Find the header on whichever page has it, apply those column
    boundaries to every subsequent line across all pages (bank statements
    usually repeat the header per page, but we don't require that), merge
    continuation lines, and return the row list.

    columns_override: a column layout inferred WITHOUT a header (see
    infer_columns_without_header) — used instead of searching for a header,
    for scans that start mid-statement. In that mode there is no header to
    skip past or to bound the table, so each page starts at its first dated
    line and ends at the first line left of the date column (page footer)."""
    columns = columns_override
    headerless = columns_override is not None
    rows = []
    last_date_x = None  # x of the most recent row-start date (see Phase 1)
    tall_gap_limit = None  # widened footer guard for tall-row layouts (see below)

    for page_lines in all_lines_by_page:
        start_idx = 0
        if headerless:
            first = _first_line_start_idx(page_lines, columns)
            if first is None:
                continue
            start_idx = first
        elif columns is None:
            header_idx, cols = find_header_and_columns(page_lines)
            if cols:
                columns = cols
                start_idx = header_idx + 1
        else:
            # If this page repeats the header, skip past it — wherever it
            # actually falls, not just when it sits at line 0. Confirmed
            # real and necessary with an IDFC FIRST Bank statement: every
            # continuation page reprints ~4 lines of boilerplate
            # ("STATEMENT OF ACCOUNT" / customer ID / account no /
            # statement period) plus an opening-balance summary line BEFORE
            # the actual "Value Date / Particulars / Debit / Credit /
            # Balance" header — unlike the IDBI statement this repeated-
            # header check was originally written for, whose header sat
            # right at the top of every page. With the old `header_idx == 0`
            # check, that never matched here, so `start_idx` stayed 0 and
            # MAX_ROW_LINE_GAP's footer/gap guard (correctly meant to stop
            # at a page's trailing footer) instead tripped on the ~55pt gap
            # between the boilerplate and the balance-summary line — a gap
            # that occurs before any real data has even been reached — and
            # silently discarded the ENTIRE rest of every continuation page.
            # That's why a real 22-page/many-transaction IDFC statement came
            # back with only the 6 transactions that happened to be on page
            # 1 (the only page whose header sits at the very top of the
            # all-lines list, since there's no earlier boilerplate to skip
            # past yet). Skipping to wherever the header is actually found —
            # exactly like the very first page does — fixes this generally,
            # not just for this one statement's specific boilerplate length.
            header_idx, cols = find_header_and_columns(page_lines)
            if cols:
                start_idx = header_idx + 1

        if columns is None:
            continue

        date_left = columns[0]["x0"]

        # MAX_ROW_LINE_GAP assumes the lines of a table sit close together.
        # A statement laid out with tall, airy rows (a real IndusInd one puts
        # ~56pt between consecutive transactions, with the wrapped date
        # half-way between) would trip that footer guard after its very
        # first row and come out with 1-3 transactions. The guard is only
        # known to be wrong when a big gap is REPEATEDLY followed by a new
        # dated row — between two real rows — so only then (2+ such gaps on
        # a page, remembered for later pages, which may hold a single row)
        # is it scaled to the observed gaps. Ordinary statements, whose
        # lines are all closer than the guard, never trigger this and keep
        # the original limit exactly.
        tall_gaps = []
        _tops = [ln for ln in page_lines[start_idx:] if ln["words"]]
        for a, b in zip(_tops, _tops[1:]):
            if (
                b["top"] - a["top"] > MAX_ROW_LINE_GAP
                and DATE_RE.match(b["words"][0]["text"].strip())
                and abs(b["words"][0]["x0"] - date_left) <= 15
            ):
                tall_gaps.append(b["top"] - a["top"])
        if len(tall_gaps) >= 2:
            tall_gap_limit = max(tall_gap_limit or 0.0, 1.5 * max(tall_gaps))
        gap_limit = max(MAX_ROW_LINE_GAP, tall_gap_limit or 0.0)

        # ---- Phase 1: walk the page's table lines, classify each as the
        # start of a new row or a dateless (wrapped narration / balance) line.
        entries = []  # (top, cell, is_start)
        prev_top = None
        started_here = False
        header_blocks = {} if headerless else {
            f: l for f, l in _find_repeat_header_blocks(page_lines, start_idx)
        }
        skip_until = -1
        for idx in range(start_idx, len(page_lines)):
            line = page_lines[idx]
            if idx <= skip_until:
                continue
            if idx in header_blocks:
                # a second table starts here: the dateless lines since the
                # last row are its title/boilerplate, not narration
                while entries and not entries[-1][2]:
                    entries.pop()
                skip_until = header_blocks[idx]
                prev_top = None
                continue
            if PAGE_FOOTER_RE.match(_line_text(line)):
                break  # page-number footer: nothing after it is table
            if prev_top is not None and line["top"] - prev_top > gap_limit:
                break  # end of the transaction table for this page — the rest is footer/disclaimer text
            if (
                gap_limit > MAX_ROW_LINE_GAP
                and prev_top is not None
                and line["top"] - prev_top > MAX_ROW_LINE_GAP
                and not DATE_RE.match(line["words"][0]["text"].strip())
                and not any(REAL_AMOUNT_RE.match(w["text"].strip()) for w in line["words"])
            ):
                # In a tall-row layout the guard was widened (see gap_limit),
                # so a big gap alone no longer proves the table ended. But
                # only a new dated/amount line may follow one — a dateless,
                # amount-free line after a large gap is the page footer
                # (disclaimer text), never a wrapped narration line.
                break
            if headerless:
                # Words right of the last inferred column are slivers of a
                # column the scan cuts off (e.g. the first digit of a cropped
                # Closing Balance, at the page's very edge) or border debris —
                # they must not be forced into the nearest real column.
                kept = [
                    w for w in line["words"]
                    if w["x0"] <= columns[-1]["x1"] + 12 and not _is_border_noise(w["text"])
                ]
                if not kept:
                    continue
                line = {"top": line["top"], "words": kept}
                if kept[0]["x0"] < date_left - 3:
                    break  # left of the date column: page footer, not table
            prev_top = line["top"]
            cell = assign_line_to_columns(line["words"], columns)
            date_val = cell.get("date", "").strip()
            has_real_date = bool(DATE_PREFIX_RE.match(date_val))
            if headerless:
                # with no header to bound the table, a stray date inside
                # boilerplate/narration text must not start a row
                has_real_date = has_real_date and bool(
                    DATE_RE.match(line["words"][0]["text"].strip())
                )
            # Only debit/credit/amount define a NEW row on a dateless line —
            # deliberately excludes "balance". Confirmed real with an IDBI
            # statement whose running balance sometimes wraps onto its own
            # line with nothing else on it (the date+narration+debit/credit
            # line runs too long, so just the balance number spills onto a
            # line by itself). That line has no date and no debit/credit/
            # amount, but it DOES have a real amount (the balance) — with
            # balance counted here, this dateless-but-has-an-amount line
            # was wrongly treated as the start of a brand new transaction,
            # producing a spurious extra row with the previous transaction's
            # balance and none of its own narration, while the real row lost
            # its balance entirely. A genuine new transaction with no date on
            # its first line still always carries a debit/credit/amount;
            # balance-only is the signature of a wrapped balance instead.
            has_txn_amount = any(
                REAL_AMOUNT_RE.match(cell.get(k, "").strip())
                for k in ("debit", "credit", "amount")
            )
            if (
                has_real_date
                and not headerless
                and last_date_x is not None
                and line["words"][0]["x0"] - last_date_x > 20
                and not any(
                    cell.get(k, "").strip()
                    for k in ("narration", "ref", "debit", "credit", "amount", "balance")
                )
            ):
                # a lone date sitting well right of where row dates start is
                # a date INSIDE wrapped narration ("... to 30-12-2023"), not
                # a new transaction
                cell = {"narration": date_val}
                has_real_date = False
            is_start = has_real_date or (has_txn_amount and (rows or started_here))
            if is_start:
                started_here = True
                if has_real_date:
                    last_date_x = line["words"][0]["x0"]
            entries.append((line["top"], cell, is_start))

        # ---- Phase 2: build rows. Each dateless run between two row-starts
        # is split between the row above (trailing) and the row below
        # (leading) — see _split_dateless_run.
        page_start_rows = []  # row dicts created on this page, in order
        start_positions = [i for i, e in enumerate(entries) if e[2]]
        if not start_positions:
            continue

        def _new_row(cell):
            return {
                "date": cell.get("date", "").strip(),
                "narration": cell.get("narration", "").strip(),
                "ref": cell.get("ref", "").strip(),
                "debit": clean_amount(cell.get("debit", "")),
                "credit": clean_amount(cell.get("credit", "")),
                "amount": clean_amount(cell.get("amount", "")),
                "balance": clean_amount(cell.get("balance", "")),
            }

        def _append_text(row, cell):
            for k in ("narration", "ref"):
                extra = cell.get(k, "").strip()
                if extra:
                    row[k] = (row[k] + " " + extra).strip()

        def _prepend_text(row, cell):
            for k in ("narration", "ref"):
                extra = cell.get(k, "").strip()
                if extra:
                    row[k] = (extra + " " + row[k]).strip()

        def _fill_wrapped_balance(row, cell):
            if not row["balance"]:
                extra_balance = clean_amount(cell.get("balance", ""))
                if extra_balance:
                    row["balance"] = extra_balance

        page_rows = [_new_row(entries[p][1]) for p in start_positions]

        # lines BEFORE the page's first row-start: only a centre-aligned
        # layout puts real leading narration there, and it mirrors the row's
        # own trailing lines (same offsets above and below the dated line).
        # Anything that doesn't mirror is page header/boilerplate and is left
        # out — a dateless line before the first row used to be dropped
        # unconditionally, which stays the default.
        first_p = start_positions[0]
        first_top = entries[first_p][0]
        trail_end = start_positions[1] if len(start_positions) > 1 else len(entries)
        trailing_block = list(range(first_p + 1, trail_end))
        # trailing lines of the first row = the ones the run split gives it
        if len(start_positions) > 1:
            ktop = _split_dateless_run(
                first_top,
                [entries[i][0] for i in trailing_block],
                entries[start_positions[1]][0],
            )
            trailing_block = trailing_block[:ktop]
        mirrored = []
        before = list(range(first_p - 1, -1, -1))  # nearest first
        for n, bi in enumerate(before):
            if n < len(trailing_block):
                up = first_top - entries[bi][0]
                down = entries[trailing_block[n]][0] - first_top
                if abs(up - down) <= 1.5:
                    mirrored.append(bi)
                    continue
            break
        # prepend in reading order, so the farthest-from-the-row line ends up first
        lead_text_cells = [entries[bi][1] for bi in sorted(mirrored)]
        for c in reversed(lead_text_cells):
            _prepend_text(page_rows[0], c)

        # runs between consecutive row-starts, and after the last one
        for si, p in enumerate(start_positions):
            nxt = start_positions[si + 1] if si + 1 < len(start_positions) else None
            run_idx = list(range(p + 1, nxt if nxt is not None else len(entries)))
            if nxt is None:
                k = len(run_idx)  # nothing follows: all trail the last row
            else:
                k = _split_dateless_run(
                    entries[p][0], [entries[i][0] for i in run_idx], entries[nxt][0]
                )
            for i in run_idx[:k]:
                _append_text(page_rows[si], entries[i][1])
                _fill_wrapped_balance(page_rows[si], entries[i][1])
            lead = run_idx[k:]
            if nxt is not None:
                for i in reversed(lead):
                    _prepend_text(page_rows[si + 1], entries[i][1])

        rows.extend(page_rows)

    return columns, rows


# ---------------------------------------------------------------------------
# Header-less tables. A scan (or a page range cut out of a longer statement)
# can start MID-statement, with no column-header row anywhere in the file —
# confirmed real with a 43-page scanned HDFC statement that begins at its
# page 10. With nothing to anchor on, parse_page_set found no table and the
# freeform fallback turned the page text into 468 garbage rows. When the
# data itself has an unmistakable grid — dated lines carrying a second (value)
# date and right-aligned amounts in two or three fixed columns — the columns
# can be inferred from where those tokens sit instead.
# ---------------------------------------------------------------------------

def _cluster_by_gap(values, max_gap):
    """Split sorted numbers into runs where neighbours are <= max_gap apart."""
    clusters = []
    for v in sorted(values):
        if clusters and v - clusters[-1][-1] <= max_gap:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return clusters


def infer_columns_without_header(all_lines_by_page, min_date_lines=6):
    """Infer a column layout from the data alone, for statements with no
    header row. Returns the same column dicts find_header_and_columns would
    ([{type, x0, x1, label, primary}, ...] sorted left to right), or None
    when the layout isn't unmistakable enough to trust.

    Deliberately narrow: it only fires for the common Indian-bank shape
    "Date | Narration | [Ref] | Value Date | Debit | Credit | [Balance]",
    requiring a second date on most dated lines and 2-3 clean right-aligned
    amount columns. Debit is assumed to be the LEFTMOST amount column and
    Credit the next one — the near-universal order (and the order on the
    HDFC statement this was built from, where every row checks out: "NEFT
    DR-..." sits in the left amount column, "IB FUNDS TRANSFER CR-..." in
    the right one)."""
    date_lines = []
    for page_lines in all_lines_by_page:
        for line in page_lines:
            words = line["words"]
            if words and DATE_RE.match(words[0]["text"].strip()):
                date_lines.append(words)
    if len(date_lines) < min_date_lines:
        return None

    first = [w[0] for w in date_lines]
    date_x0 = min(w["x0"] for w in first)
    date_x1 = max(w["x1"] for w in first)

    seconds = []
    for words in date_lines:
        for w in words[1:]:
            if DATE_RE.match(w["text"].strip()):
                seconds.append(w)
                break
    if len(seconds) < 0.6 * len(date_lines):
        return None
    vd_x0 = min(w["x0"] for w in seconds)
    vd_x1 = max(w["x1"] for w in seconds)
    if vd_x0 <= date_x1:
        return None

    amounts = []
    for words in date_lines:
        for w in words[1:]:
            if w["x0"] > vd_x1 and REAL_AMOUNT_RE.match(w["text"].strip().replace(",", "")):
                amounts.append(w)
    if len(amounts) < min_date_lines:
        return None
    clusters = _cluster_by_gap([w["x1"] for w in amounts], 12.0)
    # fold a stray tiny cluster (OCR jitter on one wide number) into its neighbour
    minimum = max(3, int(0.05 * len(amounts)))
    merged = []
    for c in clusters:
        if merged and len(c) < minimum:
            merged[-1].extend(c)
        elif merged and len(merged[-1]) < minimum:
            merged[-1].extend(c)
        else:
            merged.append(list(c))
    if not 2 <= len(merged) <= 3:
        return None
    amount_types = ["debit", "credit", "balance"][: len(merged)]
    amount_cols = []
    for cl, t in zip(merged, amount_types):
        lo, hi = min(cl), max(cl)
        members = [w for w in amounts if lo <= w["x1"] <= hi]
        amount_cols.append({
            "type": t, "x0": min(w["x0"] for w in members), "x1": hi,
            "label": t, "primary": True,
        })
    # the columns must not overlap each other or the value date
    prev_x1 = vd_x1
    for c in amount_cols:
        if c["x0"] <= prev_x1:
            return None
        prev_x1 = c["x1"]

    # narration starts at the first non-date word after the date on dated lines
    nar_starts = []
    for words in date_lines:
        for w in words[1:]:
            if w["x0"] > date_x1:
                nar_starts.append(w["x0"])
                break
    if not nar_starts:
        return None
    nar_x0 = min(nar_starts)

    # optional ref column: long, digit-bearing, right-aligned tokens just left
    # of the value date (e.g. "0000000000380882", "HDFCH01091172798")
    refs = []
    for words in date_lines:
        for w in words[1:]:
            t = w["text"].strip()
            if (
                len(t) >= 10 and any(ch.isdigit() for ch in t)
                and w["x1"] < vd_x0 and w["x0"] > nar_x0 + 60
                and not DATE_RE.match(t)
            ):
                refs.append(w)
    ref_col = None
    if len(refs) >= 0.4 * len(date_lines):
        mode_bucket = max(
            {round(w["x1"] / 6) for w in refs},
            key=lambda b: sum(1 for w in refs if round(w["x1"] / 6) == b),
        )
        members = [w for w in refs if round(w["x1"] / 6) == mode_bucket]
        ref_col = {
            "type": "ref", "x0": min(w["x0"] for w in members),
            "x1": max(w["x1"] for w in members), "label": "ref", "primary": True,
        }

    nar_x1 = (ref_col["x0"] if ref_col else vd_x0) - 2
    if nar_x1 <= nar_x0:
        return None
    columns = [
        {"type": "date", "x0": date_x0, "x1": date_x1, "label": "date", "primary": True},
        {"type": "narration", "x0": nar_x0, "x1": nar_x1, "label": "narration", "primary": True},
    ]
    if ref_col:
        columns.append(ref_col)
    columns.append({"type": "date", "x0": vd_x0, "x1": vd_x1, "label": "value date", "primary": False})
    columns.extend(amount_cols)
    return sorted(columns, key=lambda c: c["x0"])


# ---------------------------------------------------------------------------
# Fallback for statements with no detectable table header at all — some
# banks' mobile-app "mini statement" / condensed e-statement export isn't a
# proper aligned table the way the netbanking portal's PDF usually is. Two
# shapes of this have come up so far:
#   1. One line per transaction, everything packed together, e.g.
#      "16-09-2026 UPI-JOHN TRADERS-Dr Rs.500.00 Bal Rs.98,060.00".
#   2. One small *block* per transaction spread across several short lines
#      instead — a date line, then a narration line, then an amount line,
#      then a balance line (a plausible shape for a phone-width app export;
#      not confirmed against a real sample yet, this is a reasoned
#      generalisation of shape 1's logic across lines rather than a
#      guessed-at brand new format).
# Either way this reads lines as free text rather than relying on column
# x-positions, so it doesn't need a header row to work from at all: a line
# starting with a date begins a new transaction, and every line up to the
# next date-starting line — whether that's the rest of the same line or a
# handful of lines after it — gets scanned for an amount/Dr-Cr-Bal signal
# and folded into that transaction's fields, with whatever text is left
# over folded into its narration.
# ---------------------------------------------------------------------------

FREEFORM_DATE_LINE_RE = re.compile(
    r"^(\d{1,2}[-/. ](?:[A-Za-z]{3,9}|\d{1,2})[-/. ]\d{2,4})\s*[:\-]?\s*(.*)$"
)
# capture just the numeric part so a "Rs." / "INR" / "₹" prefix never leaks
# into the extracted number (that was the bug in the first version of this)
FREEFORM_AMOUNT_RE = re.compile(r"(?:Rs\.?|INR|₹)?\s*([\d,]+\.\d{2})", re.IGNORECASE)
FREEFORM_DEBIT_RE = re.compile(r"\b(dr|debit(?:ed)?|withdrawal)\b", re.IGNORECASE)
FREEFORM_CREDIT_RE = re.compile(r"\b(cr|credit(?:ed)?|deposit)\b", re.IGNORECASE)
FREEFORM_BALANCE_LABEL_RE = re.compile(r"\bbal(?:ance)?\b", re.IGNORECASE)
# structural labels that just annotate the number next to them (Dr/Cr/Bal/...)
# rather than being part of the actual transaction description
FREEFORM_LABEL_STRIP_RE = re.compile(
    r"\b(bal(?:ance)?|dr|cr|debit(?:ed)?|credit(?:ed)?|withdrawal|deposit|rs\.?|inr)\b[:.]?",
    re.IGNORECASE,
)


def _apply_freeform_fragment(row, text):
    """Pull any amount(s) and a debit/credit/balance signal out of one
    line's worth of text and fold them into `row` (only filling a field
    that's still blank, so a later line never overwrites what an earlier
    line in the same block already found), then fold whatever text is left
    over into the row's narration. Used for both the line that starts a
    transaction (date + the rest of the row all on one line — the common
    case) and any further lines up to the next date-starting line (a
    block-style export spread across several short lines instead)."""
    found_amounts = [a.replace(",", "") for a in FREEFORM_AMOUNT_RE.findall(text)]
    is_debit = bool(FREEFORM_DEBIT_RE.search(text))
    is_credit = bool(FREEFORM_CREDIT_RE.search(text))
    is_balance_labeled = bool(FREEFORM_BALANCE_LABEL_RE.search(text))

    narration = FREEFORM_AMOUNT_RE.sub("", text)
    narration = FREEFORM_LABEL_STRIP_RE.sub("", narration)
    narration = re.sub(r"[\s\-:|]+", " ", narration).strip(" -:|")

    if found_amounts:
        if len(found_amounts) >= 2:
            # two amounts on one line/fragment: the last is almost always
            # the running balance, the one before it this transaction's amount
            balance, txn_amt = found_amounts[-1], found_amounts[-2]
        elif is_debit or is_credit:
            # an explicit Dr/Cr signal means a lone amount is the
            # transaction amount, not the balance
            balance, txn_amt = "", found_amounts[0]
        elif is_balance_labeled:
            # a lone amount with no Dr/Cr signal but an explicit "Bal"
            # label (its own line in a block-style export) is the balance
            balance, txn_amt = found_amounts[0], ""
        else:
            balance, txn_amt = "", found_amounts[0]

        if balance and not row["balance"]:
            row["balance"] = balance
        if txn_amt and not (row["debit"] or row["credit"] or row["amount"]):
            if is_debit and not is_credit:
                row["debit"] = txn_amt
            elif is_credit and not is_debit:
                row["credit"] = txn_amt
            else:
                row["amount"] = txn_amt

    if narration:
        row["narration"] = (row["narration"] + " " + narration).strip()


def parse_freeform_lines(all_lines_by_page):
    """Line-by-line (and block-of-lines) fallback: no table structure
    assumed. Returns a row list (possibly empty) — the caller decides
    whether it's good enough to use in place of the "no layout recognized"
    error."""
    rows = []
    for page_lines in all_lines_by_page:
        for line in page_lines:
            text = " ".join(w["text"] for w in line["words"]).strip()
            if not text:
                continue
            m = FREEFORM_DATE_LINE_RE.match(text)
            if not m:
                # no date at the start of this line — either a wrapped
                # continuation of the previous transaction (more narration
                # text, or its amount/balance on a line of its own in a
                # block-style export), or, if no transaction has been seen
                # yet, header/account-info text at the top of the page,
                # which is safely dropped since `rows` is still empty.
                if rows:
                    _apply_freeform_fragment(rows[-1], text)
                continue

            date_val, rest = m.group(1), m.group(2)
            row = {
                "date": date_val, "narration": "", "ref": "",
                "debit": "", "credit": "", "amount": "", "balance": "",
            }
            rows.append(row)
            if rest:
                _apply_freeform_fragment(row, rest)
    return rows


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def _rows_without_amount_warning(rows):
    """Warn when some rows came out with no debit/credit/amount at all —
    on a scanned statement OCR occasionally drops a short amount outright
    (confirmed real: "1.00" and "2.84" on a scanned HDFC statement), which
    would otherwise leave a silently incomplete transaction in the Excel."""
    n = sum(1 for r in rows if not (r["debit"] or r["credit"] or r["amount"]))
    if n == 0:
        return []
    return [
        f"{n} transaction row(s) have no amount — the scan may have been too "
        "faint to read; please check those rows against the original statement."
    ]


@app.get("/health")
def health():
    return {"ok": True}


def _process_pdf(contents, password, progress=None):
    """The whole extraction. Returns (http_status, json_body). Runs OUTSIDE
    the event loop (see the endpoints below): it is CPU-heavy and, for a
    scanned statement, takes minutes."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            pages_words = extract_pages_words(tmp_path, password, progress)
        except ValueError as e:
            code = str(e)
            status = 422 if code == "bad_password" else 400
            return status, {"error": code}

        if progress:
            progress("Finding the transactions")
        lines_by_page = [group_into_lines(pw) for pw in pages_words]
        columns, rows = parse_page_set(lines_by_page)

        if columns is not None:
            detected_columns = [c["type"] for c in columns]
            result = {
                "columns": detected_columns,
                "row_count": len(rows),
                "rows": rows,
            }
            warnings = _rows_without_amount_warning(rows)
            if warnings:
                result["warnings"] = warnings
            return 200, result

        # No header row anywhere (e.g. a scan that starts mid-statement):
        # try to infer the column grid from the data itself.
        inferred = infer_columns_without_header(lines_by_page)
        if inferred is not None:
            columns, rows = parse_page_set(lines_by_page, columns_override=inferred)
            if len(rows) >= 2:
                warnings = [
                    "No column-heading row was found in this file, so the "
                    "columns were inferred from the layout (Debit = left "
                    "amount column, Credit = right amount column)."
                ]
                if "balance" not in [c["type"] for c in columns]:
                    warnings.append("No Balance column was found in this file.")
                warnings += _rows_without_amount_warning(rows)
                return 200, {
                    "columns": [c["type"] for c in columns],
                    "row_count": len(rows),
                    "rows": rows,
                    "warnings": warnings,
                }

        # No aligned table header found — some mobile-app-exported "mini
        # statements" list one transaction per line instead of a proper
        # table (no consistent column positions to key off at all). Try
        # that before giving up.
        freeform_rows = parse_freeform_lines(lines_by_page)
        if len(freeform_rows) >= 2:
            present_fields = [
                f for f in ("debit", "credit", "amount", "balance")
                if any(r[f] for r in freeform_rows)
            ]
            detected_columns = ["date", "narration"] + present_fields
            return 200, {
                "columns": detected_columns,
                "row_count": len(freeform_rows),
                "rows": freeform_rows,
            }

        return 422, {
            "error": "layout_not_recognized",
            "message": (
                "Couldn't find a Date / Narration / Amount header on this "
                "statement. It may be a bank format we don't support yet."
            ),
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        # contents only ever lived in memory + this one temp file, both gone now


def _validate_upload(file, contents):
    if len(contents) > MAX_FILE_BYTES:
        return JSONResponse(status_code=413, content={"error": "file_too_large"})
    if file.content_type not in ("application/pdf", "application/octet-stream") and not (
        file.filename or ""
    ).lower().endswith(".pdf"):
        return JSONResponse(status_code=400, content={"error": "not_a_pdf"})
    return None


@app.post("/extract")
@limiter.limit("10/minute")
async def extract(
    request: Request,  # required first positional arg for slowapi's limiter
    file: UploadFile = File(...),
    password: Optional[str] = Form(default=None),
):
    """Synchronous endpoint (kept for compatibility). The work runs in a
    worker thread so the server can still answer /health meanwhile."""
    contents = await file.read()
    bad = _validate_upload(file, contents)
    if bad:
        return bad
    status, body = await run_in_threadpool(_process_pdf, contents, password)
    return JSONResponse(status_code=status, content=body)


# ---------------------------------------------------------------------------
# Background jobs. A scanned statement can take minutes, far longer than a
# browser/proxy keeps one HTTP request open ("Network error" with no result,
# confirmed real on a 10-page scan). So the page starts a job, gets an id back
# at once, then polls for progress/result. Jobs live in memory only (nothing is
# written to disk) and are dropped JOB_TTL_SECONDS after they finish.
# ---------------------------------------------------------------------------

JOB_TTL_SECONDS = 30 * 60
_jobs = {}
_jobs_lock = threading.Lock()


def _purge_old_jobs():
    now = time.time()
    with _jobs_lock:
        for jid in [j for j, v in _jobs.items() if now - v["updated"] > JOB_TTL_SECONDS]:
            del _jobs[jid]


def _run_job(job_id, contents, password):
    def progress(msg):
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["progress"] = msg
                _jobs[job_id]["updated"] = time.time()

    try:
        status, body = _process_pdf(contents, password, progress)
    except Exception:  # never leave a job "processing" forever
        status, body = 500, {"error": "processing_failed"}
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(
                state="done" if status == 200 else "error",
                http_status=status, body=body, updated=time.time(),
            )


@app.post("/jobs")
@limiter.limit("10/minute")
async def start_job(
    request: Request,
    file: UploadFile = File(...),
    password: Optional[str] = Form(default=None),
):
    contents = await file.read()
    bad = _validate_upload(file, contents)
    if bad:
        return bad
    _purge_old_jobs()
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "state": "processing", "progress": "Starting", "updated": time.time(),
            "started": time.time(),
        }
    threading.Thread(target=_run_job, args=(job_id, contents, password), daemon=True).start()
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
@limiter.limit("120/minute")
async def job_status(request: Request, job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "job_not_found"})
        if job["state"] == "processing":
            return {
                "state": "processing", "progress": job["progress"],
                "elapsed": int(time.time() - job["started"]),
            }
        status, body = job["http_status"], job["body"]
    # the finished result stays available until JOB_TTL_SECONDS has passed
    if status == 200:
        return {"state": "done", "result": body}
    return {"state": "error", "error": body.get("error", "processing_failed"), "message": body.get("message", "")}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
