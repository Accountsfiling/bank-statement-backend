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
from typing import Optional

import pdfplumber
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

MAX_FILE_BYTES = int(os.environ.get("MAX_FILE_MB", "20")) * 1024 * 1024
MAX_PAGES = int(os.environ.get("MAX_PAGES", "60"))
OCR_DPI = int(os.environ.get("OCR_DPI", "300"))


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
    r"^\d{1,2}[-/. ]([A-Za-z]{3,9}|\d{1,2})[-/. ]\d{2,4}$"
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
    r"^\d{1,2}[-/. ]([A-Za-z]{3,9}|\d{1,2})[-/. ]\d{2,4}"
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

def extract_pages_words(pdf_path: str, password: Optional[str]):
    """Return a list of pages, each a list of word dicts:
    {text, x0, x1, top, bottom}. Falls back to OCR for pages with no
    extractable text layer (scanned/photographed statements).
    Raises ValueError("bad_password") / ValueError("too_many_pages") as needed.
    """
    pages_words = []
    needs_ocr_pages = []

    try:
        with pdfplumber.open(pdf_path, password=password or "") as pdf:
            if len(pdf.pages) > MAX_PAGES:
                raise ValueError("too_many_pages")
            for i, page in enumerate(pdf.pages):
                words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
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
        images = convert_from_path(pdf_path, dpi=OCR_DPI, userpw=password or None)
        for i in needs_ocr_pages:
            if i >= len(images):
                pages_words[i] = []
                continue
            data = pytesseract.image_to_data(
                images[i], output_type=pytesseract.Output.DICT
            )
            scale = 72.0 / OCR_DPI  # convert pixel coords back to PDF-point scale
            words = []
            for j, text in enumerate(data["text"]):
                text = text.strip()
                if not text:
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
            pages_words[i] = words

    return pages_words


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


def _next_date_top(page_lines, after_idx, columns, max_lookahead=4):
    """Scan forward from after_idx (exclusive) up to max_lookahead lines for
    the next one whose date cell is a real date, and return its `top`
    (None if none found within the window). Used only to decide whether a
    dateless line is LEADING narration for the row about to start, rather
    than TRAILING continuation of the row before it — see the leading-vs-
    trailing note in parse_page_set."""
    end = min(after_idx + 1 + max_lookahead, len(page_lines))
    for j in range(after_idx + 1, end):
        cand = assign_line_to_columns(page_lines[j]["words"], columns)
        if DATE_PREFIX_RE.match(cand.get("date", "").strip()):
            return page_lines[j]["top"]
    return None


def parse_page_set(all_lines_by_page):
    """Find the header on whichever page has it, apply those column
    boundaries to every subsequent line across all pages (bank statements
    usually repeat the header per page, but we don't require that), merge
    continuation lines, and return the row list."""
    columns = None
    rows = []
    pending_narration_extra = []

    for page_lines in all_lines_by_page:
        start_idx = 0
        if columns is None:
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

        prev_top = None
        # Whether a row has actually been added FROM THIS PAGE yet. A
        # dateless line can only be a continuation of a row that started on
        # THIS page — never one carried over from the previous page.
        # Confirmed real and necessary: this statement's closing page is
        # pure "Statement Summary" / legends / disclaimer text with no
        # transactions and no repeated header at all, so every one of its
        # lines is dateless — without this guard, all of it silently
        # attached itself to the last real transaction from the page
        # before, producing one transaction row with several paragraphs of
        # legal boilerplate stuffed into its narration.
        page_added_row = False
        # top of the last row-starting date line seen on THIS page — used
        # below to tell a TRAILING continuation line from a LEADING one.
        last_date_top = None
        # Narration/ref text pulled off a dateless line that sits closer to
        # the NEXT row than to the one before it — held here until that next
        # row actually gets created, then prepended to it. Confirmed real
        # and necessary with a Bank of Baroda statement: unlike IDBI/IDFC
        # (where a wrapped narration line only ever continues AFTER its
        # row's date+amount line), Bank of Baroda can split a transaction's
        # narration as a line BEFORE its own date+amount line too, so the
        # old "every dateless line continues whatever row is currently open"
        # assumption silently stitched that leading line onto the PREVIOUS
        # transaction instead of the one it actually belongs to.
        pending_leading = {"narration": "", "ref": ""}

        for idx in range(start_idx, len(page_lines)):
            line = page_lines[idx]
            if prev_top is not None and line["top"] - prev_top > MAX_ROW_LINE_GAP:
                break  # end of the transaction table for this page — the rest is footer/disclaimer text
            prev_top = line["top"]
            cell = assign_line_to_columns(line["words"], columns)
            date_val = cell.get("date", "").strip()
            has_real_date = bool(DATE_PREFIX_RE.match(date_val))
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
            if has_real_date or (has_txn_amount and rows):
                row = {
                    "date": date_val,
                    "narration": cell.get("narration", "").strip(),
                    "ref": cell.get("ref", "").strip(),
                    "debit": clean_amount(cell.get("debit", "")),
                    "credit": clean_amount(cell.get("credit", "")),
                    "amount": clean_amount(cell.get("amount", "")),
                    "balance": clean_amount(cell.get("balance", "")),
                }
                if pending_leading["narration"] or pending_leading["ref"]:
                    for k in ("narration", "ref"):
                        lead = pending_leading[k].strip()
                        if lead:
                            row[k] = (lead + " " + row[k]).strip()
                    pending_leading = {"narration": "", "ref": ""}
                rows.append(row)
                page_added_row = True
                last_date_top = line["top"]
            else:
                # A dateless line could be a TRAILING continuation of the
                # row before it (wrapped narration/ref text, and/or a
                # wrapped balance value — see above), or — confirmed real
                # with Bank of Baroda — LEADING narration for the row about
                # to start. Tell them apart by which neighbouring date line
                # this one actually sits closer to: IDBI/IDFC's genuine
                # continuation lines always sit much closer to the
                # preceding date line than to the next transaction's, so
                # this preserves their existing behaviour unchanged, while
                # correctly reclassifying Bank of Baroda's leading lines
                # (which sit closer to the date line still ahead of them).
                dist_back = (
                    (line["top"] - last_date_top) if last_date_top is not None else None
                )
                next_top = _next_date_top(page_lines, idx, columns)
                dist_fwd = (next_top - line["top"]) if next_top is not None else None
                is_leading = dist_fwd is not None and (
                    dist_back is None or dist_fwd < dist_back
                )
                if is_leading:
                    for k in ("narration", "ref"):
                        extra = cell.get(k, "").strip()
                        if extra:
                            pending_leading[k] = (pending_leading[k] + " " + extra).strip()
                elif rows and page_added_row:
                    # trailing continuation of the previous row — only when
                    # that previous row actually started on this same page.
                    for k in ("narration", "ref"):
                        extra = cell.get(k, "").strip()
                        if extra:
                            rows[-1][k] = (rows[-1][k] + " " + extra).strip()
                    if not rows[-1]["balance"]:
                        extra_balance = clean_amount(cell.get("balance", ""))
                        if extra_balance:
                            rows[-1]["balance"] = extra_balance

    return columns, rows


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

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/extract")
@limiter.limit("10/minute")
async def extract(
    request: Request,  # required first positional arg for slowapi's limiter
    file: UploadFile = File(...),
    password: Optional[str] = Form(default=None),
):
    contents = await file.read()
    if len(contents) > MAX_FILE_BYTES:
        return JSONResponse(status_code=413, content={"error": "file_too_large"})
    if file.content_type not in ("application/pdf", "application/octet-stream") and not (
        file.filename or ""
    ).lower().endswith(".pdf"):
        return JSONResponse(status_code=400, content={"error": "not_a_pdf"})

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            pages_words = extract_pages_words(tmp_path, password)
        except ValueError as e:
            code = str(e)
            status = 422 if code == "bad_password" else 400
            return JSONResponse(status_code=status, content={"error": code})

        lines_by_page = [group_into_lines(pw) for pw in pages_words]
        columns, rows = parse_page_set(lines_by_page)

        if columns is not None:
            detected_columns = [c["type"] for c in columns]
            return {
                "columns": detected_columns,
                "row_count": len(rows),
                "rows": rows,
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
            return {
                "columns": detected_columns,
                "row_count": len(freeform_rows),
                "rows": freeform_rows,
            }

        return JSONResponse(
            status_code=422,
            content={
                "error": "layout_not_recognized",
                "message": (
                    "Couldn't find a Date / Narration / Amount header on this "
                    "statement. It may be a bank format we don't support yet."
                ),
            },
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        # contents only ever lived in memory + this one temp file, both gone now


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
