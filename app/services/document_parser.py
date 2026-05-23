"""Document parsing and clause extraction.

Supports PDF (pypdf), DOCX (python-docx), and raw text.
Output: list of Clause objects with heading_path preserved.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass


HEADING_NUM_PATTERN = re.compile(r"^\s*((?:\d+\.){1,4}\d*)\s+(.+?)$")
HEADING_LETTER_PATTERN = re.compile(r"^\s*\(([a-z])\)\s+(.+)$", re.IGNORECASE)
BULLET_PATTERN = re.compile(r"^\s*[•·\-\*]\s+(.+)$")
ALL_CAPS_HEADING = re.compile(r"^[A-Z][A-Z0-9 ,&\-]{4,}$")

MIN_CLAUSE_CHARS = 30
MAX_CLAUSE_CHARS = 1500


@dataclass
class ParsedClause:
    position: int
    heading_path: str
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)


class ScannedPDFError(ValueError):
    """Raised when a PDF appears to be a scan with no extractable text layer."""


def parse_pdf(data: bytes) -> str:
    """Extract text from PDF preserving paragraph structure.

    Scanned PDFs (image-only, no text layer) yield empty or near-empty extractions.
    Per the spec, OCR is out of scope for the MVP — raise ScannedPDFError so the
    pipeline can surface a clear "paste the text instead" message to the user.
    """
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    pages = []
    page_count = 0
    for page in reader.pages:
        page_count += 1
        pages.append(page.extract_text() or "")
    text = "\n\n".join(pages)
    # Heuristic: a 10+ page PDF with under 50 characters of extractable text is
    # almost certainly an image-only scan.
    stripped_len = len(text.strip())
    if page_count > 0 and stripped_len < max(50, page_count * 20):
        if stripped_len < 50:
            raise ScannedPDFError(
                "This PDF appears to be a scanned image with no embedded text "
                "(OCR is not supported in the MVP). Please paste the text "
                "into a .txt file, export from Word as .docx, or upload a "
                "text-based PDF."
            )
    return text


def parse_docx(data: bytes) -> str:
    """Extract text from DOCX preserving paragraph structure and heading style names."""
    from docx import Document  # python-docx
    doc = Document(io.BytesIO(data))
    out = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        # Mark heading-styled paragraphs so the clause extractor can detect them
        if para.style and "Heading" in (para.style.name or ""):
            out.append(f"## {text}")
        else:
            out.append(text)
    return "\n\n".join(out)


def parse_document(filename: str, data: bytes, content_type: str | None = None) -> str:
    """Dispatch to the right parser based on filename/content_type."""
    name = filename.lower()
    if name.endswith(".pdf") or (content_type and "pdf" in content_type):
        return parse_pdf(data)
    if name.endswith(".docx") or (content_type and "wordprocessingml" in content_type):
        return parse_docx(data)
    # raw text
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="ignore")
    return str(data)


def extract_clauses(raw_text: str) -> list[ParsedClause]:
    """Split raw text into clauses with heading breadcrumbs.

    Strategy:
    - Detect heading lines (numbered, lettered, all-caps, or ## prefix from docx).
    - Maintain a stack of headings to build the breadcrumb.
    - Each non-heading paragraph becomes a clause (or split further on bullets/sub-numbering).
    - Reject clauses shorter than MIN_CLAUSE_CHARS.
    - Truncate clauses longer than MAX_CLAUSE_CHARS at sentence boundaries.
    """
    # Normalize whitespace
    lines = [ln.strip() for ln in raw_text.replace("\r\n", "\n").split("\n")]

    # Group into paragraphs separated by blank lines
    paragraphs: list[str] = []
    buf: list[str] = []
    for ln in lines:
        if not ln:
            if buf:
                paragraphs.append(" ".join(buf))
                buf = []
        else:
            buf.append(ln)
    if buf:
        paragraphs.append(" ".join(buf))

    heading_stack: list[tuple[int, str]] = []  # (level, text)
    clauses: list[ParsedClause] = []
    position = 0

    for para in paragraphs:
        p = para.strip()
        if not p:
            continue

        # Detect heading
        heading_match = None
        level = None

        if p.startswith("## "):
            heading_match = p[3:].strip()
            level = 1
        else:
            m = HEADING_NUM_PATTERN.match(p)
            if m:
                num = m.group(1).rstrip(".")
                title = m.group(2).strip()
                # If the rest after the heading is short and looks title-like, treat as heading
                if len(title) < 100 and not title.endswith(".") and len(p) < 150:
                    heading_match = f"{num} {title}"
                    level = num.count(".") + 1
            elif ALL_CAPS_HEADING.match(p) and len(p) < 80:
                heading_match = p
                level = 1

        if heading_match and level is not None:
            # Pop stack to current level
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_match))
            continue

        # Otherwise, treat as clause(s)
        breadcrumb = " → ".join(t for _, t in heading_stack) if heading_stack else "(no heading)"

        sub_clauses = _split_into_subclauses(p)
        for sub in sub_clauses:
            text = sub.strip()
            if len(text) < MIN_CLAUSE_CHARS:
                continue
            if len(text) > MAX_CLAUSE_CHARS:
                text = _truncate_at_sentence(text, MAX_CLAUSE_CHARS)
            position += 1
            clauses.append(ParsedClause(position=position, heading_path=breadcrumb, text=text))

    return clauses


def _split_into_subclauses(paragraph: str) -> list[str]:
    """Within a paragraph, split on lettered subsections (a) (b) (c) or bullet markers."""
    # If the paragraph contains lettered subsections embedded inline, split them
    if re.search(r"\([a-z]\)\s", paragraph):
        parts = re.split(r"(?=\s\([a-z]\)\s)", paragraph)
        return [p.strip(" ;") for p in parts if p.strip()]
    # Inline bullet markers
    if re.search(r"\s•\s", paragraph) and len(paragraph.split("•")) > 2:
        return [p.strip(" ;•-") for p in paragraph.split("•") if p.strip()]
    return [paragraph]


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    # Try to break at sentence boundary near max_chars
    snippet = text[:max_chars]
    last_period = max(snippet.rfind(". "), snippet.rfind("? "), snippet.rfind("! "))
    if last_period > max_chars * 0.5:
        return snippet[: last_period + 1]
    return snippet
