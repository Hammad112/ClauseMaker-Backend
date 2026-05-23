"""Document parser unit tests."""
from app.services.document_parser import extract_clauses, parse_document


SAMPLE = """1. RISK MANAGEMENT

1.1 Risk identification

Acme maintains a documented risk management process for all AI systems classified as high-risk.

1.2 Risk evaluation

For each identified risk, we evaluate severity and likelihood. Risks are documented in our central risk register.

2. DATA GOVERNANCE

2.1 Training data

All training datasets used in AI model development undergo a quality review.
"""


def test_extract_clauses_basic():
    clauses = extract_clauses(SAMPLE)
    assert len(clauses) >= 3, f"Expected at least 3 clauses, got {len(clauses)}"
    # Each clause should have a heading path
    for c in clauses:
        assert c.heading_path, f"Clause {c.position} missing heading_path"
        assert c.text, f"Clause {c.position} missing text"
        assert c.char_count == len(c.text)


def test_extract_clauses_heading_path():
    clauses = extract_clauses(SAMPLE)
    # The first content clause should be under "1. RISK MANAGEMENT → 1.1 Risk identification"
    breadcrumbs = [c.heading_path for c in clauses]
    assert any("RISK MANAGEMENT" in b for b in breadcrumbs), f"Expected RISK MANAGEMENT in {breadcrumbs}"
    assert any("DATA GOVERNANCE" in b for b in breadcrumbs), f"Expected DATA GOVERNANCE in {breadcrumbs}"


def test_extract_clauses_filters_short():
    text = "a\n\nb\n\nThis clause has enough characters to pass the minimum length filter."
    clauses = extract_clauses(text)
    # Only the longer one survives
    assert len(clauses) == 1


def test_parse_document_plain_text():
    data = SAMPLE.encode("utf-8")
    text = parse_document("policy.txt", data, "text/plain")
    assert "RISK MANAGEMENT" in text
