"""Confidence scorer tests."""
import pytest

from app.services.confidence_scorer import (
    fuse_confidence,
    normalize_min_max,
    primary_status_for_clause,
)


def test_normalize_min_max_basic():
    assert normalize_min_max([1, 2, 3]) == pytest.approx([0.0, 0.5, 1.0])


def test_normalize_min_max_all_equal():
    assert normalize_min_max([5, 5, 5]) == [0.5, 0.5, 0.5]


def test_normalize_empty():
    assert normalize_min_max([]) == []


def test_fuse_confidence_high():
    conf = fuse_confidence(1.0, 1.0, 90.0)
    # 0.3 + 0.3 + 0.36 = 0.96 → 96
    assert 95.0 <= conf <= 96.5


def test_fuse_confidence_low():
    conf = fuse_confidence(0.0, 0.0, 10.0)
    # 0 + 0 + 0.04 = 0.04 → 4
    assert 0.0 <= conf <= 5.0


def test_fuse_confidence_capped_for_invalid_citation():
    conf = fuse_confidence(1.0, 1.0, 95.0, citation_validated=False)
    assert conf <= 40.0


def test_fuse_confidence_clamp_range():
    # Even with extreme inputs, output is clamped to 0-100
    conf = fuse_confidence(2.0, 2.0, 200.0)
    assert 0.0 <= conf <= 100.0


def test_primary_status_compliant_wins():
    status, _ = primary_status_for_clause([
        ("Compliant", 75.0),
        ("Partial", 80.0),
        ("Gap", 60.0),
    ])
    assert status == "Compliant"


def test_primary_status_partial_when_no_strong_compliant():
    status, _ = primary_status_for_clause([
        ("Compliant", 40.0),  # below 60 threshold
        ("Partial", 70.0),
        ("Gap", 30.0),
    ])
    assert status == "Partial"


def test_primary_status_gap_fallback():
    status, _ = primary_status_for_clause([
        ("Gap", 55.0),
        ("NotApplicable", 80.0),
    ])
    assert status == "Gap"


def test_primary_status_not_applicable_if_no_signal():
    status, _ = primary_status_for_clause([
        ("Compliant", 30.0),
        ("Partial", 20.0),
        ("Gap", 15.0),
    ])
    assert status == "NotApplicable"


def test_primary_status_empty():
    status, conf = primary_status_for_clause([])
    assert status == "NotApplicable"
    assert conf == 0.0
