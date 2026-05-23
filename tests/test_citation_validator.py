"""Citation validator tests."""
from app.services import citation_validator


def test_valid_eu_ai_act_citation():
    assert citation_validator.is_valid_citation("eu_ai_act", "Article 9")
    assert citation_validator.is_valid_citation("eu_ai_act", "Article 14")
    assert citation_validator.is_valid_citation("eu_ai_act", "Annex IV")


def test_hallucinated_citation_dropped():
    # These don't exist in our indexed subset
    assert not citation_validator.is_valid_citation("eu_ai_act", "Article 999")
    assert not citation_validator.is_valid_citation("eu_ai_act", "Article 7(b)(iv)")


def test_validate_mappings_mixed():
    result = citation_validator.validate_mappings(
        "eu_ai_act",
        ["Article 9", "Article 999", "Article 14", "Made-Up Article"],
    )
    assert result.valid_count == 2
    assert set(result.dropped_article_ids) == {"Article 999", "Made-Up Article"}


def test_validate_mappings_empty():
    result = citation_validator.validate_mappings("eu_ai_act", [])
    assert result.valid_count == 0
    assert result.dropped_article_ids == []


def test_validate_unknown_framework():
    result = citation_validator.validate_mappings("nonexistent_framework", ["Article 1"])
    assert result.valid_count == 0
