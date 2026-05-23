"""Clause mapper integration tests."""
from app.services.clause_mapper import map_clauses


def test_map_compliant_clause():
    """A clause clearly addressing risk management should map to Article 9 as Compliant or Partial."""
    clauses = [(
        "Risk Management",
        "Acme maintains a documented continuous risk management process for high-risk AI systems. "
        "The process identifies and evaluates foreseeable risks to health, safety, and fundamental "
        "rights, and is reviewed and updated regularly throughout the system lifecycle.",
    )]
    results = map_clauses(clauses, framework_id="eu_ai_act")
    assert len(results) == 1
    clause_results = results[0]
    assert clause_results, "Expected at least one mapping result"

    # The top-confidence mapping should reference a real article
    top = clause_results[0]
    assert top.article_id.startswith("Article") or top.article_id == "Annex IV"
    # It should land on something related to risk
    assert "Article 9" in [r.article_id for r in clause_results[:3]], \
        f"Expected Article 9 in top 3: {[r.article_id for r in clause_results[:3]]}"


def test_map_human_oversight_clause():
    """A clause about human oversight should map to Article 14."""
    clauses = [(
        "Human Oversight",
        "Every high-risk AI feature has a designated human reviewer with the authority to override, "
        "disable, or roll back the automated decision. Reviewers receive annual training on the system "
        "capabilities and limitations and can interrupt the system at any time.",
    )]
    results = map_clauses(clauses, framework_id="eu_ai_act")
    assert results[0], "Expected mappings"
    article_ids = [r.article_id for r in results[0][:3]]
    assert "Article 14" in article_ids, f"Expected Article 14 in top 3: {article_ids}"


def test_map_off_topic_clause():
    """A clause about shipping/delivery should map as NotApplicable with low scores."""
    clauses = [(
        "Shipping",
        "For customers ordering hardware, shipping is fulfilled within 5 business days. "
        "Tracking information is provided via email and SMS notifications.",
    )]
    results = map_clauses(clauses, framework_id="eu_ai_act")
    if results[0]:
        # All mappings should be NotApplicable
        classifications = [r.classification for r in results[0]]
        assert all(c == "NotApplicable" for c in classifications), \
            f"Expected all NotApplicable, got: {classifications}"


def test_map_returns_valid_citations_only():
    """Every cited article_id in the results must exist in the indexed corpus."""
    from app.services import citation_validator
    clauses = [(
        "Data Governance",
        "Training data is reviewed for quality, representativeness, and bias before model training. "
        "Documentation of the review is approved by the Head of Data.",
    )]
    results = map_clauses(clauses, framework_id="eu_ai_act")
    for r in results[0]:
        assert citation_validator.is_valid_citation("eu_ai_act", r.article_id), \
            f"Mapper returned invalid citation: {r.article_id}"


def test_map_empty_clauses():
    assert map_clauses([], framework_id="eu_ai_act") == []


def test_confidence_in_valid_range():
    clauses = [(
        "Logs",
        "The system automatically generates event logs covering model inferences, decision overrides, "
        "and system errors. Logs are retained for a minimum of 12 months.",
    )]
    results = map_clauses(clauses, framework_id="eu_ai_act")
    for r in results[0]:
        assert 0.0 <= r.confidence <= 100.0
        assert 0.0 <= r.llm_self_score <= 100.0
