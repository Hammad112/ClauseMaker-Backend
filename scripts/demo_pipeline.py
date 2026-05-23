"""Demo script: run the full pipeline on the sample policy and print the mapping output.

Run with: python scripts/demo_pipeline.py
"""
import asyncio
import os
import sys
from pathlib import Path

# Force test mode
os.environ.setdefault("APP_MODE", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./demo_clausemark.db")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.db import reset_db  # noqa: E402
from app.services.document_parser import extract_clauses, parse_document  # noqa: E402
from app.services.clause_mapper import map_clauses  # noqa: E402
from app.services.confidence_scorer import primary_status_for_clause  # noqa: E402
from app.services.framework_loader import ensure_frameworks_loaded  # noqa: E402
from app.services import citation_validator  # noqa: E402


COLORS = {
    "Compliant": "\033[92m",  # green
    "Partial":   "\033[93m",  # yellow
    "Gap":       "\033[91m",  # red
    "NotApplicable": "\033[90m",  # grey
}
RESET = "\033[0m"


async def main():
    await reset_db()
    counts = ensure_frameworks_loaded()
    citation_validator.refresh_cache()
    print(f"\n📚 Loaded frameworks: {counts}\n")

    # Parse the sample policy
    policy_path = Path(__file__).resolve().parent.parent / "data" / "sample_acme_policy.txt"
    raw = policy_path.read_bytes()
    text = parse_document(policy_path.name, raw)
    clauses = extract_clauses(text)
    print(f"📄 Extracted {len(clauses)} clauses from {policy_path.name}\n")

    # Map all clauses
    pairs = [(c.heading_path, c.text) for c in clauses]
    results = map_clauses(pairs, framework_id="eu_ai_act")

    # Summarize and print
    by_status = {"Compliant": 0, "Partial": 0, "Gap": 0, "NotApplicable": 0}
    print("=" * 80)
    print("MAPPING RESULTS — EU AI Act")
    print("=" * 80)

    for clause, clause_results in zip(clauses, results):
        primary, conf = primary_status_for_clause(
            [(r.classification, r.confidence) for r in clause_results]
        )
        by_status[primary] += 1
        color = COLORS.get(primary, "")
        print(f"\n{color}● {primary:14s}{RESET} confidence={conf:5.1f}  "
              f"clause #{clause.position}: {clause.heading_path}")
        snippet = clause.text[:120].replace("\n", " ")
        print(f"  \"{snippet}{'...' if len(clause.text) > 120 else ''}\"")
        # Top 2 mappings
        for r in clause_results[:2]:
            tag_color = COLORS.get(r.classification, "")
            print(f"    → {r.article_id} ({r.article_title})  "
                  f"{tag_color}[{r.classification}]{RESET}  conf={r.confidence:.1f}")
            if r.gap_remediation:
                print(f"      ⚠ Remediation: {r.gap_remediation[:140]}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total = len(clauses)
    for k, v in by_status.items():
        pct = (v / total * 100) if total else 0
        c = COLORS.get(k, "")
        print(f"  {c}{k:14s}{RESET} {v:2d}  ({pct:5.1f}%)")
    print(f"  {'Total':14s} {total:2d}")

    # Verify no hallucinated citations
    all_cited = {r.article_id for clause_results in results for r in clause_results}
    invalid = [a for a in all_cited if not citation_validator.is_valid_citation("eu_ai_act", a)]
    if invalid:
        print(f"\n❌ HALLUCINATED CITATIONS DETECTED: {invalid}")
    else:
        print(f"\n✅ All {len(all_cited)} cited Articles validated against indexed corpus")

    print()


if __name__ == "__main__":
    asyncio.run(main())
