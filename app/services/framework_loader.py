"""Framework loader.

Loads regulatory frameworks into Qdrant. In production this would scrape EUR-Lex
and other sources. For the MVP / demo we ship a curated subset of the EU AI Act
as a JSON file in app/data/, which is enough for realistic demo mappings.

The same interface supports ingesting from external sources later — just point
at a different loader function.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from app.core.embeddings import get_embedder
from app.core.vector_store import get_vector_store


@dataclass
class FrameworkSpec:
    id: str
    name: str
    description: str
    source_url: str


FRAMEWORKS: dict[str, FrameworkSpec] = {
    "eu_ai_act": FrameworkSpec(
        id="eu_ai_act",
        name="EU AI Act",
        description="Regulation (EU) 2024/1689 on harmonised rules on artificial intelligence",
        source_url="https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689",
    ),
    "gdpr": FrameworkSpec(
        id="gdpr",
        name="GDPR",
        description="General Data Protection Regulation 2016/679",
        source_url="https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32016R0679",
    ),
}


# ---- Curated EU AI Act Article subset ----
# These are paraphrased summaries of the actual Articles. In production the
# ingestion script fetches verbatim text from EUR-Lex.

EU_AI_ACT_ARTICLES = [
    {
        "article_id": "Article 9",
        "article_title": "Risk management system",
        "chapter": "Chapter III — High-risk AI systems",
        "text": (
            "A risk management system shall be established, implemented, documented and maintained for "
            "high-risk AI systems. It must run as a continuous iterative process throughout the entire "
            "lifecycle of the system, requiring regular systematic review and update. The system shall "
            "identify and analyse known and foreseeable risks to health, safety and fundamental rights, "
            "estimate and evaluate risks that may emerge under reasonably foreseeable misuse, and adopt "
            "appropriate risk management measures."
        ),
    },
    {
        "article_id": "Article 10",
        "article_title": "Data and data governance",
        "chapter": "Chapter III — High-risk AI systems",
        "text": (
            "High-risk AI systems that use techniques involving the training of models with data shall be "
            "developed on the basis of training, validation and testing data sets that meet quality criteria. "
            "Data sets shall be relevant, sufficiently representative, free of errors, and complete in view "
            "of the intended purpose. Data governance practices shall address data collection, preparation, "
            "examination of bias, identification of data gaps and shortcomings, and ways to address them."
        ),
    },
    {
        "article_id": "Article 12",
        "article_title": "Record-keeping",
        "chapter": "Chapter III — High-risk AI systems",
        "text": (
            "High-risk AI systems shall technically allow for the automatic recording of events (logs) over "
            "the lifetime of the system. The logging capabilities shall ensure a level of traceability of the "
            "AI system's functioning appropriate to its intended purpose, including identification of situations "
            "that may result in the AI system presenting a risk or substantial modification."
        ),
    },
    {
        "article_id": "Article 13",
        "article_title": "Transparency and provision of information to deployers",
        "chapter": "Chapter III — High-risk AI systems",
        "text": (
            "High-risk AI systems shall be designed and developed in such a way as to ensure that their operation "
            "is sufficiently transparent to enable deployers to interpret a system's output and use it appropriately. "
            "They shall be accompanied by instructions for use that include the characteristics, capabilities and "
            "limitations of performance, including accuracy metrics, foreseeable risks, and intended purpose."
        ),
    },
    {
        "article_id": "Article 14",
        "article_title": "Human oversight",
        "chapter": "Chapter III — High-risk AI systems",
        "text": (
            "High-risk AI systems shall be designed and developed in such a way that they can be effectively overseen "
            "by natural persons during the period in which they are in use. Human oversight shall aim at preventing or "
            "minimising the risks to health, safety, fundamental rights that may emerge. Oversight measures must enable "
            "the human to fully understand capacities and limitations, monitor operation, decide not to use the system "
            "or to disregard, override or reverse its output, and intervene or interrupt the system."
        ),
    },
    {
        "article_id": "Article 15",
        "article_title": "Accuracy, robustness and cybersecurity",
        "chapter": "Chapter III — High-risk AI systems",
        "text": (
            "High-risk AI systems shall be designed and developed in such a way that they achieve an appropriate "
            "level of accuracy, robustness, and cybersecurity, and perform consistently in those respects throughout "
            "their lifecycle. Levels of accuracy and relevant accuracy metrics shall be declared in the accompanying "
            "instructions of use. Technical and organisational measures shall be taken to prevent malicious exploitation."
        ),
    },
    {
        "article_id": "Article 16",
        "article_title": "Obligations of providers of high-risk AI systems",
        "chapter": "Chapter III — High-risk AI systems",
        "text": (
            "Providers of high-risk AI systems shall ensure that their systems are compliant with the requirements set "
            "out in this Chapter, indicate their name and contact details, have a quality management system in place, "
            "draw up technical documentation, keep the logs automatically generated, ensure a conformity assessment "
            "procedure is performed, draw up an EU declaration of conformity, and affix the CE marking."
        ),
    },
    {
        "article_id": "Article 17",
        "article_title": "Quality management system",
        "chapter": "Chapter III — High-risk AI systems",
        "text": (
            "Providers of high-risk AI systems shall put a quality management system in place that ensures compliance "
            "with this Regulation. The system shall be documented in a systematic and orderly manner in the form of "
            "written policies, procedures and instructions covering at least: strategy for regulatory compliance, "
            "techniques for design, design control, verification, examination, testing and validation procedures, "
            "and the implementation of risk management."
        ),
    },
    {
        "article_id": "Article 18",
        "article_title": "Documentation keeping",
        "chapter": "Chapter III — High-risk AI systems",
        "text": (
            "Providers shall keep at the disposal of the national competent authorities the technical documentation, "
            "the documentation concerning the quality management system, the documentation concerning the changes "
            "approved by notified bodies where applicable, the decisions and other documents issued by the notified "
            "bodies where applicable, and the EU declaration of conformity, for a period of 10 years after the "
            "high-risk AI system has been placed on the market."
        ),
    },
    {
        "article_id": "Article 26",
        "article_title": "Obligations of deployers of high-risk AI systems",
        "chapter": "Chapter III — High-risk AI systems",
        "text": (
            "Deployers of high-risk AI systems shall take appropriate technical and organisational measures to ensure "
            "they use such systems in accordance with the instructions for use. Deployers shall assign human oversight "
            "to natural persons who have the necessary competence, training, and authority. Deployers shall monitor the "
            "operation, keep logs, and inform providers when they have reasons to consider the use may result in a risk."
        ),
    },
    {
        "article_id": "Article 27",
        "article_title": "Fundamental rights impact assessment",
        "chapter": "Chapter III — High-risk AI systems",
        "text": (
            "Prior to deploying a high-risk AI system, deployers that are bodies governed by public law or private "
            "operators providing public services shall perform an assessment of the impact on fundamental rights "
            "that the use of such system may produce. The assessment shall include a description of the deployer's "
            "processes, period and frequency of use, categories of natural persons affected, specific risks of harm, "
            "implementation of human oversight measures, and measures to be taken in case of materialisation of risks."
        ),
    },
    {
        "article_id": "Article 50",
        "article_title": "Transparency obligations for providers and deployers of certain AI systems",
        "chapter": "Chapter IV — Transparency obligations",
        "text": (
            "Providers shall ensure that AI systems intended to interact directly with natural persons are designed "
            "and developed in such a way that the natural persons concerned are informed they are interacting with an "
            "AI system. Providers of AI systems generating synthetic audio, image, video or text content shall ensure "
            "that the outputs are marked in a machine-readable format and detectable as artificially generated."
        ),
    },
    {
        "article_id": "Annex IV",
        "article_title": "Technical documentation",
        "chapter": "Annexes",
        "text": (
            "The technical documentation shall contain a general description of the AI system including its intended "
            "purpose, the provider's name, the version, how the system interacts with hardware or software, the "
            "versions of relevant software, the forms in which the AI system is placed on the market, a description "
            "of the elements of the AI system and of the process for its development, the design specifications, the "
            "system architecture, data requirements, the assessment of the human oversight measures needed, and where "
            "applicable a detailed description of pre-determined changes to the system and its performance."
        ),
    },
]


GDPR_ARTICLES = [
    {
        "article_id": "GDPR Article 5",
        "article_title": "Principles relating to processing of personal data",
        "chapter": "Chapter II",
        "text": (
            "Personal data shall be processed lawfully, fairly and in a transparent manner; collected for specified, "
            "explicit and legitimate purposes; adequate, relevant and limited to what is necessary in relation to the "
            "purposes; accurate and where necessary kept up to date; kept in a form which permits identification of "
            "data subjects for no longer than is necessary; and processed in a manner that ensures appropriate security."
        ),
    },
    {
        "article_id": "GDPR Article 17",
        "article_title": "Right to erasure",
        "chapter": "Chapter III",
        "text": (
            "The data subject shall have the right to obtain from the controller the erasure of personal data concerning "
            "them without undue delay where the personal data are no longer necessary in relation to the purposes for "
            "which they were collected, where the data subject withdraws consent, where the data subject objects to the "
            "processing, where the personal data have been unlawfully processed, or where erasure is required to comply "
            "with a legal obligation."
        ),
    },
]


def get_framework_articles(framework_id: str) -> list[dict]:
    if framework_id == "eu_ai_act":
        return EU_AI_ACT_ARTICLES
    if framework_id == "gdpr":
        return GDPR_ARTICLES
    return []


def load_framework_into_vector_store(framework_id: str) -> int:
    """Embed and upsert all Articles for a framework. Returns count loaded."""
    articles = get_framework_articles(framework_id)
    if not articles:
        logger.warning(f"No articles defined for framework_id={framework_id}")
        return 0

    spec = FRAMEWORKS.get(framework_id)
    if not spec:
        raise ValueError(f"Unknown framework: {framework_id}")

    embedder = get_embedder()
    store = get_vector_store()

    texts = [a["text"] for a in articles]
    vectors = embedder.encode(texts)

    points = []
    for i, article in enumerate(articles):
        # Use deterministic UUIDs derived from framework + article_id
        import hashlib
        h = hashlib.md5(f"{framework_id}::{article['article_id']}".encode()).hexdigest()
        # Format as UUID
        point_id = f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
        payload = {
            "framework_id": framework_id,
            "framework_name": spec.name,
            "article_id": article["article_id"],
            "article_title": article["article_title"],
            "chapter": article.get("chapter", ""),
            "full_text": article["text"],
            "source_url": spec.source_url,
        }
        points.append((point_id, vectors[i], payload))

    store.upsert(points)
    logger.info(f"Loaded {len(points)} articles for framework {framework_id}")
    return len(points)


def ensure_frameworks_loaded() -> dict[str, int]:
    """Ensure all known frameworks are indexed. Idempotent — re-upserts are safe."""
    store = get_vector_store()
    counts = {}
    for fid in FRAMEWORKS:
        existing = store.count(framework_id=fid)
        if existing == 0:
            counts[fid] = load_framework_into_vector_store(fid)
        else:
            counts[fid] = existing
            logger.info(f"Framework {fid} already loaded ({existing} articles)")
    return counts
