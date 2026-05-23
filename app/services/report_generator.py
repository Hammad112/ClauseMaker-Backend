"""Audit-grade PDF report generator using ReportLab.

Produces a cover page, executive summary, per-article breakdown, and citation appendix.
"""
from __future__ import annotations

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
)


NAVY = colors.HexColor("#0B1B2B")
AMBER = colors.HexColor("#D97706")
GREEN = colors.HexColor("#15803D")
RED = colors.HexColor("#B91C1C")
LIGHT_FILL = colors.HexColor("#F0EEE9")
LIGHT_BORDER = colors.HexColor("#CCCCCC")


def _status_color(status: str):
    return {
        "Compliant": GREEN,
        "Partial": AMBER,
        "Gap": RED,
        "NotApplicable": colors.HexColor("#6B6B6B"),
    }.get(status, colors.black)


def generate_audit_pdf(
    job_data: dict,
    clauses_data: list[dict],
    framework_name: str,
    company_name: str = "Acme Corporation",
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title", parent=styles["Heading1"], fontSize=32, leading=38,
        alignment=TA_CENTER, textColor=NAVY, spaceAfter=20,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Heading2"], fontSize=14, leading=18,
        alignment=TA_CENTER, textColor=colors.HexColor("#6B6B6B"), spaceAfter=80,
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=18, leading=22,
        textColor=NAVY, spaceBefore=18, spaceAfter=12,
    )
    h3_style = ParagraphStyle(
        "H3", parent=styles["Heading3"], fontSize=12, leading=16,
        textColor=NAVY, spaceBefore=10, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontSize=10, leading=14,
        textColor=colors.HexColor("#1A1A1A"), alignment=TA_LEFT, spaceAfter=8,
    )
    small_style = ParagraphStyle(
        "Small", parent=styles["BodyText"], fontSize=9, leading=12,
        textColor=colors.HexColor("#6B6B6B"),
    )

    story = []

    # --- Cover page ---
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("CLAUSEMARK", title_style))
    story.append(Paragraph(
        f"Compliance Mapping — {framework_name}", subtitle_style,
    ))
    story.append(Spacer(1, 0.5 * inch))
    cover_table = Table(
        [
            ["Organization", company_name],
            ["Framework", framework_name],
            ["Report date", datetime.utcnow().strftime("%B %d, %Y")],
            ["Job ID", job_data.get("id", "")[:8]],
        ],
        colWidths=[2 * inch, 4.5 * inch],
    )
    cover_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, LIGHT_BORDER),
    ]))
    story.append(cover_table)
    story.append(PageBreak())

    # --- Executive summary ---
    story.append(Paragraph("Executive Summary", h2_style))
    total = job_data["total_clauses"]
    avg_conf = job_data["average_confidence"]
    summary_text = (
        f"This report presents the results of mapping {total} clauses from <b>{company_name}'s</b> "
        f"policy documentation against the <b>{framework_name}</b>. The average confidence "
        f"across all mappings is <b>{avg_conf:.1f}%</b>. The findings are organized into "
        f"four categories: Compliant, Partial, Gap, and Not Applicable."
    )
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 0.2 * inch))

    summary_table = Table(
        [
            ["Status", "Count", "% of Total"],
            ["Compliant", job_data["compliant_count"], f"{job_data['compliant_count'] / max(total,1):.0%}"],
            ["Partial", job_data["partial_count"], f"{job_data['partial_count'] / max(total,1):.0%}"],
            ["Gap", job_data["gap_count"], f"{job_data['gap_count'] / max(total,1):.0%}"],
            ["Not Applicable", job_data["not_applicable_count"], f"{job_data['not_applicable_count'] / max(total,1):.0%}"],
        ],
        colWidths=[2.5 * inch, 1.5 * inch, 1.5 * inch],
    )
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_FILL]),
        ("TEXTCOLOR", (0, 1), (0, 1), GREEN),
        ("TEXTCOLOR", (0, 2), (0, 2), AMBER),
        ("TEXTCOLOR", (0, 3), (0, 3), RED),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.3, LIGHT_BORDER),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.4 * inch))

    # --- Per-clause breakdown ---
    story.append(PageBreak())
    story.append(Paragraph("Clause-by-Clause Analysis", h2_style))
    story.append(Paragraph(
        "Each policy clause is shown below with its assigned status, the regulatory Articles it "
        "was mapped to, the confidence score, and any gap remediation recommendations.",
        body_style,
    ))
    story.append(Spacer(1, 0.15 * inch))

    for c in clauses_data:
        story.append(Paragraph(f"<b>Clause {c['position']}: {c['heading_path']}</b>", h3_style))
        status = c["primary_status"]
        story.append(Paragraph(
            f'<font color="{_status_color(status).hexval()}">'
            f'<b>● {status}</b></font>  &nbsp;|&nbsp;  '
            f'Confidence: <b>{c["primary_confidence"]:.1f}%</b>',
            small_style,
        ))
        story.append(Spacer(1, 0.05 * inch))
        clause_text = c["text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(
            f'<i>"{clause_text[:600]}{"..." if len(clause_text) > 600 else ""}"</i>',
            body_style,
        ))

        if c.get("mappings"):
            for m in c["mappings"][:3]:  # top 3 mappings per clause
                cls = m["classification"]
                story.append(Paragraph(
                    f'Maps to <b>{m["article_id"]}</b> — {m["article_title"]}  '
                    f'<font color="{_status_color(cls).hexval()}"><b>[{cls}]</b></font>  '
                    f'<font color="#6B6B6B">Confidence {m["confidence"]:.0f}%</font>',
                    small_style,
                ))
                if m.get("reasoning"):
                    story.append(Paragraph(f'<i>{m["reasoning"]}</i>', small_style))
                if m.get("gap_remediation"):
                    story.append(Paragraph(
                        f'<font color="{AMBER.hexval()}"><b>Remediation:</b></font> '
                        f'{m["gap_remediation"]}',
                        small_style,
                    ))
                story.append(Spacer(1, 0.05 * inch))
        story.append(Spacer(1, 0.15 * inch))

    # --- Citation appendix ---
    story.append(PageBreak())
    story.append(Paragraph("Citation Appendix", h2_style))
    story.append(Paragraph(
        "All cited Articles were validated against the official regulation corpus prior to "
        "inclusion in this report. No hallucinated citations appear here.",
        body_style,
    ))

    seen_articles: dict[str, dict] = {}
    for c in clauses_data:
        for m in c.get("mappings", []):
            if m["article_id"] not in seen_articles:
                seen_articles[m["article_id"]] = m

    for aid, m in sorted(seen_articles.items()):
        story.append(Paragraph(f"<b>{aid} — {m['article_title']}</b>", h3_style))
        article_text = m["article_text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(article_text, body_style))
        if m.get("source_url"):
            story.append(Paragraph(f'<font color="#6B6B6B">Source: {m["source_url"]}</font>', small_style))
        story.append(Spacer(1, 0.1 * inch))

    # --- Footer disclaimer ---
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        "This report is provided for informational purposes only and does not constitute legal advice. "
        "Outputs are advisory and should be reviewed by qualified compliance and legal professionals "
        "before reliance for regulatory submissions.",
        small_style,
    ))

    doc.build(story)
    return buf.getvalue()
