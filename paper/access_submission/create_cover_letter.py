"""Render the IEEE Access cover-letter draft for author review."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "pdf" / "COVER_LETTER_DRAFT_DO_NOT_UPLOAD.pdf"


def p(text, style):
    return Paragraph(text, style)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(TTFont("ArialEmbedded", "/System/Library/Fonts/Supplemental/Arial.ttf"))
    pdfmetrics.registerFont(TTFont("ArialEmbedded-Bold", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"))
    pdfmetrics.registerFontFamily("ArialEmbedded", normal="ArialEmbedded",
                                  bold="ArialEmbedded-Bold")
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "LetterBody", parent=styles["BodyText"], fontName="ArialEmbedded",
        fontSize=10.2, leading=14.2, spaceAfter=7, alignment=TA_LEFT,
    )
    small = ParagraphStyle(
        "Small", parent=body, fontSize=8.7, leading=11.5, textColor=colors.HexColor("#333333")
    )
    heading = ParagraphStyle(
        "Heading", parent=body, fontName="ArialEmbedded-Bold", fontSize=10.2,
        leading=13, spaceBefore=3, spaceAfter=4,
    )
    warning = ParagraphStyle(
        "Warning", parent=body, fontName="ArialEmbedded-Bold", fontSize=9.3,
        leading=12.2, textColor=colors.HexColor("#7A4100"),
    )
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4, rightMargin=23 * mm, leftMargin=23 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
        title="IEEE Access Cover Letter Draft - Author Action Required",
        author="Bin Wen, Dai-Qiao Zhang, Tien-Ping Tan",
    )
    story = []
    warning_box = Table([[p(
        "AUTHOR ACTION REQUIRED: Confirm author metadata, ORCID, funding, conflicts, and the "
        "originality/exclusivity statement before submission.", warning
    )]], colWidths=[164 * mm])
    warning_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF3CD")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#C88A00")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story += [warning_box, Spacer(1, 7 * mm), p("July 16, 2026", body),
              Spacer(1, 2 * mm), p("Editor-in-Chief<br/>IEEE Access", body),
              Spacer(1, 2 * mm), p("Dear Editor-in-Chief,", body)]

    paragraphs = [
        "Please consider our manuscript, <b>“Information-Matched Auditing of Conversational "
        "Emotion-Shift Forecasting,”</b> for publication in <i>IEEE Access</i> as "
        "a Research Article.",
        "Conversational emotion forecasting is intended to predict an emotional change before "
        "the target utterance is observed. We show that common evaluation practices can silently "
        "violate this temporal contract or compare systems with unequal current-state information. "
        "The manuscript contributes executable causality and provenance audits, information-matched "
        "transition baselines, equal validation search budgets for six causal model families, "
        "seed–dialogue-aware inference, a same-architecture leakage dose response, and a complete "
        "interaction-shift versus next-own-utterance self-shift evaluation.",
        "In the deployable comparison, all six models improve in point estimate over a transition "
        "baseline driven by train-only predicted labels on all four primary text corpora; "
        "duplicate-free DailyDialog gains up to 0.167 ROC–AUC survive a 72-comparison Holm correction. "
        "When both sides receive gold current emotion, the transition baseline remains competitive "
        "on IEMOCAP and stronger on MELD, while all six DailyDialog gains remain significant. These "
        "findings are relevant to affective computing, speech and language processing, multimodal "
        "learning, and reliable machine-learning evaluation.",
    ]
    story += [p(x, body) for x in paragraphs]
    story += [p(
        "This manuscript is original, has not been published, and is not under consideration by "
        "any other journal or conference. All authors have approved the manuscript and agree with "
        "its submission to IEEE Access.", body), p(
        "The authors have disclosed the use of OpenAI Codex for language editing, LaTeX formatting, "
        "code review, and the documentation and presentation of supplementary analyses. All "
        "experimental designs, executions, interpretations, and conclusions were independently "
        "reviewed and verified by the authors, who assume full responsibility.", body
    ), p("Thank you for your consideration.", body), Spacer(1, 2 * mm), p(
        "Sincerely,<br/><br/><b>Tien-Ping Tan</b> (corresponding author)<br/>"
        "on behalf of Bin Wen, Dai-Qiao Zhang, and Tien-Ping Tan<br/>"
        "School of Computer Sciences<br/>"
        "Universiti Sains Malaysia<br/>11800 Gelugor, Penang, Malaysia<br/>"
        "tienping@usm.my", body
    )]
    doc.build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
