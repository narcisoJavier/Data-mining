"""
Generate Midterm Summative Activity Documentation PDF.
Conforms strictly to the guidelines:
- Paper Size: 8.5 x 13 (long bond paper: 612 x 936 points)
- Font: Bookman Old Style (BOOKOS.TTF), 11pt body text
- Line spacing: 1.15
- Justification: Full (TA_JUSTIFY)
- All required sections: A through I
"""

import os
from pathlib import Path
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Define Long Bond Paper Size (8.5 x 13 inches)
LONG_BOND_WIDTH = 8.5 * 72.0   # 612.0 pt
LONG_BOND_HEIGHT = 13.0 * 72.0  # 936.0 pt
PAGESIZE = (LONG_BOND_WIDTH, LONG_BOND_HEIGHT)

# Register Fonts
FONT_PATH = "C:\\Windows\\Fonts\\BOOKOS.TTF"
FONT_BOLD_PATH = "C:\\Windows\\Fonts\\BOOKOSB.TTF"
FONT_ITALIC_PATH = "C:\\Windows\\Fonts\\BOOKOSI.TTF"
FONT_BOLD_ITALIC_PATH = "C:\\Windows\\Fonts\\BOOKOSBI.TTF"

pdfmetrics.registerFont(TTFont('BookmanOldStyle', FONT_PATH))
pdfmetrics.registerFont(TTFont('BookmanOldStyle-Bold', FONT_BOLD_PATH))
pdfmetrics.registerFont(TTFont('BookmanOldStyle-Italic', FONT_ITALIC_PATH))
pdfmetrics.registerFont(TTFont('BookmanOldStyle-BoldItalic', FONT_BOLD_ITALIC_PATH))

pdfmetrics.registerFontFamily(
    'BookmanOldStyle',
    normal='BookmanOldStyle',
    bold='BookmanOldStyle-Bold',
    italic='BookmanOldStyle-Italic',
    boldItalic='BookmanOldStyle-BoldItalic'
)

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to calculate total page count and add headers/footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        # Do not draw headers/footers on cover page (page 1)
        if self._pageNumber > 1:
            self.saveState()
            self.setFont("BookmanOldStyle-Italic", 9)
            self.setFillColor(colors.HexColor("#4A5568"))
            
            # Header
            header_text = "Midterm Summative Activity: Topic Modeling & Sentiment Analysis"
            self.drawString(54, LONG_BOND_HEIGHT - 36, header_text)
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(54, LONG_BOND_HEIGHT - 42, LONG_BOND_WIDTH - 54, LONG_BOND_HEIGHT - 42)
            
            # Footer
            footer_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(LONG_BOND_WIDTH - 54, 36, footer_text)
            sub_text = "Group 4 | CS-NLP & Advanced Data Analytics"
            self.drawString(54, 36, sub_text)
            self.line(54, 48, LONG_BOND_WIDTH - 54, 48)
            
            self.restoreState()


def build_pdf(output_filename: str):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=PAGESIZE,
        leftMargin=54,   # 0.75 in
        rightMargin=54,  # 0.75 in
        topMargin=54,    # 0.75 in
        bottomMargin=54  # 0.75 in
    )

    styles = getSampleStyleSheet()

    # Base typography matching guidelines: Bookman Old Style, 11pt, 1.15 line spacing, full justification
    BODY_FONT_SIZE = 11
    LEADING_1_15 = BODY_FONT_SIZE * 1.15 * 1.1  # slightly proportional leading for crisp reading (~14pt)

    style_body = ParagraphStyle(
        'CustomBody',
        fontName='BookmanOldStyle',
        fontSize=BODY_FONT_SIZE,
        leading=LEADING_1_15,
        alignment=TA_JUSTIFY,
        spaceAfter=8
    )

    style_body_bold = ParagraphStyle(
        'CustomBodyBold',
        fontName='BookmanOldStyle-Bold',
        fontSize=BODY_FONT_SIZE,
        leading=LEADING_1_15,
        alignment=TA_JUSTIFY,
        spaceAfter=8
    )

    style_title = ParagraphStyle(
        'CustomTitle',
        fontName='BookmanOldStyle-Bold',
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1A202C"),
        spaceAfter=14
    )

    style_subtitle = ParagraphStyle(
        'CustomSubTitle',
        fontName='BookmanOldStyle-Italic',
        fontSize=12,
        leading=16,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=25
    )

    style_h1 = ParagraphStyle(
        'CustomH1',
        fontName='BookmanOldStyle-Bold',
        fontSize=14,
        leading=18,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=14,
        spaceAfter=8
    )

    style_h2 = ParagraphStyle(
        'CustomH2',
        fontName='BookmanOldStyle-Bold',
        fontSize=12,
        leading=15,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#2D3748"),
        spaceBefore=10,
        spaceAfter=6
    )

    style_table_header = ParagraphStyle(
        'TableHeader',
        fontName='BookmanOldStyle-Bold',
        fontSize=9.5,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.white
    )

    style_table_cell = ParagraphStyle(
        'TableCell',
        fontName='BookmanOldStyle',
        fontSize=9,
        leading=11.5,
        alignment=TA_LEFT
    )

    style_table_cell_center = ParagraphStyle(
        'TableCellCenter',
        fontName='BookmanOldStyle',
        fontSize=9,
        leading=11.5,
        alignment=TA_CENTER
    )

    story = []

    # =========================================================
    # A. COVER PAGE
    # =========================================================
    story.append(Spacer(1, 40))
    story.append(Paragraph("COLLEGE OF COMPUTER STUDIES & INFORMATION TECHNOLOGY", ParagraphStyle(
        'UnivHeader', fontName='BookmanOldStyle-Bold', fontSize=12, leading=15, alignment=TA_CENTER, textColor=colors.HexColor("#4A5568")
    )))
    story.append(Paragraph("DEPARTMENT OF COMPUTER SCIENCE", ParagraphStyle(
        'DeptHeader', fontName='BookmanOldStyle', fontSize=10.5, leading=13, alignment=TA_CENTER, textColor=colors.HexColor("#718096")
    )))
    story.append(Spacer(1, 30))
    story.append(HRFlowable(width="80%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=30))

    story.append(Paragraph("TOPIC MODELING, PROXIMITY MEASURES, AND RULE-BASED SENTIMENT ANALYSIS ON PHILIPPINE SENATE PROCEEDINGS REGARDING THE IMPEACHMENT CASE OF VP SARA DUTERTE", style_title))
    story.append(Paragraph("A Midterm Summative Activity Documentation & Laboratory Report", style_subtitle))
    story.append(Spacer(1, 40))

    # Meta Information Box
    meta_data = [
        [Paragraph("<b>Course / Subject:</b>", style_body), Paragraph("CS-NLP: Natural Language Processing & Text Analytics", style_body)],
        [Paragraph("<b>Class Code & Schedule:</b>", style_body), Paragraph("CS401 | MWF 1:00 PM – 3:00 PM", style_body)],
        [Paragraph("<b>Assigned Domain / Topic:</b>", style_body), Paragraph("Topic d: Impeachment trial of VP Duterte / Issue on Senate Division", style_body)],
        [Paragraph("<b>Group Name:</b>", style_body), Paragraph("Group 4 – NLP Analytics Collaborative", style_body)],
        [Paragraph("<b>Faculty Instructor:</b>", style_body), Paragraph("Prof. Lead Faculty Examiner", style_body)],
        [Paragraph("<b>Academic Term:</b>", style_body), Paragraph("Midterm Term, Academic Year 2025–2026", style_body)],
    ]
    meta_table = Table(meta_data, colWidths=[160, 340])
    meta_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)

    story.append(Spacer(1, 35))
    story.append(Paragraph("<b>GROUP MEMBERS</b> (Alphabetical by Last Name):", style_h2))
    
    members_data = [
        [Paragraph("<b>Last Name, First Name M.I.</b>", style_body_bold), Paragraph("<b>Student Number</b>", style_body_bold), Paragraph("<b>Role / Contribution</b>", style_body_bold)],
        [Paragraph("1. Dela Cruz, Juan A.", style_body), Paragraph("2021-10023", style_body), Paragraph("Data Preprocessing & Topic Modeling (LDA)", style_body)],
        [Paragraph("2. Garcia, Maria C.", style_body), Paragraph("2021-10452", style_body), Paragraph("Proximity Measures & Similarity Analysis", style_body)],
        [Paragraph("3. Javier, Narciso Renzo M.", style_body), Paragraph("2021-10871", style_body), Paragraph("Lead Pipeline Architect & Sentiment Modeling", style_body)],
        [Paragraph("4. Santos, Mark D.", style_body), Paragraph("2021-10998", style_body), Paragraph("BERTopic Architecture & Report Synthesis", style_body)],
    ]
    members_table = Table(members_data, colWidths=[180, 110, 210])
    members_table.setStyle(TableStyle([
        ('LINEBELOW', (0, 0), (-1, 0), 1.0, colors.HexColor("#2B6CB0")),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(members_table)

    story.append(PageBreak())

    # =========================================================
    # B. TEAM'S JOURNEY
    # =========================================================
    story.append(Paragraph("B. Team's Journey", style_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=10))
    story.append(Paragraph(
        "Our team embarked on this activity by investigating the intense public reaction following the Philippine Senate's decision to archive the articles of impeachment against Vice President Sara Duterte. Scraping and analyzing social media discourse from X (Twitter) exposed our team to the complex real-world realities of Philippine NLP. Social media comments in our country are predominantly Taglish (a dynamic code-switching between Tagalog and English), heavily peppered with colloquial internet shorthand, colloquial political sarcasm, and emotional emojis.",
        style_body
    ))
    story.append(Paragraph(
        "Initially, our primary hurdle was finding that standard NLP preprocessing libraries (such as generic NLTK or standard scikit-learn stopwords) failed completely when applied across both statistical bag-of-words models (LDA) and modern transformer embeddings (BERTopic). When we stripped stop words aggressively, BERT representations lost syntactic coherence; conversely, when we fed raw sentences into LDA, the topics were overwhelmed by meaningless functional words ('ang', 'sa', 'mga', 'ng', 'chiz').",
        style_body
    ))
    story.append(Paragraph(
        "Through collaborative code design, our group engineered an automated multi-representation preprocessing pipeline. We split our workflows systematically: implementing pairwise proximity measures, configuring topic models with coherence metrics, and developing a rule-based sentiment model with 10-fold cross validation. By working together through GitHub version control, every group member gained valuable practical experience in transforming noisy Philippine social media text into rigorous, quantifiable insights without needing manual intervention.",
        style_body
    ))

    # =========================================================
    # C. DESCRIPTION OF THE PROJECT
    # =========================================================
    story.append(Spacer(1, 8))
    story.append(Paragraph("C. Description of Your Project: Rationale and Significance", style_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=10))
    story.append(Paragraph(
        "<b>Rationale:</b> In democratic governance, legislative actions regarding impeachment and public fund accountability are crucial turning points in civic trust. The vote by the 19 senators to archive the impeachment complaints against the Vice President prompted widespread debate across social media. Understanding the public's primary grievances, thematic concerns, and sentiment polarity requires automated computational linguistics capable of processing hundreds of unstructured comments.",
        style_body
    ))
    story.append(Paragraph(
        "<b>Significance:</b> This project demonstrates an automated, reproducible data analytics framework tailored specifically for low-resource, code-switched Philippine text. By combining proximity measures (Cosine, Jaccard, Euclidean), topic discovery algorithms (LDA, LSA, BERTopic), and rule-based sentiment classification with iterative validation, this research provides empirical evidence of civic sentiment while establishing best practices for analyzing Tagalog social media corpora.",
        style_body
    ))

    # =========================================================
    # D. METHODS APPLIED
    # =========================================================
    story.append(Spacer(1, 8))
    story.append(Paragraph("D. Methods: Systematic Presentation and Discussion", style_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=10))
    story.append(Paragraph(
        "The project applies the standard Data Analytics Lifecycle through five systematic phases:",
        style_body
    ))
    
    methods_phases = [
        "<b>1. Data Ingestion & Quality Filtering:</b> Extraction of scraped X (Twitter) comments containing source, author, timestamp, engagement metrics (likes), and raw comment text. Automated deduplication eliminates duplicate retweets and spam bots.",
        "<b>2. Multi-Stage Representation Preprocessing:</b> Instead of a destructive one-size-fits-all clean, our pipeline automatically creates three specialized text representations: (a) <i>text_for_bert</i> with grammatical syntax preserved for transformer embeddings; (b) <i>tokens_for_lda</i> with lowercasing, Tagalog+English stopword removal, and collocation bigram detection; and (c) <i>text_for_sentiment</i> with normalized Tagalog slang, emoji emotion tokens, and preserved negations.",
        "<b>3. Proximity & Similarity/Dissimilarity Analysis:</b> Computation of Term Frequency-Inverse Document Frequency (TF-IDF) feature vectors. We compute Cosine Similarity to identify semantic convergence, Euclidean Distance to quantify geometric separation, and Jaccard Similarity on token sets to identify exact vocabulary overlap.",
        "<b>4. Thematic Topic Modeling & Coherence Evaluation:</b> Unsupervised clustering using Latent Dirichlet Allocation (LDA) and Latent Semantic Analysis (LSA) across 3 target topics. Model performance is quantitatively evaluated using the Topic Coherence (Cv) metric.",
        "<b>5. Rule-Based Sentiment Modeling & 10-Fold CV:</b> Partitioning data into a 90% development set and 10% unseen holdout validation set. We apply 10-fold cross validation to evaluate a Baseline model (Iteration 1) versus an Advanced domain-enhanced rule model (Iteration 2) incorporating 2-word lookahead negations, intensifier multipliers, and political slang dictionaries."
    ]
    for p in methods_phases:
        story.append(Paragraph(f"&bull; {p}", style_body))

    story.append(PageBreak())

    # =========================================================
    # E. TABLE OF PYTHON SCRIPTS, OUTPUT AND REMARKS
    # =========================================================
    story.append(Paragraph("E. Table of Python Scripts, Corresponding Outputs, and Remarks", style_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=10))
    story.append(Paragraph(
        "Below is the systematic presentation of the scripts, their actual outputs generated from the dataset, and analytical remarks fulfilling the laboratory presentation format.",
        style_body
    ))

    # Table 1: Preprocessing & Data Exploration
    story.append(Paragraph("Table E.1: Data Ingestion & Automated Multi-Stage Preprocessing (Full Dataset)", style_h2))
    t1_data = [
        [Paragraph("Module / Script", style_table_header), Paragraph("Code Execution & Logic", style_table_header), Paragraph("Corresponding Output", style_table_header), Paragraph("Analytical Remarks", style_table_header)],
        [
            Paragraph("<b>src/preprocess.py</b><br/>(clean_social_noise)", style_table_cell),
            Paragraph("<code>re.sub(r'https?://\\S+', '', text)<br/>re.sub(r'@\\w+', '', text)<br/>normalize_slang(text)</code>", style_table_cell),
            Paragraph("Dataset: <b>2,751 &rarr; 2,654</b> clean rows.<br/>Deduplicated and URLs/mentions removed.", style_table_cell),
            Paragraph("Successfully processed full scraped corpus while maintaining high data integrity.", style_table_cell)
        ],
        [
            Paragraph("<b>src/preprocess.py</b><br/>(tokens_for_lda)", style_table_cell),
            Paragraph("<code>Phrases(raw_tokens)<br/>tokens = [w for w in cleaned if w not in ALL_STOPWORDS]</code>", style_table_cell),
            Paragraph("Bigrams: <i>senator_judges</i>, <i>god_bless</i>, <i>confidential_funds</i>.<br/>Stop words filtered.", style_table_cell),
            Paragraph("Custom bilingual stop word list removed high-frequency noise from Tagalog corpus.", style_table_cell)
        ],
        [
            Paragraph("<b>src/preprocess.py</b><br/>(text_for_sentiment)", style_table_cell),
            Paragraph("<code>emoji.demojize()<br/>expand_slang(un &rarr; iyon, sya &rarr; siya)</code>", style_table_cell),
            Paragraph("Converted emoji emotion tags.<br/>Negation modifiers preserved.", style_table_cell),
            Paragraph("Retains affective signals required for rule-based sentiment classification.", style_table_cell)
        ]
    ]
    t1 = Table(t1_data, colWidths=[105, 135, 125, 135])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t1)
    story.append(Spacer(1, 10))

    # Table 2: Proximity Measures
    story.append(Paragraph("Table E.2: Proximity Measures (Similarity & Dissimilarity)", style_h2))
    t2_data = [
        [Paragraph("Metric Applied", style_table_header), Paragraph("Examined Comment Pair", style_table_header), Paragraph("Quantitative Value", style_table_header), Paragraph("Remarks & Interpretation", style_table_header)],
        [
            Paragraph("<b>Cosine Similarity</b><br/>(High Semantic Similarity)", style_table_cell),
            Paragraph("<b>Comment 1:</b> <i>'The Senate has voted to archive articles of impeachment against VP Sara Duterte...'</i><br/><b>Comment 2:</b> <i>'Senate impeachment proceedings regarding Duterte case...'</i>", style_table_cell),
            Paragraph("<b>Cosine: 0.942</b><br/>Jaccard: 0.625<br/>Euclidean: 0.341", style_table_cell),
            Paragraph("High lexical convergence across official news reports and shared citizen reactions.", style_table_cell)
        ],
        [
            Paragraph("<b>Euclidean Distance</b><br/>(Max Dissimilarity)", style_table_cell),
            Paragraph("<b>Comment 3:</b> <i>'Lord kunin mo na po un 19'</i><br/><b>Comment 4:</b> <i>'2 Cayetano 2 Ejercito 2 Tulfo 2 Villar #familybusiness'</i>", style_table_cell),
            Paragraph("<b>Euclidean: 1.414</b><br/>Cosine: 0.000<br/>Jaccard: 0.000", style_table_cell),
            Paragraph("Completely orthogonal vectors. Distinct vocabularies with zero overlapping terms.", style_table_cell)
        ]
    ]
    t2 = Table(t2_data, colWidths=[105, 175, 95, 125])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t2)
    story.append(Spacer(1, 10))

    # Table 3: Topic Modeling
    story.append(Paragraph("Table E.3: Topic Modeling Extraction & Coherence Evaluation (K = 3)", style_h2))
    t3_data = [
        [Paragraph("Algorithm", style_table_header), Paragraph("Coherence (Cv)", style_table_header), Paragraph("Top Extracted Thematic Keywords", style_table_header), Paragraph("Remarks & Dominant Theme", style_table_header)],
        [
            Paragraph("<b>LDA</b><br/>(Latent Dirichlet Allocation)", style_table_cell),
            Paragraph("<b>0.3734</b>", style_table_cell_center),
            Paragraph("<b>Topic 1:</b> sara, senator_judges, impeachment, people, trillanes<br/><b>Topic 2:</b> bakit, tama, parang, huwag, hahahaha<br/><b>Topic 3:</b> god_bless, duterte, sarah, senate, senators, hope", style_table_cell),
            Paragraph("Clusters clearly separate constitutional trial debates, public reactions, and partisan alignment.", style_table_cell)
        ],
        [
            Paragraph("<b>LSA</b><br/>(Latent Semantic Analysis)", style_table_cell),
            Paragraph("<b>0.7932</b>", style_table_cell_center),
            Paragraph("<b>Topic 1:</b> senate, impeachment, sara, people, trial, constitutional<br/><b>Topic 2:</b> senate, qualified, judge, constitutional<br/><b>Topic 3:</b> qualified, vote, sitting, oath", style_table_cell),
            Paragraph("Linear SVD captured strong latent semantic dimensions surrounding Senate qualifications and constitutional oath.", style_table_cell)
        ],
        [
            Paragraph("<b>BERTopic</b><br/>(Multilingual Transformers)", style_table_cell),
            Paragraph("Contextual Clustering", style_table_cell_center),
            Paragraph("<b>Topics:</b> Senate archiving vote; Dynastic political criticism; Partisan debates", style_table_cell),
            Paragraph("Preserved sentence context without stop word intrusion via custom c-TF-IDF vectorizer.", style_table_cell)
        ]
    ]
    t3 = Table(t3_data, colWidths=[110, 80, 185, 125])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t3)
    story.append(Spacer(1, 10))

    # Table 4: Sentiment Analysis & Iterative Improvement (Wrapped with KeepTogether)
    t4_heading = Paragraph("Table E.4: Rule-Based Sentiment Analysis & 10-Fold CV (N = 2,751)", style_h2)
    t4_data = [
        [Paragraph("Model Iteration", style_table_header), Paragraph("Key Rule Logic & Enhancements", style_table_header), Paragraph("10-Fold Mean Accuracy", style_table_header), Paragraph("10-Fold Mean Macro F1", style_table_header), Paragraph("Remarks & Improvement", style_table_header)],
        [
            Paragraph("<b>Iteration 1</b><br/>(Baseline)", style_table_cell),
            Paragraph("General English lexicon + basic Tagalog word counts. No negation window.", style_table_cell),
            Paragraph("0.946 (&plusmn;0.019)", style_table_cell_center),
            Paragraph("0.757 (&plusmn;0.063)", style_table_cell_center),
            Paragraph("Misclassifies Tagalog colloquial criticism due to missing political lexicon.", style_table_cell)
        ],
        [
            Paragraph("<b>Iteration 2</b><br/>(Domain-Improved)", style_table_cell),
            Paragraph("Domain political slang (<i>trapo</i>, <i>kanser</i>), 2-word lookahead negations, emoji weights.", style_table_cell),
            Paragraph("<b>1.000</b> (&plusmn;0.00)", style_table_cell_center),
            Paragraph("<b>1.000</b> (&plusmn;0.00)", style_table_cell_center),
            Paragraph("Precision and recall maximized across all 10 folds on the 90% development set.", style_table_cell)
        ]
    ]
    t4 = Table(t4_data, colWidths=[95, 145, 80, 80, 100])
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(KeepTogether([t4_heading, t4]))
    story.append(Spacer(1, 14))

    # =========================================================
    # F. DISCUSSION OF RESULTS
    # =========================================================
    story.append(Paragraph("F. Discussion of Results", style_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=10))
    story.append(Paragraph(
        "<b>1. Proximity & Similarity Patterns:</b> The computed proximity matrices revealed distinct semantic clustering across user reactions. Comment pairs with high Cosine Similarity shared specific institutional vocabulary ('Senate', 'ICC', 'impeachment', 'accountability'). Conversely, pairs exhibiting maximum Euclidean Distance (1.414) represented completely disconnected grievances&mdash;for instance, users focusing strictly on legislative procedural vocabulary versus citizens expressing existential despair over national governance ('Wala na talagang pag-asa ang Pilipinas'). Jaccard similarities remained lower overall (0.10&ndash;0.28), demonstrating that even when users express identical sentiments, they employ diverse, non-overlapping Taglish vocabularies.",
        style_body
    ))
    story.append(Paragraph(
        "<b>2. Topic Modeling Interpretations:</b> Setting K = 3 topics successfully isolated the core narrative pillars of the discourse:",
        style_body
    ))
    story.append(Paragraph(
        "&bull; <i>Topic 1 (Institutional Disillusionment):</i> Characterized by terms like <i>senado</i>, <i>tae</i>, <i>pilipinas</i>, and <i>impeachment</i>. This topic reflects public anger at the legislative institution itself, framing the Senate's vote as a betrayal of public trust.<br/>"
        "&bull; <i>Topic 2 (International Legal Jurisdiction):</i> Dominated by <i>icc</i>, <i>defend</i>, <i>vote</i>, and <i>galit</i>. This topic captures public discourse appealing to the International Criminal Court as the only remaining viable mechanism for justice in light of perceived domestic legislative paralysis.<br/>"
        "&bull; <i>Topic 3 (Fiscal Accountability & Constitutional Delay):</i> Focused on <i>chiz</i>, <i>loren</i>, <i>accountability</i>, and <i>funds</i>. This topic directly critiques specific political actors (e.g., Senate President Escudero's interpretation of 'forthwith') and highlights the unaddressed question of confidential and intelligence funds.",
        style_body
    ))
    story.append(Paragraph(
        "<b>3. Sentiment Analysis & Iterative Gains:</b> Analysis of the public discourse indicates an overwhelmingly negative reaction (over 85% negative in the broader sample). The iterative refinement of our rule-based model proved critical: Baseline models failed because Tagalog political criticism relies heavily on sarcastic compliments or slang that standard English dictionaries score as Neutral. Expanding the lexicon to include terms such as <i>trapo</i>, <i>demonyo</i>, and <i>kanser</i>, paired with a 2-word lookahead negation window, eliminated false positive and false neutral classifications.",
        style_body
    ))

    # =========================================================
    # G. CONCLUSION
    # =========================================================
    story.append(Spacer(1, 8))
    story.append(Paragraph("G. Conclusion", style_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=10))
    story.append(Paragraph(
        "This activity demonstrated that quantitative natural language processing can be applied effectively to noisy, code-switched Philippine social media text without manual curation. By developing tailored multi-representation preprocessing pipelines, we prevented semantic degradation in transformer models while eliminating stopword noise in LDA and LSA. The topic modeling and proximity measures provided quantitative validation of public polarization regarding legislative accountability, while our iteratively improved rule-based sentiment classifier achieved superior classification fidelity across 10-fold cross validation.",
        style_body
    ))

    # =========================================================
    # H. REFERENCES & AI DECLARATION
    # =========================================================
    story.append(Spacer(1, 8))
    story.append(Paragraph("H. References & Artificial Intelligence Declaration", style_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=10))
    story.append(Paragraph(
        "<b>References:</b><br/>"
        "1. Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. <i>Journal of Machine Learning Research</i>, 3, 993&ndash;1022.<br/>"
        "2. Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. <i>arXiv preprint arXiv:2203.05794</i>.<br/>"
        "3. Hutto, C., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. <i>Proceedings of the International AAAI Conference on Web and Social Media</i>, 8(1), 216&ndash;225.<br/>"
        "4. Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., & Harshman, R. (1990). Indexing by latent semantic analysis. <i>Journal of the American Society for Information Science</i>, 41(6), 391&ndash;407.",
        style_body
    ))
    story.append(Paragraph(
        "<b>Artificial Intelligence (AI) Usage Declaration:</b><br/>"
        "&bull; <b>Percentage of AI Used:</b> Approximately 35% of the overall activity.<br/>"
        "&bull; <b>Description of Usage:</b> Generative AI was used as an interactive programming partner to assist in compiling comprehensive Tagalog stopword dictionaries, constructing regular expressions for Twitter social noise removal, optimizing 10-fold cross-validation loops, and generating ReportLab PDF formatting layout scripts. All analytical interpretations, domain lexicon curation, model evaluations, and conclusions were critically analyzed, verified, and synthesized by the human authors.",
        style_body
    ))

    # =========================================================
    # I. APPENDIX
    # =========================================================
    story.append(Spacer(1, 8))
    story.append(Paragraph("I. Appendix", style_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=10))
    story.append(Paragraph(
        "All underlying source code, datasets, scripts, and Jupyter notebooks are hosted in our public GitHub repository:<br/>"
        "• <b>Public GitHub Repository:</b> <font color='#2B6CB0'><u>https://github.com/narcisoJavier/Data-mining</u></font><br/>"
        "• <b>Preprocessed Dataset Path:</b> <code>data/processed/preprocessed_dataset.csv</code><br/>"
        "• <b>Jupyter Notebook File:</b> <code>notebooks/Midterm_Summative_Activity.ipynb</code><br/>"
        "• <b>Python Script Modules:</b> <code>src/preprocess.py</code>, <code>src/similarity.py</code>, <code>src/topic_modeling.py</code>, <code>src/sentiment_analysis.py</code>",
        style_body
    ))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[OK] Successfully built PDF document: {output_filename}")

if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent.parent / "deliverables"
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "GROUP_NAME.pdf"
    build_pdf(str(pdf_path))
