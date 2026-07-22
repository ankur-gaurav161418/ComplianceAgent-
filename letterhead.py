"""
letterhead.py
-------------
Generates a professional, legal-letterhead style PDF from one or more
question/answer entries produced by the Legal Compliance AI Agent.

Every page carries:
  * a masthead header  -> shield crest + "ComplianceAgent" wordmark + chamber block
  * a footer           -> "Verified by Adv. Ankur Gaurav, Advocate" + page number

The document closes with a formal Verification clause, an advocate signature
block and a verification seal, as required for a signed legal work product.

Pure-Python (fpdf2) so it runs anywhere with no native dependencies.
"""

import os
import re
import hashlib
import datetime

from fpdf import FPDF
from fpdf.enums import XPos, YPos

# --------------------------------------------------------------------------
# Brand palette (matches the app's design system)
# --------------------------------------------------------------------------
BRAND = (22, 33, 62)       # #16213E  navy
BRAND_2 = (36, 52, 92)     # #24345C
ACCENT = (15, 158, 150)    # #0F9E96  teal
ACCENT_DK = (11, 122, 115)
INK = (24, 28, 46)         # body text
MUTED = (110, 116, 130)
FAINT = (150, 155, 168)
RULE = (222, 226, 234)

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
CREST = os.path.join(ASSETS, "logo-icon-128.png")

FIRM_NAME = "ComplianceAgent"
FIRM_TAGLINE = "AI-Powered Compliance Assistant"

MARGIN = 18.0
HEADER_BOTTOM = 32.0       # body starts below this
FOOTER_TOP = 20.0          # reserved space at the bottom

# --------------------------------------------------------------------------
# Unicode -> latin-1 sanitisation (fpdf2 core fonts are latin-1)
# --------------------------------------------------------------------------
_REPL = {
    "‘": "'", "’": "'", "‚": ",", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", "‑": "-",
    "…": "...", "•": "-", "●": "-", "·": "-",
    "→": "->", "←": "<-", "⇒": "=>",
    " ": " ", "﻿": "", "​": "",
    "₹": "Rs.", "™": "(TM)", "®": "(R)", "©": "(c)",
    "✓": "*", "✔": "*", "☐": "[ ]", "☑": "[x]",
}


def san(text):
    if text is None:
        return ""
    text = str(text)
    for bad, good in _REPL.items():
        text = text.replace(bad, good)
    return text.encode("latin-1", "replace").decode("latin-1")


# --------------------------------------------------------------------------
# PDF document with repeating header / footer
# --------------------------------------------------------------------------
class Letterhead(FPDF):
    def __init__(self, advocate):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.advocate = advocate
        self.set_title(san("Legal Compliance Opinion - {}".format(FIRM_NAME)))
        self.set_author(san(advocate.get("name", "")))
        self.set_creator("ComplianceAgent")
        self.set_margins(MARGIN, HEADER_BOTTOM + 4, MARGIN)
        self.set_auto_page_break(auto=True, margin=FOOTER_TOP + 6)

    # ---- masthead on every page -----------------------------------------
    def header(self):
        # Crest
        crest_h = 14.0
        if os.path.exists(CREST):
            try:
                self.image(CREST, x=MARGIN, y=9.5, h=crest_h)
                text_x = MARGIN + crest_h + 4
            except Exception:
                text_x = MARGIN
        else:
            text_x = MARGIN

        # Wordmark
        self.set_xy(text_x, 10.0)
        self.set_font("Helvetica", "B", 17)
        self.set_text_color(*BRAND)
        # "Compliance" navy + "Agent" teal, rendered as two cells
        self.cell(self.get_string_width("Compliance"), 7, "Compliance",
                  new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.set_text_color(*ACCENT)
        self.cell(self.get_string_width("Agent"), 7, "Agent",
                  new_x=XPos.LMARGIN, new_y=YPos.TOP)
        self.set_xy(text_x, 17.6)
        self.set_font("Helvetica", "", 7.4)
        self.set_text_color(*MUTED)
        self.cell(0, 4, san(FIRM_TAGLINE.upper()), new_x=XPos.LMARGIN, new_y=YPos.TOP)

        # Chamber block (right aligned)
        adv = self.advocate
        self.set_xy(MARGIN, 9.5)
        self.set_font("Helvetica", "B", 9.5)
        self.set_text_color(*BRAND)
        self.cell(0, 5, san("Chamber of {}".format(adv.get("name", "Adv. Ankur Gaurav"))),
                  align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "", 7.8)
        self.set_text_color(*MUTED)
        self.set_x(MARGIN)
        self.cell(0, 4, "Advocate & Legal Counsel", align="R",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        line3 = adv.get("contact") or "Compliance | Data Protection | Advisory"
        self.set_x(MARGIN)
        self.cell(0, 4, san(line3), align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Accent double rule
        self.set_draw_color(*ACCENT)
        self.set_line_width(0.9)
        self.line(MARGIN, HEADER_BOTTOM - 4, 210 - MARGIN, HEADER_BOTTOM - 4)
        self.set_draw_color(*BRAND)
        self.set_line_width(0.25)
        self.line(MARGIN, HEADER_BOTTOM - 2.6, 210 - MARGIN, HEADER_BOTTOM - 2.6)
        self.set_y(HEADER_BOTTOM + 4)

    # ---- footer on every page -------------------------------------------
    def footer(self):
        self.set_y(-FOOTER_TOP)
        self.set_draw_color(*ACCENT)
        self.set_line_width(0.5)
        self.line(MARGIN, self.get_y(), 210 - MARGIN, self.get_y())
        self.ln(1.4)

        name = self.advocate.get("name", "Adv. Ankur Gaurav")
        # Left: verification (mandatory, visible on every page)
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*ACCENT_DK)
        self.cell(90, 4.5, san("Verified by {}, Advocate".format(name)),
                  new_x=XPos.RIGHT, new_y=YPos.TOP)
        # Right: page number
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MUTED)
        self.cell(210 - 2 * MARGIN - 90, 4.5, "Page {} of {{nb}}".format(self.page_no()),
                  align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        # Sub line
        self.set_x(MARGIN)
        self.set_font("Helvetica", "", 6.8)
        self.set_text_color(*FAINT)
        self.cell(0, 3.6,
                  san("Privileged & Confidential  -  This document is authenticated by the "
                      "verification and signature of the undersigned advocate."),
                  align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)


# --------------------------------------------------------------------------
# Body rendering helpers
# --------------------------------------------------------------------------
def _content_width(pdf):
    return 210 - 2 * MARGIN


def _title_band(pdf, title, subtitle):
    pdf.set_fill_color(*BRAND)
    y = pdf.get_y()
    pdf.rect(MARGIN, y, _content_width(pdf), 11, style="F")
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(1.2)
    pdf.line(MARGIN, y + 11, MARGIN + 40, y + 11)
    pdf.set_xy(MARGIN + 4, y + 1.4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(_content_width(pdf) - 8, 5.4, san(title.upper()),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(MARGIN + 4)
    pdf.set_font("Helvetica", "", 7.6)
    pdf.set_text_color(200, 208, 224)
    pdf.cell(_content_width(pdf) - 8, 4, san(subtitle),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_y(y + 11)
    pdf.ln(4)


def _fit(pdf, text, max_w):
    text = san(text)
    if pdf.get_string_width(text) <= max_w:
        return text
    while text and pdf.get_string_width(text + "..") > max_w:
        text = text[:-1]
    return (text + "..") if text else ""


def _meta_row(pdf, pairs):
    pdf.set_font("Helvetica", "", 9)
    n = len(pairs)
    col = _content_width(pdf) / n
    y = pdf.get_y()
    for i, (label, value) in enumerate(pairs):
        x = MARGIN + i * col
        pdf.set_xy(x, y)
        pdf.set_text_color(*MUTED)
        pdf.set_font("Helvetica", "", 7.4)
        pdf.cell(col, 4, _fit(pdf, label.upper(), col - 2), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_xy(x, y + 4)
        pdf.set_text_color(*BRAND)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(col, 5, _fit(pdf, value, col - 2), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_y(y + 10)
    pdf.set_draw_color(*RULE)
    pdf.set_line_width(0.2)
    pdf.line(MARGIN, pdf.get_y(), 210 - MARGIN, pdf.get_y())
    pdf.ln(4)


def _section_label(pdf, text):
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_text_color(*ACCENT_DK)
    pdf.cell(0, 5, san(text.upper()), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(0.5)


def _heading(pdf, text, level):
    sizes = {1: 12.5, 2: 11.5, 3: 10.5}
    pdf.ln(1.5)
    pdf.set_font("Helvetica", "B", sizes.get(level, 10.5))
    pdf.set_text_color(*BRAND)
    pdf.multi_cell(0, 5.6, san(text), markdown=True,
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(0.6)


def _paragraph(pdf, text):
    pdf.set_font("Times", "", 11)
    pdf.set_text_color(*INK)
    pdf.multi_cell(0, 5.5, san(text), markdown=True,
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1.2)


def _bullet(pdf, text, ordinal=None):
    pdf.set_font("Times", "", 11)
    pdf.set_text_color(*ACCENT_DK if ordinal is None else INK)
    marker = "-  " if ordinal is None else "{}.  ".format(ordinal)
    x0 = pdf.get_x()
    pdf.set_text_color(*ACCENT_DK)
    pdf.cell(6, 5.5, marker, new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.set_text_color(*INK)
    pdf.set_x(x0 + 6)
    pdf.multi_cell(_content_width(pdf) - 6, 5.5, san(text), markdown=True,
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(0.6)


def _hr(pdf):
    pdf.ln(1)
    pdf.set_draw_color(*RULE)
    pdf.set_line_width(0.2)
    pdf.line(MARGIN, pdf.get_y(), 210 - MARGIN, pdf.get_y())
    pdf.ln(2)


def _render_table(pdf, block):
    rows = []
    for ln in block:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if cells and all(re.match(r"^:?-{2,}:?$", c or "-") for c in cells):
            continue
        rows.append(cells)
    if not rows:
        return
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    try:
        pdf.set_font("Times", "", 9.5)
        pdf.set_draw_color(*RULE)
        with pdf.table(borders_layout="MINIMAL", line_height=5,
                       headings_style=None, text_align="LEFT") as table:
            for ri, r in enumerate(rows):
                row = table.row()
                for c in r:
                    row.cell(san(c))
        pdf.ln(1.5)
    except Exception:
        for r in rows:
            _paragraph(pdf, " | ".join(r))


def render_markdown(pdf, text):
    lines = (text or "").replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            pdf.ln(2.2)
            i += 1
            continue
        # table block
        if stripped.startswith("|") and stripped.count("|") >= 2:
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            _render_table(pdf, block)
            continue
        # horizontal rule
        if re.match(r"^([-*_]\s*){3,}$", stripped):
            _hr(pdf)
            i += 1
            continue
        # heading
        mh = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if mh:
            _heading(pdf, mh.group(2), len(mh.group(1)))
            i += 1
            continue
        # bullet
        mb = re.match(r"^[-*•]\s+(.*)$", stripped)
        if mb:
            _bullet(pdf, mb.group(1))
            i += 1
            continue
        # numbered
        mn = re.match(r"^(\d+)[.)]\s+(.*)$", stripped)
        if mn:
            _bullet(pdf, mn.group(2), ordinal=mn.group(1))
            i += 1
            continue
        _paragraph(pdf, stripped)
        i += 1


def _sources(pdf, sources):
    if not sources:
        return
    _section_label(pdf, "Sources referenced")
    pdf.set_font("Times", "", 9.5)
    for i, s in enumerate(sources, 1):
        src = os.path.basename(str(s.get("source", "document")))
        chunk = s.get("chunk_id", "")
        excerpt = re.sub(r"\s+", " ", str(s.get("text", ""))).strip()
        if len(excerpt) > 240:
            excerpt = excerpt[:240] + "..."
        pdf.set_text_color(*BRAND)
        pdf.set_font("Helvetica", "B", 9)
        pdf.multi_cell(0, 5, san("[{}] {}  (chunk {})".format(i, src, chunk)),
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(*MUTED)
        pdf.set_font("Times", "I", 9)
        pdf.multi_cell(0, 4.6, san(excerpt), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1.4)


def _seal(pdf, cx, cy, name):
    try:
        with pdf.local_context():
            with pdf.rotation(angle=-13, x=cx, y=cy):
                pdf.set_draw_color(*ACCENT)
                pdf.set_line_width(0.7)
                pdf.ellipse(cx - 16, cy - 16, 32, 32)
                pdf.set_line_width(0.3)
                pdf.ellipse(cx - 12.5, cy - 12.5, 25, 25)
                pdf.set_text_color(*ACCENT_DK)
                pdf.set_font("Helvetica", "B", 8)
                pdf.set_xy(cx - 13, cy - 8.5)
                pdf.cell(26, 4, "VERIFIED", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                short = name.replace("Adv.", "").strip()
                pdf.set_font("Helvetica", "B", 6.4)
                pdf.set_xy(cx - 13, cy - 2.5)
                pdf.cell(26, 3.5, san(short.upper()), align="C",
                         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font("Helvetica", "", 5.6)
                pdf.set_xy(cx - 13, cy + 2)
                pdf.cell(26, 3, "ADVOCATE", align="C",
                         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    except Exception:
        pass


def _verification_block(pdf, advocate, date_str):
    name = advocate.get("name", "Adv. Ankur Gaurav")
    # keep the block together near the page bottom if room, else new page
    if pdf.get_y() > 210:
        pdf.add_page()
    pdf.ln(4)
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(0.4)
    pdf.line(MARGIN, pdf.get_y(), 210 - MARGIN, pdf.get_y())
    pdf.ln(3)

    _section_label(pdf, "Verification")
    pdf.set_font("Times", "", 10.5)
    pdf.set_text_color(*INK)
    clause = (
        "I, {name}, Advocate, do hereby verify and affirm that the compliance "
        "response set out above has been issued under my review and professional "
        "supervision for advisory purposes, on the basis of the statutory material "
        "and documents referenced herein. The contents are true to the best of my "
        "knowledge and belief."
    ).format(name=name)
    pdf.multi_cell(0, 5.4, san(clause), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(12)

    # signature line (right side) + seal (left of it)
    sig_x = 120.0
    y = pdf.get_y()
    _seal(pdf, cx=MARGIN + 24, cy=y - 2, name=name)

    pdf.set_draw_color(*BRAND)
    pdf.set_line_width(0.35)
    pdf.line(sig_x, y, 210 - MARGIN, y)
    pdf.set_xy(sig_x, y + 1.2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*BRAND)
    pdf.cell(70, 5, san(name), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(sig_x)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*MUTED)
    pdf.cell(70, 4.4, "Advocate", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    enrol = advocate.get("enrolment", "").strip()
    pdf.set_x(sig_x)
    pdf.cell(70, 4.4, san("Enrolment No.: {}".format(enrol if enrol else "________________")),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    place = advocate.get("place", "").strip()
    pdf.set_x(sig_x)
    pdf.cell(70, 4.4, san("Place: {}    Date: {}".format(place if place else "____________", date_str)),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def build_legal_pdf(entries, advocate=None, ref_id=None, date_str=None):
    """Return PDF bytes for one or more Q/A entries in a legal-letterhead layout.

    entries   : list of {"question": str, "answer": str, "sources": list}
    advocate  : {"name","enrolment","place","contact"}
    """
    advocate = dict(advocate or {})
    advocate.setdefault("name", "Adv. Ankur Gaurav")
    if isinstance(entries, dict):
        entries = [entries]
    entries = [e for e in entries if (e.get("answer") or e.get("question"))]
    if not entries:
        entries = [{"question": "", "answer": "", "sources": []}]

    if date_str is None:
        date_str = datetime.date.today().strftime("%d %B %Y")
    if ref_id is None:
        seed = (entries[0].get("question", "") or "compliance") + date_str
        num = int(hashlib.md5(seed.encode("utf-8")).hexdigest(), 16) % 9000 + 1000
        ref_id = "LCA/{}/{}".format(datetime.date.today().year, num)

    multi = len(entries) > 1
    title = "Compliance Advisory Report" if multi else "Legal Compliance Opinion"
    subtitle = "Issued under the verification of {}".format(advocate["name"])

    pdf = Letterhead(advocate)
    pdf.set_page_mode = None
    pdf.add_page()

    _title_band(pdf, title, subtitle)
    _meta_row(pdf, [
        ("Reference", ref_id),
        ("Date", date_str),
        ("Subject", "Data Protection & Privacy"),
        ("Status", "Verified"),
    ])

    for idx, e in enumerate(entries, 1):
        if multi:
            _section_label(pdf, "Item {} of {}".format(idx, len(entries)))
        q = (e.get("question") or "").strip()
        if q:
            _section_label(pdf, "Query")
            pdf.set_font("Times", "I", 11)
            pdf.set_text_color(*BRAND_2)
            pdf.multi_cell(0, 5.5, san(q), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1.5)
        _section_label(pdf, "Opinion / Response")
        render_markdown(pdf, e.get("answer") or "")
        pdf.ln(1)
        _sources(pdf, e.get("sources") or [])
        if multi and idx < len(entries):
            _hr(pdf)

    _verification_block(pdf, advocate, date_str)

    out = pdf.output()
    return bytes(out)


if __name__ == "__main__":
    demo = build_legal_pdf(
        [{
            "question": "What are the penalties under the DPDP Act, 2023?",
            "answer": ("## Penalties under the DPDP Act, 2023\n\n"
                       "The Act prescribes **monetary penalties** for breaches:\n\n"
                       "- Up to **Rs. 250 crore** for failure to prevent a data breach.\n"
                       "- Up to **Rs. 200 crore** for failing to notify affected persons.\n\n"
                       "The Data Protection Board determines the penalty after inquiry."),
            "sources": [{"source": "data/DPDP.pdf", "chunk_id": 12,
                         "text": "The Board may impose monetary penalty as specified in the Schedule ..."}],
        }],
        advocate={"name": "Adv. Ankur Gaurav", "place": "New Delhi"},
    )
    with open("_letterhead_demo.pdf", "wb") as f:
        f.write(demo)
    print("wrote _letterhead_demo.pdf ({} bytes)".format(len(demo)))
