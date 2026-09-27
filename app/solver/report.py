"""PDF solution report (ReportLab + Matplotlib mathtext for equations)."""

from __future__ import annotations

import io
import re
from pathlib import Path
from datetime import datetime
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .utils import pretty_text, sci  # noqa: E402
from .visualization import temperature_history_png, temperature_profile_png  # noqa: E402

NAVY = colors.HexColor("#0f172a")
SLATE = colors.HexColor("#475569")
BORDER = colors.HexColor("#cbd5e1")
HEAD_BG = colors.HexColor("#f1f5f9")
GREEN = colors.HexColor("#15803d")
RED = colors.HexColor("#b91c1c")

MAX_MATRIX = 10  # print dense matrices only up to this size
LOGO = Path(__file__).resolve().parent.parent / "assets" / "logo.png"


def _mathtext_png(latex: str) -> tuple[bytes, float, float] | None:
    """Render a LaTeX snippet with Matplotlib mathtext; None if unsupported."""
    if "begin{" in latex or "bigm" in latex or "vdots" in latex:
        return None
    expr = re.sub(r"\\text\{([^}]*)\}", lambda m: r"\mathrm{" + m.group(1).replace(" ", r"\ ") + "}", latex)
    expr = expr.replace(r"\left.", "").replace(r"\right|", "|").replace(r"\left(", "(").replace(r"\right)", ")")
    expr = expr.replace(r"\left[", "[").replace(r"\right]", "]")
    fig = plt.figure(figsize=(0.01, 0.01))
    try:
        fig.text(0, 0, f"${expr}$", fontsize=12)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", pad_inches=0.04, transparent=True)
    except Exception:  # noqa: BLE001 — mathtext rejects some macros; fall back to text
        plt.close(fig)
        return None
    plt.close(fig)
    data = buf.getvalue()
    from PIL import Image as PILImage

    w, h = PILImage.open(io.BytesIO(data)).size
    return data, w / 200 * 72, h / 200 * 72  # points


_GLYPHS = [("ⁿ⁺¹", "<super>n+1</super>"), ("ⁿ", "<super>n</super>"),
           ("ᵢ", "<sub>i</sub>"), ("≫", "&gt;&gt;"), ("≪", "&lt;&lt;"), ("≈", "~"), ("∞", "inf")]


_SUPER_RUN = re.compile("[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+")
_FROM_SUP = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")


def _clean(text: str) -> str:
    """Replace glyphs missing from the base PDF fonts with ReportLab markup."""
    for a, b in _GLYPHS:
        text = text.replace(a, b)
    text = _SUPER_RUN.sub(lambda m: f"<super>{m.group(0).translate(_FROM_SUP)}</super>", text)
    return text


class ReportBuilder:
    """Builds the PDF report from a solver result dictionary."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.r = result
        self.unit = result["summary"]["unit_symbol"]
        ss = getSampleStyleSheet()
        self.h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=NAVY, fontSize=20, alignment=0, spaceAfter=4)
        self.h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=NAVY, fontSize=13, spaceBefore=10, spaceAfter=4)
        self.h3 = ParagraphStyle("h3", parent=ss["Heading4"], textColor=SLATE, spaceBefore=6, spaceAfter=2)
        self.body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.5, leading=13)
        self.small = ParagraphStyle("small", parent=self.body, fontSize=8, textColor=SLATE)
        self.mono = ParagraphStyle("mono", parent=self.body, fontName="Courier", fontSize=8.5)

    # ------------------------------------------------------------- helpers
    def _table(self, rows: list[list[Any]], header: bool = True, widths: list[float] | None = None,
               font: float = 8.5) -> Table:
        t = Table(rows, colWidths=widths, hAlign="LEFT", repeatRows=1 if header else 0)
        style = [
            ("FONTSIZE", (0, 0), (-1, -1), font),
            ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ]
        if header:
            style += [("BACKGROUND", (0, 0), (-1, 0), HEAD_BG), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
        t.setStyle(TableStyle(style))
        return t

    def _eq(self, latex: str) -> Any:
        rendered = _mathtext_png(latex)
        if rendered is None:
            return Paragraph(latex.replace("<", "&lt;"), self.mono)
        data, w, h = rendered
        max_w = 170 * mm
        if w > max_w:
            h, w = h * max_w / w, max_w
        return Image(io.BytesIO(data), width=w, height=h, hAlign="LEFT")

    def _matrix(self, matrix: list[list[float]], split: int | None = None) -> Table:
        rows = [[sci(v, 5) for v in row] for row in matrix]
        t = self._table(rows, header=False, font=7.5)
        if split is not None:
            t.setStyle(TableStyle([("LINEBEFORE", (split, 0), (split, -1), 1.2, NAVY),
                                   ("ALIGN", (0, 0), (-1, -1), "RIGHT")]))
        return t

    # --------------------------------------------------------------- build
    def build(self) -> bytes:
        r, s = self.r, self.r["summary"]
        story: list[Any] = []
        if LOGO.exists():
            story.append(Image(str(LOGO), width=62 * mm, height=62 * mm * 237 / 900, hAlign="LEFT"))
            story.append(Paragraph("Solution report", self.h1))
        else:
            story.append(Paragraph("ThermoSolve — Solution Report", self.h1))
        story.append(Paragraph(f"{s['title']} &nbsp;·&nbsp; generated {datetime.now():%Y-%m-%d %H:%M}", self.small))
        story.append(Spacer(1, 6))

        # Problem definition / inputs
        story.append(Paragraph("1. Problem Definition & Inputs", self.h2))
        story.append(Paragraph(_clean(r["explanation"]["summary"]), self.body))
        story.append(Spacer(1, 4))
        inp = r["inputs"]
        rows = [["Quantity", "Value"],
                ["Length L", f"{inp['length']} {inp['length_unit']}  ({sci(s['length_m'], 6)} m)"],
                ["Nodes N / intervals", f"{s['nodes']} / {s['intervals']}"],
                ["Δx", f"{sci(s['dx'], 6)} m"],
                ["Thermal conductivity k", f"{inp['thermal_conductivity']} W/m·K"],
                ["Heat generation q'''", f"{inp['heat_generation']} W/m³"]]
        if r["problem_type"] == "transient":
            rows += [["Density ρ", f"{inp['density']} kg/m³"], ["Specific heat Cp", f"{inp['specific_heat']} J/kg·K"],
                     ["Diffusivity α", f"{sci(s['alpha'], 5)} m²/s"], ["Fourier number Fo", sci(s["fourier"], 6)],
                     ["Time step Δt", f"{sci(s['dt'], 6)} s"], ["Time steps", str(s["time_steps"])],
                     ["Final time", f"{sci(s['final_time'], 6)} s"]]
        rows += [["Left BC (x = 0)", s["left_bc"]], ["Right BC (x = L)", s["right_bc"]],
                 ["Boundary scheme", s["scheme"].replace("_", " ")],
                 ["Sign convention", "positive q'' = heat entering the slab"]]
        rows = [rows[0]] + [[Paragraph(_clean(pretty_text(str(c))), self.small) for c in r] for r in rows[1:]]
        story.append(self._table(rows, widths=[55 * mm, 110 * mm]))

        # Derivation
        story.append(Paragraph("2. Step-by-Step Derivation", self.h2))
        for i, step in enumerate(r["explanation"]["steps"], 1):
            block = [Paragraph(f"Step {i} — {step['title']}", self.h3), Paragraph(_clean(step["text"]), self.body)]
            for latex in step["latex"]:
                if "begin{array}" in latex or "begin{bmatrix}" in latex:
                    continue  # matrices / vectors are printed as tables below
                block.append(self._eq(latex))
            story.append(KeepTogether(block))

        # Matrix
        story.append(Paragraph("3. Matrix System", self.h2))
        n = s["nodes"]
        if r.get("matrix") and n <= MAX_MATRIX:
            aug = [row + [b] for row, b in zip(r["matrix"], r["rhs"])]
            story.append(Paragraph("Augmented matrix [A | b]" + (" for the first time step" if r["problem_type"] == "transient" else ""), self.body))
            story.append(self._matrix(aug, split=n))
        else:
            bands = r["bands"]
            story.append(Paragraph(f"The {n}×{n} matrix is tridiagonal. First rows of the bands:", self.body))
            k = min(n, 8)
            rows = [["Row", "sub-diag", "diag", "super-diag", "b"]]
            for i in range(k):
                rows.append([i + 1, sci(bands["lower"][i - 1], 5) if i > 0 else "—", sci(bands["diag"][i], 5),
                             sci(bands["upper"][i], 5) if i < n - 1 else "—", sci(r["rhs"][i], 6)])
            story.append(self._table(rows))

        # Gauss elimination
        g = r["gaussian"]
        story.append(Paragraph("4. Gaussian Elimination", self.h2))
        story.append(Paragraph(f"{g['total_row_operations']} row operations, {g['total_swaps']} row swaps.", self.body))
        if g["recorded"] and n <= MAX_MATRIX:
            for st in g["steps"]:
                ops = "; ".join(o["description"] for o in st["operations"]) or "no elimination needed (zeros below pivot)"
                swap = f" (rows {st['pivot_row']} ↔ {st['swap_with']} swapped)" if st["swapped"] else ""
                story.append(KeepTogether([
                    Paragraph(f"<b>Step {st['step']}</b> — pivot a[{st['pivot_row']},{st['pivot_column']}] = "
                              f"{sci(st['pivot_value'], 6)}{swap}. {ops}", self.body),
                    self._matrix(st["matrix"], split=n), Spacer(1, 4)]))
        else:
            for st in g["steps"][:10]:
                ops = "; ".join(o["description"] for o in st["operations"]) or "none"
                story.append(Paragraph(f"Step {st['step']}: pivot {sci(st['pivot_value'], 6)} — {ops}", self.small))
        story.append(Paragraph("Backward substitution", self.h3))
        for bs in g["back_substitution"][:MAX_MATRIX + 2]:
            story.append(Paragraph(
                f"T{bs['index']} = ({sci(bs['rhs'], 6)} − {sci(bs['sum'], 6)}) / {sci(bs['diagonal'], 6)} = "
                f"<b>{sci(bs['value'], 6)}</b>", self.mono))

        # Results
        story.append(PageBreak())
        story.append(Paragraph("5. Final Temperature Distribution", self.h2))
        rows = [["Node", "x [m]", f"T [{self.unit}]"]]
        nodes = r["nodes"]
        shown = nodes if len(nodes) <= 60 else nodes[:30] + nodes[-10:]
        for nd in shown:
            rows.append([nd["index"], f"{nd['x']:.6g}", f"{nd['temperature']:.4f}"])
        if r["problem_type"] == "transient":
            story.append(Paragraph(f"At t = {sci(s['final_time'], 6)} s", self.body))
        story.append(self._table(rows, widths=[20 * mm, 40 * mm, 40 * mm]))
        story.append(Spacer(1, 8))
        story.append(Image(io.BytesIO(temperature_profile_png(r)), width=160 * mm, height=85 * mm))
        if r["problem_type"] == "transient":
            story.append(Image(io.BytesIO(temperature_history_png(r)), width=160 * mm, height=85 * mm))

        # Validation
        v = r["validation"]
        story.append(Paragraph("6. Validation", self.h2))
        story.append(Paragraph(f"<font color='{'#15803d' if v['status'] == 'validated' else '#b91c1c'}'><b>"
                               f"{v['message']}</b></font> — max residual {v['residual_max']:.3e}, "
                               f"boundary error {v['boundary_error']:.3e}, execution time "
                               f"{r['execution_time_ms']:.2f} ms.", self.body))
        rows = [["Check", "Result", "Detail"]]
        for c in v["checks"]:
            res = "info" if c["passed"] is None else ("PASS" if c["passed"] else "FAIL")
            rows.append([Paragraph(c["name"], self.small), res, Paragraph(_clean(c["detail"]), self.small)])
        story.append(self._table(rows, widths=[50 * mm, 15 * mm, 100 * mm]))
        if r.get("warnings"):
            story.append(Paragraph("Notes", self.h3))
            for w in r["warnings"]:
                story.append(Paragraph("• " + _clean(w), self.small))

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                                topMargin=16 * mm, bottomMargin=16 * mm, title="ThermoSolve Report")
        doc.build(story, onFirstPage=self._footer, onLaterPages=self._footer)
        return buf.getvalue()

    @staticmethod
    def _footer(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(SLATE)
        canvas.drawString(18 * mm, 10 * mm, "ThermoSolve — 1-D Heat Conduction Numerical Solver")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()


def build_pdf_report(result: dict[str, Any]) -> bytes:
    """Render a result dictionary to PDF bytes."""
    return ReportBuilder(result).build()
