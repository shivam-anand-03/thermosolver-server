"""Deterministic, human-readable derivations generated from the actual inputs.

No LLM is involved: every sentence and equation is built from the solver's own
numbers so that the "Step-by-Step" tab always matches what was computed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .boundary_conditions import TEMP_SYMBOL
from .finite_difference import AssembledSystem
from .gaussian_elimination import GaussianResult
from .utils import latex_num, pretty_latex, pretty_text, sci

if TYPE_CHECKING:
    from .finite_difference import FiniteDifferenceSolver

MAX_LISTED = 12  # nodes / operations listed explicitly before summarising


def _vector_latex(values: np.ndarray | list[float], name: str = "T", sup: str = "") -> str:
    vals = list(values)
    if len(vals) > MAX_LISTED:
        head = r" \\ ".join(sci(v, 6) for v in vals[:5])
        tail = r" \\ ".join(sci(v, 6) for v in vals[-3:])
        body = head + r" \\ \vdots \\ " + tail
    else:
        body = r" \\ ".join(sci(v, 6) for v in vals)
    return rf"\{{{name}\}}{sup} = \begin{{bmatrix}} {body} \end{{bmatrix}}"


def _matrix_latex(A: np.ndarray, b: np.ndarray) -> str:
    n = A.shape[0]
    if n > 8:
        return r"[A]_{%d\times%d}\{T\} = \{b\}\quad(\text{tridiagonal; see Matrix tab})" % (n, n)
    rows = []
    for i in range(n):
        rows.append(" & ".join(sci(v, 5) for v in A[i]) + r" & \bigm| & " + sci(b[i], 6))
    cols = "c" * n + "cc"
    return r"\left[\begin{array}{%s}%s\end{array}\right]" % (cols, r" \\ ".join(rows))


class ExplanationBuilder:
    """Builds the list of derivation steps shown in the UI and the PDF."""

    def __init__(self, solver: "FiniteDifferenceSolver", system: AssembledSystem, gauss: GaussianResult,
                 T: np.ndarray, extra: dict[str, Any] | None = None) -> None:
        self.s = solver
        self.system = system
        self.gauss = gauss
        self.T = T
        self.extra = extra or {}
        self.unit = TEMP_SYMBOL[solver.req.temperature_unit]

    # ----------------------------------------------------------- shared parts
    def _discretization(self) -> dict:
        m = self.s.mesh
        req = self.s.req
        L_in = f"{sci(req.length, 6)} {req.length_unit}"
        return {
            "title": "Domain Discretization",
            "text": (f"The slab of length L = {L_in} is divided into N = {m.nodes} equally spaced nodes "
                     f"({m.intervals} intervals). Node i sits at xᵢ = (i − 1)Δx."),
            "latex": [
                rf"\Delta x = \frac{{L}}{{N-1}} = \frac{{{sci(m.length, 6)}\ \mathrm{{m}}}}{{{m.nodes}-1}} = {latex_num(m.dx, 6)}\ \mathrm{{m}}",
                r"x_1 = 0,\; x_2 = \Delta x,\; \ldots,\; x_N = L",
            ],
        }

    def _bc_step(self, number: int) -> dict:
        eqs = self.system.equations
        first, last = eqs[0], eqs[-1]
        scheme = ("first-order one-sided differences" if self.s.scheme.value == "first_order"
                  else "second-order ghost-node (half control-volume) equations")
        return {
            "title": "Boundary Conditions",
            "text": (f"Left (x = 0): {self.s.left.describe()}. Right (x = L): {self.s.right.describe()}. "
                     f"Derivative conditions are discretised with {scheme}. Sign convention: a positive "
                     "heat flux q'' means heat ENTERING the slab."),
            "latex": [
                self.s.left.condition_latex(), first.general_latex, rf"\text{{Node 1: }}\; {first.numeric_latex}",
                self.s.right.condition_latex(), last.general_latex,
                rf"\text{{Node {last.node}: }}\; {last.numeric_latex}",
            ],
        }

    def _gauss_step(self) -> dict:
        g = self.gauss
        ops = [op for s in g.steps for op in s.operations]
        listed = [op.describe() for op in ops[:MAX_LISTED]]
        more = f" … and {len(ops) - MAX_LISTED} more" if len(ops) > MAX_LISTED else ""
        text = (f"The {len(self.T)}×{len(self.T)} system was solved by Gaussian elimination with partial pivoting. "
                f"Forward elimination used {g.total_row_operations} row operation(s) and {g.total_swaps} row swap(s)"
                f"{' (zero entries below a pivot need no operation)' if g.total_row_operations < len(self.T) * (len(self.T) - 1) / 2 else ''}. "
                "Backward substitution then recovered the temperatures from the last node upward.")
        latex = []
        if listed:
            latex.append(r"\text{Row operations: }" + r",\; ".join(
                rf"\text{{{d.replace('·', ' ')}}}" for d in listed) + (rf"\text{{{more}}}" if more else ""))
        latex.append(r"x_n = \frac{b_n^*}{a_{nn}^*},\qquad x_i = \frac{b_i^* - \sum_{j>i} a_{ij}^* x_j}{a_{ii}^*}")
        for bs in g.back_substitution[:6]:
            terms = " + ".join(rf"({sci(t[1], 5)})({sci(t[2], 6)})" for t in bs.terms) or "0"
            latex.append(rf"T_{{{bs.index + 1}}} = \frac{{{sci(bs.rhs, 6)} - [{terms}]}}{{{sci(bs.diagonal, 6)}}} = {sci(bs.value, 6)}")
        if len(g.back_substitution) > 6:
            latex.append(r"\vdots")
        return {"title": "Gaussian Elimination", "text": text, "latex": latex}

    # ----------------------------------------------------------------- steady
    def steady(self) -> dict:
        s, m, T = self.s, self.s.mesh, self.T
        k, q = s.k, s.q_gen
        src = q / k * m.dx ** 2
        steps = [
            {
                "title": "Governing Equation",
                "text": ("One-dimensional steady conduction with constant conductivity k and uniform "
                         f"volumetric heat generation q''' (k = {sci(k, 6)} W/m·K, q''' = {sci(q, 6)} W/m³)."),
                "latex": [r"\frac{d}{dx}\left(k\frac{dT}{dx}\right) + q''' = 0",
                          r"\frac{d^2T}{dx^2} + \frac{q'''}{k} = 0"],
            },
            self._discretization(),
            {
                "title": "Central Difference Approximation",
                "text": "The second derivative at every interior node is replaced by the second-order central difference.",
                "latex": [r"\left.\frac{d^2T}{dx^2}\right|_i \approx \frac{T_{i+1} - 2T_i + T_{i-1}}{(\Delta x)^2}"],
            },
            {
                "title": "Interior Node Equation",
                "text": (f"Substituting into the governing equation and multiplying by −Δx² gives one equation per "
                         f"interior node (i = 2 … {m.nodes - 1}). Here (q'''/k)Δx² = {sci(src, 6)}."),
                "latex": [r"\frac{T_{i+1} - 2T_i + T_{i-1}}{(\Delta x)^2} + \frac{q'''}{k} = 0",
                          r"-T_{i-1} + 2T_i - T_{i+1} = \frac{q'''}{k}(\Delta x)^2"]
                + ([rf"\text{{Node {e.node}: }}\; {e.numeric_latex}" for e in self.system.equations[1:-1][:4]]
                   if m.nodes > 2 else []),
            },
            self._bc_step(5),
            {
                "title": "Matrix System",
                "text": f"Collecting all {m.nodes} equations gives the linear system [A]{{T}} = {{b}}.",
                "latex": [_matrix_latex(self.system.A, self.system.b)],
            },
            self._gauss_step(),
            {
                "title": "Final Temperature Distribution",
                "text": (f"Temperatures range from {sci(T.min(), 6)} {self.unit} to {sci(T.max(), 6)} {self.unit}."),
                "latex": [_vector_latex(T)],
            },
        ]
        summary = (f"The domain was divided into {m.nodes} equally spaced nodes. Since L = {sci(m.length, 6)} m, "
                   f"Δx = {sci(m.dx, 6)} m. The interior nodes were discretized using the central difference "
                   f"approximation, and the boundary conditions ({s.left.describe()} at x = 0; "
                   f"{s.right.describe()} at x = L) supplied the first and last equations. The resulting linear "
                   f"system contains {m.nodes} equations and {m.nodes} unknown temperatures, which were solved by "
                   "Gaussian elimination with partial pivoting followed by backward substitution.")
        return self._finalize(summary, steps)

    @staticmethod
    def _finalize(summary: str, steps: list[dict]) -> dict:
        """Replace raw e-notation with proper scientific notation."""
        for st in steps:
            st["text"] = pretty_text(st["text"])
            st["latex"] = [pretty_latex(l) for l in st["latex"]]
        return {"summary": pretty_text(summary), "steps": steps}

    # -------------------------------------------------------------- transient
    def transient(self) -> dict:
        s, m = self.s, self.s.mesh
        dt, fo, nsteps = self.extra["dt"], self.extra["fo"], self.extra["steps"]
        T0, b1 = self.extra["T0"], self.extra["b1"]
        ic = s.req.initial_condition
        ic_text = (f"T(x, 0) = {sci(ic.value, 6)} {self.unit} at every node" if ic.type == "uniform"
                   else "a custom nodal temperature distribution")
        stencil = ""
        if abs(fo - 1.0) < 1e-9:
            stencil = " For Fo = 1 this is the familiar −1, 3, −1 stencil."
        steps = [
            {
                "title": "Governing Equation",
                "text": ("One-dimensional transient conduction with constant properties. Thermal diffusivity "
                         "combines conductivity, density and specific heat."),
                "latex": [r"\frac{\partial T}{\partial t} = \alpha\frac{\partial^2 T}{\partial x^2}"
                          + (r" + \frac{q'''}{\rho C_p}" if s.q_gen else ""),
                          rf"\alpha = \frac{{k}}{{\rho C_p}} = \frac{{{sci(s.k, 6)}}}{{({sci(s.rho, 6)})({sci(s.cp, 6)})}}"
                          rf" = {latex_num(s.alpha, 5)}\ \mathrm{{m^2/s}}"],
            },
            self._discretization(),
            {
                "title": "Time Step and Fourier Number",
                "text": (f"{'Δt was computed from the chosen Fourier number' if s.req.time_step_mode == 'fourier' else 'Δt was specified'}; "
                         f"{nsteps} step(s) reach t = {sci(nsteps * dt, 6)} s. The fully implicit scheme is "
                         "unconditionally stable, so Fo > 0.5 is allowed (accuracy, not stability, limits Δt)."),
                "latex": [rf"Fo = \frac{{\alpha\,\Delta t}}{{(\Delta x)^2}} = {sci(fo, 6)}",
                          rf"\Delta t = \frac{{Fo\,(\Delta x)^2}}{{\alpha}} = {latex_num(dt, 6)}\ \mathrm{{s}}"],
            },
            {
                "title": "Initial Condition",
                "text": f"The simulation starts from {ic_text}.",
                "latex": [_vector_latex(T0, sup="^{0}")],
            },
            {
                "title": "Fully Implicit Interior Equation",
                "text": ("The time derivative uses a backward difference and the spatial derivative is evaluated "
                         "at the new time level n+1 with the central difference." + stencil),
                "latex": [r"\frac{T_i^{n+1} - T_i^{n}}{\Delta t} = \alpha\,\frac{T_{i+1}^{n+1} - 2T_i^{n+1} + T_{i-1}^{n+1}}{(\Delta x)^2}",
                          r"-Fo\,T_{i-1}^{n+1} + (1+2Fo)\,T_i^{n+1} - Fo\,T_{i+1}^{n+1} = T_i^{n}"]
                + ([rf"\text{{Node {e.node}: }}\; {e.numeric_latex}" for e in self.system.equations[1:-1][:4]]
                   if m.nodes > 2 else []),
            },
            self._bc_step(6),
            {
                "title": "Matrix System (first time step)",
                "text": ("The coefficient matrix is the same at every time step; only the right-hand side changes "
                         "because it contains the previous temperatures Tⁿ. For step 1 the RHS uses T⁰."),
                "latex": [_matrix_latex(self.system.A, b1)],
            },
            self._gauss_step(),
            {
                "title": "Solution after the First Time Step",
                "text": f"At t = Δt = {sci(dt, 6)} s:",
                "latex": [_vector_latex(self.T, sup="^{1}")],
            },
            {
                "title": "Time Marching",
                "text": (f"The procedure (update RHS → forward elimination → backward substitution) was repeated "
                         f"{nsteps} times. Because [A] is constant, the elimination multipliers were computed once "
                         "and re-applied to each new right-hand side."),
                "latex": [r"\{b\}^{n} = \{T\}^{n} \;\Rightarrow\; [A]\{T\}^{n+1} = \{b\}^{n}"],
            },
        ]
        summary = (f"The slab (L = {sci(m.length, 6)} m) was divided into {m.nodes} nodes with Δx = {sci(m.dx, 6)} m. "
                   f"With α = {sci(s.alpha, 5)} m²/s and Fo = {sci(fo, 5)}, the time step is Δt = {sci(dt, 6)} s. "
                   f"Starting from {ic_text}, the fully implicit equations were assembled into a {m.nodes}×{m.nodes} "
                   f"system and solved by Gaussian elimination at each of {nsteps} time steps, reaching "
                   f"t = {sci(nsteps * dt, 6)} s.")
        return self._finalize(summary, steps)
