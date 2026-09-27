"""Mesh generation and finite-difference matrix assembly.

Interior nodes use the second-order central difference

    d²T/dx² ≈ (T_{i+1} - 2 T_i + T_{i-1}) / Δx²

which gives, for the steady equation d²T/dx² + q'''/k = 0,

    -T_{i-1} + 2 T_i - T_{i+1} = (q'''/k) Δx²

and for the fully implicit transient equation ∂T/∂t = α ∂²T/∂x² + q'''/(ρCp)

    -Fo T_{i-1}^{n+1} + (1 + 2Fo) T_i^{n+1} - Fo T_{i+1}^{n+1} = T_i^n + Fo (q'''/k) Δx²

with Fo = α Δt / Δx². (For Fo = 1 this is the familiar  -1, 3, -1  stencil.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..models import BoundaryScheme
from .boundary_conditions import BoundaryCondition, BoundaryRow
from .utils import pretty_latex, sci


@dataclass
class Mesh:
    """Uniform 1-D mesh with N nodes on [0, L]."""

    length: float
    nodes: int

    @property
    def dx(self) -> float:
        return self.length / (self.nodes - 1)

    @property
    def intervals(self) -> int:
        return self.nodes - 1

    @property
    def x(self) -> np.ndarray:
        # i * dx rather than linspace so that x_i = (i-1)Δx exactly as written in the notes
        x = np.arange(self.nodes) * self.dx
        x[-1] = self.length
        return x


@dataclass
class NodeEquation:
    """The finite-difference equation written for one node (for display)."""

    node: int  # 1-based
    kind: str  # "interior" | "boundary"
    general_latex: str
    numeric_latex: str
    note: str = ""


@dataclass
class AssembledSystem:
    """A linear system ``A T = b`` plus per-node equation descriptions.

    For transient problems ``b = time_coeff * T^n + b_const``.
    """

    A: np.ndarray
    b: np.ndarray
    equations: list[NodeEquation]
    time_coeff: np.ndarray | None = None
    b_const: np.ndarray | None = None
    lower: np.ndarray = field(init=False)
    diag: np.ndarray = field(init=False)
    upper: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.lower = np.diag(self.A, -1).copy()
        self.diag = np.diag(self.A).copy()
        self.upper = np.diag(self.A, 1).copy()


def equation_latex(coeffs: dict[int, float], rhs: float | str, sup: str = "") -> str:
    """Render ``Σ c_j T_j = rhs`` in LaTeX with 1-based node indices."""
    parts: list[str] = []
    for j in sorted(coeffs):
        c = coeffs[j]
        if c == 0:
            continue
        mag = abs(c)
        coef = "" if abs(mag - 1.0) < 1e-12 else sci(mag, 6)
        term = rf"{coef}T_{{{j + 1}}}{sup}"
        if not parts:
            parts.append(("-" if c < 0 else "") + term)
        else:
            parts.append(("- " if c < 0 else "+ ") + term)
    r = rhs if isinstance(rhs, str) else sci(rhs, 6)
    return pretty_latex(" ".join(parts) + f" = {r}")


class MatrixAssembler:
    """Builds the coefficient matrix and right-hand side from the FD equations."""

    def __init__(self, mesh: Mesh, left: BoundaryCondition, right: BoundaryCondition, scheme: BoundaryScheme) -> None:
        self.mesh = mesh
        self.left = left
        self.right = right
        self.scheme = scheme

    def _place(self, A: np.ndarray, b: np.ndarray, i: int, row: BoundaryRow) -> None:
        for j, c in row.coeffs.items():
            A[i, j] = c
        b[i] = row.rhs

    def assemble_steady(self, k: float, q_gen: float) -> AssembledSystem:
        """Assemble ``[A][T] = [b]`` for the steady problem."""
        n, dx = self.mesh.nodes, self.mesh.dx
        A = np.zeros((n, n))
        b = np.zeros(n)
        eqs: list[NodeEquation] = []
        source = q_gen / k * dx * dx

        lrow = self.left.steady_row(n, dx, k, q_gen, self.scheme)
        rrow = self.right.steady_row(n, dx, k, q_gen, self.scheme)
        self._place(A, b, 0, lrow)
        eqs.append(NodeEquation(1, "boundary", lrow.general_latex, equation_latex(lrow.coeffs, lrow.rhs), lrow.note))
        for i in range(1, n - 1):
            A[i, i - 1], A[i, i], A[i, i + 1] = -1.0, 2.0, -1.0
            b[i] = source
            eqs.append(NodeEquation(
                i + 1, "interior",
                r"-T_{i-1} + 2T_i - T_{i+1} = \frac{q'''}{k}\Delta x^2",
                equation_latex({i - 1: -1.0, i: 2.0, i + 1: -1.0}, source),
            ))
        self._place(A, b, n - 1, rrow)
        eqs.append(NodeEquation(n, "boundary", rrow.general_latex, equation_latex(rrow.coeffs, rrow.rhs), rrow.note))
        return AssembledSystem(A, b, eqs)

    def assemble_transient(self, fo: float, k: float, q_gen: float) -> AssembledSystem:
        """Assemble the (time-invariant) implicit matrix and the RHS recipe."""
        n, dx = self.mesh.nodes, self.mesh.dx
        A = np.zeros((n, n))
        b_const = np.zeros(n)
        tc = np.zeros(n)
        eqs: list[NodeEquation] = []
        source = fo * q_gen / k * dx * dx
        sup = "^{n+1}"

        lrow = self.left.transient_row(n, dx, fo, k, q_gen, self.scheme)
        rrow = self.right.transient_row(n, dx, fo, k, q_gen, self.scheme)
        self._place(A, b_const, 0, lrow)
        tc[0] = lrow.time_coeff
        eqs.append(NodeEquation(1, "boundary", lrow.general_latex,
                                equation_latex(lrow.coeffs, self._rhs_label(0, lrow.time_coeff, lrow.rhs), sup), lrow.note))
        for i in range(1, n - 1):
            A[i, i - 1], A[i, i], A[i, i + 1] = -fo, 1.0 + 2.0 * fo, -fo
            b_const[i] = source
            tc[i] = 1.0
            eqs.append(NodeEquation(
                i + 1, "interior",
                r"-Fo\,T_{i-1}^{n+1} + (1+2Fo)\,T_i^{n+1} - Fo\,T_{i+1}^{n+1} = T_i^{n}"
                + (r" + Fo\,\frac{q'''}{k}\Delta x^2" if q_gen else ""),
                equation_latex({i - 1: -fo, i: 1.0 + 2.0 * fo, i + 1: -fo}, self._rhs_label(i, 1.0, source), sup),
            ))
        self._place(A, b_const, n - 1, rrow)
        tc[n - 1] = rrow.time_coeff
        eqs.append(NodeEquation(n, "boundary", rrow.general_latex,
                                equation_latex(rrow.coeffs, self._rhs_label(n - 1, rrow.time_coeff, rrow.rhs), sup), rrow.note))
        return AssembledSystem(A, b_const.copy(), eqs, time_coeff=tc, b_const=b_const)

    @staticmethod
    def _rhs_label(i: int, time_coeff: float, const: float) -> str:
        if time_coeff == 0:
            return sci(const, 6)
        label = rf"T_{{{i + 1}}}^{{n}}"
        if const:
            label += (" + " if const > 0 else " - ") + sci(abs(const), 6)
        return label


class FiniteDifferenceSolver:
    """Common base for the steady and transient finite-difference solvers."""

    problem_type: str = ""

    def __init__(self, length_m: float, nodes: int, left: BoundaryCondition, right: BoundaryCondition,
                 scheme: BoundaryScheme) -> None:
        self.mesh = Mesh(length_m, nodes)
        self.left = left
        self.right = right
        self.scheme = scheme
        self.assembler = MatrixAssembler(self.mesh, left, right, scheme)

    def node_table(self, temperatures: np.ndarray) -> list[dict[str, float]]:
        """Rows of the Node | x | T table."""
        return [
            {"index": i + 1, "x": float(x), "temperature": float(t)}
            for i, (x, t) in enumerate(zip(self.mesh.x, temperatures))
        ]
