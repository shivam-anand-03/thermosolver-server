"""Boundary-condition objects and their finite-difference matrix rows.

Sign convention (shown in the UI as well)
-----------------------------------------
A **positive heat flux q'' always means heat ENTERING the slab** through that
boundary, regardless of the side. With Fourier's law q_x = -k dT/dx:

* left  boundary (x = 0, outward normal -x):  q''_in =  q_x = -k dT/dx  →  dT/dx = -q''/k
* right boundary (x = L, outward normal +x):  q''_in = -q_x = +k dT/dx  →  dT/dx = +q''/k

So the reference example ``k (dT/dx)|_{x=L} = 1000 W/m²`` is a right-boundary
heat flux of +1000 W/m² flowing *into* the slab.

Derivative discretisation
-------------------------
``first_order`` (default – the method used in the course solutions)::

    left : (T2 - T1)/Δx       = g   →   T1 - T2       = -g Δx
    right: (TN - T(N-1))/Δx   = g   →  -T(N-1) + TN   =  g Δx

``ghost_node`` (second-order): a fictitious node outside the domain is
eliminated using the central difference g ≈ (T2 - T0)/(2Δx); this is
equivalent to an energy balance on the half control volume at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import BoundaryConditionInput, BoundaryScheme, BoundaryType
from .utils import pretty_latex, pretty_text, sci

TEMP_SYMBOL = {"C": "°C", "K": "K"}
TEMP_LATEX = {"C": r"^\circ\mathrm{C}", "K": r"\,\mathrm{K}"}


@dataclass
class BoundaryRow:
    """One boundary row of the linear system.

    ``coeffs`` maps 0-based column → coefficient. For transient problems the RHS
    is ``time_coeff * T_i^n + rhs`` so that the row can be rebuilt every step.
    """

    coeffs: dict[int, float]
    rhs: float
    time_coeff: float = 0.0
    general_latex: str = ""
    note: str = ""


@dataclass
class BoundaryCondition:
    """A boundary condition at one end of the slab."""

    side: str  # "left" | "right"
    type: BoundaryType
    value: float
    temperature_unit: str = "K"

    @classmethod
    def from_input(cls, side: str, bc: BoundaryConditionInput, temperature_unit: str) -> "BoundaryCondition":
        value = 0.0 if bc.type == BoundaryType.adiabatic else float(bc.value)
        return cls(side, bc.type, value, temperature_unit)

    # ----------------------------------------------------------- properties
    @property
    def is_dirichlet(self) -> bool:
        return self.type == BoundaryType.temperature

    @property
    def x_label(self) -> str:
        return "x=0" if self.side == "left" else "x=L"

    def gradient(self, k: float) -> float:
        """Temperature gradient dT/dx [K/m] imposed at this boundary."""
        if self.type == BoundaryType.gradient:
            return self.value
        if self.type in (BoundaryType.heat_flux, BoundaryType.adiabatic):
            return -self.value / k if self.side == "left" else self.value / k
        raise ValueError("Dirichlet boundary has no prescribed gradient")

    def heat_in(self, k: float) -> float:
        """Heat flux entering the slab [W/m²] for a derivative-type condition."""
        g = self.gradient(k)
        return -k * g if self.side == "left" else k * g

    # ---------------------------------------------------------- description
    def describe(self) -> str:
        """Short plain-text description, e.g. ``T = 373 K``."""
        return pretty_text(self._describe())

    def _describe(self) -> str:
        u = TEMP_SYMBOL[self.temperature_unit]
        if self.type == BoundaryType.temperature:
            return f"T = {sci(self.value, 6)} {u}"
        if self.type == BoundaryType.heat_flux:
            return f"q'' = {sci(self.value, 6)} W/m² (into slab)"
        if self.type == BoundaryType.gradient:
            return f"dT/dx = {sci(self.value, 6)} K/m"
        return "Adiabatic (q'' = 0)"

    def condition_latex(self) -> str:
        """The physical boundary condition in LaTeX."""
        return pretty_latex(self._condition_latex())

    def _condition_latex(self) -> str:
        x = "0" if self.side == "left" else "L"
        if self.type == BoundaryType.temperature:
            return rf"T({x}) = {sci(self.value, 6)}{TEMP_LATEX[self.temperature_unit]}"
        if self.type == BoundaryType.gradient:
            return rf"\left.\frac{{dT}}{{dx}}\right|_{{x={x}}} = {sci(self.value, 6)}\ \mathrm{{K/m}}"
        sign = "-" if self.side == "left" else ""
        q = "0" if self.type == BoundaryType.adiabatic else sci(self.value, 6)
        return rf"{sign}k\left.\frac{{dT}}{{dx}}\right|_{{x={x}}} = q''_{{in}} = {q}\ \mathrm{{W/m^2}}"

    # ----------------------------------------------------------------- rows
    def steady_row(self, n: int, dx: float, k: float, q_gen: float, scheme: BoundaryScheme) -> BoundaryRow:
        """Matrix row for the steady equation d²T/dx² + q'''/k = 0."""
        i, nb = (0, 1) if self.side == "left" else (n - 1, n - 2)
        I, NB = i + 1, nb + 1  # 1-based labels
        if self.is_dirichlet:
            return BoundaryRow({i: 1.0}, self.value, general_latex=rf"T_{{{I}}} = T_{{\text{{{self.side}}}}}",
                               note="Dirichlet: the boundary temperature is known.")
        g = self.gradient(k)
        s = 1.0 if self.side == "right" else -1.0  # outward direction
        if scheme == BoundaryScheme.first_order:
            if self.side == "left":
                gl = r"\frac{T_2 - T_1}{\Delta x} = g_{0} \;\Rightarrow\; T_1 - T_2 = -g_{0}\,\Delta x"
            else:
                gl = rf"\frac{{T_{{{I}}} - T_{{{NB}}}}}{{\Delta x}} = g_L \;\Rightarrow\; -T_{{{NB}}} + T_{{{I}}} = g_L\,\Delta x"
            return BoundaryRow({i: 1.0, nb: -1.0}, s * g * dx, general_latex=gl,
                               note="First-order one-sided difference for the boundary derivative.")
        # ghost node / half control volume (divided by 2)
        src = q_gen / k * dx * dx / 2.0
        gl = (rf"T_{{{I}}} - T_{{{NB}}} = \frac{{q'''\Delta x^2}}{{2k}} {'+' if s > 0 else '-'} g\,\Delta x"
              r"\quad(\text{ghost node eliminated})")
        return BoundaryRow({i: 1.0, nb: -1.0}, src + s * g * dx, general_latex=gl,
                           note="Second-order ghost-node (half control-volume) boundary equation.")

    def transient_row(self, n: int, dx: float, fo: float, k: float, q_gen: float, scheme: BoundaryScheme) -> BoundaryRow:
        """Matrix row for the fully implicit transient equation."""
        i, nb = (0, 1) if self.side == "left" else (n - 1, n - 2)
        I, NB = i + 1, nb + 1
        if self.is_dirichlet:
            return BoundaryRow({i: 1.0}, self.value,
                               general_latex=rf"T_{{{I}}}^{{n+1}} = T_{{\text{{{self.side}}}}}",
                               note="Dirichlet: the boundary temperature is held fixed at every time level.")
        g = self.gradient(k)
        s = 1.0 if self.side == "right" else -1.0
        if scheme == BoundaryScheme.first_order:
            gl = (rf"\frac{{T_{{{max(I, NB)}}}^{{n+1}} - T_{{{min(I, NB)}}}^{{n+1}}}}{{\Delta x}} = g"
                  rf"\;\Rightarrow\; T_{{{I}}}^{{n+1}} - T_{{{NB}}}^{{n+1}} = {'+' if s > 0 else '-'}g\,\Delta x")
            return BoundaryRow({i: 1.0, nb: -1.0}, s * g * dx, 0.0, gl,
                               "First-order one-sided difference (no storage term at the boundary node).")
        src = fo * q_gen / k * dx * dx
        gl = (rf"(1+2Fo)\,T_{{{I}}}^{{n+1}} - 2Fo\,T_{{{NB}}}^{{n+1}} = T_{{{I}}}^{{n}} "
              rf"{'+' if s > 0 else '-'} 2Fo\,g\,\Delta x")
        return BoundaryRow({i: 1.0 + 2.0 * fo, nb: -2.0 * fo}, src + s * 2.0 * fo * g * dx, 1.0, gl,
                           "Second-order ghost-node boundary equation (includes the storage term).")
