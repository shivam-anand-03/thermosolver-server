"""Phase III-II — closed-form solutions used to validate the numerical results.

Available cases
---------------
1. Effectively 1-D steady problems (1-D, or 2-D/3-D with insulated / outflow
   side faces and velocity along x only):

       k T'' − ρc_p u T' + βT + q0 = 0

   with any pair of temperature / flux / convective / insulated end conditions.
   The homogeneous basis comes from the roots of k r² − ρc_p u r + β = 0
   (real, repeated or complex), so conduction with q''' = q0 + βT,
   convection–diffusion (exponential boundary layer) and their combination are
   all covered. Pure convection (k = 0) integrates ρc_p u T' = q0 + βT from the inlet.
2. Effectively 1-D transient conduction (β = 0, ends at fixed temperature or
   insulated): steady part + Fourier series of the initial deviation.
3. 2-D steady Laplace equation on a rectangle with a constant temperature on
   each of the four sides: superposition of four sinh–sine series.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .discretization import DIRICHLET_TYPES, PhysicalModel
from .schema import FaceBC

Func = Callable[[np.ndarray], np.ndarray]


@dataclass
class ExactSolution:
    name: str
    latex: str
    evaluate: Callable[[np.ndarray, float | None], np.ndarray]  # (coords (n, dim), t) -> T
    exclude: Callable[[np.ndarray], np.ndarray] | None = None  # points excluded from error norms


def _is_effectively_1d(m: PhysicalModel) -> bool:
    if m.dim == 1:
        return True
    if any(abs(v) > 0 for v in m.velocity[1:]):
        return False
    return all(bc.type in ("insulated", "outflow") for f, bc in m.bcs.items() if not f.startswith("x"))


# --------------------------------------------------------------- 1-D steady
def _basis(k: float, c: float, beta: float, L: float) -> tuple[list[tuple[Func, Func]], str]:
    """Two homogeneous solutions (value, derivative) of k T'' − c T' + β T = 0, overflow-safe."""
    disc = c * c - 4 * k * beta
    tol = 1e-12 * max(c * c, abs(4 * k * beta), 1e-300)
    if abs(disc) <= tol:
        r = c / (2 * k)
        ref = L if r > 0 else 0.0
        e = lambda x: np.exp(r * (x - ref))  # noqa: E731
        if r == 0:
            return [(lambda x: np.ones_like(x), lambda x: np.zeros_like(x)), (lambda x: x, lambda x: np.ones_like(x))], "linear"
        return [(e, lambda x: r * e(x)), (lambda x: x * e(x), lambda x: e(x) * (1 + r * x))], "repeated"
    if disc > 0:
        sq = math.sqrt(disc)
        out = []
        for r in ((c + sq) / (2 * k), (c - sq) / (2 * k)):
            ref = L if r > 0 else 0.0
            out.append((lambda x, r=r, ref=ref: np.exp(r * (x - ref)), lambda x, r=r, ref=ref: r * np.exp(r * (x - ref))))
        return out, "exponential"
    a, w = c / (2 * k), math.sqrt(-disc) / (2 * k)
    ref = L if a > 0 else 0.0
    E = lambda x: np.exp(a * (x - ref))  # noqa: E731
    return [
        (lambda x: E(x) * np.cos(w * x), lambda x: E(x) * (a * np.cos(w * x) - w * np.sin(w * x))),
        (lambda x: E(x) * np.sin(w * x), lambda x: E(x) * (a * np.sin(w * x) + w * np.cos(w * x))),
    ], "oscillatory"


def _particular(k: float, c: float, beta: float, q0: float) -> tuple[Func, Func]:
    if beta != 0:
        return (lambda x: np.full_like(x, -q0 / beta), lambda x: np.zeros_like(x))
    if c != 0:
        return (lambda x: q0 * x / c, lambda x: np.full_like(x, q0 / c))
    return (lambda x: -q0 * x * x / (2 * k), lambda x: -q0 * x / k)


def _bc_row(bc: FaceBC, side: str, k: float, basis, part, xb: float) -> tuple[list[float], float] | None:
    """Linear condition α·C = r on the two basis coefficients at one end."""
    X = np.array([xb])
    vals = [float(f(X)[0]) for f, _ in basis]
    ders = [float(d(X)[0]) for _, d in basis]
    pv, pd = float(part[0](X)[0]), float(part[1](X)[0])
    sgn = -1.0 if side == "min" else 1.0  # heat in = sgn·k·T'
    if bc.type in DIRICHLET_TYPES:
        return vals, bc.value - pv
    if bc.type in ("insulated", "outflow"):
        return ders, -pd
    if bc.type == "heat_flux":
        return [sgn * k * d for d in ders], bc.value - sgn * k * pd
    if bc.type == "convection":  # sgn k T' = h (T∞ − T)
        return [sgn * k * d + bc.h * v for d, v in zip(ders, vals)], bc.h * bc.t_inf - sgn * k * pd - bc.h * pv
    return None


def steady_1d(m: PhysicalModel) -> tuple[Func, str] | None:
    L = m.lengths[0]
    left, right = m.bcs["x_min"], m.bcs["x_max"]
    c = m.rho_cp * m.velocity[0]
    k, beta, q0 = m.k, m.beta, m.q0

    if k == 0:  # pure convection: c T' = q0 + βT from the inlet
        if c == 0:
            return None
        inlet = left if c > 0 else right
        if inlet.type not in DIRICHLET_TYPES:
            return None
        Tin, cc = inlet.value, abs(c)
        dist = (lambda x: x) if c > 0 else (lambda x: L - x)
        if beta == 0:
            return (lambda x: Tin + q0 * dist(x) / cc), r"T(s) = T_{in} + \frac{q_0\,s}{\rho c_p |u|}"
        return (lambda x: (Tin + q0 / beta) * np.exp(beta * dist(x) / cc) - q0 / beta,
                r"T(s) = \left(T_{in} + \frac{q_0}{\beta}\right)e^{\beta s/(\rho c_p |u|)} - \frac{q_0}{\beta}")

    basis, kind = _basis(k, c, beta, L)
    part = _particular(k, c, beta, q0)
    r1 = _bc_row(left, "min", k, basis, part, 0.0)
    r2 = _bc_row(right, "max", k, basis, part, L)
    if r1 is None or r2 is None:
        return None
    M = np.array([r1[0], r2[0]])
    rhs = np.array([r1[1], r2[1]])
    if abs(np.linalg.det(M)) <= 1e-14 * max(1.0, float(np.max(np.abs(M))) ** 2):
        return None
    C1, C2 = np.linalg.solve(M, rhs)
    f = lambda x: C1 * basis[0][0](x) + C2 * basis[1][0](x) + part[0](x)  # noqa: E731

    if c == 0 and beta == 0:
        latex = r"T(x) = -\frac{q_0}{2k}x^2 + C_1 x + C_2"
    elif c == 0 and beta > 0:
        latex = r"T(x) = C_1\cos mx + C_2\sin mx - \frac{q_0}{\beta},\quad m=\sqrt{\beta/k}"
    elif c == 0:
        latex = r"T(x) = C_1\cosh mx + C_2\sinh mx - \frac{q_0}{\beta},\quad m=\sqrt{-\beta/k}"
    elif beta == 0 and q0 == 0:
        latex = r"T(x) = C_1 + C_2\,e^{Pe\,x/L},\quad Pe = \frac{\rho c_p u L}{k}"
    else:
        latex = r"T(x) = C_1 e^{r_1 x} + C_2 e^{r_2 x} + T_p,\quad k r^2 - \rho c_p u\, r + \beta = 0"
    return f, latex


# ------------------------------------------------------------ 1-D transient
def transient_1d(m: PhysicalModel, T0: float) -> tuple[Callable[[np.ndarray, float], np.ndarray], str] | None:
    if m.beta != 0 or any(m.velocity) or m.k == 0:
        return None
    L, alpha = m.lengths[0], m.alpha
    left, right = m.bcs["x_min"], m.bcs["x_max"]
    zero_flux = ("insulated", "outflow")
    kinds = []
    for bc in (left, right):
        if bc.type in DIRICHLET_TYPES:
            kinds.append("D")
        elif bc.type in zero_flux:
            kinds.append("N")
        else:
            return None
    if kinds == ["N", "N"]:
        rate = m.q0 / m.rho_cp
        return (lambda x, t: np.full_like(x, T0 + rate * t)), r"T(t) = T_0 + \frac{q_0}{\rho c_p}\,t"
    st = steady_1d(m)
    if st is None:
        return None
    Ts = st[0]
    if kinds == ["D", "D"]:
        lam = lambda n: n * math.pi / L  # noqa: E731
        X = lambda n, x: np.sin(lam(n) * x)  # noqa: E731
    elif kinds == ["N", "D"]:
        lam = lambda n: (n - 0.5) * math.pi / L  # noqa: E731
        X = lambda n, x: np.cos(lam(n) * x)  # noqa: E731
    else:
        lam = lambda n: (n - 0.5) * math.pi / L  # noqa: E731
        X = lambda n, x: np.sin(lam(n) * x)  # noqa: E731
    xs = np.linspace(0.0, L, 4001)
    theta0 = T0 - Ts(xs)
    terms = 400
    coeffs = []
    for n in range(1, terms + 1):
        y = theta0 * X(n, xs)
        coeffs.append(2.0 / L * float(np.trapezoid(y, xs)))

    def evaluate(x: np.ndarray, t: float) -> np.ndarray:
        out = Ts(x).astype(float)
        for n, b in enumerate(coeffs, start=1):
            decay = math.exp(-alpha * lam(n) ** 2 * t)
            if decay < 1e-16:
                break
            out = out + b * decay * X(n, x)
        return out

    return evaluate, r"T(x,t) = T_s(x) + \sum_{n} b_n X_n(x)\,e^{-\alpha\lambda_n^2 t}"


# ---------------------------------------------------------------- 2-D Laplace
def laplace_2d(m: PhysicalModel) -> ExactSolution | None:
    if m.dim != 2 or m.q0 or m.beta or any(m.velocity):
        return None
    if not all(bc.type in DIRICHLET_TYPES for bc in m.bcs.values()):
        return None
    Lx, Ly = m.lengths
    Tl, Tr = m.bcs["x_min"].value, m.bcs["x_max"].value
    Tb, Tt = m.bcs["y_min"].value, m.bcs["y_max"].value

    def ratio(a: float, s: np.ndarray, S: float) -> np.ndarray:
        """sinh(a s)/sinh(a S) without overflow."""
        return np.exp(a * (s - S)) * (1 - np.exp(-2 * a * s)) / (1 - math.exp(-2 * a * S))

    def evaluate(xy: np.ndarray, _t: float | None = None) -> np.ndarray:
        x, y = xy[:, 0], xy[:, 1]
        T = np.zeros(x.size)
        for n in range(1, 400, 2):
            cn = 4.0 / (n * math.pi)
            ax, ay = n * math.pi / Lx, n * math.pi / Ly
            T += cn * np.sin(ax * x) * (Tt * ratio(ax, y, Ly) + Tb * ratio(ax, Ly - y, Ly))
            T += cn * np.sin(ay * y) * (Tr * ratio(ay, x, Lx) + Tl * ratio(ay, Lx - x, Lx))
        # on the faces the series suffers from Gibbs overshoot; the exact value there is the face temperature
        for on, val in ((np.isclose(x, 0), Tl), (np.isclose(x, Lx), Tr), (np.isclose(y, 0), Tb), (np.isclose(y, Ly), Tt)):
            T[on] = val
        return T

    def corners(xy: np.ndarray) -> np.ndarray:
        x, y = xy[:, 0], xy[:, 1]
        ex = np.isclose(x, 0) | np.isclose(x, Lx)
        ey = np.isclose(y, 0) | np.isclose(y, Ly)
        return ex & ey

    return ExactSolution("2-D Laplace (Fourier series)",
                         r"T(x,y) = \sum_{n\ \mathrm{odd}} \frac{4T_s}{n\pi}\sin\frac{n\pi x}{L}\,"
                         r"\frac{\sinh(n\pi y/L)}{\sinh(n\pi W/L)} + \ldots\ \text{(one series per side)}",
                         evaluate, corners)


# ---------------------------------------------------------------- dispatcher
def find_exact(m: PhysicalModel, transient: bool, T0: float) -> ExactSolution | None:
    if transient:
        if not _is_effectively_1d(m):
            return None
        tr = transient_1d(m, T0)
        if tr is None:
            return None
        f, latex = tr
        return ExactSolution("1-D transient (separation of variables)", latex,
                             lambda xyz, t: f(xyz[:, 0], t or 0.0))
    if _is_effectively_1d(m):
        st = steady_1d(m)
        if st is None:
            return None
        f, latex = st
        name = {"conduction": "1-D steady conduction", "convection_diffusion": "1-D steady convection–diffusion",
                "convection": "1-D steady convection"}[m.physics]
        return ExactSolution(name, latex, lambda xyz, _t=None: f(xyz[:, 0]))
    return laplace_2d(m)
