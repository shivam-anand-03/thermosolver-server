"""Phase II-II — matrix inversion techniques for [A]{T} = {B}.

All three are written out explicitly (no ``numpy.linalg.solve``):

* Gauss elimination  — forward elimination of [A | B] with partial pivoting,
  then back substitution. Repeated in full for every right-hand side.
* LU decomposition   — the same elimination stored as PA = LU once, then
  forward (Ly = Pb) and backward (Ux = y) substitution for each right-hand side.
* Gauss–Seidel / SOR — point iteration T_i ← (B_i − Σ_{j≠i} a_ij T_j)/a_ii using
  the newest values, with relaxation factor ω and a convergence criterion.

The direct solvers exploit the band structure of grid matrices: with lower
bandwidth p and upper bandwidth q, pivoting keeps every non-zero inside
columns k … k+p+q, so elimination costs O(n·p·(p+q)) instead of O(n³).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..solver.utils import SolverError
from .grid import SparseMatrix

SINGULAR_TOL = 1e-13


@dataclass
class LinearSolveResult:
    x: np.ndarray
    method: str
    iterations: int | None = None
    converged: bool = True
    history: list[float] = field(default_factory=list)
    residual: float = 0.0
    swaps: int = 0
    time_ms: float = 0.0


def relative_residual(A: SparseMatrix, x: np.ndarray, b: np.ndarray) -> float:
    r = b - A.matvec(x)
    ref = max(float(np.linalg.norm(b)), float(np.linalg.norm(A.matvec(x))), 1e-300)
    return float(np.linalg.norm(r)) / ref


# ------------------------------------------------------------------ direct
@dataclass
class BandLU:
    """Packed result of banded elimination (multipliers per column + upper factor)."""

    n: int
    U: np.ndarray
    pivots: list[int]
    multipliers: list[tuple[np.ndarray, np.ndarray]]  # per column k: (rows, factors)
    width: int
    swaps: int

    def solve(self, b: np.ndarray) -> np.ndarray:
        y = np.array(b, dtype=float)
        for k, p in enumerate(self.pivots):  # forward: L y = P b
            if p != k:
                y[k], y[p] = y[p], y[k]
            rows, f = self.multipliers[k]
            if rows.size:
                y[rows] -= f * y[k]
        return back_substitute(self.U, y, self.width)


def back_substitute(U: np.ndarray, y: np.ndarray, width: int) -> np.ndarray:
    n = y.size
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        hi = min(n, i + width + 1)
        x[i] = (y[i] - U[i, i + 1:hi] @ x[i + 1:hi]) / U[i, i]
    return x


def _eliminate(A: np.ndarray, bl: int, bu: int, b: np.ndarray | None = None) -> BandLU:
    """Forward elimination with partial pivoting, restricted to the band."""
    n = A.shape[0]
    M = np.array(A, dtype=float)
    width = bl + bu
    scale = float(np.max(np.abs(M))) or 1.0
    pivots: list[int] = []
    mults: list[tuple[np.ndarray, np.ndarray]] = []
    swaps = 0
    for k in range(n):
        r_end = min(n, k + bl + 1)
        c_end = min(n, k + width + 1)
        p = k + int(np.argmax(np.abs(M[k:r_end, k])))
        if abs(M[p, k]) <= SINGULAR_TOL * scale:
            raise SolverError(
                f"The coefficient matrix is singular (zero pivot in column {k + 1}). The temperature level is "
                "not fixed by the boundary conditions, or β makes the operator singular.")
        if p != k:
            M[[k, p], k:c_end] = M[[p, k], k:c_end]
            if b is not None:
                b[k], b[p] = b[p], b[k]
            swaps += 1
        pivots.append(p)
        rows = np.arange(k + 1, r_end)
        if rows.size:
            f = M[rows, k] / M[k, k]
            nz = f != 0.0
            rows, f = rows[nz], f[nz]
            if rows.size:
                M[rows, k:c_end] -= np.outer(f, M[k, k:c_end])
                M[rows, k] = 0.0
                if b is not None:
                    b[rows] -= f * b[k]
        mults.append((rows, f if rows.size else np.zeros(0)))
    return BandLU(n, M, pivots, mults, width, swaps)


def gauss_elimination(A: SparseMatrix, b: np.ndarray, dense: np.ndarray | None = None) -> LinearSolveResult:
    t0 = time.perf_counter()
    bl, bu = A.bandwidth()
    y = np.array(b, dtype=float)
    lu = _eliminate(A.to_dense() if dense is None else dense, bl, bu, y)
    x = back_substitute(lu.U, y, lu.width)
    _check_finite(x)
    return LinearSolveResult(x, "gauss_elimination", residual=relative_residual(A, x, b), swaps=lu.swaps,
                             time_ms=(time.perf_counter() - t0) * 1000)


def lu_factorize(A: SparseMatrix) -> BandLU:
    bl, bu = A.bandwidth()
    return _eliminate(A.to_dense(), bl, bu)


def lu_solve(A: SparseMatrix, lu: BandLU, b: np.ndarray) -> LinearSolveResult:
    t0 = time.perf_counter()
    x = lu.solve(b)
    _check_finite(x)
    return LinearSolveResult(x, "lu", residual=relative_residual(A, x, b), swaps=lu.swaps,
                             time_ms=(time.perf_counter() - t0) * 1000)


def explicit_lu(A: np.ndarray) -> dict:
    """Dense Doolittle PA = LU with row swaps, for display of small systems."""
    n = A.shape[0]
    U = np.array(A, dtype=float)
    L = np.eye(n)
    P = np.arange(n)
    for k in range(n - 1):
        p = k + int(np.argmax(np.abs(U[k:, k])))
        if p != k:
            U[[k, p]] = U[[p, k]]
            P[[k, p]] = P[[p, k]]
            L[[k, p], :k] = L[[p, k], :k]
        if U[k, k] == 0:
            continue
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
            U[i, k] = 0.0
    return {"L": L.tolist(), "U": U.tolist(), "P": P.tolist()}


# --------------------------------------------------------------- iterative
def optimal_omega(counts: tuple[int, ...]) -> float:
    """SOR factor of the model Poisson problem, 2/(1 + sin(π/(N−1))) for the finest direction."""
    N = max(counts)
    return 2.0 / (1.0 + math.sin(math.pi / max(N - 1, 1)))


def gauss_seidel(A: SparseMatrix, b: np.ndarray, x0: np.ndarray, *, tol: float, max_iter: int,
                 omega: float = 1.0, criterion: str = "residual", rows=None) -> LinearSolveResult:
    t0 = time.perf_counter()
    off, diag = rows if rows is not None else A.row_lists()
    zero = [i for i, d in enumerate(diag) if d == 0.0]
    if zero:
        raise SolverError(
            f"Gauss–Seidel needs non-zero diagonal coefficients; row {zero[0] + 1} has a_ii = 0. "
            "Use upwind advection or a direct solver (Gauss elimination / LU).")
    n = A.n
    x = [float(v) for v in x0]
    bl = [float(v) for v in b]
    history: list[float] = []
    converged = False
    it = 0
    start = relative_residual(A, np.array(x), b)
    for it in range(1, max_iter + 1):
        max_change = 0.0
        for i in range(n):
            s = bl[i]
            for j, a in off[i]:
                s -= a * x[j]
            new = s / diag[i]
            if omega != 1.0:
                new = x[i] + omega * (new - x[i])
            d = abs(new - x[i])
            if d > max_change:
                max_change = d
            x[i] = new
        xa = np.array(x)
        res = relative_residual(A, xa, b)
        metric = res if criterion == "residual" else max_change / max(1.0, float(np.max(np.abs(xa))))
        history.append(metric)
        if not math.isfinite(metric) or res > 1e8 * max(start, 1e-12):
            raise SolverError(
                f"Gauss–Seidel diverged after {it} iterations. The matrix is not diagonally dominant "
                "(e.g. central differencing at cell Péclet > 2, or ω ≥ 2). Use upwind, lower ω, or a direct solver.")
        if metric < tol:
            converged = True
            break
    xa = np.array(x)
    return LinearSolveResult(xa, "gauss_seidel", it, converged, history, relative_residual(A, xa, b),
                             time_ms=(time.perf_counter() - t0) * 1000)


def _check_finite(x: np.ndarray) -> None:
    if not np.all(np.isfinite(x)):
        raise SolverError("The solution contains non-finite values; the system is ill-conditioned.")
