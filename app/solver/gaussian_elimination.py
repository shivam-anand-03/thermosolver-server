"""Hand-written Gaussian elimination with partial pivoting.

This module is the numerical heart of ThermoSolve. It deliberately does *not*
call ``numpy.linalg.solve``: every row swap, every multiplier and every
back-substitution step is performed explicitly so that it can be recorded and
shown to the user.

Two usage patterns are supported:

* :meth:`GaussianEliminationSolver.solve` – a one-off solve of ``A x = b`` that
  optionally records every intermediate augmented matrix.
* :meth:`GaussianEliminationSolver.factorize` followed by
  :meth:`EliminationRecord.solve` – the forward-elimination multipliers are
  computed once and then replayed on many right-hand sides. The transient
  solver uses this because the coefficient matrix is identical at every time
  step; replaying the *same* row operations on a new ``b`` is exactly what
  Gaussian elimination would do, just without recomputing the multipliers.

Performance note: rows whose entry below the pivot is already zero need no
row operation, so they are skipped. For the tridiagonal systems produced by the
finite-difference method this reduces the cost from O(n³) to O(n²) for the
elimination and makes N = 1000 practical.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .utils import SolverError

#: Relative tolerance below which a pivot is considered to be zero.
SINGULAR_TOL = 1e-13
#: Relative tolerance below which a pivot triggers a "near-zero" warning.
NEAR_ZERO_TOL = 1e-8


@dataclass
class RowOperation:
    """A single elementary row operation ``R_target -> R_target - factor * R_pivot``."""

    target: int
    pivot_row: int
    factor: float

    def describe(self) -> str:
        """Human readable form using 1-based row numbers, e.g. ``R2 → R2 + 1.0000·R1``."""
        t, p = self.target + 1, self.pivot_row + 1
        if self.factor < 0:
            return f"R{t} → R{t} + {abs(self.factor):.6g}·R{p}"
        return f"R{t} → R{t} − {self.factor:.6g}·R{p}"


@dataclass
class EliminationStep:
    """Everything that happened while processing one pivot column."""

    step: int
    pivot_column: int
    pivot_row_before_swap: int
    pivot_value: float
    swapped: bool
    operations: list[RowOperation]
    matrix: list[list[float]] | None = None  # augmented matrix after this step


@dataclass
class BackSubstitutionStep:
    """One line of backward substitution: ``x_i = (b_i - Σ a_ij x_j) / a_ii``."""

    index: int
    rhs: float
    sum_term: float
    diagonal: float
    value: float
    terms: list[tuple[int, float, float]]  # (j, a_ij, x_j) used in the sum


@dataclass
class EliminationRecord:
    """Result of forward elimination; can be replayed on new right-hand sides.

    The row operations and the non-zero structure of the upper-triangular
    matrix are stored as plain Python lists because the transient solver
    replays them thousands of times and scalar Python arithmetic is faster than
    NumPy for the one or two entries per row of a tridiagonal system.
    """

    n: int
    upper: np.ndarray
    swaps: list[tuple[int, int]]
    operations: list[list[RowOperation]]
    warnings: list[str] = field(default_factory=list)
    _flat_ops: list[tuple[int, int, int, float]] = field(default_factory=list, repr=False)
    _upper_rows: list[tuple[list[int], list[float], float]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        # (step k, swap partner p, target row, factor); target = -1 marks a swap-only entry
        for k, (_, p) in enumerate(self.swaps):
            self._flat_ops.append((k, p, -1, 0.0))
            for op in self.operations[k]:
                self._flat_ops.append((k, p, op.target, op.factor))
        for i in range(self.n):
            cols = (i + 1 + np.nonzero(self.upper[i, i + 1 :])[0]).tolist()
            self._upper_rows.append((cols, [float(self.upper[i, j]) for j in cols], float(self.upper[i, i])))

    def forward(self, b: np.ndarray) -> list[float]:
        """Apply the recorded swaps and row operations to a new RHS vector."""
        y = [float(v) for v in b]
        for k, p, target, factor in self._flat_ops:
            if target < 0:
                if p != k:
                    y[k], y[p] = y[p], y[k]
            else:
                y[target] -= factor * y[k]
        return y

    def back_substitute(self, y: list[float]) -> np.ndarray:
        """Solve the upper-triangular system ``U x = y``."""
        x = [0.0] * self.n
        for i in range(self.n - 1, -1, -1):
            cols, vals, diag = self._upper_rows[i]
            s = 0.0
            for j, a in zip(cols, vals):
                s += a * x[j]
            x[i] = (y[i] - s) / diag
        return np.array(x)

    def solve(self, b: np.ndarray) -> np.ndarray:
        """Solve ``A x = b`` re-using the stored elimination."""
        return self.back_substitute(self.forward(b))


@dataclass
class GaussianResult:
    """Full output of :meth:`GaussianEliminationSolver.solve`."""

    solution: np.ndarray
    initial_augmented: list[list[float]] | None
    steps: list[EliminationStep]
    back_substitution: list[BackSubstitutionStep]
    upper_augmented: list[list[float]] | None
    warnings: list[str]
    recorded: bool
    total_row_operations: int
    total_swaps: int


class GaussianEliminationSolver:
    """Gaussian elimination with partial pivoting and step recording.

    Parameters
    ----------
    record_limit:
        Intermediate augmented matrices are only stored when ``n`` is at most
        this value; beyond it only the pivots/operations of the first
        ``op_log_limit`` steps are kept to avoid huge payloads.
    op_log_limit:
        Maximum number of pivot steps whose row operations are logged when the
        system is too large to store matrices.
    """

    def __init__(self, record_limit: int = 12, op_log_limit: int = 40) -> None:
        self.record_limit = record_limit
        self.op_log_limit = op_log_limit

    # ------------------------------------------------------------------ core
    def _eliminate(
        self, M: np.ndarray, n: int, record: bool, log_ops: bool
    ) -> tuple[list[EliminationStep], list[tuple[int, int]], list[list[RowOperation]], list[str]]:
        """Forward elimination in place on ``M`` (``n`` rows, ≥ ``n`` columns)."""
        steps: list[EliminationStep] = []
        swaps: list[tuple[int, int]] = []
        all_ops: list[list[RowOperation]] = []
        warnings: list[str] = []
        scale = float(np.max(np.abs(M[:, :n]))) or 1.0

        for k in range(n - 1):
            # --- partial pivoting: largest |a_ik| on or below the diagonal
            p = k + int(np.argmax(np.abs(M[k:, k])))
            pivot = M[p, k]
            if abs(pivot) <= SINGULAR_TOL * scale:
                raise SolverError(
                    "Cannot solve: the assembled coefficient matrix is singular "
                    f"(zero pivot in column {k + 1}). Check your boundary conditions — "
                    "at least one boundary must fix the temperature for a steady problem."
                )
            if abs(pivot) <= NEAR_ZERO_TOL * scale:
                warnings.append(
                    f"Near-zero pivot ({pivot:.3e}) in column {k + 1}; results may be inaccurate."
                )
            swapped = p != k
            if swapped:
                M[[k, p]] = M[[p, k]]
            swaps.append((k, p))

            # --- eliminate only rows with a non-zero entry below the pivot
            below = M[k + 1 :, k]
            rows = k + 1 + np.nonzero(below)[0]
            ops: list[RowOperation] = []
            if rows.size:
                factors = M[rows, k] / M[k, k]
                M[rows, k:] -= np.outer(factors, M[k, k:])
                M[rows, k] = 0.0  # exact zero instead of round-off
                ops = [RowOperation(int(r), k, float(f)) for r, f in zip(rows, factors)]
            all_ops.append(ops)

            if record or (log_ops and k < self.op_log_limit):
                steps.append(
                    EliminationStep(
                        step=k + 1,
                        pivot_column=k,
                        pivot_row_before_swap=p,
                        pivot_value=float(M[k, k]),
                        swapped=swapped,
                        operations=ops,
                        matrix=M.tolist() if record else None,
                    )
                )

        if abs(M[n - 1, n - 1]) <= SINGULAR_TOL * scale:
            raise SolverError(
                "Matrix became singular: the last pivot is zero. Check the boundary "
                "conditions — the temperature level may be undetermined."
            )
        return steps, swaps, all_ops, warnings

    @staticmethod
    def _validate(A: np.ndarray, b: np.ndarray | None = None) -> None:
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise SolverError("Coefficient matrix must be square.")
        if b is not None and b.shape != (A.shape[0],):
            raise SolverError("Right-hand side length must match the matrix size.")
        if not np.all(np.isfinite(A)) or (b is not None and not np.all(np.isfinite(b))):
            raise SolverError("The linear system contains NaN or infinite values.")

    # ------------------------------------------------------------ public API
    def solve(self, A: np.ndarray, b: np.ndarray, record: bool | None = None) -> GaussianResult:
        """Solve ``A x = b`` and return the solution with a full step log."""
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        self._validate(A, b)
        n = A.shape[0]
        record = (n <= self.record_limit) if record is None else record

        M = np.hstack([A, b.reshape(-1, 1)])
        initial = M.tolist() if record else None

        if n == 1:
            if abs(M[0, 0]) <= SINGULAR_TOL:
                raise SolverError("Cannot solve: the 1×1 system has a zero coefficient.")
            steps, swaps, ops, warnings = [], [], [], []
        else:
            steps, swaps, ops, warnings = self._eliminate(M, n, record, log_ops=True)

        # --- backward substitution
        x = np.zeros(n)
        back: list[BackSubstitutionStep] = []
        detail = record or n <= 60
        for i in range(n - 1, -1, -1):
            cols = i + 1 + np.nonzero(M[i, i + 1 : n])[0]
            s = float(M[i, cols] @ x[cols]) if cols.size else 0.0
            x[i] = (M[i, n] - s) / M[i, i]
            if detail:
                back.append(
                    BackSubstitutionStep(
                        index=i,
                        rhs=float(M[i, n]),
                        sum_term=s,
                        diagonal=float(M[i, i]),
                        value=float(x[i]),
                        terms=[(int(j), float(M[i, j]), float(x[j])) for j in cols],
                    )
                )

        if not np.all(np.isfinite(x)):
            raise SolverError("The solution contains non-finite values; the system is ill-conditioned.")

        return GaussianResult(
            solution=x,
            initial_augmented=initial,
            steps=steps,
            back_substitution=back,
            upper_augmented=M.tolist() if record else None,
            warnings=warnings,
            recorded=record,
            total_row_operations=sum(len(o) for o in ops),
            total_swaps=sum(1 for k, p in swaps if k != p),
        )

    def factorize(self, A: np.ndarray) -> EliminationRecord:
        """Run forward elimination on ``A`` only and keep the multipliers."""
        A = np.asarray(A, dtype=float)
        self._validate(A)
        n = A.shape[0]
        M = A.copy()
        if n == 1:
            if abs(M[0, 0]) <= SINGULAR_TOL:
                raise SolverError("Cannot solve: the 1×1 system has a zero coefficient.")
            swaps, ops, warnings = [], [], []
        else:
            _, swaps, ops, warnings = self._eliminate(M, n, record=False, log_ops=False)
        return EliminationRecord(n, M, swaps, ops, warnings)


def gaussian_elimination(A: np.ndarray, b: np.ndarray) -> GaussianResult:
    """Convenience wrapper: solve ``A x = b`` with default recording settings."""
    return GaussianEliminationSolver().solve(A, b)
