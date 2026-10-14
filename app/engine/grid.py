"""Structured grids and a small sparse-matrix container.

Unknowns are numbered lexicographically, x fastest:

    p = i + Nx·(j + Ny·k)

so a field reshaped to ``grid.shape`` (= (Nz, Ny, Nx)[-dim:]) is indexed ``[k, j, i]``.

* ``vertex`` grids (FDM, FEM): N points per axis including both faces, h = L/(N−1)
* ``cell`` grids (FVM): N control volumes per axis, nodes at the cell centres, h = L/N
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

AXES = "xyz"
FACE_AXIS = {"x_min": (0, "min"), "x_max": (0, "max"), "y_min": (1, "min"),
             "y_max": (1, "max"), "z_min": (2, "min"), "z_max": (2, "max")}


@dataclass(frozen=True)
class Grid:
    dim: int
    lengths: tuple[float, ...]  # (Lx[, Ly[, Lz]])
    counts: tuple[int, ...]  # (Nx[, Ny[, Nz]])
    kind: str  # "vertex" | "cell"

    @property
    def n(self) -> int:
        return int(np.prod(self.counts))

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(reversed(self.counts))

    @cached_property
    def spacing(self) -> tuple[float, ...]:
        if self.kind == "vertex":
            return tuple(L / (N - 1) for L, N in zip(self.lengths, self.counts))
        return tuple(L / N for L, N in zip(self.lengths, self.counts))

    @cached_property
    def axis_coords(self) -> list[np.ndarray]:
        out = []
        for L, N, h in zip(self.lengths, self.counts, self.spacing):
            if self.kind == "vertex":
                c = np.arange(N) * h
                c[-1] = L
            else:
                c = (np.arange(N) + 0.5) * h
            out.append(c)
        return out

    @cached_property
    def node_coords(self) -> np.ndarray:
        """(n, dim) array of node coordinates in unknown order."""
        mesh = np.meshgrid(*reversed(self.axis_coords), indexing="ij")  # [k, j, i]
        return np.stack([m.ravel() for m in reversed(mesh)], axis=1)

    @cached_property
    def multi_index(self) -> np.ndarray:
        """(n, dim) integer array of (i, j, k) for every unknown."""
        idx = np.indices(self.shape).reshape(self.dim, -1)  # rows: k, j, i
        return idx[::-1].T.copy()

    def flat(self, ijk: tuple[int, ...] | list[int]) -> int:
        p, stride = 0, 1
        for a in range(self.dim):
            p += ijk[a] * stride
            stride *= self.counts[a]
        return p

    def stride(self, axis: int) -> int:
        return int(np.prod(self.counts[:axis])) if axis else 1

    @property
    def cell_volume(self) -> float:
        return float(np.prod(self.spacing))

    def on_face(self, p_ijk: np.ndarray, face: str) -> bool:
        a, side = FACE_AXIS[face]
        return bool(p_ijk[a] == (0 if side == "min" else self.counts[a] - 1))

    def faces(self) -> list[str]:
        return [f for f in FACE_AXIS if FACE_AXIS[f][0] < self.dim]


class SparseBuilder:
    """COO accumulator; duplicate entries are summed when finalised."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.rows: list[int] = []
        self.cols: list[int] = []
        self.vals: list[float] = []

    def add(self, i: int, j: int, v: float) -> None:
        if v != 0.0:
            self.rows.append(i)
            self.cols.append(j)
            self.vals.append(v)

    def build(self) -> "SparseMatrix":
        return SparseMatrix.from_coo(self.n, np.array(self.rows, dtype=np.int64),
                                     np.array(self.cols, dtype=np.int64), np.array(self.vals, dtype=float))


@dataclass
class SparseMatrix:
    """Minimal CSR-like matrix: enough for mat-vec, Gauss–Seidel and densification."""

    n: int
    rows: np.ndarray
    cols: np.ndarray
    vals: np.ndarray

    @classmethod
    def from_coo(cls, n: int, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray) -> "SparseMatrix":
        if rows.size == 0:
            return cls(n, rows, cols, vals)
        keys = rows * n + cols
        uniq, inv = np.unique(keys, return_inverse=True)
        data = np.bincount(inv, weights=vals)
        keep = data != 0.0
        uniq, data = uniq[keep], data[keep]
        return cls(n, uniq // n, uniq % n, data)  # sorted by row, then column

    @classmethod
    def identity_rows(cls, n: int, rows: np.ndarray) -> "SparseMatrix":
        return cls(n, rows.astype(np.int64), rows.astype(np.int64), np.ones(rows.size))

    def __add__(self, other: "SparseMatrix") -> "SparseMatrix":
        return SparseMatrix.from_coo(self.n, np.concatenate([self.rows, other.rows]),
                                     np.concatenate([self.cols, other.cols]),
                                     np.concatenate([self.vals, other.vals]))

    def scale(self, s: float) -> "SparseMatrix":
        return SparseMatrix(self.n, self.rows, self.cols, self.vals * s)

    @property
    def nnz(self) -> int:
        return int(self.vals.size)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        return np.bincount(self.rows, weights=self.vals * x[self.cols], minlength=self.n)

    def diagonal(self) -> np.ndarray:
        d = np.zeros(self.n)
        m = self.rows == self.cols
        d[self.rows[m]] = self.vals[m]
        return d

    def to_dense(self) -> np.ndarray:
        A = np.zeros((self.n, self.n))
        A[self.rows, self.cols] = self.vals
        return A

    def bandwidth(self) -> tuple[int, int]:
        """(lower, upper) bandwidth."""
        if self.nnz == 0:
            return 0, 0
        d = self.cols - self.rows
        return int(max(0, -d.min())), int(max(0, d.max()))

    def row_lists(self) -> tuple[list[list[tuple[int, float]]], list[float]]:
        """Off-diagonal (col, val) lists and the diagonal, for Gauss–Seidel sweeps."""
        off: list[list[tuple[int, float]]] = [[] for _ in range(self.n)]
        diag = [0.0] * self.n
        for i, j, v in zip(self.rows.tolist(), self.cols.tolist(), self.vals.tolist()):
            if i == j:
                diag[i] = v
            else:
                off[i].append((j, v))
        return off, diag

    def is_symmetric(self, tol: float = 1e-12) -> bool:
        A = {(i, j): v for i, j, v in zip(self.rows.tolist(), self.cols.tolist(), self.vals.tolist())}
        scale = float(np.max(np.abs(self.vals))) if self.nnz else 1.0
        return all(abs(v - A.get((j, i), 0.0)) <= tol * scale for (i, j), v in A.items())

    def diagonal_dominance(self) -> tuple[bool, bool]:
        """(weakly dominant in every row, strictly dominant in at least one row)."""
        d = np.abs(self.diagonal())
        off = np.bincount(self.rows, weights=np.abs(self.vals), minlength=self.n) - d
        weak = bool(np.all(d >= off * (1 - 1e-12)))
        strict = bool(np.any(d > off * (1 + 1e-12)))
        return weak, strict
