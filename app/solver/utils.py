"""Shared helpers: exceptions, unit conversion and small numeric utilities."""

from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np


class SolverError(Exception):
    """An engineering-level error that is safe to show to the user."""

    def __init__(self, message: str, details: list[str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or []


class InputValidationError(SolverError):
    """Raised when user input is physically or numerically invalid."""


LENGTH_FACTORS = {"m": 1.0, "mm": 1e-3, "cm": 1e-2}


def length_to_si(value: float, unit: str) -> float:
    """Convert a length in ``unit`` to metres."""
    return value * LENGTH_FACTORS[unit]


def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert an absolute temperature between °C (``"C"``) and K (``"K"``)."""
    if from_unit == to_unit:
        return value
    return value + 273.15 if from_unit == "C" else value - 273.15


def is_finite(*values: float | None) -> bool:
    """True when every non-None value is a finite real number."""
    return all(v is None or (isinstance(v, (int, float)) and math.isfinite(v)) for v in values)


def tridiagonal_bands(A: np.ndarray) -> dict[str, list[float]]:
    """Extract sub-, main- and super-diagonals of a (tridiagonal) matrix."""
    return {
        "lower": np.diag(A, -1).tolist(),
        "diag": np.diag(A).tolist(),
        "upper": np.diag(A, 1).tolist(),
    }


def tridiagonal_matvec(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute ``A @ x`` for a tridiagonal ``A`` given by its three bands, in O(n)."""
    y = diag * x
    if x.size > 1:
        y[1:] += lower * x[:-1]
        y[:-1] += upper * x[1:]
    return y


def sci(value: float, digits: int = 4) -> str:
    """Format a number compactly: fixed-point for moderate magnitudes, else scientific."""
    if value == 0:
        return "0"
    a = abs(value)
    if 1e-3 <= a < 1e5:
        return f"{value:.{digits}g}"
    return f"{value:.{digits - 1}e}"


def latex_num(value: float, digits: int = 4) -> str:
    """Format a number for LaTeX, e.g. ``9.615 \\times 10^{-8}``."""
    s = sci(value, digits)
    if "e" in s:
        mant, exp = s.split("e")
        return rf"{mant} \times 10^{{{int(exp)}}}"
    return s


_SCI = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)e([+-]?\d+)\b")
_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def pretty_text(s: str) -> str:
    """Rewrite ``9.615e-08`` as ``9.615 × 10⁻⁸`` in human-readable text."""
    return _SCI.sub(lambda m: f"{m.group(1)} × 10{str(int(m.group(2))).translate(_SUP)}", s)


def pretty_latex(s: str) -> str:
    """Rewrite ``9.615e-08`` as ``9.615 \\times 10^{-8}`` inside LaTeX."""
    return _SCI.sub(lambda m: rf"{m.group(1)} \times 10^{{{int(m.group(2))}}}", s)


def max_abs(values: Iterable[float]) -> float:
    """Maximum absolute value of an iterable (0 for empty)."""
    return max((abs(v) for v in values), default=0.0)
