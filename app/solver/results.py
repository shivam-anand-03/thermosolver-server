"""Result containers and JSON serialisation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..config import settings
from .finite_difference import AssembledSystem
from .gaussian_elimination import GaussianResult
from .utils import pretty_text


def serialize_gaussian(g: GaussianResult) -> dict[str, Any]:
    """Convert a :class:`GaussianResult` to a JSON-friendly dict for the Algorithm tab."""
    return {
        "recorded": g.recorded,
        "initial_augmented": g.initial_augmented,
        "upper_augmented": g.upper_augmented,
        "total_row_operations": g.total_row_operations,
        "total_swaps": g.total_swaps,
        "warnings": g.warnings,
        "steps": [
            {
                "step": s.step,
                "pivot_row": s.pivot_column + 1,
                "pivot_column": s.pivot_column + 1,
                "pivot_value": s.pivot_value,
                "swapped": s.swapped,
                "swap_with": s.pivot_row_before_swap + 1,
                "operations": [
                    {"target": o.target + 1, "pivot_row": o.pivot_row + 1, "factor": o.factor,
                     "description": o.describe()}
                    for o in s.operations
                ],
                "matrix": s.matrix,
            }
            for s in g.steps
        ],
        "back_substitution": [
            {
                "index": s.index + 1,
                "rhs": s.rhs,
                "sum": s.sum_term,
                "diagonal": s.diagonal,
                "value": s.value,
                "terms": [{"j": j + 1, "a": a, "x": x} for j, a, x in s.terms],
            }
            for s in g.back_substitution
        ],
    }


def serialize_system(system: AssembledSystem, b: np.ndarray) -> dict[str, Any]:
    """Matrix payload: dense matrix for small N, tridiagonal bands always."""
    n = system.A.shape[0]
    small = n <= settings.matrix_payload_limit
    return {
        "matrix": system.A.tolist() if small else None,
        "matrix_truncated": not small,
        "bands": {"lower": system.lower.tolist(), "diag": system.diag.tolist(), "upper": system.upper.tolist()},
        "rhs": b.tolist(),
        "equations": [asdict(e) for e in system.equations] if n <= 200 else
        [asdict(e) for e in system.equations[:3]] + [asdict(e) for e in system.equations[-3:]],
    }


@dataclass
class SimulationResult:
    """Common envelope for steady and transient results."""

    problem_type: str
    inputs: dict[str, Any]
    summary: dict[str, Any]
    payload: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": True,
            "problem_type": self.problem_type,
            "inputs": self.inputs,
            "summary": self.summary,
            "warnings": [pretty_text(w) for w in self.warnings],
            "execution_time_ms": self.execution_time_ms,
            **self.payload,
        }
