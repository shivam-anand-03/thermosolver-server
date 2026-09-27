"""Steady-state 1-D conduction solver:  d²T/dx² + q'''/k = 0."""

from __future__ import annotations

import time

import numpy as np

from ..config import settings
from ..models import SteadyRequest
from .boundary_conditions import TEMP_SYMBOL, BoundaryCondition
from .explanation import ExplanationBuilder
from .finite_difference import FiniteDifferenceSolver
from .gaussian_elimination import GaussianEliminationSolver
from .results import SimulationResult, serialize_gaussian, serialize_system
from .utils import length_to_si
from .validation import InputValidator, ValidationEngine


class SteadyStateSolver(FiniteDifferenceSolver):
    """Assembles and solves the steady finite-difference system with Gaussian elimination."""

    problem_type = "steady"

    def __init__(self, req: SteadyRequest) -> None:
        self.req = req
        # validate first so that no derived quantity is computed from invalid input
        self.report = InputValidator().validate_steady(req)
        self.report.raise_if_errors()
        self.k = req.thermal_conductivity
        self.q_gen = req.heat_generation
        left = BoundaryCondition.from_input("left", req.left_boundary, req.temperature_unit)
        right = BoundaryCondition.from_input("right", req.right_boundary, req.temperature_unit)
        super().__init__(length_to_si(req.length, req.length_unit), req.nodes, left, right, req.boundary_scheme)

    def solve(self) -> SimulationResult:
        """Validate → assemble → eliminate → back-substitute → validate the solution."""
        t0 = time.perf_counter()
        report = self.report

        mesh = self.mesh
        system = self.assembler.assemble_steady(self.k, self.q_gen)
        gauss = GaussianEliminationSolver(record_limit=settings.record_limit).solve(
            system.A, system.b, record=self.req.record_steps)
        T = gauss.solution

        residual = system.A @ T - system.b
        engine = ValidationEngine()
        n = mesh.nodes
        left_err = engine.boundary_error(self.left, system.A, system.b, T, 0)
        right_err = engine.boundary_error(self.right, system.A, system.b, T, n - 1)
        extra = [self._energy_check(T)]
        validation = engine.check(residual=residual, b=system.b, T=T, n_expected=n,
                                  left_error=left_err, right_error=right_err, extra=extra)
        elapsed = (time.perf_counter() - t0) * 1000

        explanation = ExplanationBuilder(self, system, gauss, T).steady()
        unit = TEMP_SYMBOL[self.req.temperature_unit]
        summary = {
            "title": "1-D Steady-State Heat Conduction",
            "length_m": mesh.length,
            "nodes": n,
            "intervals": mesh.intervals,
            "dx": mesh.dx,
            "left_bc": self.left.describe(),
            "right_bc": self.right.describe(),
            "temperature_unit": self.req.temperature_unit,
            "t_min": float(T.min()),
            "t_max": float(T.max()),
            "unit_symbol": unit,
            "scheme": self.scheme.value,
        }
        payload = {
            "dx": mesh.dx,
            "length_m": mesh.length,
            "x": mesh.x.tolist(),
            "nodes": self.node_table(T),
            "solution": T.tolist(),
            **serialize_system(system, system.b),
            "gaussian": serialize_gaussian(gauss),
            "validation": validation,
            "explanation": explanation,
            "boundary_conditions": {
                "left": {"description": self.left.describe(), "latex": self.left.condition_latex(),
                         "type": self.left.type.value, "value": self.left.value},
                "right": {"description": self.right.describe(), "latex": self.right.condition_latex(),
                          "type": self.right.type.value, "value": self.right.value},
            },
        }
        return SimulationResult("steady", self.req.model_dump(mode="json"), summary, payload,
                                report.warnings + gauss.warnings, round(elapsed, 3))

    def _energy_check(self, T: np.ndarray) -> dict:
        """Informational global heat balance using one-sided boundary gradients."""
        dx, k = self.mesh.dx, self.k
        if T.size < 2:
            return {"name": "Global energy balance", "passed": None, "detail": "n/a"}
        q_in_left = -k * (T[1] - T[0]) / dx
        q_in_right = k * (T[-1] - T[-2]) / dx
        gen = self.q_gen * self.mesh.length
        net = q_in_left + q_in_right + gen
        return {
            "name": "Global energy balance (informational)",
            "passed": None,
            "detail": (f"q''in,left = {q_in_left:.4g} W/m², q''in,right = {q_in_right:.4g} W/m², "
                       f"q'''·L = {gen:.4g} W/m², net = {net:.3e} W/m² "
                       "(non-zero net with generation reflects O(Δx) one-sided boundary gradients)"),
        }
