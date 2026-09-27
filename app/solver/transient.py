"""Transient 1-D conduction solver, fully implicit (backward Euler) in time.

At every time step:

1. Build the RHS vector ``b`` from the previous time level  (b = c ⊙ Tⁿ + d)
2. Apply the boundary conditions (already built into the rows of A and d)
3. Forward elimination
4. Backward substitution
5. Store Tⁿ⁺¹ and advance

The coefficient matrix A does not depend on time, so the forward-elimination
multipliers are computed once and replayed on each new RHS; the very first
step is additionally solved with full step recording for the Algorithm tab.
"""

from __future__ import annotations

import math
import time

import numpy as np

from ..config import settings
from ..models import TransientRequest
from .boundary_conditions import TEMP_SYMBOL, BoundaryCondition
from .explanation import ExplanationBuilder
from .finite_difference import FiniteDifferenceSolver
from .gaussian_elimination import GaussianEliminationSolver
from .results import SimulationResult, serialize_gaussian, serialize_system
from .utils import SolverError, length_to_si, tridiagonal_matvec
from .validation import InputValidator, ValidationEngine


class TransientSolver(FiniteDifferenceSolver):
    """Fully implicit finite-difference solver for ∂T/∂t = α ∂²T/∂x² + q'''/(ρCp)."""

    problem_type = "transient"

    def __init__(self, req: TransientRequest) -> None:
        self.req = req
        # validate first so that no derived quantity is computed from invalid input
        self.report = InputValidator().validate_transient(req)
        self.report.raise_if_errors()
        self.k = req.thermal_conductivity
        self.rho = req.density
        self.cp = req.specific_heat
        self.q_gen = req.heat_generation
        left = BoundaryCondition.from_input("left", req.left_boundary, req.temperature_unit)
        right = BoundaryCondition.from_input("right", req.right_boundary, req.temperature_unit)
        super().__init__(length_to_si(req.length, req.length_unit), req.nodes, left, right, req.boundary_scheme)
        self.alpha = self.k / (self.rho * self.cp)

    # ------------------------------------------------------------ time setup
    def time_parameters(self) -> tuple[float, float, int, list[str]]:
        """Return (Δt, Fo, number of steps, notes)."""
        dx2 = self.mesh.dx ** 2
        notes: list[str] = []
        if self.req.time_step_mode == "manual":
            dt = float(self.req.time_step)
            fo = self.alpha * dt / dx2
        else:
            fo = float(self.req.fourier_number)
            dt = fo * dx2 / self.alpha
        if self.req.num_steps is not None:
            steps = int(self.req.num_steps)
        elif self.req.final_time is not None:
            steps = max(1, math.ceil(self.req.final_time / dt - 1e-9))
            if abs(steps * dt - self.req.final_time) > 1e-9 * max(1.0, self.req.final_time):
                notes.append(f"Final time rounded up to a whole number of steps: {steps} × Δt = {steps * dt:.6g} s "
                             f"(requested {self.req.final_time:.6g} s).")
        else:
            steps = 10
            notes.append("No end time given; running 10 time steps.")
        return dt, fo, steps, notes

    def initial_field(self) -> np.ndarray:
        ic = self.req.initial_condition
        if ic.type == "custom" and ic.values:
            return np.array(ic.values, dtype=float)
        return np.full(self.mesh.nodes, float(ic.value))

    def output_interval(self, steps: int) -> int:
        if self.req.output_interval:
            interval = int(self.req.output_interval)
        else:
            interval = max(1, math.ceil(steps / settings.max_snapshots))
        cap = math.ceil(steps * self.mesh.nodes / settings.max_snapshot_values)
        return max(interval, cap, 1)

    # ------------------------------------------------------------------ solve
    def solve(self) -> SimulationResult:
        t0 = time.perf_counter()
        report = self.report

        mesh, n = self.mesh, self.mesh.nodes
        dt, fo, steps, notes = self.time_parameters()
        system = self.assembler.assemble_transient(fo, self.k, self.q_gen)
        A, tc, d = system.A, system.time_coeff, system.b_const

        gsolver = GaussianEliminationSolver(record_limit=settings.record_limit)
        T_prev = self.initial_field()
        T0 = T_prev.copy()

        # --- first time step: full, recorded Gaussian elimination
        b1 = tc * T_prev + d
        first = gsolver.solve(A, b1, record=self.req.record_steps)
        record = gsolver.factorize(A) if steps > 1 else None

        interval = self.output_interval(steps)
        times = [0.0]
        snapshots = [T0.tolist()]
        engine = ValidationEngine()
        max_res = 0.0
        max_bc = 0.0
        steady_at: float | None = None
        scale = max(1.0, float(np.max(np.abs(T0))))

        T = first.solution
        for step in range(1, steps + 1):
            if step > 1:
                b = tc * T_prev + d
                T = record.solve(b)
            else:
                b = b1
            Ab = tridiagonal_matvec(system.lower, system.diag, system.upper, T)
            res = Ab - b
            max_res = max(max_res, float(np.max(np.abs(res))))
            le = engine.boundary_error(self.left, None, b, T, 0, float(Ab[0]))
            re = engine.boundary_error(self.right, None, b, T, n - 1, float(Ab[-1]))
            max_bc = max(max_bc, le, re)
            if not np.all(np.isfinite(T)):
                raise SolverError(f"Non-finite temperatures appeared at time step {step}.")
            if steady_at is None and float(np.max(np.abs(T - T_prev))) < 1e-9 * scale:
                steady_at = step * dt
            if step % interval == 0 or step == steps:
                times.append(step * dt)
                snapshots.append(T.tolist())
            T_prev = T

        T_final = T
        b_last = tc * T_prev + d  # only for scale in the validation block
        validation = engine.check(
            residual=np.array([max_res]), b=b_last, T=T_final, n_expected=n,
            left_error=max_bc, right_error=max_bc,
            extra=[{"name": "Time steps completed", "passed": True, "detail": f"{steps} of {steps}"}],
        )
        validation["checks"][2]["name"] = "Max residual over all time steps"
        validation["checks"][3]["detail"] = validation["checks"][4]["detail"] = \
            f"max boundary residual over all steps {max_bc:.3e}"
        validation["residual_first_step"] = (A @ first.solution - b1).tolist() if n <= 200 else None
        elapsed = (time.perf_counter() - t0) * 1000

        explanation = ExplanationBuilder(self, system, first, first.solution, extra={
            "dt": dt, "fo": fo, "steps": steps, "T0": T0, "b1": b1}).transient()

        if steady_at is not None:
            notes.append(f"Temperature field reached steady state (max change < 1e-9) at t ≈ {steady_at:.6g} s.")

        summary = {
            "title": "1-D Transient Heat Conduction (Fully Implicit)",
            "length_m": mesh.length,
            "nodes": n,
            "intervals": mesh.intervals,
            "dx": mesh.dx,
            "alpha": self.alpha,
            "dt": dt,
            "fourier": fo,
            "time_steps": steps,
            "final_time": steps * dt,
            "left_bc": self.left.describe(),
            "right_bc": self.right.describe(),
            "temperature_unit": self.req.temperature_unit,
            "unit_symbol": TEMP_SYMBOL[self.req.temperature_unit],
            "t_min": float(min(min(s) for s in snapshots)),
            "t_max": float(max(max(s) for s in snapshots)),
            "scheme": self.scheme.value,
            "output_interval": interval,
            "steady_state_time": steady_at,
        }
        payload = {
            "dx": mesh.dx,
            "length_m": mesh.length,
            "alpha": self.alpha,
            "dt": dt,
            "fourier": fo,
            "num_steps": steps,
            "final_time": steps * dt,
            "x": mesh.x.tolist(),
            "initial": T0.tolist(),
            "nodes": self.node_table(T_final),
            "solution": T_final.tolist(),
            "first_step_solution": first.solution.tolist(),
            "times": times,
            "snapshots": snapshots,
            "time_coefficients": tc.tolist(),
            "rhs_constant": d.tolist(),
            **serialize_system(system, b1),
            "gaussian": serialize_gaussian(first),
            "validation": validation,
            "explanation": explanation,
            "boundary_conditions": {
                "left": {"description": self.left.describe(), "latex": self.left.condition_latex(),
                         "type": self.left.type.value, "value": self.left.value},
                "right": {"description": self.right.describe(), "latex": self.right.condition_latex(),
                          "type": self.right.type.value, "value": self.right.value},
            },
        }
        return SimulationResult("transient", self.req.model_dump(mode="json"), summary, payload,
                                report.warnings + notes + first.warnings, round(elapsed, 3))
