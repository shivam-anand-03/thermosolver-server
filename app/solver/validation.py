"""Input validation (before solving) and solution validation (after solving)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..config import settings
from ..models import BoundaryType, SteadyRequest, TransientRequest
from .boundary_conditions import BoundaryCondition
from .utils import InputValidationError, is_finite, length_to_si, pretty_text, sci

ABS_ZERO = {"C": -273.15, "K": 0.0}


@dataclass
class ValidationReport:
    """Outcome of input validation: blocking errors and non-blocking warnings."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_errors(self) -> None:
        if self.errors:
            errors = [pretty_text(e) for e in self.errors]
            raise InputValidationError(errors[0], errors)

    def to_dict(self) -> dict:
        return {"valid": self.ok, "errors": [pretty_text(e) for e in self.errors],
                "warnings": [pretty_text(w) for w in self.warnings]}


class InputValidator:
    """Physical and numerical sanity checks with engineer-friendly messages."""

    # ---------------------------------------------------------------- shared
    def _common(self, req: SteadyRequest | TransientRequest, rep: ValidationReport) -> None:
        numbers = {
            "Length": req.length,
            "Thermal conductivity": req.thermal_conductivity,
            "Heat generation": req.heat_generation,
            "Left boundary value": req.left_boundary.value,
            "Right boundary value": req.right_boundary.value,
        }
        for name, v in numbers.items():
            if not is_finite(v):
                rep.errors.append(f"{name} must be a finite number (NaN/∞ is not allowed).")
        if is_finite(req.length) and req.length <= 0:
            rep.errors.append("Length must be greater than zero.")
        if req.nodes < 2:
            rep.errors.append("Number of nodes must be at least 2.")
        elif req.nodes > settings.max_nodes:
            rep.errors.append(f"Number of nodes must not exceed {settings.max_nodes}.")
        if is_finite(req.thermal_conductivity) and req.thermal_conductivity <= 0:
            rep.errors.append("Thermal conductivity must be positive.")

        tmin = ABS_ZERO[req.temperature_unit]
        for side, bc in (("Left", req.left_boundary), ("Right", req.right_boundary)):
            if bc.type == BoundaryType.temperature and is_finite(bc.value) and bc.value < tmin:
                rep.errors.append(f"{side} boundary temperature is below absolute zero.")
            if bc.type == BoundaryType.adiabatic and bc.value not in (0, 0.0):
                rep.warnings.append(f"{side} boundary is adiabatic; the entered value is ignored (q'' = 0).")

    # ---------------------------------------------------------------- steady
    def validate_steady(self, req: SteadyRequest) -> ValidationReport:
        rep = ValidationReport()
        self._common(req, rep)
        if rep.errors:
            return rep
        left = BoundaryCondition.from_input("left", req.left_boundary, req.temperature_unit)
        right = BoundaryCondition.from_input("right", req.right_boundary, req.temperature_unit)
        if not left.is_dirichlet and not right.is_dirichlet:
            k = req.thermal_conductivity
            L = length_to_si(req.length, req.length_unit)
            q_in = left.heat_in(k) + right.heat_in(k)
            q_gen = req.heat_generation * L
            net = q_in + q_gen
            scale = max(abs(left.heat_in(k)), abs(right.heat_in(k)), abs(q_gen), 1e-30)
            balance = (
                f"Heat balance per unit area: q''in,left + q''in,right + q'''·L = "
                f"{sci(left.heat_in(k))} + {sci(right.heat_in(k))} + {sci(q_gen)} = {sci(net)} W/m²."
            )
            if abs(net) <= 1e-9 * scale:
                rep.errors.append(
                    "At least one boundary condition must constrain the temperature field. "
                    "The flux conditions are compatible (net heat input is zero), but a steady "
                    "solution is then only defined up to an arbitrary constant. Specify a "
                    "temperature at one boundary."
                )
            else:
                rep.errors.append(
                    "Neumann-Neumann boundary conditions require a compatible heat balance. "
                    "The net heat input is not zero, so no steady solution exists "
                    "(the slab would keep heating or cooling)."
                )
            rep.errors.append(balance)
        return rep

    # ------------------------------------------------------------- transient
    def validate_transient(self, req: TransientRequest) -> ValidationReport:
        rep = ValidationReport()
        self._common(req, rep)
        for name, v in (("Density", req.density), ("Specific heat", req.specific_heat),
                        ("Fourier number", req.fourier_number), ("Time step", req.time_step),
                        ("Final time", req.final_time), ("Initial temperature", req.initial_condition.value)):
            if not is_finite(v):
                rep.errors.append(f"{name} must be a finite number (NaN/∞ is not allowed).")
        if rep.errors:
            return rep
        if req.density <= 0:
            rep.errors.append("Density must be positive.")
        if req.specific_heat <= 0:
            rep.errors.append("Specific heat must be positive.")
        if req.time_step_mode == "manual":
            if req.time_step is None or req.time_step <= 0:
                rep.errors.append("Time step must be positive.")
        elif req.fourier_number <= 0:
            rep.errors.append("Fourier number must be positive.")
        if req.num_steps is not None and req.num_steps < 1:
            rep.errors.append("Number of time steps must be at least 1.")
        if req.num_steps is None and req.final_time is not None and req.final_time <= 0:
            rep.errors.append("Final simulation time must be greater than zero.")
        if req.output_interval is not None and req.output_interval < 1:
            rep.errors.append("Output interval must be at least 1 time step.")

        ic = req.initial_condition
        tmin = ABS_ZERO[req.temperature_unit]
        if ic.type == "custom":
            if not ic.values or len(ic.values) != req.nodes:
                rep.errors.append(f"Custom initial condition must provide exactly N = {req.nodes} nodal temperatures.")
            elif not all(is_finite(v) for v in ic.values):
                rep.errors.append("Custom initial temperatures must be finite numbers.")
            elif min(ic.values) < tmin:
                rep.errors.append("An initial temperature is below absolute zero.")
        elif ic.value < tmin:
            rep.errors.append("Initial temperature is below absolute zero.")
        if rep.errors:
            return rep

        # time-step / Fourier number diagnostics
        dx = length_to_si(req.length, req.length_unit) / (req.nodes - 1)
        alpha = req.thermal_conductivity / (req.density * req.specific_heat)
        if req.time_step_mode == "manual":
            dt = float(req.time_step)
            fo = alpha * dt / dx**2
        else:
            fo = req.fourier_number
            dt = fo * dx**2 / alpha
        steps = req.num_steps if req.num_steps is not None else (
            max(1, math.ceil(req.final_time / dt - 1e-9)) if req.final_time else 10)
        if steps > settings.max_time_steps:
            rep.errors.append(
                f"The run needs {steps} time steps (limit {settings.max_time_steps}). "
                "Increase Δt / Fo or shorten the simulation time.")
        elif steps * req.nodes > settings.max_node_steps:
            rep.errors.append(
                f"N × time steps = {steps * req.nodes:,} exceeds the limit of {settings.max_node_steps:,}. "
                "Use fewer nodes or a larger time step.")
        if fo > 0.5:
            rep.warnings.append(
                f"Fo = {sci(fo)} > 0.5. This would be unstable for the explicit scheme, but the fully "
                "implicit method used here is unconditionally stable for the diffusion equation.")
        if fo > 10:
            rep.warnings.append(
                "Fourier number is outside the recommended range for accuracy (Fo ≫ 1): the solution "
                "stays stable but early-time transients may be smeared. Consider a smaller Δt.")
        if fo < 1e-4:
            rep.warnings.append("Fourier number is very small; many time steps are needed to see any change.")
        return rep


class ValidationEngine:
    """Checks a computed solution: residual, boundary conditions, finiteness."""

    def __init__(self, tolerance: float = 1e-8) -> None:
        self.tolerance = tolerance

    def boundary_error(self, bc: BoundaryCondition, A: np.ndarray | None, b: np.ndarray, T: np.ndarray,
                       row: int, matvec_row: float | None = None) -> float:
        """Residual of the boundary equation row (for Dirichlet: |T - T_bc|)."""
        if bc.is_dirichlet:
            return abs(float(T[row]) - bc.value)
        lhs = matvec_row if matvec_row is not None else float(A[row] @ T)
        return abs(lhs - float(b[row]))

    def check(self, *, residual: np.ndarray, b: np.ndarray, T: np.ndarray, n_expected: int,
              left_error: float, right_error: float, extra: list[dict] | None = None) -> dict:
        """Assemble the validation block returned to the client."""
        finite = bool(np.all(np.isfinite(T)))
        res_max = float(np.max(np.abs(residual))) if residual.size else 0.0
        scale = max(1.0, float(np.max(np.abs(b))) if b.size else 1.0, float(np.max(np.abs(T))) if T.size else 1.0)
        tol = self.tolerance * scale
        bc_err = max(left_error, right_error)
        checks = [
            {"name": "Solution size", "passed": int(T.size) == n_expected,
             "detail": f"{T.size} temperatures solved for N = {n_expected} nodes"},
            {"name": "All temperatures finite", "passed": finite,
             "detail": "No NaN or infinite values" if finite else "Solution contains NaN/∞"},
            {"name": "Residual max|A·T − b|", "passed": res_max <= tol,
             "detail": f"{res_max:.3e} (tolerance {tol:.1e})"},
            {"name": "Left boundary condition", "passed": left_error <= tol,
             "detail": f"boundary residual {left_error:.3e}"},
            {"name": "Right boundary condition", "passed": right_error <= tol,
             "detail": f"boundary residual {right_error:.3e}"},
        ]
        if extra:
            checks.extend(extra)
        passed = all(c["passed"] for c in checks if c["passed"] is not None)
        return {
            "status": "validated" if passed else "failed",
            "message": "Numerical solution validated" if passed else "Validation failed — inspect the checks below",
            "residual_max": res_max,
            "residual_l2": float(np.linalg.norm(residual)),
            "residual_vector": residual.tolist() if residual.size <= 200 else None,
            "boundary_error": bc_err,
            "boundary_error_left": left_error,
            "boundary_error_right": right_error,
            "tolerance": tol,
            "all_finite": finite,
            "checks": checks,
        }
