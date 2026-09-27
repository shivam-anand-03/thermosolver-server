"""Pre-loaded example problems (including the ME46220 MST-1 reference problems).

Reference values are stored ONLY for comparison in the UI — the solver never
reads them. See the README ("Reference examples") for why the steady-state
reference values cannot be reproduced exactly.
"""

from __future__ import annotations

from typing import Any

STEADY_NOTE = (
    "Reference values as printed in the supplied MST-1 solution. The last interval matches the flux "
    "condition exactly (T5 − T4 = q''Δx/k = 125 K), but the interior values are not consistent with "
    "d²T/dx² + q'''/k = 0 for q''' = 0 (which gives a linear profile) nor with the printed 2.015 "
    "diagonal — their successive differences (58.86, 145.45, 134.23) cannot come from a single "
    "uniform stencil. ThermoSolve therefore computes the result consistently from the governing "
    "equation instead of forcing the printed numbers."
)
TRANSIENT_NOTE = (
    "Reference values for the first time step from the supplied MST-1 solution. They are reproduced "
    "exactly by the fully implicit scheme with Fo = 1 and the first-order one-sided adiabatic "
    "condition (T1 = T2)."
)


class ExampleManager:
    """Provides the catalogue of example problems."""

    def __init__(self) -> None:
        self._examples: list[dict[str, Any]] = [
            {
                "id": "mst-steady",
                "title": "MST-1 · Steady slab with boundary heat flux",
                "description": "L = 0.5 m, N = 5, k = 1 W/m·K, T(0) = 373 K, q''(L) = 1000 W/m² into the slab.",
                "problem_type": "steady",
                "is_reference": True,
                "request": {
                    "length": 0.5, "length_unit": "m", "nodes": 5,
                    "thermal_conductivity": 1.0, "heat_generation": 0.0,
                    "left_boundary": {"type": "temperature", "value": 373.0},
                    "right_boundary": {"type": "heat_flux", "value": 1000.0},
                    "temperature_unit": "K", "boundary_scheme": "first_order",
                },
                "reference": {"label": "Printed in MST-1 solution", "time": None,
                              "values": [373.0, 431.86, 577.31, 711.54, 836.54], "note": STEADY_NOTE},
            },
            {
                "id": "mst-transient",
                "title": "MST-1 · Transient slab, insulated / fixed temperature",
                "description": "L = 5 mm, N = 6, k = 0.25, ρ = 1300, Cp = 2000, T⁰ = 30 °C, adiabatic left, T(L) = 250 °C, Fo = 1.",
                "problem_type": "transient",
                "is_reference": True,
                "request": {
                    "length": 5.0, "length_unit": "mm", "nodes": 6,
                    "thermal_conductivity": 0.25, "density": 1300.0, "specific_heat": 2000.0,
                    "heat_generation": 0.0,
                    "left_boundary": {"type": "adiabatic", "value": 0.0},
                    "right_boundary": {"type": "temperature", "value": 250.0},
                    "temperature_unit": "C", "boundary_scheme": "first_order",
                    "initial_condition": {"type": "uniform", "value": 30.0},
                    "time_step_mode": "fourier", "fourier_number": 1.0, "num_steps": 40,
                },
                "reference": {"label": "MST-1 solution, first time step (t = Δt)", "time_step": 1,
                              "values": [36.47, 36.47, 42.94, 62.36, 114.12, 250.0], "note": TRANSIENT_NOTE},
            },
            {
                "id": "generation-wall",
                "title": "Wall with internal heat generation",
                "description": "L = 0.1 m, N = 11, k = 20 W/m·K, q''' = 5×10⁵ W/m³, both faces at 100 °C.",
                "problem_type": "steady",
                "is_reference": False,
                "request": {
                    "length": 0.1, "length_unit": "m", "nodes": 11,
                    "thermal_conductivity": 20.0, "heat_generation": 5e5,
                    "left_boundary": {"type": "temperature", "value": 100.0},
                    "right_boundary": {"type": "temperature", "value": 100.0},
                    "temperature_unit": "C", "boundary_scheme": "first_order",
                },
                "reference": None,
            },
            {
                "id": "quench",
                "title": "Steel plate — sudden surface cooling",
                "description": "L = 20 mm, N = 21, steel (k = 45, ρ = 7850, Cp = 460), T⁰ = 500 °C, both faces at 25 °C.",
                "problem_type": "transient",
                "is_reference": False,
                "request": {
                    "length": 20.0, "length_unit": "mm", "nodes": 21,
                    "thermal_conductivity": 45.0, "density": 7850.0, "specific_heat": 460.0,
                    "heat_generation": 0.0,
                    "left_boundary": {"type": "temperature", "value": 25.0},
                    "right_boundary": {"type": "temperature", "value": 25.0},
                    "temperature_unit": "C", "boundary_scheme": "first_order",
                    "initial_condition": {"type": "uniform", "value": 500.0},
                    "time_step_mode": "manual", "time_step": 0.1, "final_time": 30.0,
                },
                "reference": None,
            },
        ]

    def list(self) -> list[dict[str, Any]]:
        return self._examples

    def get(self, example_id: str) -> dict[str, Any] | None:
        return next((e for e in self._examples if e["id"] == example_id), None)
