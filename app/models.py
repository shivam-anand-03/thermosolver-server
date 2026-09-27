"""Pydantic request models for the ThermoSolve API.

Structural validation (types, enums) is done by Pydantic. Physical validation
("length must be positive", "at least one boundary must fix temperature", …) is
done by :mod:`app.solver.validation` so the error messages read like an
engineer wrote them.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BoundaryType(str, Enum):
    """Supported boundary-condition types."""

    temperature = "temperature"  # Dirichlet: T = value
    heat_flux = "heat_flux"  # q'' = value (positive = heat ENTERING the slab)
    gradient = "gradient"  # dT/dx = value
    adiabatic = "adiabatic"  # q'' = 0


class BoundaryScheme(str, Enum):
    """How derivative (Neumann/flux) boundary conditions are discretised."""

    first_order = "first_order"  # one-sided difference (T2 - T1)/Δx — course method
    ghost_node = "ghost_node"  # second-order fictitious-node / half-cell method


LengthUnit = Literal["m", "mm", "cm"]
TemperatureUnit = Literal["C", "K"]


class BoundaryConditionInput(BaseModel):
    """A boundary condition at x = 0 or x = L."""

    type: BoundaryType
    value: float = 0.0


class _BaseRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    length: float = Field(description="Slab length, in `length_unit`.")
    length_unit: LengthUnit = "m"
    nodes: int = Field(description="Number of nodes N (Δx = L/(N-1)).")
    thermal_conductivity: float = Field(description="k [W/m·K]")
    heat_generation: float = Field(0.0, description="q''' [W/m³]")
    left_boundary: BoundaryConditionInput
    right_boundary: BoundaryConditionInput
    temperature_unit: TemperatureUnit = "K"
    boundary_scheme: BoundaryScheme = BoundaryScheme.first_order
    record_steps: bool | None = Field(
        None, description="Force/disable storing every augmented matrix (default: auto for N ≤ 12)."
    )


class SteadyRequest(_BaseRequest):
    """Input for a 1-D steady-state conduction problem."""


class InitialCondition(BaseModel):
    """Initial temperature field T(x, 0)."""

    type: Literal["uniform", "custom"] = "uniform"
    value: float = 30.0
    values: list[float] | None = None


class TransientRequest(_BaseRequest):
    """Input for a 1-D transient conduction problem (fully implicit scheme)."""

    temperature_unit: TemperatureUnit = "C"
    density: float = Field(description="ρ [kg/m³]")
    specific_heat: float = Field(description="Cp [J/kg·K]")
    initial_condition: InitialCondition = InitialCondition()
    time_step_mode: Literal["fourier", "manual"] = "fourier"
    fourier_number: float = 1.0
    time_step: float | None = Field(None, description="Δt [s] when time_step_mode = manual")
    num_steps: int | None = Field(None, description="Number of time steps (takes priority over final_time).")
    final_time: float | None = Field(None, description="Simulation end time [s].")
    output_interval: int | None = Field(None, description="Store every k-th time step (auto if omitted).")


class SolveEnvelope(BaseModel):
    """Body for POST /api/validate — either problem type."""

    problem_type: Literal["steady", "transient"]
    steady: SteadyRequest | None = None
    transient: TransientRequest | None = None


class ReportRequest(BaseModel):
    """Body for POST /api/report/pdf — the request plus its problem type."""

    problem_type: Literal["steady", "transient"]
    steady: SteadyRequest | None = None
    transient: TransientRequest | None = None
