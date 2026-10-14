"""Input model for the three-phase workflow (Preprocessor → Solver → Post-processor).

The structure mirrors the course workflow sheet:

Phase I — Preprocessor
    I.   Problem statement: q''' = q0 + βT; diffusion / convection–diffusion /
         convection; steady / unsteady; 1-D / 2-D / 3-D
    II.  Geometry and grid generation: L, W, H and Nx, Ny, Nz
    III. Physical phenomena mode: conduction / convection–conduction / convection
    IV.  Material properties: fluid and solid
    V.   Initial and boundary conditions: temperature, heat flux, convective
         heat transfer; inflow, wall and outflow boundaries

Phase II — Solver
    I.   Discretisation technique: FDM / FVM / FEM
    II.  Matrix inversion technique for [A]{T} = {B}: Gauss–Seidel,
         Gauss elimination, LU decomposition
    III. Convergence criteria and initialisation
    IV.  Solve — run the calculation

Phase III — Post-processing
    I.   Results: Node | x | T_numerical | T_exact
    II.  Validation against the exact solution
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PhysicsMode = Literal["conduction", "convection_diffusion", "convection"]
TimeMode = Literal["steady", "transient"]
FaceName = Literal["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
BCType = Literal["temperature", "heat_flux", "convection", "insulated", "inflow", "outflow"]
Discretization = Literal["fdm", "fvm", "fem"]
AdvectionScheme = Literal["upwind", "central", "hybrid"]
TimeScheme = Literal["implicit", "crank_nicolson", "explicit"]
LinearSolver = Literal["gauss_seidel", "gauss_elimination", "lu"]
ConvergenceCriterion = Literal["residual", "max_change"]
InitialGuess = Literal["uniform", "zero", "bc_mean"]

FACES: tuple[FaceName, ...] = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


# ----------------------------------------------------------------- Phase I
class SourceTerm(_Model):
    """Volumetric heat generation q''' = q0 + β·T  [W/m³]."""

    q0: float = 0.0
    beta: float = Field(0.0, description="β [W/m³·K]; positive β makes generation grow with T.")


class ProblemStatement(_Model):
    title: str = "Untitled problem"
    physics: PhysicsMode = "conduction"
    time_mode: TimeMode = "steady"
    dimension: Literal[1, 2, 3] = 1
    source: SourceTerm = SourceTerm()


class Geometry(_Model):
    """Box domain [0, L] × [0, W] × [0, H] and the number of grid points per axis."""

    length: float = Field(1.0, description="L along x [m]")
    width: float = Field(1.0, description="W along y [m]")
    height: float = Field(1.0, description="H along z [m]")
    nx: int = 11
    ny: int = 11
    nz: int = 11


class Velocity(_Model):
    """Uniform prescribed velocity field [m/s] (used by the convective modes)."""

    u: float = 0.0
    v: float = 0.0
    w: float = 0.0


class Material(_Model):
    name: str = ""
    k: float = Field(1.0, description="Thermal conductivity [W/m·K]")
    rho: float = Field(1000.0, description="Density [kg/m³]")
    cp: float = Field(1000.0, description="Specific heat [J/kg·K]")


class FaceBC(_Model):
    """Boundary condition on one face of the box.

    * temperature — T = value
    * heat_flux   — conductive heat flux q'' = value entering the domain [W/m²]
    * convection  — q''_in = h (T∞ − T_s)
    * insulated   — wall, q'' = 0
    * inflow      — fluid enters at T = value
    * outflow     — fully developed outflow, ∂T/∂n = 0
    """

    type: BCType = "insulated"
    value: float = 0.0
    h: float = 10.0
    t_inf: float = 25.0


class Preprocessor(_Model):
    problem: ProblemStatement = ProblemStatement()
    geometry: Geometry = Geometry()
    velocity: Velocity = Velocity()
    solid: Material = Material(name="Solid", k=45.0, rho=7850.0, cp=460.0)
    fluid: Material = Material(name="Fluid", k=0.6, rho=1000.0, cp=4180.0)
    initial_temperature: float = 25.0
    boundaries: dict[FaceName, FaceBC] = Field(default_factory=dict)
    temperature_unit: Literal["C", "K"] = "C"


# ---------------------------------------------------------------- Phase II
class SolverSettings(_Model):
    discretization: Discretization = "fdm"
    advection_scheme: AdvectionScheme = "upwind"
    time_scheme: TimeScheme = "implicit"
    time_step: float = Field(1.0, description="Δt [s]")
    num_steps: int = 50
    linear_solver: LinearSolver = "gauss_elimination"
    tolerance: float = 1e-6
    max_iterations: int = 5000
    relaxation: float = Field(1.0, description="ω (1 = Gauss–Seidel, 1 < ω < 2 = SOR)")
    auto_relaxation: bool = False
    criterion: ConvergenceCriterion = "residual"
    initial_guess: InitialGuess = "uniform"
    initial_guess_value: float = 25.0


class SimulationSetup(_Model):
    preprocessor: Preprocessor = Preprocessor()
    solver: SolverSettings = SolverSettings()
