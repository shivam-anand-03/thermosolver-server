"""Runs the three phases for one :class:`SimulationSetup` and packages the results.

Phase I   — validate the problem statement, generate the grid, derive Pe, α, Fo …
Phase II  — discretise, assemble [A]{T} = {B}, initialise, solve (marching in time
            with the θ-scheme for unsteady problems)
Phase III — node table with the exact solution, error norms and solver diagnostics
"""

from __future__ import annotations

import math
import time

import numpy as np

from ..config import settings
from ..solver.utils import InputValidationError, SolverError, latex_num, pretty_latex, sci
from .discretization import DIRICHLET_TYPES, Discretized, PhysicalModel, discretize
from .exact import ExactSolution, find_exact
from .grid import FACE_AXIS, SparseMatrix
from .linear_solvers import (
    LinearSolveResult,
    explicit_lu,
    gauss_elimination,
    gauss_seidel,
    lu_factorize,
    lu_solve,
    optimal_omega,
)
from .schema import SimulationSetup

ABS_ZERO = {"C": -273.15, "K": 0.0}
THETA = {"implicit": 1.0, "crank_nicolson": 0.5, "explicit": 0.0}
FACE_LABEL = {"x_min": "x = 0", "x_max": "x = L", "y_min": "y = 0", "y_max": "y = W", "z_min": "z = 0", "z_max": "z = H"}
BC_LABEL = {"temperature": "Temperature specified", "heat_flux": "Heat flux specified",
            "convection": "Convective heat transfer", "insulated": "Wall (insulated)",
            "inflow": "Inflow", "outflow": "Outflow"}
METHOD_LABEL = {"fdm": "Finite difference method", "fvm": "Finite volume method", "fem": "Finite element method"}
SOLVER_LABEL = {"gauss_seidel": "Gauss–Seidel", "gauss_elimination": "Gauss elimination", "lu": "LU decomposition"}


# ======================================================================= Phase I
def validate_setup(setup: SimulationSetup) -> tuple[list[str], list[str]]:
    pre, sol = setup.preprocessor, setup.solver
    prob, geo = pre.problem, pre.geometry
    dim, physics = prob.dimension, prob.physics
    errors: list[str] = []
    warnings: list[str] = []

    def finite(*vals: float) -> bool:
        return all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)

    lengths = (geo.length, geo.width, geo.height)[:dim]
    counts = (geo.nx, geo.ny, geo.nz)[:dim]
    for name, L in zip("LWH", lengths):
        if not finite(L) or L <= 0:
            errors.append(f"{name} must be a positive length.")
    for ax, N in zip("xyz", counts):
        if N < 2:
            errors.append(f"N{ax} must be at least 2.")
        elif N > settings.engine_max_per_axis:
            errors.append(f"N{ax} must not exceed {settings.engine_max_per_axis}.")
    n = int(np.prod(counts))
    limit = settings.engine_max_unknowns_iterative if sol.linear_solver == "gauss_seidel" \
        else settings.engine_max_unknowns_direct
    if n > limit:
        errors.append(f"The grid has {n:,} unknowns; {SOLVER_LABEL[sol.linear_solver]} is limited to {limit:,}. "
                      "Use fewer nodes" + (" or Gauss–Seidel." if sol.linear_solver != "gauss_seidel" else "."))

    mat = pre.solid if physics == "conduction" else pre.fluid
    if not finite(mat.k, mat.rho, mat.cp, prob.source.q0, prob.source.beta):
        errors.append("Material properties and q''' coefficients must be finite numbers.")
    else:
        if mat.rho <= 0 or mat.cp <= 0:
            errors.append(f"{'Solid' if physics == 'conduction' else 'Fluid'} density and specific heat must be positive.")
        if physics != "convection" and mat.k <= 0:
            errors.append("Thermal conductivity must be positive.")

    vel = (pre.velocity.u, pre.velocity.v, pre.velocity.w)[:dim]
    if physics == "conduction" and any(vel):
        warnings.append("Conduction mode: the velocity field is ignored.")
    if physics != "conduction" and not any(vel):
        errors.append("Convective modes need a non-zero velocity component.")
    if physics == "convection" and sol.discretization == "fem":
        errors.append("Pure convection is unstable with the Galerkin finite element method; use FDM or FVM.")

    tmin = ABS_ZERO[pre.temperature_unit]
    faces = [f for f in FACE_AXIS if FACE_AXIS[f][0] < dim]
    has_level = prob.source.beta < 0
    for f in faces:
        bc = pre.boundaries.get(f)
        if bc is None:
            continue
        if not finite(bc.value, bc.h, bc.t_inf):
            errors.append(f"Boundary {FACE_LABEL[f]}: values must be finite.")
            continue
        if bc.type in DIRICHLET_TYPES:
            has_level = True
            if bc.value < tmin:
                errors.append(f"Boundary {FACE_LABEL[f]}: temperature is below absolute zero.")
        if bc.type == "convection":
            if bc.h <= 0:
                errors.append(f"Boundary {FACE_LABEL[f]}: the heat transfer coefficient h must be positive.")
            has_level = True
            if bc.t_inf < tmin:
                errors.append(f"Boundary {FACE_LABEL[f]}: T∞ is below absolute zero.")
        if physics == "conduction" and bc.type in ("inflow", "outflow"):
            warnings.append(f"Boundary {FACE_LABEL[f]}: in conduction mode an {bc.type} face acts as "
                            + ("a fixed temperature." if bc.type == "inflow" else "an insulated wall."))
    if physics == "convection":
        for a, u in enumerate(vel):
            if u == 0:
                continue
            f = ("xyz"[a] + "_min") if u > 0 else ("xyz"[a] + "_max")
            bc = pre.boundaries.get(f)
            if bc is None or bc.type not in DIRICHLET_TYPES:
                errors.append(f"Pure convection: fluid enters through {FACE_LABEL[f]}, so that face needs an "
                              "inflow (or temperature) condition.")
    if prob.time_mode == "steady" and not has_level:
        errors.append("The steady temperature level is undetermined: specify a temperature, inflow or convective "
                      "boundary on at least one face.")
    if prob.source.beta > 0:
        warnings.append("β > 0: generation grows with temperature. A steady solution exists only below the "
                        "thermal-runaway limit.")

    if prob.time_mode == "transient":
        if not finite(sol.time_step) or sol.time_step <= 0:
            errors.append("Time step Δt must be positive.")
        if sol.num_steps < 1:
            errors.append("Number of time steps must be at least 1.")
        elif sol.num_steps * n > settings.engine_max_node_steps:
            errors.append(f"Nodes × time steps = {sol.num_steps * n:,} exceeds {settings.engine_max_node_steps:,}.")
        if pre.initial_temperature < tmin:
            errors.append("Initial temperature is below absolute zero.")

    if sol.linear_solver == "gauss_seidel":
        if not (sol.tolerance > 0):
            errors.append("Convergence tolerance must be positive.")
        if not (1 <= sol.max_iterations <= 200_000):
            errors.append("Maximum iterations must be between 1 and 200,000.")
        if not sol.auto_relaxation and not (0 < sol.relaxation < 2):
            errors.append("Relaxation factor ω must satisfy 0 < ω < 2.")

    if not errors:
        m = PhysicalModel.from_setup(setup)
        kind = "cell" if sol.discretization == "fvm" else "vertex"
        h = [L / (N if kind == "cell" else N - 1) for L, N in zip(lengths, counts)]
        if m.k > 0 and any(vel) and sol.advection_scheme == "central":
            pe = max(abs(m.rho_cp * u) * hi / m.k for u, hi in zip(vel, h))
            if pe > 2:
                warnings.append(f"Cell Péclet number {pe:.3g} > 2 with central differencing: expect wiggles and a "
                                "non-diagonally-dominant matrix. Upwind or hybrid is safer.")
        if physics == "convection" and sol.advection_scheme != "upwind":
            warnings.append("Pure convection always uses first-order upwind differencing.")
        if prob.time_mode == "transient" and sol.time_scheme == "explicit" and finite(sol.time_step) and sol.time_step > 0:
            alpha = m.alpha
            lim_d = 1 / (2 * alpha * sum(1 / x**2 for x in h)) if alpha > 0 else math.inf
            lim_c = 1 / sum(abs(u) / x for u, x in zip(m.velocity, h)) if any(m.velocity) else math.inf
            lim = min(lim_d, lim_c)
            if sol.time_step > lim:
                warnings.append(f"Explicit scheme: Δt = {sci(sol.time_step)} s exceeds the stability limit "
                                f"≈ {sci(lim)} s. The solution will blow up; use the implicit scheme or reduce Δt.")
    return errors, warnings


def governing_latex(physics: str, transient: bool, dim: int) -> str:
    lap = {1: r"\frac{\partial^2 T}{\partial x^2}",
           2: r"\left(\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2}\right)",
           3: r"\nabla^2 T"}[dim]
    adv = {1: r"u\frac{\partial T}{\partial x}", 2: r"u\frac{\partial T}{\partial x} + v\frac{\partial T}{\partial y}",
           3: r"\mathbf{u}\cdot\nabla T"}[dim]
    storage = r"\rho c_p\frac{\partial T}{\partial t}"
    src = r"q_0 + \beta T"
    if physics == "conduction":
        return (storage + " = " if transient else "0 = ") + rf"k\,{lap} + {src}"
    if physics == "convection":
        lhs = rf"\rho c_p\left(\frac{{\partial T}}{{\partial t}} + {adv}\right)" if transient else rf"\rho c_p\left({adv}\right)"
        return rf"{lhs} = {src}"
    lhs = rf"\rho c_p\left(\frac{{\partial T}}{{\partial t}} + {adv}\right)" if transient else rf"\rho c_p\left({adv}\right)"
    return rf"{lhs} = k\,{lap} + {src}"


# ====================================================================== helpers
NEIGHBOUR = {(0, -1): "W", (0, 1): "E", (1, -1): "S", (1, 1): "N", (2, -1): "B", (2, 1): "T"}


def _neighbour_label(d: np.ndarray) -> str:
    if not np.any(d):
        return "P"
    parts = [NEIGHBOUR[(a, int(np.sign(d[a])))] * abs(int(d[a])) for a in (2, 1, 0) if a < d.size and d[a]]
    return "".join(parts)


def _sample_equation(A: SparseMatrix, b: np.ndarray, disc: Discretized) -> dict | None:
    grid = disc.grid
    centre = [N // 2 for N in grid.counts]
    free = np.nonzero(~disc.dirichlet)[0]
    if free.size == 0:
        return None
    ijk = grid.multi_index
    dist = np.abs(ijk[free] - np.array(centre)).sum(axis=1)
    p = int(free[np.argmin(dist)])
    m = A.rows == p
    cols, vals = A.cols[m], A.vals[m]
    terms = []
    for j, v in sorted(zip(cols.tolist(), vals.tolist()), key=lambda t: (t[0] != p, t[0])):
        lab = _neighbour_label(ijk[j] - ijk[p])
        terms.append({"label": lab, "coefficient": v, "node": j + 1})
    latex_parts = []
    for i, t in enumerate(terms):
        v = t["coefficient"]
        mag = latex_num(abs(v), 5)
        sign = ("-" if v < 0 else "") if i == 0 else (" - " if v < 0 else " + ")
        latex_parts.append(rf"{sign}{mag}\,T_{{{t['label']}}}")
    latex = pretty_latex("".join(latex_parts) + f" = {latex_num(float(b[p]), 6)}")
    return {"node": p + 1, "ijk": (ijk[p] + 1).tolist(), "terms": terms, "rhs": float(b[p]), "latex": latex}


def _initial_guess(setup: SimulationSetup, disc: Discretized) -> np.ndarray:
    sol = setup.solver
    pre = setup.preprocessor
    n = disc.grid.n
    if sol.initial_guess == "zero":
        x0 = np.zeros(n)
    elif sol.initial_guess == "bc_mean":
        vals = [bc.value for bc in pre.boundaries.values() if bc.type in DIRICHLET_TYPES] + \
               [bc.t_inf for bc in pre.boundaries.values() if bc.type == "convection"]
        x0 = np.full(n, float(np.mean(vals)) if vals else sol.initial_guess_value)
    else:
        x0 = np.full(n, sol.initial_guess_value)
    x0[disc.dirichlet] = disc.g[disc.dirichlet]
    return x0


class _LinearSolver:
    """Applies the chosen matrix-inversion technique, re-using work across time steps where valid."""

    def __init__(self, setup: SimulationSetup, A: SparseMatrix, counts: tuple[int, ...]) -> None:
        s = setup.solver
        self.setup, self.A, self.method = setup, A, s.linear_solver
        self.omega = optimal_omega(counts) if s.auto_relaxation else s.relaxation
        self._lu = None
        self._dense = None
        self._rows = None
        self.factor_ms = 0.0
        if self.method == "lu":
            t0 = time.perf_counter()
            self._lu = lu_factorize(A)
            self.factor_ms = (time.perf_counter() - t0) * 1000
        elif self.method == "gauss_elimination":
            self._dense = A.to_dense()
        else:
            self._rows = A.row_lists()

    def solve(self, b: np.ndarray, x0: np.ndarray) -> LinearSolveResult:
        s = self.setup.solver
        if self.method == "lu":
            return lu_solve(self.A, self._lu, b)
        if self.method == "gauss_elimination":
            return gauss_elimination(self.A, b, self._dense)
        return gauss_seidel(self.A, b, x0, tol=s.tolerance, max_iter=s.max_iterations, omega=self.omega,
                            criterion=s.criterion, rows=self._rows)


def _errors(T: np.ndarray, Te: np.ndarray, mask: np.ndarray) -> dict:
    d = (T - Te)[mask]
    if d.size == 0:
        return {}
    ref = float(np.max(Te[mask]) - np.min(Te[mask]))
    ref = ref if ref > 1e-12 else max(float(np.max(np.abs(Te[mask]))), 1e-12)
    max_abs = float(np.max(np.abs(d)))
    return {"max_abs": max_abs, "rms": float(np.sqrt(np.mean(d * d))), "mean_abs": float(np.mean(np.abs(d))),
            "relative_pct": 100 * max_abs / ref, "reference_range": ref, "points": int(d.size),
            "excluded": int(mask.size - d.size)}


# ====================================================================== run
def run(setup: SimulationSetup) -> dict:
    t_start = time.perf_counter()
    errors, warnings = validate_setup(setup)
    if errors:
        raise InputValidationError(errors[0], errors)

    pre, sol = setup.preprocessor, setup.solver
    transient = pre.problem.time_mode == "transient"
    log: list[dict] = []

    def note(phase: str, msg: str) -> None:
        log.append({"t_ms": round((time.perf_counter() - t_start) * 1000, 2), "phase": phase, "message": msg})

    # ---------------- Phase I
    t0 = time.perf_counter()
    model, disc = discretize(setup)
    grid = disc.grid
    assemble_ms = (time.perf_counter() - t0) * 1000
    note("Preprocessor", f"{grid.dim}-D {'cell-centred' if grid.kind == 'cell' else 'vertex'} grid: "
                         f"{' × '.join(map(str, grid.counts))} = {grid.n:,} nodes, "
                         f"h = {', '.join(sci(h, 4) for h in grid.spacing)} m")
    nd = int(disc.dirichlet.sum())
    note("Solver", f"{METHOD_LABEL[disc.method]}: assembled {grid.n:,} equations"
                   + (f" ({nd} fixed-temperature nodes)" if nd else ""))
    warnings.extend(disc.notes[1:])

    derived = _derived(model, grid, setup)

    # ---------------- Phase II
    dirichlet_rows = np.nonzero(disc.dirichlet)[0]
    I_dir = SparseMatrix.identity_rows(grid.n, dirichlet_rows)
    x0 = _initial_guess(setup, disc)

    if not transient:
        A = disc.K + I_dir
        b = disc.F.copy()
        b[disc.dirichlet] = disc.g[disc.dirichlet]
        matrix_info = _matrix_info(A)
        note("Solver", f"[A]{{T}} = {{B}}: {grid.n:,} × {grid.n:,}, {A.nnz:,} non-zeros, "
                       f"bandwidth {matrix_info['bandwidth']}")
        if sol.linear_solver == "gauss_seidel" and not matrix_info["diagonally_dominant"]:
            warnings.append("The matrix is not diagonally dominant; Gauss–Seidel may converge slowly or diverge.")
        solver = _LinearSolver(setup, A, grid.counts)
        res = solver.solve(b, x0)
        T = res.x
        _note_solve(note, res, solver)
        if res.method == "gauss_seidel" and not res.converged:
            warnings.append(f"Gauss–Seidel stopped at the iteration limit ({sol.max_iterations}) before reaching the "
                            f"tolerance {sci(sol.tolerance)}; final value {sci(res.history[-1])}.")
        convergence = {"iterations": res.iterations, "converged": res.converged, "history": _thin(res.history),
                       "residual": res.residual, "omega": solver.omega if res.method == "gauss_seidel" else None,
                       "factor_ms": solver.factor_ms, "solve_ms": res.time_ms, "row_swaps": res.swaps}
        times = None
        snapshots = None
        monitor = None
        rhs_display = last_rhs = b
    else:
        theta = THETA[sol.time_scheme]
        dt = sol.time_step
        Cdt = disc.C / dt
        diagC = SparseMatrix(grid.n, np.arange(grid.n), np.arange(grid.n), Cdt)
        A = SparseMatrix.from_coo(grid.n, *_concat(diagC, disc.K.scale(theta), I_dir))
        Kexp = disc.K.scale(1.0 - theta)
        matrix_info = _matrix_info(A)
        note("Solver", f"θ-scheme (θ = {theta:g}), Δt = {sci(dt)} s, {sol.num_steps} steps; "
                       f"[A] is {grid.n:,} × {grid.n:,} with {A.nnz:,} non-zeros")
        solver = _LinearSolver(setup, A, grid.counts)
        T = np.full(grid.n, pre.initial_temperature, dtype=float)
        T[disc.dirichlet] = disc.g[disc.dirichlet]
        T_init = T.copy()
        interval = max(1, math.ceil(sol.num_steps / settings.engine_max_snapshots),
                       math.ceil(sol.num_steps * grid.n / settings.engine_max_snapshot_values))
        times, snapshots = [0.0], [_r(T_init)]
        centre = _monitor_node(disc)
        monitor = {"node": centre + 1, "coords": grid.node_coords[centre].tolist(),
                   "times": [0.0], "values": [float(T[centre])]}
        iters_per_step: list[int] = []
        first_history: list[float] = []
        total_solve = 0.0
        swaps = 0
        res = None
        rhs_display = None
        for step in range(1, sol.num_steps + 1):
            rhs = Cdt * T - Kexp.matvec(T) + disc.F
            rhs[disc.dirichlet] = disc.g[disc.dirichlet]
            if rhs_display is None:
                rhs_display = rhs.copy()
            res = solver.solve(rhs, T)
            last_rhs = rhs
            total_solve += res.time_ms
            swaps = res.swaps
            if res.iterations is not None:
                iters_per_step.append(res.iterations)
                if step == 1:
                    first_history = res.history
                if not res.converged:
                    warnings.append(f"Gauss–Seidel did not converge within {sol.max_iterations} iterations at step "
                                    f"{step}.")
                    break
            T = res.x
            if not np.all(np.isfinite(T)) or float(np.max(np.abs(T))) > 1e12:
                raise SolverError(f"The solution blew up at time step {step}. The explicit scheme is unstable for "
                                  "this Δt; use the implicit scheme or a smaller time step.")
            monitor["times"].append(step * dt)
            monitor["values"].append(float(T[centre]))
            if step % interval == 0 or step == sol.num_steps:
                times.append(step * dt)
                snapshots.append(_r(T))
        note("Solver", f"Marched {len(monitor['times']) - 1} time steps to t = {sci(monitor['times'][-1])} s "
                       f"({total_solve:.1f} ms in the linear solver)")
        if iters_per_step:
            note("Solver", f"Gauss–Seidel: {sum(iters_per_step):,} iterations in total "
                           f"(mean {np.mean(iters_per_step):.1f} per step)")
        convergence = {"iterations": int(sum(iters_per_step)) if iters_per_step else None,
                       "converged": bool(res.converged) if res else True,
                       "history": _thin(first_history), "iterations_per_step": _thin(iters_per_step),
                       "residual": res.residual if res else 0.0,
                       "omega": solver.omega if sol.linear_solver == "gauss_seidel" else None,
                       "factor_ms": solver.factor_ms, "solve_ms": total_solve, "row_swaps": swaps}

    # ---------------- Phase III
    final_time = monitor["times"][-1] if transient else None
    exact = find_exact(model, transient, pre.initial_temperature)
    coords = grid.node_coords
    exact_field = None
    excluded: list[int] = []
    err = {}
    if exact is not None:
        exact_field = exact.evaluate(coords, final_time)
        mask = np.ones(grid.n, bool) if exact.exclude is None else ~exact.exclude(coords)
        excluded = np.nonzero(~mask)[0].tolist()
        if transient and final_time == 0:
            mask[:] = False
        err = _errors(T, exact_field, mask)
        note("Post-processor", f"Exact solution ({exact.name}): max |error| = {sci(err.get('max_abs', 0))}, "
                               f"RMS = {sci(err.get('rms', 0))}")
        if transient:
            monitor["times"], monitor["values"] = _thin(monitor["times"], 600), _thin(monitor["values"], 600)
            point = coords[[monitor["node"] - 1]]
            monitor["exact"] = [float(exact.evaluate(point, t)[0]) for t in monitor["times"]]
            snaps_exact = [_r(exact.evaluate(coords, t)) for t in times]
        else:
            snaps_exact = None
    else:
        snaps_exact = None
        note("Post-processor", "No closed-form exact solution for this configuration; showing residual checks only.")

    residual_abs = float(np.max(np.abs(A.matvec(T) - last_rhs)))
    total_ms = (time.perf_counter() - t_start) * 1000
    note("Post-processor", f"Done in {total_ms:.1f} ms")

    A_dense_small = A.to_dense() if grid.n <= settings.engine_matrix_preview else None
    return {
        "success": True,
        "title": pre.problem.title,
        "warnings": list(dict.fromkeys(warnings)),
        "log": log,
        "execution_time_ms": round(total_ms, 2),
        "phase1": {
            "physics": model.physics,
            "time_mode": pre.problem.time_mode,
            "dimension": grid.dim,
            "governing_latex": governing_latex(model.physics, transient, grid.dim),
            "source": {"q0": model.q0, "beta": model.beta},
            "material": {"k": model.k, "rho_cp": model.rho_cp, "alpha": model.alpha},
            "velocity": list(model.velocity),
            "grid": {"kind": grid.kind, "counts": list(grid.counts), "lengths": list(grid.lengths),
                     "spacing": list(grid.spacing), "n": grid.n, "axis_coords": [c.tolist() for c in grid.axis_coords]},
            "boundaries": {f: {"type": bc.type, "label": BC_LABEL[bc.type], "face": FACE_LABEL[f],
                               "value": bc.value, "h": bc.h, "t_inf": bc.t_inf} for f, bc in model.bcs.items()},
            "derived": derived,
        },
        "phase2": {
            "discretization": disc.method,
            "discretization_label": METHOD_LABEL[disc.method],
            "advection": disc.advection_used,
            "time_scheme": sol.time_scheme if transient else None,
            "linear_solver": sol.linear_solver,
            "linear_solver_label": SOLVER_LABEL[sol.linear_solver],
            "matrix": {**matrix_info,
                       "dense": A_dense_small.tolist() if A_dense_small is not None else None,
                       "rhs": rhs_display.tolist() if A_dense_small is not None else None,
                       "pattern": _pattern(A) if grid.n <= settings.engine_pattern_limit else None,
                       "lu": explicit_lu(A_dense_small) if A_dense_small is not None and grid.n <= 12 else None},
            "sample_equation": _sample_equation(A, rhs_display, disc),
            "convergence": convergence,
            "assemble_ms": assemble_ms,
        },
        "phase3": {
            "coords": coords.tolist() if grid.dim > 1 else coords[:, 0].tolist(),
            "temperature": T.tolist(),
            "exact": exact_field.tolist() if exact_field is not None else None,
            "exact_info": {"available": exact is not None, "name": exact.name if exact else None,
                           "latex": exact.latex if exact else None},
            "errors": err,
            "stats": {"min": float(T.min()), "max": float(T.max()), "mean": float(T.mean())},
            "residual_max": residual_abs,
            "final_time": final_time,
            "times": times,
            "snapshots": snapshots,
            "snapshots_exact": snaps_exact,
            "monitor": monitor,
            "fixed_nodes": dirichlet_rows.tolist() if grid.n <= 20000 else None,
            "excluded_nodes": excluded,
        },
    }


def _note_solve(note, res: LinearSolveResult, solver: _LinearSolver) -> None:
    if res.method == "gauss_seidel":
        note("Solver", f"Gauss–Seidel (ω = {solver.omega:.4g}): {res.iterations} iterations, "
                       f"{'converged' if res.converged else 'NOT converged'}, relative residual {sci(res.residual)}")
    elif res.method == "lu":
        note("Solver", f"LU decomposition: factorised in {solver.factor_ms:.1f} ms, forward/back substitution "
                       f"{res.time_ms:.1f} ms, residual {sci(res.residual)}")
    else:
        note("Solver", f"Gauss elimination: {res.swaps} row swaps, {res.time_ms:.1f} ms, residual {sci(res.residual)}")


def _concat(*mats: SparseMatrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.concatenate([m.rows for m in mats]), np.concatenate([m.cols for m in mats]),
            np.concatenate([m.vals for m in mats]))


def _monitor_node(disc: Discretized) -> int:
    grid = disc.grid
    centre = np.array([N // 2 for N in grid.counts])
    return int(np.argmin(np.abs(grid.multi_index - centre).sum(axis=1)))


def _matrix_info(A: SparseMatrix) -> dict:
    weak, strict = A.diagonal_dominance()
    bl, bu = A.bandwidth()
    return {"n": A.n, "nnz": A.nnz, "bandwidth": f"{bl} / {bu}", "lower_bandwidth": bl, "upper_bandwidth": bu,
            "fill_pct": 100 * A.nnz / A.n**2, "symmetric": A.is_symmetric() if A.nnz <= 200_000 else None,
            "diagonally_dominant": weak and strict}


def _pattern(A: SparseMatrix) -> list[list[int]]:
    return np.stack([A.rows, A.cols], axis=1).tolist()


def _r(values) -> list[float]:
    """Round to 7 significant digits to keep the JSON payload compact."""
    return [float(f"{v:.7g}") for v in np.asarray(values, dtype=float).ravel().tolist()]


def _thin(values: list, limit: int = 2000) -> list:
    if len(values) <= limit:
        return list(values)
    step = math.ceil(len(values) / limit)
    out = list(values[::step])
    if out[-1] != values[-1]:
        out.append(values[-1])
    return out


def _derived(m: PhysicalModel, grid, setup: SimulationSetup) -> dict:
    out: dict = {"alpha": m.alpha, "rho_cp": m.rho_cp, "k": m.k}
    L = max(grid.lengths)
    speed = math.sqrt(sum(u * u for u in m.velocity))
    if speed and m.k > 0:
        out["peclet"] = m.rho_cp * speed * L / m.k
        out["cell_peclet"] = max(m.rho_cp * abs(u) * h / m.k for u, h in zip(m.velocity, grid.spacing))
    if setup.preprocessor.problem.time_mode == "transient":
        dt = setup.solver.time_step
        if m.alpha > 0:
            out["fourier_cell"] = m.alpha * dt * sum(1 / h**2 for h in grid.spacing)
        if speed:
            out["courant"] = dt * sum(abs(u) / h for u, h in zip(m.velocity, grid.spacing))
        out["final_time"] = dt * setup.solver.num_steps
    for f, bc in m.bcs.items():
        if bc.type == "convection" and m.k > 0:
            out.setdefault("biot", {})[f] = bc.h * grid.lengths[FACE_AXIS[f][0]] / m.k
    return out


# ============================================================ refinement study
def refinement_study(setup: SimulationSetup) -> dict:
    """Solve on successively halved grids and report the observed order of accuracy."""
    errors, _ = validate_setup(setup)
    if errors:
        raise InputValidationError(errors[0], errors)
    if setup.preprocessor.problem.time_mode != "steady":
        raise InputValidationError("The mesh refinement study is available for steady problems only.")
    base = setup.model_copy(deep=True)
    base.solver.linear_solver = "lu"
    model = PhysicalModel.from_setup(base)
    exact = find_exact(model, False, 0.0)
    if exact is None:
        raise InputValidationError("The refinement study needs an exact solution, which this configuration lacks.")
    dim = base.preprocessor.problem.dimension
    geo = base.preprocessor.geometry
    cell = base.solver.discretization == "fvm"
    counts = [geo.nx, geo.ny, geo.nz][:dim]
    # only refine the axes that carry the variation for effectively 1-D problems
    refine_axes = list(range(dim)) if dim == 1 or exact.exclude is not None else [0]
    refine = lambda cs: [(2 * c if cell else 2 * c - 1) if a in refine_axes else c for a, c in enumerate(cs)]  # noqa: E731
    coarsen = lambda cs: [(c // 2 if cell else (c + 1) // 2) if a in refine_axes else c for a, c in enumerate(cs)]  # noqa: E731
    fits = lambda cs: int(np.prod(cs)) <= settings.engine_max_unknowns_direct and max(cs) <= settings.engine_max_per_axis  # noqa: E731
    # start coarser when the requested grid leaves no room for at least two refinements
    while not fits(refine(refine(counts))) and min(counts[a] for a in refine_axes) >= 6:
        counts = coarsen(counts)
    levels = []
    for _ in range(4):
        n = int(np.prod(counts))
        if n > settings.engine_max_unknowns_direct or any(c > settings.engine_max_per_axis for c in counts):
            break
        s = base.model_copy(deep=True)
        for ax, c in zip("xyz", counts):
            setattr(s.preprocessor.geometry, f"n{ax}", c)
        r = run(s)
        h = r["phase1"]["grid"]["spacing"][0]
        levels.append({"counts": list(counts), "n": n, "h": h, "max_abs": r["phase3"]["errors"].get("max_abs"),
                       "rms": r["phase3"]["errors"].get("rms")})
        counts = refine(counts)
    for i in range(1, len(levels)):
        a, b = levels[i - 1], levels[i]
        for key in ("max_abs", "rms"):
            if a[key] and b[key] and a[key] > 0 and b[key] > 0:
                b[f"order_{key}"] = math.log(a[key] / b[key]) / math.log(a["h"] / b["h"])
    if len(levels) < 2:
        raise InputValidationError("The grid is too fine to refine further; start the study from a coarser grid.")
    return {"levels": levels, "method": base.solver.discretization}
