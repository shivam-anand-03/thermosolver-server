"""Three-phase workflow engine: discretisations, matrix solvers and exact-solution validation."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.engine.grid import SparseBuilder
from app.engine.linear_solvers import gauss_elimination, gauss_seidel, lu_factorize, lu_solve
from app.engine.runner import refinement_study, run
from app.engine.schema import SimulationSetup
from app.main import app
from app.solver.utils import InputValidationError, SolverError

METHODS = ("fdm", "fvm", "fem")
SOLVERS = ("gauss_elimination", "lu", "gauss_seidel")


def T(v):
    return {"type": "temperature", "value": v}


def setup(physics="conduction", dim=1, time_mode="steady", bcs=None, n=11, q0=0.0, beta=0.0, u=0.0, v=0.0,
          method="fdm", solver="gauss_elimination", **solver_kw):
    return SimulationSetup.model_validate({
        "preprocessor": {
            "problem": {"physics": physics, "dimension": dim, "time_mode": time_mode,
                        "source": {"q0": q0, "beta": beta}},
            "geometry": {"length": 1, "width": 1, "height": 1, "nx": n, "ny": n, "nz": n},
            "velocity": {"u": u, "v": v},
            "solid": {"k": 1, "rho": 1, "cp": 1}, "fluid": {"k": 1, "rho": 1, "cp": 1},
            "initial_temperature": 0.0,
            "boundaries": bcs or {},
        },
        "solver": {"discretization": method, "linear_solver": solver, "tolerance": 1e-10,
                   "max_iterations": 20000, **solver_kw},
    })


# ------------------------------------------------------------ matrix solvers
def _random_banded(n=30, seed=0):
    rng = np.random.default_rng(seed)
    B = SparseBuilder(n)
    for i in range(n):
        for j in range(max(0, i - 2), min(n, i + 3)):
            B.add(i, j, rng.normal() + (6.0 if i == j else 0.0))
    return B.build(), rng.normal(size=n)


def test_direct_solvers_match_numpy():
    A, b = _random_banded()
    ref = np.linalg.solve(A.to_dense(), b)
    assert np.allclose(gauss_elimination(A, b).x, ref)
    assert np.allclose(lu_solve(A, lu_factorize(A), b).x, ref)


def test_gauss_elimination_pivots():
    B = SparseBuilder(3)
    for i, j, v in [(0, 1, 1.0), (1, 0, 1.0), (1, 2, 1.0), (2, 1, 1.0), (2, 2, 1.0)]:
        B.add(i, j, v)
    A = B.build()
    b = np.array([1.0, 2.0, 3.0])
    res = gauss_elimination(A, b)
    assert res.swaps >= 1
    assert np.allclose(A.to_dense() @ res.x, b)


def test_singular_matrix_detected():
    B = SparseBuilder(2)
    for i, j, v in [(0, 0, 1.0), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 1.0)]:
        B.add(i, j, v)
    with pytest.raises(SolverError):
        gauss_elimination(B.build(), np.zeros(2))


def test_gauss_seidel_converges_and_sor_is_faster():
    A, b = _random_banded()
    plain = gauss_seidel(A, b, np.zeros(A.n), tol=1e-12, max_iter=5000)
    assert plain.converged
    assert np.allclose(plain.x, np.linalg.solve(A.to_dense(), b), atol=1e-9)
    assert plain.history[-1] < 1e-12


# ---------------------------------------------------- exact-solution checks
@pytest.mark.parametrize("method", ["fdm", "fem"])
@pytest.mark.parametrize("solver", SOLVERS)
def test_parabola_is_exact_for_fdm_and_fem(method, solver):
    r = run(setup(q0=10, bcs={"x_min": T(0), "x_max": T(0)}, method=method, solver=solver))
    assert r["phase3"]["exact_info"]["available"]
    assert r["phase3"]["errors"]["max_abs"] < 1e-8
    assert abs(max(r["phase3"]["temperature"]) - 10 / 8) < 1e-8  # q0 L²/8k


@pytest.mark.parametrize("method", METHODS)
def test_linear_source_q0_plus_beta_T(method):
    r = run(setup(q0=10, beta=2, bcs={"x_min": T(0), "x_max": T(1)}, method=method, n=41))
    assert r["phase3"]["errors"]["relative_pct"] < 0.2


@pytest.mark.parametrize("method", METHODS)
def test_convective_and_flux_boundaries(method):
    r = run(setup(q0=5, beta=-3, method=method, n=41,
                  bcs={"x_min": {"type": "heat_flux", "value": 50}, "x_max": {"type": "convection", "h": 5, "t_inf": 20}}))
    assert r["phase3"]["errors"]["relative_pct"] < 0.5


@pytest.mark.parametrize("method", METHODS)
def test_convection_diffusion_boundary_layer(method):
    r = run(setup("convection_diffusion", u=10, n=81, method=method, advection_scheme="central",
                  bcs={"x_min": {"type": "inflow", "value": 0}, "x_max": T(1)}))
    assert r["phase1"]["derived"]["peclet"] == pytest.approx(10)
    assert r["phase3"]["errors"]["relative_pct"] < 1.0


@pytest.mark.parametrize("method", ["fdm", "fvm"])
def test_pure_convection_upwind(method):
    r = run(setup("convection", u=2, q0=5, n=81, method=method,
                  bcs={"x_min": {"type": "inflow", "value": 1}, "x_max": {"type": "outflow"}}))
    assert r["phase3"]["errors"]["relative_pct"] < 1.0


def test_pure_convection_rejects_fem_and_missing_inflow():
    with pytest.raises(InputValidationError):
        run(setup("convection", u=1, method="fem", bcs={"x_min": {"type": "inflow", "value": 1}}))
    with pytest.raises(InputValidationError):
        run(setup("convection", u=1, bcs={"x_max": {"type": "inflow", "value": 1}}))


@pytest.mark.parametrize("method", METHODS)
def test_2d_laplace_series(method):
    r = run(setup(dim=2, n=31, method=method, bcs={"x_min": T(0), "x_max": T(0), "y_min": T(0), "y_max": T(100)}))
    assert r["phase3"]["exact_info"]["name"].startswith("2-D Laplace")
    assert r["phase3"]["errors"]["rms"] < 0.5


@pytest.mark.parametrize("dim", [2, 3])
def test_multi_d_reduces_to_1d_exact(dim):
    r = run(setup(dim=dim, n=7, q0=10, bcs={"x_min": T(0), "x_max": T(5)}))
    assert r["phase3"]["errors"]["max_abs"] < 1e-8


@pytest.mark.parametrize("scheme", ["implicit", "crank_nicolson", "explicit"])
def test_transient_series_solution(scheme):
    r = run(setup(time_mode="transient", n=21, time_scheme=scheme, time_step=0.001, num_steps=100,
                  bcs={"x_min": T(100), "x_max": T(0)}))
    assert r["phase3"]["errors"]["relative_pct"] < 0.5
    assert len(r["phase3"]["snapshots"]) == len(r["phase3"]["times"])
    assert r["phase3"]["monitor"]["exact"]


def test_explicit_instability_reported():
    with pytest.raises(SolverError):
        run(setup(time_mode="transient", n=21, time_scheme="explicit", time_step=0.01, num_steps=200,
                  bcs={"x_min": T(100), "x_max": T(0)}))


def test_undetermined_steady_level_rejected():
    with pytest.raises(InputValidationError):
        run(setup(bcs={"x_min": {"type": "heat_flux", "value": 1}}))


def test_refinement_study_second_order():
    study = refinement_study(setup(q0=10, beta=2, bcs={"x_min": T(0), "x_max": T(1)}))
    assert len(study["levels"]) >= 3
    assert study["levels"][-1]["order_max_abs"] == pytest.approx(2.0, abs=0.05)


def test_small_system_exposes_matrix_and_lu():
    r = run(setup(n=5, bcs={"x_min": T(0), "x_max": T(1)}, solver="lu"))
    m = r["phase2"]["matrix"]
    assert len(m["dense"]) == 5 and m["lu"] is not None
    assert r["phase2"]["sample_equation"]["latex"]


# ------------------------------------------------------------------------ API
def test_v2_api_roundtrip():
    client = TestClient(app)
    body = setup(q0=10, bcs={"x_min": T(0), "x_max": T(0)}).model_dump()
    ok = client.post("/api/v2/simulate", json=body)
    assert ok.status_code == 200 and ok.json()["success"]
    assert client.post("/api/v2/validate", json=body).json()["valid"]
    bad = client.post("/api/v2/simulate", json=setup(bcs={}).model_dump())
    assert bad.status_code == 422 and bad.json()["kind"] == "input"
    assert client.post("/api/v2/refine", json=body).status_code == 200


def test_fvm_hybrid_switches_to_upwind_at_high_cell_peclet():
    """Hybrid must judge the cell Péclet number from ρc_p·u·h/k, not from the face flux (which includes area)."""
    bcs = {"x_min": {"type": "inflow", "value": 0}, "x_max": T(1), "y_min": {"type": "insulated"}, "y_max": {"type": "insulated"}}
    kw = dict(physics="convection_diffusion", dim=2, n=11, u=100, method="fvm", bcs=bcs)
    hybrid = run(setup(advection_scheme="hybrid", **kw))
    upwind = run(setup(advection_scheme="upwind", **kw))
    assert np.allclose(hybrid["phase3"]["temperature"], upwind["phase3"]["temperature"])
    assert hybrid["phase2"]["advection"] == "upwind"
