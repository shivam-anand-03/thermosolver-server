"""Steady-state solver tests."""

import numpy as np
import pytest

from app.models import SteadyRequest
from app.solver.finite_difference import Mesh
from app.solver.steady_state import SteadyStateSolver
from app.solver.utils import InputValidationError


def req(**kw) -> SteadyRequest:
    base = dict(length=1.0, nodes=5, thermal_conductivity=1.0, heat_generation=0.0,
                left_boundary={"type": "temperature", "value": 100.0},
                right_boundary={"type": "temperature", "value": 0.0}, temperature_unit="C")
    base.update(kw)
    return SteadyRequest(**base)


def solve(**kw) -> dict:
    return SteadyStateSolver(req(**kw)).solve().to_dict()


def test_dx_calculation():
    assert Mesh(0.5, 5).dx == pytest.approx(0.125)
    assert Mesh(5e-3, 6).dx == pytest.approx(1e-3)
    assert Mesh(1.0, 2).dx == 1.0
    x = Mesh(0.5, 5).x
    np.testing.assert_allclose(x, [0, 0.125, 0.25, 0.375, 0.5])


def test_length_unit_conversion():
    r = solve(length=500, length_unit="mm")
    assert r["dx"] == pytest.approx(0.125)


def test_dirichlet_dirichlet_is_linear():
    r = solve(nodes=11)
    np.testing.assert_allclose(r["solution"], np.linspace(100, 0, 11), atol=1e-10)
    assert r["validation"]["status"] == "validated"


def test_dirichlet_neumann_flux():
    # T(0)=373 K, q''=1000 W/m² INTO the slab at x=L, k=1 → dT/dx = +1000 K/m (linear, q'''=0)
    r = solve(length=0.5, nodes=5, temperature_unit="K",
              left_boundary={"type": "temperature", "value": 373},
              right_boundary={"type": "heat_flux", "value": 1000})
    np.testing.assert_allclose(r["solution"], [373, 498, 623, 748, 873], atol=1e-9)


def test_gradient_and_adiabatic_boundaries():
    r = solve(left_boundary={"type": "gradient", "value": -50}, right_boundary={"type": "temperature", "value": 0})
    np.testing.assert_allclose(r["solution"], [50, 37.5, 25, 12.5, 0], atol=1e-10)
    r = solve(left_boundary={"type": "adiabatic"}, right_boundary={"type": "temperature", "value": 20})
    np.testing.assert_allclose(r["solution"], [20] * 5, atol=1e-12)


def test_heat_flux_sign_convention_left():
    # heat entering at x=0 must make the left end hotter than the right
    r = solve(left_boundary={"type": "heat_flux", "value": 100}, right_boundary={"type": "temperature", "value": 0})
    assert r["solution"][0] > r["solution"][-1]
    assert r["solution"][0] == pytest.approx(100.0)  # q''L/k


def test_heat_generation_matches_analytic_parabola():
    # T = Ts + q'''/(2k) x (L - x) — exact for the 3-point stencil
    L, k, q, Ts = 0.1, 20.0, 5e5, 100.0
    r = solve(length=L, nodes=11, thermal_conductivity=k, heat_generation=q,
              left_boundary={"type": "temperature", "value": Ts}, right_boundary={"type": "temperature", "value": Ts})
    x = np.array(r["x"])
    np.testing.assert_allclose(r["solution"], Ts + q / (2 * k) * x * (L - x), atol=1e-9)
    assert max(r["solution"]) == pytest.approx(131.25)


def test_ghost_node_scheme_exact_for_generation_with_insulation():
    # insulated left, T(L)=0, generation: exact T = q/(2k)(L² - x²); ghost-node scheme is exact
    L, k, q = 1.0, 1.0, 2.0
    r = solve(length=L, nodes=6, thermal_conductivity=k, heat_generation=q, boundary_scheme="ghost_node",
              left_boundary={"type": "adiabatic"}, right_boundary={"type": "temperature", "value": 0})
    x = np.array(r["x"])
    np.testing.assert_allclose(r["solution"], q / (2 * k) * (L**2 - x**2), atol=1e-10)


def test_neumann_neumann_incompatible_rejected():
    with pytest.raises(InputValidationError, match="compatible heat balance"):
        solve(left_boundary={"type": "heat_flux", "value": 100}, right_boundary={"type": "heat_flux", "value": 50})


def test_neumann_neumann_compatible_but_non_unique_rejected():
    with pytest.raises(InputValidationError, match="constrain the temperature"):
        solve(left_boundary={"type": "heat_flux", "value": 100}, right_boundary={"type": "heat_flux", "value": -100})
    with pytest.raises(InputValidationError, match="constrain the temperature"):
        solve(left_boundary={"type": "adiabatic"}, right_boundary={"type": "adiabatic"})


@pytest.mark.parametrize("kw, msg", [
    (dict(length=0), "Length must be greater than zero"),
    (dict(length=-1), "Length must be greater than zero"),
    (dict(nodes=1), "at least 2"),
    (dict(nodes=0), "at least 2"),
    (dict(thermal_conductivity=0), "Thermal conductivity must be positive"),
    (dict(length=float("nan")), "finite"),
    (dict(left_boundary={"type": "temperature", "value": -300}), "absolute zero"),
])
def test_invalid_inputs(kw, msg):
    with pytest.raises(InputValidationError, match=msg):
        solve(**kw)


@pytest.mark.parametrize("n", [2, 5, 10, 50, 100, 500, 1000])
def test_mesh_sizes(n):
    r = solve(nodes=n, heat_generation=10.0)
    assert len(r["solution"]) == n
    assert r["validation"]["status"] == "validated"
    assert (r["matrix"] is None) == (n > 100)


def test_reference_mst_steady_example():
    """MST-1 steady example: computed from the governing equation, not the printed values."""
    r = solve(length=0.5, nodes=5, temperature_unit="K",
              left_boundary={"type": "temperature", "value": 373},
              right_boundary={"type": "heat_flux", "value": 1000})
    T = r["solution"]
    assert r["dx"] == pytest.approx(0.125)
    assert T[0] == pytest.approx(373.0)
    # the part of the printed answer that is consistent with the BC: T5 - T4 = q''Δx/k = 125 K
    assert T[4] - T[3] == pytest.approx(125.0)
    assert r["matrix"][1] == [-1.0, 2.0, -1.0, 0.0, 0.0]
    assert r["matrix"][4] == [0.0, 0.0, 0.0, -1.0, 1.0]
    assert r["rhs"][4] == pytest.approx(125.0)
