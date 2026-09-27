"""Transient solver tests."""

import numpy as np
import pytest

from app.models import TransientRequest
from app.solver.transient import TransientSolver
from app.solver.utils import InputValidationError

MST = dict(length=5.0, length_unit="mm", nodes=6, thermal_conductivity=0.25, density=1300.0,
           specific_heat=2000.0, left_boundary={"type": "adiabatic"},
           right_boundary={"type": "temperature", "value": 250.0}, temperature_unit="C",
           initial_condition={"type": "uniform", "value": 30.0}, time_step_mode="fourier",
           fourier_number=1.0, num_steps=1)


def solve(**kw) -> dict:
    base = dict(MST)
    base.update(kw)
    return TransientSolver(TransientRequest(**base)).solve().to_dict()


def test_reference_mst_transient_first_step():
    r = solve()
    assert r["alpha"] == pytest.approx(9.6154e-8, rel=1e-4)
    assert r["fourier"] == pytest.approx(1.0)
    assert r["dt"] == pytest.approx(10.40, rel=1e-3)
    expected = [36.47, 36.47, 42.94, 62.35, 114.12, 250.0]  # PDF prints 62.36 (exact 62.353)
    np.testing.assert_allclose(r["solution"], expected, atol=0.01)
    assert r["matrix"][2] == [0.0, -1.0, 3.0, -1.0, 0.0, 0.0]  # -1, 3, -1 stencil for Fo = 1
    assert r["rhs"] == [0.0, 30.0, 30.0, 30.0, 30.0, 250.0]


def test_implicit_time_marching_reaches_steady_state():
    r = solve(num_steps=400)
    np.testing.assert_allclose(r["solution"], [250.0] * 6, atol=1e-6)
    assert r["summary"]["steady_state_time"] is not None


def test_final_time_and_sampling():
    r = solve(num_steps=None, final_time=104.0, output_interval=2)
    assert r["num_steps"] == 10
    assert r["times"][0] == 0.0 and r["times"][-1] == pytest.approx(104.0)
    assert len(r["times"]) == 6  # t=0 + every 2nd of 10 steps
    assert len(r["snapshots"]) == len(r["times"])


def test_manual_time_step_and_fourier():
    r = solve(time_step_mode="manual", time_step=5.2, num_steps=2)
    assert r["fourier"] == pytest.approx(0.5, rel=1e-6)


def test_dirichlet_both_sides_symmetry_and_energy():
    r = solve(left_boundary={"type": "temperature", "value": 0.0},
              right_boundary={"type": "temperature", "value": 0.0}, nodes=21,
              initial_condition={"type": "uniform", "value": 100.0}, num_steps=20)
    T = np.array(r["solution"])
    np.testing.assert_allclose(T, T[::-1], atol=1e-10)
    assert T.max() < 100.0


def test_neumann_neumann_allowed_in_transient_and_conserves_energy():
    # insulated both ends with ghost-node scheme: mean temperature (trapezoid) is conserved
    ic = list(np.linspace(0, 100, 11))
    r = solve(nodes=11, left_boundary={"type": "adiabatic"}, right_boundary={"type": "adiabatic"},
              boundary_scheme="ghost_node", initial_condition={"type": "custom", "values": ic}, num_steps=50)
    def trap(v):
        v = np.asarray(v)
        return (v[1:-1].sum() + 0.5 * (v[0] + v[-1])) / (len(v) - 1)
    assert trap(r["solution"]) == pytest.approx(trap(ic), rel=1e-10)


def test_heat_generation_transient_converges_to_steady_parabola():
    L, k, q = 0.1, 20.0, 5e5
    r = solve(length=L, length_unit="m", nodes=11, thermal_conductivity=k, density=1000, specific_heat=1000,
              heat_generation=q, left_boundary={"type": "temperature", "value": 100},
              right_boundary={"type": "temperature", "value": 100},
              initial_condition={"type": "uniform", "value": 100}, fourier_number=5.0, num_steps=500)
    x = np.array(r["x"])
    np.testing.assert_allclose(r["solution"], 100 + q / (2 * k) * x * (L - x), atol=1e-6)


def test_large_problem_runs_quickly():
    r = solve(nodes=1000, num_steps=200, left_boundary={"type": "temperature", "value": 0})
    assert len(r["solution"]) == 1000
    assert r["matrix"] is None and len(r["snapshots"]) <= 201
    assert r["validation"]["status"] == "validated"


@pytest.mark.parametrize("kw, msg", [
    (dict(density=0), "Density must be positive"),
    (dict(specific_heat=-1), "Specific heat must be positive"),
    (dict(time_step_mode="manual", time_step=0), "Time step must be positive"),
    (dict(time_step_mode="manual", time_step=None), "Time step must be positive"),
    (dict(fourier_number=0), "Fourier number must be positive"),
    (dict(initial_condition={"type": "custom", "values": [1, 2]}), "exactly N = 6"),
    (dict(num_steps=100000), "time steps"),
])
def test_invalid_transient_inputs(kw, msg):
    with pytest.raises(InputValidationError, match=msg):
        solve(**kw)


def test_large_fourier_warns_accuracy_not_instability():
    r = solve(fourier_number=50)
    joined = " ".join(r["warnings"])
    assert "unconditionally stable" in joined
    assert "accuracy" in joined
