"""Independent tests of the hand-written Gaussian elimination engine."""

import numpy as np
import pytest

from app.solver.gaussian_elimination import GaussianEliminationSolver, gaussian_elimination
from app.solver.utils import SolverError


def test_solves_small_dense_system():
    A = np.array([[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]])
    b = np.array([8.0, -11.0, -3.0])
    res = gaussian_elimination(A, b)
    np.testing.assert_allclose(res.solution, [2.0, 3.0, -1.0], atol=1e-12)


def test_records_steps_and_back_substitution():
    A = np.array([[1.0, 0, 0], [-1.0, 2.0, -1.0], [0, -1.0, 1.0]])
    b = np.array([10.0, 0.0, 1.0])
    res = GaussianEliminationSolver().solve(A, b)
    assert res.recorded
    assert len(res.initial_augmented) == 3 and len(res.initial_augmented[0]) == 4
    assert len(res.steps) == 2
    first_op = res.steps[0].operations[0]
    assert first_op.describe() == "R2 → R2 + 1·R1"
    assert [s.index for s in res.back_substitution] == [2, 1, 0]
    np.testing.assert_allclose(A @ res.solution, b, atol=1e-12)


def test_partial_pivoting_swaps_rows_for_zero_diagonal():
    A = np.array([[0.0, 1.0], [1.0, 1.0]])
    b = np.array([1.0, 3.0])
    res = gaussian_elimination(A, b)
    assert res.steps[0].swapped
    assert res.total_swaps == 1
    np.testing.assert_allclose(res.solution, [2.0, 1.0])


def test_partial_pivoting_picks_largest_pivot():
    A = np.array([[1e-3, 1.0], [1.0, 1.0]])
    res = gaussian_elimination(A, np.array([1.0, 2.0]))
    assert res.steps[0].swapped and res.steps[0].pivot_value == 1.0


def test_singular_matrix_detected():
    A = np.array([[1.0, -1.0], [-1.0, 1.0]])
    with pytest.raises(SolverError, match="singular"):
        gaussian_elimination(A, np.array([1.0, 2.0]))


def test_nan_rejected():
    with pytest.raises(SolverError, match="NaN"):
        gaussian_elimination(np.array([[np.nan]]), np.array([1.0]))


def test_matches_numpy_on_random_systems():
    rng = np.random.default_rng(0)
    for n in (2, 5, 17, 40):
        A = rng.normal(size=(n, n)) + n * np.eye(n)
        b = rng.normal(size=n)
        res = gaussian_elimination(A, b)
        np.testing.assert_allclose(res.solution, np.linalg.solve(A, b), rtol=1e-10, atol=1e-10)


def test_factorization_replay_equals_direct_solve():
    rng = np.random.default_rng(1)
    n = 30
    A = rng.normal(size=(n, n)) + n * np.eye(n)
    solver = GaussianEliminationSolver()
    rec = solver.factorize(A)
    for _ in range(3):
        b = rng.normal(size=n)
        np.testing.assert_allclose(rec.solve(b), solver.solve(A, b).solution, atol=1e-10)


def test_large_tridiagonal_not_recorded():
    n = 1000
    A = np.diag(np.full(n, 2.0)) + np.diag(np.full(n - 1, -1.0), 1) + np.diag(np.full(n - 1, -1.0), -1)
    b = np.ones(n)
    res = gaussian_elimination(A, b)
    assert not res.recorded and res.initial_augmented is None
    assert np.max(np.abs(A @ res.solution - b)) < 1e-6
