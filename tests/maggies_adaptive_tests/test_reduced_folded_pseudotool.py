"""Algebraic verification for the weighted two-sided reduced pencil."""

import numpy as np

from reduced_folded_pseudotool import ReducedFoldedPencil, project_two_sided


class _Vec:
    """Small PETSc-Vec stand-in for exercising the projection assembly."""

    def __init__(self, values):
        self.values = np.array(values, dtype=complex)

    def copy(self):
        return _Vec(self.values.copy())

    def duplicate(self):
        return _Vec(np.zeros_like(self.values))

    def axpy(self, alpha, other):
        self.values += alpha * other.values

    def scale(self, alpha):
        self.values *= alpha

    def dot(self, other):
        return np.vdot(self.values, other.values)

    def norm(self):
        return np.linalg.norm(self.values)


class _Mat:
    """Small PETSc-Mat stand-in for exercising ``Mat.mult`` calls."""

    def __init__(self, values):
        self.values = np.array(values, dtype=complex)

    def mult(self, x, y):
        y.values[:] = self.values @ x.values


def _spd(rng, n):
    B = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return B.conj().T @ B + np.eye(n)


def test_full_two_sided_reduction_matches_weighted_folded_problem():
    rng = np.random.default_rng(8)
    n = 8
    K = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    C, H = _spd(rng, n), _spd(rng, n)
    z = 0.35 - 0.61j
    U, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))

    reduced = ReducedFoldedPencil(
        U.conj().T @ K @ V,
        U.conj().T @ M @ V,
        U.conj().T @ C @ U,
        V.conj().T @ H @ V,
    )

    T = K - z * M
    Lc, Lh = np.linalg.cholesky(C), np.linalg.cholesky(H)
    That = np.linalg.solve(Lc, T)
    That = np.linalg.solve(Lh.conj(), That.T).T
    direct = np.linalg.svd(That, compute_uv=False)[-1] ** 2
    assert np.isclose(reduced.smallest_eigenvalue(z), direct, rtol=1e-11, atol=1e-11)


def test_petsc_projection_assembly_matches_full_weighted_problem():
    rng = np.random.default_rng(21)
    n = 5
    K = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    C, H = _spd(rng, n), _spd(rng, n)
    z = -0.2 + 0.4j
    coordinate_basis = [_Vec(np.eye(n)[:, i]) for i in range(n)]

    reduced, _, _ = project_two_sided(
        _Mat(K), _Mat(M), _Mat(C), _Mat(H),
        coordinate_basis, coordinate_basis,
    )

    T = K - z * M
    Lc, Lh = np.linalg.cholesky(C), np.linalg.cholesky(H)
    That = np.linalg.solve(Lc, T)
    That = np.linalg.solve(Lh.conj(), That.T).T
    direct = np.linalg.svd(That, compute_uv=False)[-1] ** 2
    assert np.isclose(reduced.smallest_eigenvalue(z), direct, rtol=1e-10, atol=1e-10)
