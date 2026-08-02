"""Serial Firedrake/PETSc integration test for the reduced folded helper.

This is deliberately tiny: it verifies the projection code against matrices
assembled from a real weak ``H_0^1 -> H^{-1}`` Laplacian problem.
"""

import numpy as np
from firedrake import (DirichletBC, FunctionSpace, TestFunction, TrialFunction,
                       UnitSquareMesh, assemble, dx, grad, inner)

from reduced_folded_pseudotool import project_two_sided


def _dense(mat):
    dense = mat.convert("dense")
    return dense.getDenseArray().copy()


def _coordinate_basis(mat):
    n, _ = mat.getSize()
    vectors = []
    for i in range(n):
        vector = mat.createVecRight()
        vector.setValue(i, 1.0)
        vector.assemble()
        vectors.append(vector)
    return vectors


def test_firedrake_petsc_weighted_projection_matches_full_problem():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    bc = DirichletBC(V, 0, "on_boundary")

    # K represents -Delta: H_0^1 -> H^{-1}; M represents the weak identity;
    # R is the H^1 Riesz map, used for both domain and codomain weighting.
    K = assemble(inner(grad(u), grad(v)) * dx, bcs=bc).petscmat
    M = assemble(u * v * dx, bcs=bc).petscmat
    R = assemble((inner(grad(u), grad(v)) + u * v) * dx, bcs=bc).petscmat
    z = 1.3 + 0.2j

    basis = _coordinate_basis(K)
    reduced, _, _ = project_two_sided(K, M, R, R, basis, basis)

    Kd, Md, Rd = _dense(K), _dense(M), _dense(R)
    T = Kd - z * Md
    L = np.linalg.cholesky(Rd)
    That = np.linalg.solve(L, T)
    That = np.linalg.solve(L.conj(), That.T).T
    direct = np.linalg.svd(That, compute_uv=False)[-1] ** 2

    assert np.isclose(reduced.smallest_eigenvalue(z), direct,
                      rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    test_firedrake_petsc_weighted_projection_matches_full_problem()
    print("Firedrake/PETSc weighted reduced-folded integration test: PASS")
