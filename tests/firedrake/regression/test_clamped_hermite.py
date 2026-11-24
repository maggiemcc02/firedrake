"""Check complex clamped solves and preservation of BCs under enrichment."""

from math import sqrt

import pytest
from firedrake import (
    Function, FunctionSpace, NonlinearVariationalProblem,
    NonlinearVariationalSolver, SpatialCoordinate, TestFunction,
    UnitIntervalMesh, assemble, ds, dx, inner,
)
from firedrake.biharmonic_adapt import ClampedHermiteBC
from firedrake.petsc import PETSc


@pytest.mark.skipreal
@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("degree", [
    pytest.param(3, id="cubic"),
    pytest.param(5, id="enriched"),
])
def test_clamped_hermite_complex(degree: int) -> None:
    mesh = UnitIntervalMesh(8)
    x, = SpatialCoordinate(mesh)
    exact = x**2 * (1 - x)**2 * (1 + 1j*x)
    f = 24 + 1j*(120*x - 48)

    V = FunctionSpace(mesh, "Hermite", 3)
    u = Function(V)
    v = TestFunction(V)
    bc = ClampedHermiteBC(V)
    F = (inner(u.dx(0).dx(0), v.dx(0).dx(0)) - inner(f, v))*dx
    problem = NonlinearVariationalProblem(F, u, bcs=[bc])

    # Keep the cubic check independent of higher-degree element support.
    # The enriched case must remain a separate, visible failure until
    # the installed element library accepts quintic interval Hermite.
    if degree != 3:
        u = Function(V.reconstruct(degree=degree))
        problem = problem.rediscretise(u=u)
        assert isinstance(problem.bcs[0], ClampedHermiteBC)

    parameters = {
        "snes_type": "ksponly",
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    NonlinearVariationalSolver(problem, solver_parameters=parameters).solve()

    # The quintic exact solution belongs to the enriched space.
    tolerance = 1e-4 if degree == 3 else 1e-9
    error = exact - u
    l2_error = sqrt(abs(assemble(inner(error, error)*dx)))
    boundary_squared = assemble(
        (inner(u, u) + inner(u.dx(0), u.dx(0)))*ds
    )
    goal = assemble(inner(f, u)*dx)
    curvature_error = error.dx(0).dx(0)
    energy_error_squared = assemble(
        inner(curvature_error, curvature_error)*dx
    ).real

    PETSc.Sys.Print(
        f"degree={degree}: "
        f"L2 error={l2_error:.3e}, "
        f"boundary squared norm={abs(boundary_squared):.3e}, "
        f"J={goal.real:.12f}{goal.imag:+.3e}j"
    )
    assert l2_error < tolerance, f"L2 error: {l2_error}"
    assert abs(boundary_squared) < 1e-20, f"Boundary residual: {boundary_squared}"
    assert abs(goal.imag) < 1e-10, f"Imaginary compliance: {goal.imag}"
    # Galerkin orthogonality gives J(exact)-J(uh) = ||exact-uh||_a^2.
    assert abs((8/7 - goal.real) - energy_error_squared) < 1e-8
