import numpy as np
from firedrake import *
from firedrake.petsc import PETSc

mesh = UnitSquareMesh(2, 2)
V = FunctionSpace(mesh, "CG", 3)
u = Function(V)
v = TestFunction(V)
x, y = SpatialCoordinate(mesh)

# A known complex solution and its corresponding forcing.
shape = x*(1-x)*y*(1-y)
q = 2*(x*(1-x) + y*(1-y))
exact = (1+1j)*shape
F = inner((1+2j)*grad(u), grad(v))*dx - inner((1+2j)*(1+1j)*q, v)*dx
problem = NonlinearVariationalProblem(
    F, u, bcs=DirichletBC(V, 0, "on_boundary"))
marker = GoalOrientedMarker((2-1j)*q*u*dx, exact_solution=exact)

parameters = {
    "snes_type": "ksponly",
    "snes_grid_sequence": 1,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "goal_cell_ksp_type": "preonly",
    "goal_cell_pc_type": "jacobi",
    "goal_facet_ksp_type": "preonly",
    "goal_facet_pc_type": "jacobi",
}
solver = NonlinearVariationalSolver(
    problem, solver_parameters=parameters, marking_callback=marker)
solver.solve()

# The exact goal is (3+1j)/45.
estimate = marker.estimates[0]
true_error = (3+1j)/45 - estimate.goal
assert abs(true_error.imag) > 1e-10
np.testing.assert_allclose(
    estimate.true_error, true_error, rtol=1e-7, atol=1e-12)
np.testing.assert_allclose(
    estimate.error_estimate, true_error, rtol=1e-6, atol=1e-12)

PETSc.Sys.Print("True goal error:", true_error)
PETSc.Sys.Print("Estimated error:", estimate.error_estimate)
PETSc.Sys.Print("Complex goal-estimator check passed.")
