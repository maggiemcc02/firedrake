from firedrake import *
from netgen.occ import *
from firedrake.maggies_adaptivefoldedeigensolver import GoalAdaptiveNonlinearVariationalSolver

print('Running poisson example')

# Netgen mesh
square = WorkPlane().Rectangle(1, 1).Face().bc("all")
square.edges.Max(Y).name = "top"
geo = OCCGeometry(square, dim=2)
ngmesh = geo.GenerateMesh(maxh=0.1)
mesh = Mesh(ngmesh)

# # Set the problem
# degree = 3
# V = FunctionSpace(mesh, "CG", degree)
# (x, y) = SpatialCoordinate(mesh)
# p = Constant(5)
# u_exact = x*(1-x)*y*(1-y)*exp(2*pi*x)*cos(pi*y)
# f = -div(inner(grad(u_exact), grad(u_exact))**((p-2)/2) * grad(u_exact))

# # Since the problem is highly nonlinear, for the purposes of this demo we will
# # cheat and pick our initial guess really close to the exact solution.
# u = Function(V, name="Solution")
# u.interpolate(0.99*u_exact)

# v = TestFunction(V)
# F = (inner(inner(grad(u), grad(u))**((p-2)/2) * grad(u), grad(v)) * dx(degree=degree+10)
#         - inner(f, v) * dx(degree=degree+10)
# )
# bcs = DirichletBC(V, u_exact, "on_boundary")
# solver_parameters = {
#                 "snes_monitor": None,
#                 "snes_atol": 1e-6,
#                 "snes_rtol": 1e-1, # very coarse!
#                 "snes_linesearch_monitor": None,
#                 "snes_linesearch_type": "l2",
#                 "snes_linesearch_maxlambda": 1}


# Rewriting (chat help) for complex mode
degree = 3
V = FunctionSpace(mesh, "CG", degree)

x, y = SpatialCoordinate(mesh)
p = Constant(5)

u_exact = (
    x * (1 - x)
    * y * (1 - y)
    * exp(2 * pi * x)
    * cos(pi * y)
)

grad_u_exact_sq = dot(grad(u_exact), grad(u_exact))
f = -div(grad_u_exact_sq**((p - 2) / 2)* grad(u_exact))

u = Function(V, name="Solution")
u.interpolate(0.99 * u_exact)

v = TestFunction(V)

grad_u_sq = dot(grad(u), grad(u))
F = (inner(grad_u_sq**((p - 2) / 2) * grad(u),grad(v),)
    * dx(degree=degree + 10)
    - inner(f, v) * dx(degree=degree + 10))

bcs = DirichletBC(V, u_exact, "on_boundary")

solver_parameters = {
                "snes_monitor": None,
                "snes_atol": 1e-6,
                "snes_rtol": 1e-1, # very coarse!
                "snes_linesearch_monitor": None,
                "snes_linesearch_type": "l2",
                "snes_linesearch_maxlambda": 1}

# To apply goal-based adaptivity, we need a goal functional. For this we will employ the integral of the normal derivative of the solution on the top boundary: ::

top = tuple(i + 1 for (i, name) in enumerate(ngmesh.GetRegionNames(codim=1)) if name == "top")
n = FacetNormal(mesh)
J = inner(grad(u), n)*ds(top)

# We now specify options for how the goal-based adaptivity should proceed.
# We set the absolute tolerance on the error estimate, and the maximum number of iterations.
# We choose to use an expensive/robust approach,
# where the adjoint solution is approximated in a higher-degree function space, and where both the adjoint and primal residuals
# are employed for the error estimate. This requires four solves on every grid (primal and adjoint solutions with degree :math:`p`
# and :math:`p+1`), and gives a provably efficient and reliable error estimator under a saturation assumption up to a term that is cubic in the error :cite:`Endtmayer2024`.
# It is possible to employ cheaper and more practical approximations by setting the options for the :code:`GoalAdaptiveNonlinearVariationalSolver`
# appropriately, as discussed below. ::

solver_parameters["goal_adaptive"] = {
    "tolerance": 1e-4,
    "max_it": 100,
    "dorfler_alpha": 0.5,
    "use_adjoint_residual": True,
    "dual_low_method": "solve",
    "primal_low_method": "solve",
    "dual_extra_degree": 1,
}

# We then solve the problem, passing the goal functional :math:`J`. We also pass the exact solution, so that
# the DWR automation can compute effectivity indices, but this is not generally required: ::

problem = NonlinearVariationalProblem(F, u, bcs)

adaptive_solver = GoalAdaptiveNonlinearVariationalSolver(problem, J,
                                                            solver_parameters=solver_parameters,
                                                            exact_solution=u_exact)
adaptive_solution = adaptive_solver.solve()
error_estimate = adaptive_solver.get_error_estimate()