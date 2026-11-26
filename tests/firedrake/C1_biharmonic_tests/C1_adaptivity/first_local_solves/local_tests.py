from firedrake import *
from firedrake.biharmonic_adapt import (ClampedHermiteBC, _C1_residual_indicators)
from petsctools import inserted_options
from firedrake.adapt import _both
from firedrake.petsc import PETSc


# Create the problem
##########################################

def lap(w):
    return div(grad(w))

primal_params = {
    "mat_type": "aij", # assemble and store A as sparse
    "ksp_type": "preonly", # apply the preconditioner once, not iterative
    # "ksp_monitor_true_residual": None,
    "pc_factor_mat_solver_type": "mumps", # Use MUMPS library
    "pc_type": "lu" # use an LU factorisation
}

# Mesh, start on 10x10
N = 160
mesh = UnitIntervalMesh(N)

# Exact manufactured solution
x = SpatialCoordinate(mesh)[0]
u_exact = x**2 * (1 - x)**2 * (1 + 10*x**3)
f = 8400*x**3 - 7200*x**2 + 1200*x + 24

# Solve for uh
V = FunctionSpace(mesh, "Hermite", 3)
u = TrialFunction(V)
v = TestFunction(V)
a = inner(lap(u), lap(v))*dx
L = inner(f, v) * dx

# Strongly enforce u = u' = 0 at x = 0, 1
# bc = DirichletBC(V, 0, "on_boundary")
# Include derivative DOFs, which standard DirichletBC skips (chats suggestion)
# bc.nodes = V.boundary_nodes("on_boundary")
bc = ClampedHermiteBC(V, 0, "on_boundary") # use my clamped BC

# Solve
uh = Function(V)
solve(a == L, uh, bcs=[bc], solver_parameters=primal_params)


# Localisation
####################################################################


# Form F
F = (inner(lap(uh), lap(v))*dx - inner(f, v)*dx)


# Exact residuals
n = FacetNormal(mesh)
Rcell_exact = lap(lap(uh)) - f # should be -f
Rfacet1_exact = lap(uh)
Rfacet0_exact = -dot(grad(lap(uh)), n)

# Local solver parameters
local_options = {}
for stage in ("goal_cell_", "goal_facet_"): # just use the primal solvers
    for key, value in primal_params.items():
        local_options[stage + key] = value


# Call our localisation

# Make these solver options avilable to PETSC while the indented code runs
with inserted_options(parameters=local_options, options_prefix="manufactured_"):
    diagnostics = _C1_residual_indicators(
        F,
        u_exact - uh,
        residual_degree=1,
        options_prefix="manufactured_",
        return_diagnostics=True,)

# Save the computed results
Rcell = diagnostics["cell_residual"]
Rfacet1 = diagnostics["facet_residual_1"]
Rfacet0 = diagnostics["facet_residual_0"]

# Errors
########################################################################## 

def facet_norm(w):
    return abs(assemble(_both(inner(w, w)) * dS + inner(w, w) * ds))**0.5

BLUE = "\033[94m"
RESET = "\033[0m"

PETSc.Sys.Print(
    f"{BLUE}Cell reconstruction error: "
    f"{norm(Rcell - Rcell_exact)}{RESET}"
)

PETSc.Sys.Print(
    f"{BLUE}Derivative-facet reconstruction error: "
    f"{facet_norm(Rfacet1 - Rfacet1_exact)}{RESET}"
)

PETSc.Sys.Print(
    f"{BLUE}Value-facet reconstruction error: "
    f"{facet_norm(Rfacet0 - Rfacet0_exact)}{RESET}"
)



