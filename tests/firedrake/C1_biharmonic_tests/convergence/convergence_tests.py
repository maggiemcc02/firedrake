from firedrake import *
from firedrake.biharmonic_adapt import ClampedHermiteBC


# ============================================================
# 1. MANUFACTURED BIHARMONIC PROBLEM in 1D
# ============================================================

def lap(w):
    return div(grad(w))


def biharm_1D(N):

    mesh = UnitIntervalMesh(N)

    # Exact manufactured solution
    x = SpatialCoordinate(mesh)[0]
    u_exact = x**2 * (1 - x)**2 * (1 + 10*x**3)
    f = 8400*x**3 - 7200*x**2 + 1200*x + 24

    # Solve for uh
    V = FunctionSpace(mesh, "Hermite", 3)

    u = TrialFunction(V)
    v = TestFunction(V)

    # a = inner(u.dx(0).dx(0), v.dx(0).dx(0)) * dx
    a = inner(lap(u), lap(v))*dx
    L = inner(f, v) * dx

    # Strongly enforce u = u' = 0 at x = 0, 1
    # bc = DirichletBC(V, 0, "on_boundary")
    # Include derivative DOFs, which standard DirichletBC skips (chats suggestion)
    # bc.nodes = V.boundary_nodes("on_boundary")
    bc = ClampedHermiteBC(V, 0, "on_boundary") # use my clamped BC


    uh = Function(V)
    solve(a == L, uh, bcs=[bc])

    # Check the errors
    e = uh - u_exact
    # L2 norm
    L2_error = norm(e).real
    H1_error = norm(e, 'H1').real
    H2_semi = norm(grad(grad(e)), "L2").real
    H2_error = sqrt(H1_error**2 + H2_semi**2)

    # Check the bc enforcement
    value_error = sqrt(abs(assemble(inner(uh, uh) * ds)))
    slope = uh.dx(0)
    slope_error = sqrt(abs(assemble(inner(slope, slope) * ds)))

    PETSc.Sys.Print(f"Boundary value error: {value_error:.3e}; " f"boundary slope error: {slope_error:.3e}")
    PETSc.Sys.Print()

    return L2_error, H1_error, H2_error



# ============================================================
# 2. Convergence Tests
# ============================================================

mesh_sizes = [10, 20, 40, 80, 160]
L2_errors = []
H1_errors = []
H2_errors = []

for N in mesh_sizes:

    L2_error, H1_error, H2_error = biharm_1D(N)
    L2_errors.append(L2_error)
    H1_errors.append(H1_error)
    H2_errors.append(H2_error)



# Plot
import matplotlib.pyplot as plt
# Reference curves
L2ref = [L2_errors[0] * (mesh_sizes[0] / N)**4 for N in mesh_sizes]
H1ref = [H1_errors[0] * (mesh_sizes[0] / N)**3 for N in mesh_sizes]
H2ref = [H2_errors[0] * (mesh_sizes[0] / N)**2 for N in mesh_sizes]


plt.loglog(mesh_sizes, L2_errors, '-o', color = 'lightblue', label=r"$L^2$ norm")
plt.loglog(mesh_sizes, H1_errors, '-o', color = 'orange',label=r"$H^1$ norm")
plt.loglog(mesh_sizes, H2_errors, '-o', color = 'lightgreen', label=r"$H^2$ norm")
plt.loglog(mesh_sizes, L2ref, '--*', color = 'darkblue', label = r'$O(h^{-4})$')
plt.loglog(mesh_sizes, H1ref, '--*', color = 'red', label = r'$O(h^{-3})$')
plt.loglog(mesh_sizes, H2ref, '--*', color = 'darkgreen',label = r'$O(h^{-2})$')
plt.legend()
plt.xlabel(r"$N$")
plt.ylabel("Error Norm")
plt.title('Convergence Rates for Manufactured 1D Biharmonic Problem')
plt.savefig("biharmonic_convergence.png", dpi=200)