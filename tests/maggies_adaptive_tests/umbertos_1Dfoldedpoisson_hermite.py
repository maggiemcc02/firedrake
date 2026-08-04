from firedrake import *
import numpy as np
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import os

# We fold the Dirichlet Laplacian L = -d^2/dx^2 on (0, pi) directly, so the
# folded quadratic form is

#     q_z(u) = ||(L - z) u||^2_{L^2} = int |u'' + z u|^2 dx

# whose form domain is D(L) = H^2 \cap H^1_0.  That needs C^1 elements, which
# is what cubic Hermite gives (u'' is piecewise linear, hence in L^2).

def folded_dist(N, xs):
    mesh = IntervalMesh(N, pi)
    V = FunctionSpace(mesh, "Hermite", 3)   # C^1; converges at the rate O(h^2)

    # D(L) constrains u but not u', and Firedrake does pick out just the
    # point-value dofs.  Clamping u' too gives u(0) = u'(0) = 0, no spectrum
    bc = [DirichletBC(V, 0, "on_boundary")]

    u = TrialFunction(V)
    v = TestFunction(V)

    # Placeholder for eigenvalue
    z = Constant(0)

    L = lambda w: -w.dx(0).dx(0)

    a = inner(L(u) - z*u, L(v) - z*v)*dx     # ||(L - z) u||^2, verbatim
    b = inner(u, v)*dx

    # restrict=True returns mesh-independent nonsense with Hermite
    problem = LinearEigenproblem(a, b, bcs=bc, bc_shift=1e8, restrict=False)
    sp = {"eps_gen_hermitian": None,   # solver parameters, passed to SLEPc
          "eps_type": "krylovschur",
          #"eps_monitor": None,        # uncomment to watch convergence
          "eps_smallest_magnitude": None,
          #"eps_view": None,           # uncomment to see the solver
          "eps_target": 0,
          "eps_target_real": None,
          "st_type": "sinvert",
          }
    solver = LinearEigensolver(problem, n_evals=1, solver_parameters=sp)
    cache_guess = Function(V)

    ys = []
    for x in xs:
        z.assign(x)
        solver.solve()
        ys.append(sqrt(solver.eigenvalue(0)).real)
        print(BLUE % f"N = {N}, x = {x}: |dist(x, spectrum)| <= {ys[-1]}")

        # Cycle eigenspace from one solve to another. Krylov-Schur only takes
        # one initial guess as input, so we just need to cycle one vector
        with cache_guess.dat.vec_wo as vec_r:
            solver.es.getEigenvector(0, vec_r)
            solver.es.setInitialSpace(vec_r)

    return np.array(ys)


xs = np.array(list(np.arange(0.05, 64, 0.05)) + [64])

# spec(L) = {n^2}, so the exact distance function is a sawtooth with gaps
# that widen as n grows
nn = np.array([n**2 for n in range(1, 12)])
exact = np.min(np.abs(np.subtract.outer(xs, nn)), axis=1)

err = None
for N in [8, 16, 32, 64]:
    ys = folded_dist(N, xs)
    plt.plot(xs, ys, linewidth=2, label=f"$N = {N}$")

    prev, err = err, np.max(np.abs(ys - exact))
    rate = "" if prev is None else f", rate {np.log2(prev/err):.2f}"
    print(GREEN % f"N = {N}: max error = {err}{rate}")

plt.plot(xs, exact, 'k--', linewidth=1, label="exact")
plt.plot(nn[nn <= xs[-1]], 0*nn[nn <= xs[-1]], 'ok', markersize=5)

os.makedirs("umberto_output/of_poisson_1d/", exist_ok=True)
plt.xlabel("$x$")
plt.legend()
plt.title(r"Approximation of $\mathrm{dist}(x, \text{spectrum})$, 1D Poisson (Hermite)")
plt.savefig("umberto_output/of_poisson_1d/hermite.pdf")