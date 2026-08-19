from firedrake import *
import numpy as np
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import os



def folded_dist(N, xs):


    mesh = IntervalMesh(N, 1)
    deg=2

    V0 = FunctionSpace(mesh, "CG", deg)     # potential u
    V1 = FunctionSpace(mesh, "CG", deg)     # flux sigma ~ u'
    Z = MixedFunctionSpace([V0, V1])
    bc = [DirichletBC(Z.sub(0), 0, "on_boundary")]   # no BC on sigma

    nu = Constant(1)
    j = Constant(1j)
    U = TrialFunction(Z)
    (u, sigma) = split(U)
    V = TestFunction(Z)
    (v, tau) = split(V)


    # Placeholder for eigenvalue
    z = Constant(0)

    
    # Folded operator 1
    A1 = (
      inner(nu*sigma.dx(0) + sigma,
            nu*tau.dx(0) + tau) * dx

    + inner(u.dx(0), v.dx(0)) * dx

    - conj(z) * inner(-j*(nu*sigma.dx(0) + sigma), v) * dx
    - conj(z) * inner(-j*u.dx(0), tau) * dx

    - z * inner(u, -j*(nu*tau.dx(0) + tau)) * dx
    - z * inner(sigma, -j*v.dx(0)) * dx

    + conj(z)*z * inner(u, v) * dx
    + conj(z)*z * inner(sigma, tau) * dx)


    # Folded operator 2
    A2 = (
      inner(sigma.dx(0), tau.dx(0)) * dx

    + inner(nu*u.dx(0) - u,
            nu*v.dx(0) - v) * dx

    - z * inner(-j*sigma.dx(0), v) * dx
    - z * inner(-j*(nu*u.dx(0) - u), tau) * dx

    - conj(z) * inner(u, -j*tau.dx(0)) * dx
    - conj(z) * inner(sigma, -j*(nu*v.dx(0) - v)) * dx

    + conj(z)*z * inner(u, v) * dx
    + conj(z)*z * inner(sigma, tau) * dx)


    # RHS
    M = inner(u, v)*dx + inner(sigma, tau)*dx






    # restrict=True returns mesh-independent nonsense with Hermite
    problem1 = LinearEigenproblem(A1, M, bcs=bc, bc_shift=1e8, restrict=False)
    problem2 = LinearEigenproblem(A2, M, bcs=bc, bc_shift=1e8, restrict=False)


    # sp = {"eps_gen_hermitian": None,   # solver parameters, passed to SLEPc
    #       "eps_type": "krylovschur",
    #       #"eps_monitor": None,        # uncomment to watch convergence
    #       "eps_smallest_magnitude": None,
    #       #"eps_view": None,           # uncomment to see the solver
    #       "eps_target": 0,
    #       "eps_target_real": None,
    #       "st_type": "sinvert",
    #       }

    sp = {
    "eps_gen_hermitian": None,
    "eps_type": "lapack",
    "eps_smallest_real": None,}


    solver1 = LinearEigensolver(problem1, n_evals=1, solver_parameters=sp)
    solver2 = LinearEigensolver(problem2, n_evals=1, solver_parameters=sp)
    #cache_guess = Function(V)

    ys = []

    for x in xs:

        z.assign(x)
        solver1.solve()
        solver2.solve()
        eig1 = solver1.eigenvalue(0)
        eig2 = solver2.eigenvalue(0)
        phi = np.sqrt(min(eig1.real, eig2.real))
        ys.append(phi)
        print(BLUE % f"N = {N}, x = {x}: |dist(x, spectrum)| <= {ys[-1]}")

        # # Cycle eigenspace from one solve to another. Krylov-Schur only takes
        # # one initial guess as input, so we just need to cycle one vector
        # with cache_guess.dat.vec_wo as vec_r:
        #     solver.es.getEigenvector(0, vec_r)
        #     solver.es.setInitialSpace(vec_r)

    return np.array(ys)


xs = np.array(list(np.arange(1, 6.0, 0.05)) + [6.0])

# spec(L) = {n^2}, so the exact distance function is a sawtooth with gaps
# that widen as n grows
nu_value = 0.015
positive_spec = np.array([ np.sqrt(1/(4*nu_value) + nu_value*(n*np.pi)**2) for n in range(1, 100 + 1) ]) 
spec = np.sort(np.concatenate([ -positive_spec, [0.0], positive_spec ]))
dist = np.min(np.abs(np.subtract.outer(xs, spec)), axis=1)

err = None
for N in [8, 16, 32, 64, 128, 256]:
    ys = folded_dist(N, xs)
    plt.plot(xs, ys, linewidth=2, label=f"$N = {N}$")
    prev, err = err, np.max(np.abs(ys - dist))
    rate = "" if prev is None else f", rate {np.log2(prev/err):.2f}"
    print(GREEN % f"N = {N}: max error = {err}{rate}")

spec_plot = spec[(spec >= xs[0]) & (spec <= xs[-1])]
plt.plot(spec_plot, np.zeros_like(spec_plot), 'ok', markersize=5)
plt.plot( xs, dist, "k--", linewidth=1, label=r"$\operatorname{dist}(x,\operatorname{Sp}(L))$")

os.makedirs("more_debugging_advection", exist_ok=True)
plt.xlabel("$x$")
plt.legend()
plt.title(r"Approximation of $\mathrm{dist}(x, \text{spectrum})$")
plt.savefig("more_debugging_advection/plot.pdf")