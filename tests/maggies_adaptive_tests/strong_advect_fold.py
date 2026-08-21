
from firedrake import *
import numpy as np
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import os



def folded_dist(N, xs):


    mesh = IntervalMesh(N, 1)


    V = FunctionSpace(mesh, "Hermite", 3)   # C^1; converges at the rate O(h^2)


    # D(L) constrains u but not u', and Firedrake does pick out just the
    # point-value dofs.  Clamping u' too gives u(0) = u'(0) = 0, no spectrum
    bc = [DirichletBC(V.sub(0), 0, "on_boundary")]

    u = TrialFunction(V)
    v = TestFunction(V)

    # Placeholder for eigenvalue
    z = Constant(0)

    # Operator and its adjoint
    L = lambda w: -w.dx(0).dx(0) - w.dx(0)
    Ls = lambda w: -w.dx(0).dx(0) + w.dx(0)

    # Eigprobs
    a1 = inner(L(u) - z*u, L(v) - z*v)*dx     
    a2 = inner(Ls(u) - z*u, Ls(v) - z*v)*dx
    b = inner(u, v)*dx

    # restrict=True returns mesh-independent nonsense with Hermite
    problem1 = LinearEigenproblem(a1, b, bcs=bc, bc_shift=1e8, restrict=False)
    problem2 = LinearEigenproblem(a2, b, bcs=bc, bc_shift=1e8, restrict=False)

    # Parameters 
    sp = {"eps_gen_hermitian": None,   # solver parameters, passed to SLEPc
          "eps_type": "krylovschur",
          #"eps_monitor": None,        # uncomment to watch convergence
          "eps_smallest_magnitude": None,
          #"eps_view": None,           # uncomment to see the solver
          "eps_target": 0,
          "eps_target_real": None,
          "st_type": "sinvert",
          }

    # Eigprobs
    solver1 = LinearEigensolver(problem1, n_evals=1, solver_parameters=sp)
    solver2 = LinearEigensolver(problem2, n_evals=1, solver_parameters=sp)
    
    #cache_guess = Function(V)

    ys = []
    for x in xs:

        z.assign(x)

        solver1.solve()
        solver2.solve()

        eig1 = solver1.eigenvalue(0)
        eig2 = solver1.eigenvalue(0)

        phi = np.sqrt(min(eig1, eig2))

        ys.append(phi)

        print(BLUE % f"N = {N}, x = {x}: |dist(x, spectrum)| <= {ys[-1]}")

        # Cycle eigenspace from one solve to another. Krylov-Schur only takes
        # one initial guess as input, so we just need to cycle one vector
        # with cache_guess.dat.vec_wo as vec_r:
        #     solver.es.getEigenvector(0, vec_r)
        #     solver.es.setInitialSpace(vec_r)

    return np.array(ys)


xs = np.array(list(np.arange(0.05, 64, 0.05)) + [64])

# spec(L) = {n^2}, so the exact distance function is a sawtooth with gaps
# that widen as n grows
nu = 1
positive_spec = np.array([ np.sqrt(1/(4*nu) + nu*(n*np.pi)**2) for n in range(1, 100 + 1) ]) 
nn = np.sort(np.concatenate([ -positive_spec, [0.0], positive_spec ]))
exact = np.min(np.abs(np.subtract.outer(xs, nn)), axis=1)

err = None

for N in [8, 16, 32, 64]:


    ys = folded_dist(N, xs)
    plt.plot(xs, ys, linewidth=2, label=f"$N = {N}$")

    # prev, err = err, np.max(np.abs(ys - exact))
    # rate = "" if prev is None else f", rate {np.log2(prev/err):.2f}"
    # print(GREEN % f"N = {N}: max error = {err}{rate}")

plt.plot(xs, exact, 'k--', linewidth=1, label="exact")
mask = (nn >= xs[0]) & (nn <= xs[-1])
plt.plot(nn[mask], np.zeros_like(nn[mask]), 'ok', markersize=5)
# plt.plot(nn[nn <= xs[-1]], 0*nn[nn <= xs[-1]], 'ok', markersize=5)

os.makedirs("strong_advect/", exist_ok=True)
plt.xlabel("$x$")
plt.xlim(xs[0], xs[-1])
plt.legend()
plt.title(r"Approximation of $\mathrm{dist}(x, \text{spectrum})$, 1D Advection Diffusion (Hermite)")
plt.savefig("strong_advect/hermite.pdf")
plt.show()