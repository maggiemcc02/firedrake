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

    # Set the pacman domain (Umberto's spectral course)
    geo = SplineGeometry()
    pnts = [(0, 0), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0),
            (-1, -1), (0, -1)]
    p1, p2, p3, p4, p5, p6, p7, p8 = [geo.AppendPoint(*pnt) for pnt in pnts]
    curves = [[["line", p1, p2], "line"],
            [["spline3", p2, p3, p4], "curve"],
            [["spline3", p4, p5, p6], "curve"],
            [["spline3", p6, p7, p8], "curve"],
            [["line", p8, p1], "line"]]
    [geo.Append(c, bc=bc) for c, bc in curves]
    ngmesh = geo.GenerateMesh(maxh=1/N)
    mesh = Mesh(ngmesh)

    # mesh = IntervalMesh(N, pi)
    V = FunctionSpace(mesh, "Argyris", 1)   

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

def folded_distance(N, zs):
    geo = SplineGeometry()

    points = [
        (0, 0), (1, 0), (1, 1), (0, 1),
        (-1, 1), (-1, 0), (-1, -1), (0, -1)
    ]

    p = [geo.AppendPoint(*x) for x in points]

    curves = [
        [["line", p[0], p[1]], "boundary"],
        [["spline3", p[1], p[2], p[3]], "boundary"],
        [["spline3", p[3], p[4], p[5]], "boundary"],
        [["spline3", p[5], p[6], p[7]], "boundary"],
        [["line", p[7], p[0]], "boundary"],
    ]

    for curve, marker in curves:
        geo.Append(curve, bc=marker)

    ngmesh = geo.GenerateMesh(maxh=1/N)
    mesh = Mesh(ngmesh)

    # Argyris is a C^1 quintic element.
    V = FunctionSpace(mesh, "Argyris", 5)

    bc = DirichletBC(V, 0, "on_boundary")

    u = TrialFunction(V)
    v = TestFunction(V)

    mass = inner(u, v) * dx

    distances = []

    for z_value in zs:
        z = Constant(z_value)

        # Poisson operator: L u = -Delta u
        Lu_minus_zu = -div(grad(u)) - z*u
        Lv_minus_zv = -div(grad(v)) - z*v

        folded = inner(Lu_minus_zu, Lv_minus_zv) * dx

        problem = LinearEigenproblem(
            folded,
            mass,
            bcs=bc,
            bc_shift=1e8,
            restrict=False,
        )

        solver_parameters = {
            "eps_type": "krylovschur",
            "eps_gen_hermitian": None,
            "eps_smallest_magnitude": None,
            "eps_target": 0,
            "st_type": "sinvert",
        }

        solver = LinearEigensolver(
            problem,
            n_evals=1,
            solver_parameters=solver_parameters,
        )

        solver.solve()

        # Folded eigenvalue = distance(z, spectrum)^2
        d = sqrt(abs(solver.eigenvalue(0).real))
        distances.append(d)

        print(f"N = {N}, z = {z_value}: distance = {d}")

    return np.asarray(distances)




for N in [8, 16, 32]:

    exact_eig = 3.375610652693620492628**2
    print(f"N = {N}: first eigenvalue ≈ {exact_eig}")
    zs = np.linspace(0.8 * exact_eig, 1.2 * exact_eig, 41)
    distances = folded_distance(N, zs)
    estimate = zs[np.argmin(distances)]
    plt.plot(zs, distances, linewidth=2, label=f"$N = {N}$")
    print(f"Matt estimate of first eigenvalue ≈ {estimate}")


# xs = np.array(list(np.arange(2, 3, 0.01)) + [3])
# exact_eig = 3.375610652693620492628**2
# err = None
# for N in [8, 16, 32, 64]:
#     ys = folded_dist(N, xs)
#     plt.plot(xs, ys, linewidth=2, label=f"$N = {N}$")
#     prev, err = err, np.max(np.abs(ys - exact))
#     rate = "" if prev is None else f", rate {np.log2(prev/err):.2f}"
#     print(GREEN % f"N = {N}: max error = {err}{rate}")

# plt.plot(xs, exact, 'k--', linewidth=1, label="exact")
# plt.plot(nn[nn <= xs[-1]], 0*nn[nn <= xs[-1]], 'ok', markersize=5)

os.makedirs("pacman_test/", exist_ok=True)
plt.xlabel("$x$")
plt.legend()
plt.title(r"Approximation of $\mathrm{dist}(x, \text{spectrum})$, 1D Poisson (Pacman)")
plt.savefig("pacman_test/matt_plot.pdf")