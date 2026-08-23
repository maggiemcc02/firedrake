
from firedrake import *
import numpy as np
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.tri import Triangulation
import os

nu = 1


def folded_dist(N, xs, ys):


    mesh = IntervalMesh(N, 1)


    V = FunctionSpace(mesh, "Hermite", 3)   # C^1; converges at the rate O(h^2)


    # D(L) constrains u but not u', and Firedrake does pick out just the
    # point-value dofs.  Clamping u' too gives u(0) = u'(0) = 0, no spectrum
    bc = [DirichletBC(V.sub(0), 0, "on_boundary")]

    u = TrialFunction(V)
    v = TestFunction(V)

    # Placeholder for z
    z = Constant(0.0 + 0.0j)
    # Separate constant for conjugate(z), used in the adjoint folding.
    zbar = Constant(0.0 + 0.0j)

    # Operator and adjoint
    L = lambda w: -nu*w.dx(0).dx(0) - w.dx(0)
    Ls = lambda w: -nu*w.dx(0).dx(0) + w.dx(0)

    # # Operator and its adjoint
    # L = lambda w: -w.dx(0).dx(0) - w.dx(0)
    # Ls = lambda w: -w.dx(0).dx(0) + w.dx(0)

    # Eigprobs
    a1 = inner(L(u) - z*u, L(v) - z*v)*dx     
    a2 = inner(Ls(u) - zbar*u, Ls(v) - zbar*v)*dx
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

    field_samples = {}
    total = len(xvals)*len(yvals)
    count = 0

    for y in ys:
        for x in xs:

            zz = complex(x, y)

            z.assign(zz)
            zbar.assign(np.conjugate(zz))

            solver1.solve()
            solver2.solve()

            eig1 = solver1.eigenvalue(0)
            eig2 = solver2.eigenvalue(0)

            # These should be real and nonnegative because the
            # two folded problems are Hermitian positive semidefinite.
            mu1 = float(np.real(eig1))
            mu2 = float(np.real(eig2))
            mu = min(mu1, mu2)

            # Compute phi
            phi = np.sqrt(mu)

            # Save it
            field_samples[(float(x), float(y))] = phi

            # Logging
            count += 1
            if count % 100 == 0:
                print(BLUE %f"N={N}: {count}/{total}, "f"z={zz:.3g}, Phi={phi:.6e}")

    return field_samples




# Set the grid
# xmin, xmax = 0.0, 160
# ymin, ymax = -80.0, 80.0
# Nx = 181
# Ny = 181
# xmin, xmax = 14, 35
# ymin, ymax = -8, 8
# Nx = 300
# Ny = 161
# xvals = np.linspace(xmin, xmax, Nx)
# yvals = np.linspace(ymin, ymax, Ny)
N = 128
xmin, xmax = 0.0, 50.0
ymin, ymax = -35.0, 35.0
Nx = 201
Ny = 201
xvals = np.linspace(xmin, xmax, Nx)
yvals = np.linspace(ymin, ymax, Ny)

# Compute the results
field_samples = folded_dist(N, xvals, yvals)




###############################################################################
# PLOT Phi_N(z,L) AND PSEUDOSPECTRAL CONTOURS (Umbertos Code Updated by Chat)
###############################################################################


# Colour map: same "Mana (Extended)" used by the render scripts of the
# advection_diffusion_forms paper, so the plates look consistent.
_PV_MANA_STOPS = [
    (0.00000, (0.098039, 0.137255, 0.352941)),
    (0.03125, (0.207283, 0.138936, 0.373669)),
    (0.06250, (0.316527, 0.140616, 0.394398)),
    (0.09375, (0.425770, 0.142297, 0.415126)),
    (0.12500, (0.535014, 0.143978, 0.435854)),
    (0.15625, (0.644258, 0.145658, 0.456583)),
    (0.18750, (0.753501, 0.147339, 0.477311)),
    (0.21875, (0.862745, 0.149020, 0.498039)),
    (0.25000, (0.803922, 0.200490, 0.560784)),
    (0.28125, (0.745098, 0.251961, 0.623529)),
    (0.31250, (0.686275, 0.303431, 0.686275)),
    (0.34375, (0.627451, 0.354902, 0.749020)),
    (0.37500, (0.568627, 0.406373, 0.811765)),
    (0.40625, (0.509804, 0.457843, 0.874510)),
    (0.43750, (0.450980, 0.509314, 0.937255)),
    (0.46875, (0.392157, 0.560784, 1.000000)),
    (0.50000, (0.343137, 0.611765, 0.916667)),
    (0.53125, (0.294118, 0.662745, 0.833333)),
    (0.56250, (0.245098, 0.713725, 0.750000)),
    (0.59375, (0.196078, 0.764706, 0.666667)),
    (0.62500, (0.464052, 0.797386, 0.699346)),
    (0.65625, (0.732026, 0.830065, 0.732026)),
    (0.68750, (1.000000, 0.862745, 0.764706)),
    (0.71875, (0.993464, 0.790850, 0.679739)),
    (0.75000, (0.986928, 0.718954, 0.594771)),
    (0.78125, (0.980392, 0.647059, 0.509804)),
    (0.81250, (0.933333, 0.560784, 0.435294)),
    (0.84375, (0.886275, 0.474510, 0.360784)),
    (0.87500, (0.839216, 0.388235, 0.286275)),
    (0.90625, (0.729412, 0.291176, 0.245098)),
    (0.93750, (0.619608, 0.194118, 0.203922)),
    (0.96875, (0.509804, 0.097059, 0.162745)),
    (1.00000, (0.400000, 0.000000, 0.121569)),
]
PV_MANA = LinearSegmentedColormap.from_list("mana", _PV_MANA_STOPS, N=256)




# ----------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------- 

OUT = "strong_advect"
os.makedirs(OUT, exist_ok=True)


# ----------------------------------------------------------------
# Convert sampled Phi values to arrays
# ----------------------------------------------------------------

pts = []
vals = []

for (px, py), phi in field_samples.items():
    pts.append(complex(px, py))
    vals.append(phi)

pts = np.asarray(pts)
phi_f = np.asarray(vals, dtype=float)

print(
    GREEN % (
        f"field assembled from {len(pts)} "
        "complex-plane evaluations"
    )
)


# ----------------------------------------------------------------
# Construct triangulation for plotting
#
# The sampled z-points are treated as vertices of a plotting
# triangulation. This triangulation is unrelated to the physical
# finite-element mesh.
# ----------------------------------------------------------------

triang = Triangulation(pts.real, pts.imag)


# ----------------------------------------------------------------
# Exact eigenvalues
#
# For L = -u'' - u', with nu = 1:
#
#     lambda_n = (n*pi)^2 + 1/4
# ----------------------------------------------------------------

n = np.arange(1, 101)

spec = nu*(n*np.pi)**2 + 1/(4*nu)

eigs = spec[
    (spec >= xmin) &
    (spec <= xmax)
]


# ----------------------------------------------------------------
# Colour range for Phi_N
# ----------------------------------------------------------------

vmin = float(phi_f.min())
vmax = float(phi_f.max())

levels = np.linspace(vmin, vmax, 25)


# ----------------------------------------------------------------
# Create figure
# ----------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(11, 4.9),
    constrained_layout=True
)


# ----------------------------------------------------------------
# Filled contour plot of Phi_N(z,L)
# ----------------------------------------------------------------

cf = ax.tricontourf(
    triang,
    phi_f,
    levels=levels,
    cmap=PV_MANA,
    vmin=vmin,
    vmax=vmax
)


# ----------------------------------------------------------------
# Light reference contours of Phi_N
# ----------------------------------------------------------------

iso_levels = np.linspace(vmin, vmax, 9)[1:-1]

ax.tricontour(
    triang,
    phi_f,
    levels=iso_levels,
    colors="k",
    linewidths=0.35,
    alpha=0.4
)


# ----------------------------------------------------------------
# Pseudospectral boundaries
#
# Each contour
#
#        Phi_N(z,L) = epsilon
#
# approximates the boundary of the epsilon-pseudospectrum.
#
# Change these values depending on the range of Phi in the plot.
# ----------------------------------------------------------------

epsilon_levels = [
    0.1, 
    0.15,
    0.2, 
    0.25,
    0.5,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0
]

# Only ask matplotlib to draw levels that actually occur in the field
epsilon_levels = [
    eps for eps in epsilon_levels
    if vmin < eps < vmax
]

if epsilon_levels:

    ps = ax.tricontour(
        triang,
        phi_f,
        levels=epsilon_levels,
        colors="k",
        linewidths=1.35,
        zorder=4
    )

    ax.clabel(
        ps,
        inline=True,
        fontsize=8,
        fmt=lambda eps: rf"$\epsilon={eps:g}$"
    )


# ----------------------------------------------------------------
# Overlay exact eigenvalues
# ----------------------------------------------------------------

ax.scatter(
    eigs,
    np.zeros_like(eigs),
    s=28,
    marker="o",
    facecolors="white",
    edgecolors="black",
    linewidths=1.2,
    zorder=5
)

# ----------------------------------------------------------------
# Reference parabola
# ----------------------------------------------------------------




# yy = np.linspace(ymin, ymax, 1000)

# xx_W = nu*yy**2 + nu*np.pi**2

# ax.plot(
#     xx_W,
#     yy,
#     "k--",
#     linewidth=1.2,
#     label=r"$\partial W(A)$"
# )


# ----------------------------------------------------------------
# Axes and labels
#
# These are direct z-grid evaluations rather than cell barycentres,
# so we do NOT trim the edges of the plotting region.
# ----------------------------------------------------------------

ax.set_xlabel(r"$\operatorname{Re}(z)$")
ax.set_ylabel(r"$\operatorname{Im}(z)$")

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# ax.set_aspect("equal")
# ax.set_xlim(xmin, xmax)
# ax.set_ylim(ymin, ymax)
ax.set_aspect("equal", adjustable="box")

ax.tick_params(
    direction="in",
    which="both"
)


# ----------------------------------------------------------------
# Colour bar
# ----------------------------------------------------------------

cbar = fig.colorbar(
    cf,
    ax=ax,
    pad=0.02,
    extend="neither"
)

cbar.set_label(r"$\Phi_N(z,L)$")


# ----------------------------------------------------------------
# Title
# ----------------------------------------------------------------

ax.set_title(
    rf"1D advection--diffusion, Hermite$_3$, $N={N}$",
    fontsize=10
)


# ----------------------------------------------------------------
# Save
# ----------------------------------------------------------------

for ext in ("png", "pdf"):

    fig.savefig(
        f"{OUT}/updated_pseudospectra_field_N={N}.{ext}",
        dpi=300
    )

plt.close(fig)

print(
    GREEN %
    f"wrote pseudospectral field plots to {OUT}/"
)


#     ys = []
#     for x in xs:

#         z.assign(x)

#         solver1.solve()
#         solver2.solve()

#         eig1 = solver1.eigenvalue(0)
#         eig2 = solver2.eigenvalue(0)

#         phi = np.sqrt(min(eig1, eig2))

#         ys.append(phi)

#         print(BLUE % f"N = {N}, x = {x}: |dist(x, spectrum)| <= {ys[-1]}")

#         # Cycle eigenspace from one solve to another. Krylov-Schur only takes
#         # one initial guess as input, so we just need to cycle one vector
#         # with cache_guess.dat.vec_wo as vec_r:
#         #     solver.es.getEigenvector(0, vec_r)
#         #     solver.es.setInitialSpace(vec_r)

#     return np.array(ys)


# xs = np.array(list(np.arange(0.05, 64, 0.05)) + [64])

# # spec(L) = {n^2}, so the exact distance function is a sawtooth with gaps
# # that widen as n grows
# # nu = 1
# # positive_spec = np.array([ np.sqrt(1/(4*nu) + nu*(n*np.pi)**2) for n in range(1, 100 + 1) ]) 
# # nn = np.sort(np.concatenate([ -positive_spec, [0.0], positive_spec ]))
# # exact = np.min(np.abs(np.subtract.outer(xs, nn)), axis=1)

# nu = 1.0
# n = np.arange(1, 101)
# nn = nu*(n*np.pi)**2 + 1/(4*nu)
# exact = np.min(np.abs(np.subtract.outer(xs, nn)),axis=1)

# err = None

# for N in [8, 16, 32, 64]:


#     ys = folded_dist(N, xs)
#     plt.plot(xs, ys, linewidth=2, label=f"$N = {N}$")

#     # prev, err = err, np.max(np.abs(ys - exact))
#     # rate = "" if prev is None else f", rate {np.log2(prev/err):.2f}"
#     # print(GREEN % f"N = {N}: max error = {err}{rate}")

# plt.plot(xs, exact, 'k--', linewidth=1, label="exact")
# mask = (nn >= xs[0]) & (nn <= xs[-1])
# plt.plot(nn[mask], np.zeros_like(nn[mask]), 'ok', markersize=5)
# # plt.plot(nn[nn <= xs[-1]], 0*nn[nn <= xs[-1]], 'ok', markersize=5)

# os.makedirs("strong_advect/", exist_ok=True)
# plt.xlabel("$x$")
# plt.xlim(xs[0], xs[-1])
# plt.legend()
# plt.title(r"Approximation of $\mathrm{dist}(x, \text{spectrum})$, 1D Advection Diffusion (Hermite)")
# plt.savefig("strong_advect/hermite.pdf")
# plt.show()