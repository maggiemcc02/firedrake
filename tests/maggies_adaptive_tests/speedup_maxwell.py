from firedrake import *
import numpy as np
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.tri import Triangulation
from matplotlib.colors import LinearSegmentedColormap
from netgen.geom2d import SplineGeometry
from ufl import conj
import os
import re
import gc
import sys
import resource

# import outerloop_adaptivefoldedeigensolver as _maggie
# from outerloop_adaptivefoldedeigensolver import GoalAdaptiveFoldedEigenSolver
import firedrake.umbertos_adaptivefoldedeigensolver as _maggie
from firedrake.umbertos_adaptivefoldedeigensolver import GoalAdaptiveFoldedEigenSolver

# Pseudospectra of the 2D Maxwell operator through Maggie's FULL inner loop,
# with genuinely LOCAL mesh adaptivity -- the Maxwell counterpart of
# pseudospectra_poisson_dwr_local.py.  Everything from SOLVE through the
# markers is her code verbatim, including her bubble/cone cell and facet
# residual indicators, which need no 1D fork here: _compute_residual_indicators
# already reconstructs a TensorFunctionSpace for vector-valued subspaces.
#
# Only the REFINE tail is replaced, and only because stock Firedrake does not
# export firedrake.mg.ufl_utils.refine (her branch adds it).  The mesh
# refinement itself IS hers: netgen's refine_marked_elements on her markers.
# What the driver supplies is the problem transfer onto the refined mesh,
# which is a one-line rebuild since the forms are constructed here.
#
# TWO CHOICES THAT ARE NOT COSMETIC
#
# 1. Domain and element.  On the SQUARE (convex) the nodal CG1 x CG1 pair of
#    her scripts is legitimate: the folding kills the spurious eigenvalues that
#    make nodal Maxwell unusable, which is Colbrook's point.  On the L-SHAPE it
#    is NOT, and folding cannot save it: the reentrant corner makes the Maxwell
#    eigenfunction singular, and the H^1-conforming closure is a PROPER closed
#    subspace of the relevant part of H(curl) there (Costabel-Dauge).  Nodal
#    elements then converge, cleanly and confidently, to the WRONG spectrum --
#    an approximation failure, not a spurious-mode failure, so no amount of
#    folding fixes it.  The default therefore uses N1curl on the L-shape and
#    nodal on the square.  Set ELEMENT to override and see it happen.
#
# 2. Complex z.  Her sweep forms carry -2 z <Au, v>, which is right on the real
#    axis but not Hermitian off it.  Here the cross term is -(conj(z) + z),
#    the correct Hermitian generalisation (= -2 Re z), as in her parallel outer
#    loop and in the 1D companions.
#
# The L-shape is the case that justifies local adaptivity at all: the corner
# singularity concentrates the DWR indicator near the reentrant vertex, so her
# marking is selective.  On the square (and in 1D Poisson) the eigenfunctions
# are smooth, the error is spread uniformly, and her marking correctly selects
# nearly everything -- the machinery is validated but not showcased.  The run
# records the marked fraction per refinement so this is measured, not asserted.



# ADDED PARALLEL STUFF
##########################################################################
from mpi4py import MPI
WORLD = MPI.COMM_WORLD # distributes z values
RANK = WORLD.rank 
NPROCS = WORLD.size
# Every rank solves its assigned z values independently and serially.
# So, we need a communicator
LOCAL_COMM = MPI.COMM_SELF


# SETUP
########################################################################

DOMAIN = os.environ.get("DOMAIN", "square")            # "lshape" | "square"
ELEMENT = os.environ.get("ELEMENT",
                         "N1curl" if DOMAIN == "lshape" else "CG")
SMOKE = int(os.environ.get("SMOKE", 0))

epsilon = float(os.environ.get("EPS", 0.2))     # pseudospectral level
maxh0 = float(os.environ.get("MAXH0", 0.3))    # initial mesh for every solve
deg = int(os.environ.get("DEG", 1 if ELEMENT == "CG" else 1))

# if DOMAIN == "square":
#     xmin, xmax, ymin, ymax = -0.25, 4.0, -0.9, 0.9
# else:
#     xmin, xmax, ymin, ymax = 0.0, 4.0, -0.8, 0.8
if DOMAIN == "square":
    xmin, xmax, ymin, ymax = 2.0, 4.0, -1.0, 1.0
else:
    xmin, xmax, ymin, ymax = 0.0, 4.0, -0.8, 0.8

# Nx = int(os.environ.get("NX", 6 if SMOKE else 10))
# Ny = int(os.environ.get("NY", 2 if SMOKE else 4))
Nx = int(os.environ.get("NX", 5 if SMOKE else 10))
Ny = int(os.environ.get("NY", 5 if SMOKE else 10))
# diam_tol = float(os.environ.get("DIAM_TOL", 0.4 if SMOKE else 0.005))
diam_tol = float(os.environ.get("DIAM_TOL", 0.4 if SMOKE else 0.01))
max_sweeps = int(os.environ.get("MAX_SWEEPS", 1 if SMOKE else 2))
max_it = int(os.environ.get("MAX_IT", 2 if SMOKE else 50)) # Make it high for now
max_dofs = int(os.environ.get("MAX_DOFS", 10000)) # Use a max dof to speed things up

# OUTDIR overrides the destination -- point a smoke test somewhere scratch so
# it cannot overwrite the plates of a real run into the same domain's directory
OUT = os.environ.get(
    "OUTDIR",
    f"output/of_maxwell_2d_dwr_local{'_' + DOMAIN if DOMAIN != 'lshape' else ''}")
os.makedirs(OUT, exist_ok=True)

# A run of this size belongs in the background, where its stdout is easy to
# lose, so every milestone also lands in OUT/progress.txt -- plain text, no
# colour codes, one line per event.  `tail -f` on it is the cheapest way to
# watch a sweep from another shell.
_PROGRESS = f"{OUT}/progress.txt"
# open(_PROGRESS, "w").close()
# def log_line(msg):
#     PETSc.Sys.Print(msg)
#     with open(_PROGRESS, "a") as f:
#         f.write(re.sub(r"\x1b\[[0-9;]*m", "", str(msg)) + "\n")

if RANK == 0:
    os.makedirs(OUT, exist_ok=True)
    open(_PROGRESS, "w").close()

WORLD.barrier()
def log_line(msg):
    if RANK == 0:
        PETSc.Sys.Print(msg)
        with open(_PROGRESS, "a") as f:
            f.write(re.sub(r"\x1b\[[0-9;]*m", "", str(msg)) + "\n")

# log_line(GREEN % (
#     f"domain = {DOMAIN}, element = {ELEMENT}{deg}, epsilon = {epsilon}, "
#     f"maxh0 = {maxh0}, z-grid {Nx}x{Ny}, diam_tol = {diam_tol}, "
#     f"max_sweeps = {max_sweeps}, inner max_it = {max_it}, max_dof = {max_dofs}"))

log_line("RUN PARAMETERS:")
log_line(f"  domain = {DOMAIN}")
log_line(f"  element = {ELEMENT}{deg}")
log_line(f"  epsilon = {epsilon}")
log_line(f"  maxh0 = {maxh0}")
log_line(f"  region = [{xmin}, {xmax}] x [{ymin}, {ymax}]")
log_line(f"  Nx = {Nx}")
log_line(f"  Ny = {Ny}")
log_line(f"  diam_tol = {diam_tol}")
log_line(f"  max_sweeps = {max_sweeps}")
log_line(f"  max_dofs = {max_dofs}")
log_line(f"  MPI ranks = {NPROCS}")


# diam_tol is a PER-CELL stopping test, not a loop bound: a z-cell is only
# labelled "contour" once its own diameter falls below it, and what actually
# ends the loop is the sweep counter.  Ask for a diam_tol the sweeps cannot
# reach and no cell ever qualifies -- every one of them keeps refining until
# max_sweeps cuts in and dumps the whole band into "contour" at whatever size
# it happens to have.  That silently produces an under-resolved contour, so
# say it up front instead.

# Maggie note
# This will check if in under max_sweeps, we will have refined the outer grid cells enough to quit
# Oterwise, we will just always hit max sweeps
_h0 = float(np.hypot((xmax - xmin) / Nx, (ymax - ymin) / Ny))
_sweeps_needed = (int(np.floor(np.log2(_h0 / diam_tol))) + 2
                  if diam_tol < _h0 else 1)
if _sweeps_needed > max_sweeps:
    log_line(RED % (
        f"WARNING: diam_tol = {diam_tol} is out of reach.  The initial z-cell "
        f"diameter is {_h0:.4f} and halves every sweep, so no cell can qualify "
        f"as 'contour' before sweep {_sweeps_needed - 1}, i.e. it needs "
        f"max_sweeps >= {_sweeps_needed}, not {max_sweeps}.  As set, the run "
        f"will refine every band cell for {max_sweeps} sweeps and then label "
        f"the remainder 'contour' at h = {_h0 / 2**max_sweeps:.4f}, "
        f"{_h0 / 2**max_sweeps / diam_tol:.0f}x coarser than requested."))

# THE GEOMETRY (netgen, so that her refine_marked_elements applies)
##################################################################


# Maggie note - this is a bit different than how I made the mesh
# But I trust umberto can deal with netgen better than me lol

def make_mesh(maxh, comm=LOCAL_COMM):
    g = SplineGeometry()
    if DOMAIN == "square":
        pts = [(0, 0), (pi, 0), (pi, pi), (0, pi)]
    else:
        pts = [(0, 0), (1, 0), (1, 1), (-1, 1), (-1, -1), (0, -1)]
    ids = [g.AppendPoint(*p) for p in pts]
    n = len(pts)
    # The boundary alternates horizontal, vertical, ... on both domains, and
    # netgen numbers the segments 1..n in order of definition, so the odd
    # markers are the horizontal edges and the even ones the vertical edges.
    for i in range(n):
        g.Append(["line", ids[i], ids[(i + 1) % n]],
                 bc=("horiz" if i % 2 == 0 else "vert"))
    return Mesh(g.GenerateMesh(maxh=maxh), comm=comm)

NEDGE = 4 if DOMAIN == "square" else 6
HORIZ = tuple(range(1, NEDGE + 1, 2))   # E_x = 0 there (tangent is e_x)
VERT = tuple(range(2, NEDGE + 1, 2))    # E_y = 0 there (tangent is e_y)

# THE FOLDED MAXWELL PROBLEM
##################################################################

z = Constant(0)   # spectral parameter, shared by every mesh's forms
j = Constant(1j)

def rot_s(E):
    return E[1].dx(0) - E[0].dx(1)

def rot_v(H):
    return as_vector([H.dx(1), -H.dx(0)])

def make_problem(mesh):
    if ELEMENT == "N1curl":
        V0 = FunctionSpace(mesh, "N1curl", deg)
    else:
        V0 = VectorFunctionSpace(mesh, "CG", deg, dim=2)
    V1 = FunctionSpace(mesh, "CG", deg)
    Z = MixedFunctionSpace([V0, V1])

    if ELEMENT == "N1curl":
        # edge elements constrain only the tangential trace, which is what
        # E x n = 0 asks for
        bc = [DirichletBC(Z.sub(0), Constant((0, 0)), "on_boundary")]
    else:
        # nodal elements carry both components, so the tangential trace has to
        # be imposed component by component against the edge orientation
        bc = [DirichletBC(Z.sub(0).sub(0), 0, HORIZ),
              DirichletBC(Z.sub(0).sub(1), 0, VERT)]

    U = TrialFunction(Z)
    (E, H) = split(U)
    V = TestFunction(Z)
    (F, G) = split(V)


    A = (
          # A^* M^{-1} A terms
          inner(rot_s(E), rot_s(F))*dx
        + inner(rot_v(H), rot_v(G))*dx
          # -2 Re(z) A terms (Hermitian for complex z)
        - (conj(z) + z) * inner(j*rot_v(H), F)*dx
        + (conj(z) + z) * inner(j*rot_s(E), G)*dx
          # |z|^2 M terms
        + conj(z) * z * inner(E, F)*dx
        + conj(z) * z * inner(H, G)*dx
        )
    M = inner(E, F)*dx + inner(H, G)*dx

    problem = LinearEigenproblem(A, M, bcs=bc, restrict=True)
    # her branch's LinearEigenproblem keeps the unrestricted forms; stock does not
    problem._original_A = A
    problem._original_M = M
    problem._original_bcs = bc
    return problem


# CHANGES TO SOLVER FOR SPEED UP
##################################################################


# Helper for pulling needed mesh
# In mixed spaces, this will give us the one mesh we need to work with
def unique_mesh(V):
    m = V.mesh()
    return m.unique() if hasattr(m, "unique") else m

class GoalAdaptiveMaxwellFoldedEigenSolver(GoalAdaptiveFoldedEigenSolver):
    """Her solver with the problem-transfer half of REFINE supplied locally.

    The marking and the mesh refinement are hers; only the reconstruction of
    the eigenproblem on the refined mesh is done here, because it goes through
    firedrake.mg.ufl_utils.refine, which her branch adds and stock lacks."""

    def refine_problem(self, markers, coef_map=None):
        mesh = unique_mesh(self.problem.output_space)
        marker = markers.subfunctions[0] if hasattr(markers, "subfunctions") \
            else markers # if markers is mixed, use first subfunction
        marked = np.real(marker.dat.data_ro) > 0.5 # converts markers into boolean array (real part used)
        # store marker fractions
        self.marked_fractions = getattr(self, "marked_fractions", [])
        self.marked_fractions.append(float(marked.sum()) / max(len(marked), 1))

        # Noe actually refine
        new_mesh = mesh.refine_marked_elements(marker)   # hers, via netgen
        try: # Check for and use hierarchy
            amh, _ = _maggie.get_level(mesh)
            if amh is not None:
                amh.add_mesh(new_mesh)
                self._hierarchical = True
        except Exception: # if failed, note it
            self._hierarchical = False
        self.problem = make_problem(new_mesh) # Use the given make problem

    # This updated version adds more fall backs
    # hierarchy prolongation
    #   ↓ if unavailable
    # cross-mesh interpolation
    #   ↓ if transfer fails
    # cold start
    
    def post_refinement(self):
        """Warm start the next eigensolve from the previous eigenfunctions.

        Her version prolongs through the AdaptiveTransferManager; that needs
        the refined mesh registered in the hierarchy, which does not survive
        the MixedFunctionSpace rebuild here, so fall back to cross-mesh
        interpolation and, failing that, to a cold start."""
        V_new = self.problem.output_space
        initial_space = []
        for old_u in getattr(self, "vecs", []):
            try:
                new_u = Function(V_new)
                for old_sub, new_sub in zip(old_u.subfunctions,
                                            new_u.subfunctions):
                    if getattr(self, "_hierarchical", False):
                        self.atm.prolong(old_sub, new_sub)
                    else:
                        new_sub.interpolate(old_sub, allow_missing_dofs=True)
                initial_space.append(new_u)
            except Exception:
                initial_space = []
                break
        self.initial_space = tuple(initial_space)

def m_form(a, b):
    # As in parallel_outerloop_maxwell.py: the L2 inner product
    if hasattr(a, "subfunctions") and len(a.subfunctions) > 1:
        return assemble(sum(inner(ai, bi) * dx
                            for ai, bi in zip(a.subfunctions, b.subfunctions)))
    else:
        return assemble(inner(a, b) * dx)

# ADAPTIVE PARAMETERS
##################################################################

sp = {
    "goal_adaptive": {
        "tolerance": 1.0e-5,
        "max_it": max_it,
        "max_dofs" : max_dofs,
        "dorfler_alpha": 0.5,
        "primal_extra_degree": (1, 1),
        "dual_extra_degree": (1, 1),
        "cell_residual_extra_degree": (1, 1),
        "facet_residual_extra_degree": (1, 1),
        "self_adjoint": True,
        "nev": 5,
        "verbose": False,
    },
    "eps_type": "krylovschur",
    "eps_tol": 1.0e-8,
    "st_type": "sinvert",
    "eps_target": 0,
}

# Log the setup
log_line("SOLVER PARAMETERS:")
log_line(f"  tolerance = {sp['goal_adaptive']['tolerance']}")
log_line(f"  max_it = {sp['goal_adaptive']['max_it']}")
log_line(f"  max_dofs = {sp['goal_adaptive']['max_dofs']}")
log_line(f"  dorfler_alpha = {sp['goal_adaptive']['dorfler_alpha']}")
log_line(f"  primal_extra_degree = {sp['goal_adaptive']['primal_extra_degree']}")
log_line(f"  dual_extra_degree = {sp['goal_adaptive']['dual_extra_degree']}")
log_line(f"  cell_residual_extra_degree = {sp['goal_adaptive']['cell_residual_extra_degree']}")
log_line(f"  facet_residual_extra_degree = {sp['goal_adaptive']['facet_residual_extra_degree']}")
log_line(f"  self_adjoint = {sp['goal_adaptive']['self_adjoint']}")
log_line(f"  nev = {sp['goal_adaptive']['nev']}")
log_line(f"  eps_type = {sp['eps_type']}")
log_line(f"  eps_tol = {sp['eps_tol']}")
log_line(f"  st_type = {sp['st_type']}")
log_line(f"  eps_target = {sp['eps_target']}")


# THE BASE PROBLEM
##################################################################

# The base problem is permanent, so its enriched reconstruction is worth
# caching; the adapted meshes are transient and caching them would only leak
base_mesh = make_mesh(maxh0, comm=COMM_SELF)
base_problem = make_problem(base_mesh)
_reconstruct_orig = _maggie._reconstruct_eig_degree
_base_enriched = {}
def _reconstruct_memo(problem, extra_degree):
    if problem is base_problem:
        key = tuple(extra_degree)
        if key not in _base_enriched:
            _base_enriched[key] = _reconstruct_orig(problem, extra_degree)
        return _base_enriched[key]
    return _reconstruct_orig(problem, extra_degree)
_maggie._reconstruct_eig_degree = _reconstruct_memo

log_line(GREEN % (
    f"base mesh: {base_mesh.num_cells()} cells, "
    f"{base_problem.output_space.dim()} dofs"))

_ncalls = [0]


# THE INNER LOOP
##################################################################

def dwr_solve(zval, hK):
    _ncalls[0] += 1
    if _ncalls[0] % 50 == 0:
        gc.collect()   # adapted meshes, enriched spaces and SLEPc
                       # factorisations sit in cyclic garbage
    z.assign(zval)
    solver = GoalAdaptiveMaxwellFoldedEigenSolver(
        base_problem, m_form=m_form, initial_space=(), target=0.0,
        epsilon=epsilon, diam_cond=hK**2, diameter=hK,
        imag_tol=1e-12, mult_tol=1e-2, solver_parameters=sp)
    solver.solve()
    # at z on the spectrum SLEPc can return lambda_min ~ -1e-16 and her class
    # takes sqrt without clamping
    # phi = float(np.real(solver.matts_phi))
    # if not np.isfinite(phi):
    #     phi = float(np.sqrt(max(np.real(solver._lam_h), 0.0)))
    # corr = float(np.real(solver.corrected_phi))
    # if not np.isfinite(corr):
    #     corr = phi
    # eta = float(abs(solver.signed_error))
    # return phi, corr, eta, solver
    # phi = solver.matts_phi
    # if not np.isfinite(phi): 
    #     print(RED % f"WARNING - the phi was saved as NAN or INF!!")
    #     # phi = float(np.sqrt(max((solver._lam_h), 0.0)))
    # corr = solver.corrected_phi # pull the corrected phi
    # if not np.isfinite(corr): 
    #     print(RED % f"WARNING - the corrected phi was saved as NAN or INF!!")
    #     # corr = phi
    # eta = abs(solver.signed_error) # pull |error est|
    # if not np.isfinite(eta): 
    #     print(RED % f"WARNING - the signed error estimate was saved as NAN or INF!!")
    # return phi, corr, eta, solver

    phi = solver.matts_phi
    corr = solver.corrected_phi
    eta = abs(solver.signed_error)

    status = {
        "phi_nonfinite": not np.isfinite(phi),
        "corr_nonfinite": not np.isfinite(corr),
        "eta_nonfinite": not np.isfinite(eta),}

    return phi, corr, eta, solver, status



# A REFERENCE SPECTRUM, INDEPENDENT OF THE FOLDING
##################################################################

def reference_spectrum():
    """Trusted omega values, for markers and for validation.

    Square: analytic, omega = sqrt(n^2 + m^2).  L-shape: a standard N1curl
    curl-curl eigensolve on a fine mesh -- edge elements are spectrally
    correct on non-convex domains, which is exactly the property the nodal
    pair lacks there."""
    if DOMAIN == "square":
        vals = sorted({np.sqrt(n**2 + m**2)
                       for n in range(8) for m in range(8)
                       if 0 < n**2 + m**2 <= 20})
        return np.array([0.0] + vals)
    # If domain is L then we do a better solve 
    # The resulting eigenvalues are treated as our spectral values
    # because we dont know exact values on L
    m = make_mesh(maxh0 / 3)
    W = FunctionSpace(m, "N1curl", 2)
    Ef, Ff = TrialFunction(W), TestFunction(W)
    bcw = [DirichletBC(W, Constant((0, 0)), "on_boundary")]
    prob = LinearEigenproblem(inner(rot_s(Ef), rot_s(Ff))*dx,
                              inner(Ef, Ff)*dx, bcs=bcw, restrict=True)
    es = LinearEigensolver(prob, n_evals=14, solver_parameters={
        "eps_gen_hermitian": None, "eps_type": "krylovschur",
        "eps_target": 3.0, "eps_target_real": None, "st_type": "sinvert",
        "eps_tol": 1e-10})
    nconv = es.solve()
    lam = sorted(float(np.real(es.eigenvalue(i))) for i in range(nconv))
    return np.array([0.0] + [np.sqrt(v) for v in lam if v > 1e-3])

try:
    spec = reference_spectrum()
    log_line(GREEN % ("reference omega in view: " + ", ".join(
        f"{w:.4f}" for w in spec if xmin - 0.5 <= w <= xmax + 0.5)))
except Exception as exc:
    spec = np.array([])
    log_line(RED % f"reference spectrum unavailable ({exc}); "
                    "plates will carry no markers and no exact validation")

def exact_gamma(zval):
    return float(np.min(np.abs(zval - spec))) if len(spec) else np.nan

# THE TRIANGULATION OF THE COMPLEX PLANE (as in z_grid_helper.py)
##################################################################

def initial_triangulation(x_min, y_min, x_max, y_max, Nx, Ny):
    x = np.linspace(x_min, x_max, Nx + 1)
    y = np.linspace(y_min, y_max, Ny + 1)
    triangles = []
    for jj in range(Ny):
        for ii in range(Nx):
            z00 = x[ii]     + 1j * y[jj]
            z10 = x[ii + 1] + 1j * y[jj]
            z01 = x[ii]     + 1j * y[jj + 1]
            z11 = x[ii + 1] + 1j * y[jj + 1]
            triangles.append([z00, z10, z11])
            triangles.append([z00, z11, z01])
    return np.array(triangles, dtype=complex)

def get_barycentres(triangles):
    return triangles.mean(axis=1)

def get_diameters(triangles):
    return np.max(np.abs(triangles - np.roll(triangles, 1, axis=1)), axis=1)

def refine_tri(triangle):
    a, b, c = triangle
    ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
    return [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]

# Colour map: the "Mana (Extended)" of the advection_diffusion_forms plates
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

# VTK WRITERS (as in pseudo_tool.py)
##################################################################

def _write_triangulated_vtk(path, x, y, triangles, residual):
    n = x.size
    n_tri = triangles.shape[0]
    eps = np.finfo(float).tiny
    log_residual = np.log10(np.maximum(residual, eps))
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Pseudo-spectra residual sigma_min(A - z M)\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {n} float\n")
        for i in range(n):
            f.write(f"{x[i]:.8e} {y[i]:.8e} 0.0\n")
        f.write(f"CELLS {n_tri} {4*n_tri}\n")
        for t in triangles:
            f.write(f"3 {int(t[0])} {int(t[1])} {int(t[2])}\n")
        f.write(f"CELL_TYPES {n_tri}\n")
        for _ in range(n_tri):
            f.write("5\n")
        f.write(f"POINT_DATA {n}\n")
        f.write("SCALARS residual float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for v in residual:
            f.write(f"{v:.8e}\n")
        f.write("SCALARS log10_residual float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for v in log_residual:
            f.write(f"{v:.8e}\n")

def _write_eigenvalues_vtk(path, lams):
    lams = list(lams) if lams is not None else []
    n = len(lams)
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Eigenvalues in the complex plane\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {n} float\n")
        for lam in lams:
            f.write(f"{float(lam.real):.8e} {float(lam.imag):.8e} 0.0\n")
        if n > 0:
            f.write(f"VERTICES {n} {2*n}\n")
            for i in range(n):
                f.write(f"1 {i}\n")
            f.write(f"POINT_DATA {n}\n")
            f.write("SCALARS real_part float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for lam in lams:
                f.write(f"{float(lam.real):.8e}\n")
            f.write("SCALARS imag_part float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for lam in lams:
                f.write(f"{float(lam.imag):.8e}\n")
            f.write("SCALARS magnitude float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for lam in lams:
                f.write(f"{abs(complex(lam)):.8e}\n")

def in_view(w):
    return (spec >= xmin) & (spec <= xmax) if len(spec) else np.array([], bool)

# PER-SWEEP PROGRESS SNAPSHOTS
##################################################################

FIGSIZE = (11, 11 * (ymax - ymin) / (xmax - xmin) + 0.9)

def plot_progress(sweep, records, active, field_samples):
    eigs = spec[in_view(spec)] if len(spec) else np.array([])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = {"in": "#b6e57f", "out": "0.92", "contour": "#2e7d32"}
    for cls, color in colors.items():
        polys = [np.column_stack([r["tri"].real, r["tri"].imag])
                 for r in records if r["cls"] == cls]
        if polys:
            ax.add_collection(PolyCollection(polys, facecolor=color,
                                             edgecolor="0.7", linewidth=0.15))
    if len(active):
        polys = [np.column_stack([t.real, t.imag]) for t in active]
        ax.add_collection(PolyCollection(polys, facecolor="none",
                                         edgecolor="0.4", linewidth=0.3))
    ax.plot(eigs, 0*eigs, 'ok', markersize=4)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\mathrm{Re}\, z$")
    ax.set_ylabel(r"$\mathrm{Im}\, z$")
    #ax.set_title(rf"Sweep {sweep}: classification state, $\epsilon = {epsilon}$")
    fig.tight_layout()
    fig.savefig(f"{OUT}/outer_state_sweep{sweep}.pdf")
    plt.close(fig)

    if len(field_samples) > 3:
        p = np.array([complex(px, py) for (px, py) in field_samples])
        vv = np.array(list(field_samples.values()))
        tr = Triangulation(p.real, p.imag)
        fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
        vmn, vmx = float(vv.min()), float(vv.max())
        cf = ax.tricontourf(tr, vv, levels=np.linspace(vmn, vmx, 25),
                            cmap=PV_MANA)
        if len(eigs):
            ax.scatter(eigs, 0*eigs, s=28, marker="o", facecolors="white",
                       edgecolors="black", linewidths=1.2, zorder=5)
        s0 = (xmax - xmin) / Nx
        ax.set_xlim(xmin + s0/2, xmax - s0/2)
        ax.set_ylim(ymin + s0/2, ymax - s0/2)
        ax.set_aspect("equal")
        ax.tick_params(direction="in", which="both")
        fig.colorbar(cf, ax=ax, pad=0.02, extend="neither")
        ax.set_xlabel(r"$\operatorname{Re}(z)$")
        ax.set_ylabel(r"$\operatorname{Im}(z)$")
        ax.set_title(rf"Sweep {sweep}: DWR-corrected field (local adaptivity)")
        fig.savefig(f"{OUT}/field_sweep{sweep}.pdf")
        plt.close(fig)


# Chatgpt help - save data per sweep so we can remake the plots later
def save_outer_sweep(sweep, records, active, field_samples, counts, its_hist, dofs_hist, marked_hist, cells_hist, filename=None): 

    """
    sweep : current sweep number
    records: classifications 
    active: active triangles
    field_samples: corrected phi values at barycentres
    its_hist: inner solve iteration
    counts: count ins, outs, contours, refines
    dofs_hist: inner dof list
    marked_hist: marking fractions from DWR
    cells_hist: final adapted mesh cell count

    """

    # converts [z1, z2, z3] to coordinate pairs
    # ex: [1+0.2j, 2+0.2j, 1.5+0.8j] becomes [[1, 0.2], [2, 0.2], [1.5, 0.8]]
    # resulting shape is (num tri, 3 vertices, two coords)
    completed_tri = np.asarray([[[p.real, p.imag] for p in r["tri"]] for r in records],dtype=float).reshape((-1, 3, 2)) 

    # Save the classifications for each triangle
    completed_cls = np.asarray([r["cls"] for r in records],dtype="U8")

    # Do same conversion for active triangles
    active_tri = np.asarray([[[p.real, p.imag] for p in t] for t in active],dtype=float).reshape((-1, 3, 2))

    # Extract barycentre coordinates
    field_points = np.asarray([[x, y] for x, y in field_samples.keys()],dtype=float).reshape((-1, 2))

    # Extract corrected phi values
    field_values = np.asarray(list(field_samples.values()))

    # Check filename
    if filename is None:
        filename = f"{OUT}/outer_sweep_{sweep:03d}.npz"

    # Save it all for later
    np.savez_compressed(
        filename,
        completed_triangles=completed_tri,
        completed_classes=completed_cls,
        completed_phi=np.asarray([r["phi"] for r in records]),
        completed_R=np.asarray([r["R"] for r in records]),
        active_triangles=active_tri,
        field_points=field_points,
        field_values=field_values,
        counts=np.asarray([
            counts["in"], counts["out"],
            counts["contour"], counts["refine"]
        ]),
        inner_iterations=np.asarray(its_hist),
        final_dofs=np.asarray(dofs_hist),
        marked_fractions=np.asarray(marked_hist),
        final_cells=np.asarray(cells_hist),
    )



# SELF-TEST: Phi_h(x+iy)^2 = Phi_h(x)^2 + y^2 holds exactly on V_h
##################################################################

if RANK == 0:
    dev = 0.0
    # probe = [1.1 + 0.35j, 2.3 - 0.4j] if SMOKE else \
    #         [0.7 + 0.3j, 1.6 - 0.35j, 2.6 + 0.3j, 3.4 - 0.25j]
    probe = [2.1 + 0.35j, 3.3 - 0.4j] if SMOKE else \
            [2.7 + 0.3j, 2.6 - 0.35j, 3.6 + 0.3j, 3.4 - 0.25j]
    for zval in probe:
        phi_c, corr_c, _, _, _ = dwr_solve(zval, 0.35)
        phi_r, _, _, _, _ = dwr_solve(zval.real, 0.35)
        dev = max(dev, abs((phi_c**2 - zval.imag**2) - phi_r**2))
    log_line(GREEN % f"self-test max |Phi(x+iy)^2 - y^2 - Phi(x)^2| = {dev:.3e}")
WORLD.barrier() # wait runtil rank 0 has tested

# THE OUTER LOOP
##################################################################

triangles = initial_triangulation(xmin, ymin, xmax, ymax, Nx, Ny)
records = []
field_samples = {}
# showcase = {}     # representative adapted meshes for the inner-mesh figure (not using right now)
sweep = 0

# THE FOLLOWING ITS CHATS SUGGESTED SPEED UP STRATEGY WHERE WE ONLY
# NEED TO DO AN EXPENSIVE EIGENSOLVE FOR EACH RE(Z) AND REUSE IT FOR DIFFERENT IMAGINARY PARTS!


while len(triangles) > 0 and sweep < max_sweeps:


    barycentres = get_barycentres(triangles)
    diameters = get_diameters(triangles)
    log_line(f"--- sweep {sweep}: {len(triangles)} active cells on "f"{NPROCS} ranks ---")

    # ------------------------------------------------------------
    # Group active outer cells by Re(z)
    # ------------------------------------------------------------

    groups = {}
    for k, zK in enumerate(barycentres):
        # round only for constructing the dictionary key
        xkey = round(float(np.real(zK)), 12)
        if xkey not in groups:
            groups[xkey] = []
        groups[xkey].append(k)
    # Fixed ordering is important so every MPI rank sees
    # exactly the same list of groups
    columns = sorted(groups.items())

    log_line(
    f"sweep {sweep}: {len(triangles)} active triangles, "
    f"{len(columns)} distinct Re(z) values, "
    f"{len(triangles) - len(columns)} eigensolves avoided")

    # ------------------------------------------------------------
    # One independent inner eigensolve per Re(z)
    # ------------------------------------------------------------

    local_results = []
    for col_idx in range(RANK, len(columns), NPROCS):


        xkey, cell_indices = columns[col_idx]

        # Use the actual x-coordinate rather than the rounded key
        x = float(np.real(barycentres[cell_indices[0]]))

        # Strictest outer tolerance required anywhere in this column.
        #
        # In your present outer algorithm these should normally all
        # be the same anyway, but this makes the code safe.
        h_column = min(diameters[k] for k in cell_indices)

        # ONE expensive adaptive eigensolve for this whole vertical column
        z_real = complex(x, 0.0)
        phi_x, corr_x, eta, solver, status = dwr_solve(z_real,h_column)

        final_mesh = unique_mesh(solver.problem.output_space)

        inner_hit_maxit = (getattr(solver, "termination_reason", None) == "max_it_reached")
        dof_limited = (getattr(solver, "termination_reason", None)== "max_dofs_reached")

        # --------------------------------------------------------
        # Quantities shared by the entire vertical column
        # --------------------------------------------------------

        phi_x = float(np.real(phi_x))
        corr_x = float(np.real(corr_x))
        eta = float(np.real(eta))

        # phi_x^2 is the discrete folded eigenvalue mu_h(x)
        mu_x = phi_x**2

        # If corr is your corrected Phi value, corr_x^2 is the
        # corresponding corrected folded eigenvalue at x
        mu_corr_x = corr_x**2

         # --------------------------------------------------------
        # Create an ordinary result for EVERY outer triangle
        # in this vertical column
        # --------------------------------------------------------

        for k in cell_indices:

            zK = barycentres[k]
            hK = diameters[k]
            y = float(np.imag(zK))

            # Exact self-adjoint shift:
            #
            # mu_h(x + iy) = mu_h(x) + y^2
            #
            # hence
            #
            # Phi_h(x + iy) = sqrt(mu_h(x) + y^2)
            phi = np.sqrt(mu_x + y**2)
            # Same transformation for your corrected Phi
            corr = np.sqrt(mu_corr_x + y**2)

            # eta is the DWR estimator for the folded eigenvalue mu.
            # It is IDENTICAL along the whole vertical column.
            eta_target = hK**2
            eta_ratio = eta / eta_target if eta_target > 0 else np.inf

            # Save the local results
            local_results.append({
                "cell": int(k),
                "zK": complex(zK),
                "hK": float(hK),
                "phi": float(np.real(phi)),
                "corr": float(np.real(corr)),
                "eta": float(np.real(eta)),
                "eta_target": float(eta_target),
                "eta_ratio": float(eta_ratio),
                "iterations": len(solver.Ndofs_vec),
                "dofs": solver.Ndofs_vec[-1],
                "cells": final_mesh.num_cells(),
                "max_it_limited": inner_hit_maxit,
                "dof_limited": dof_limited,
                "phi_nonfinite": status["phi_nonfinite"],
                "corr_nonfinite": status["corr_nonfinite"],
                "eta_nonfinite": status["eta_nonfinite"],
                "marked_fractions": [
                    float(x) for x in getattr(solver, "marked_fractions", [])],})






    # Measure usage
    # Every rank measures its own peak memory after its assigned z-solves
    rss_local = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux commonly reports kilobytes
    if sys.platform == "darwin":
        rss_local_gb = rss_local / 1e9
    else:
        rss_local_gb = rss_local / 1e6
    rss_max = WORLD.reduce( rss_local_gb,op=MPI.MAX,root=0 )


    # Still inside the sweep, but after the k loop
    gathered = WORLD.gather(local_results, root=0)

    # combine results from each rank
    if RANK == 0:

        # Save the results over each rank
        # First list comprehension is same as 
        # for rank_results in gathered:
        #    for result in rank_results:
        #       results.append(result)
        results = [result for rank_results in gathered for result in rank_results]
        results.sort(key=lambda result: result["cell"])
        its_hist = [result["iterations"] for result in results]
        dofs_hist = [result["dofs"] for result in results]
        cells_hist = [result["cells"] for result in results]
        marked_hist = [fraction for result in results for fraction in result["marked_fractions"]]



        # Classify time!!
        new_triangles = []
        counts = {"in": 0, "out": 0,"contour": 0,"refine": 0,}

         # Classify every completed z-cell
        for result in results:

            # Set k and zk
            k = result["cell"]
            zK = barycentres[k]


            # Check inner loop max its and dof
            if result["max_it_limited"]:
                log_line(RED % (f"WARNING: max_it reached at cell "f"{result['cell']}"))

            if result["dof_limited"]:
                log_line(RED % (
                        f"WARNING: max_dofs reached at cell "
                        f"{result['cell']}, "
                        f"dofs = {result['dofs']}"))

            # Check for nans and infs
            if result["phi_nonfinite"]:
                log_line(RED % (
                f"WARNING: WE QUIT because phi is NaN or infinite at "
                f"cell {k}, z = {zK:.3f}"))
                WORLD.Abort(1)

            if result["corr_nonfinite"]:
                log_line(RED % (
                    f"WARNING: WE QUIT because corrected phi is NaN or infinite at "
                    f"cell {k}, z = {zK:.3f}"))
                WORLD.Abort(1)


            if result["eta_nonfinite"]:
                log_line(RED % (
                    f"WARNING: WE QUIT because error estimate is NaN or infinite at "
                    f"cell {k}, z = {zK:.3f}"))
                WORLD.Abort(1)


            # Pull needed results
            triangle = triangles[k]
            hK = diameters[k]
            phi = result["phi"]
            corr = result["corr"]
            eta = result["eta"]

            

            # save correction
            field_samples[(round(zK.real, 8), round(zK.imag, 8))] = corr

            # Refinement radius
            R = hK + np.sqrt(eta)

            # Classify
            if phi + R < epsilon:
                cls = "in"
            elif phi - R > epsilon:
                cls = "out"
            elif hK < diam_tol:
                cls = "contour"
            else: # still active (refine)
                cls = "refine"
                new_triangles.extend(refine_tri(triangle))

            # Update classification counts
            counts[cls] += 1
            # Save classified triangles
            if cls != "refine":records.append({"tri": triangle,"phi": phi,"R": R,"cls": cls,})


        
        # Check how many times we hit maxits or maxdof
        n_max_it = sum(result["max_it_limited"] for result in results)
        n_max_dofs = sum(result["dof_limited"] for result in results)

        # Extra checks 
        ratios = np.array([r["eta_ratio"] for r in results], dtype=float)

        dof_limited_results = [r for r in results if r["dof_limited"]]
        dof_ratios = np.array(
            [r["eta_ratio"] for r in dof_limited_results],
            dtype=float
        )

        log_line(
            f"eta/target ratio: "
            f"median {np.median(ratios):.2e}, "
            f"p90 {np.quantile(ratios, 0.90):.2e}, "
            f"max {np.max(ratios):.2e}"
        )

        if len(dof_ratios):
            log_line(
                f"max-dof eta/target ratio: "
                f"min {np.min(dof_ratios):.2e}, "
                f"median {np.median(dof_ratios):.2e}, "
                f"p90 {np.quantile(dof_ratios, 0.90):.2e}, "
                f"max {np.max(dof_ratios):.2e}"
            )

        
        # print DOF report
        if dof_limited_results:
            
            sample_idx = np.linspace(
                0,
                len(dof_limited_results) - 1,
                min(5, len(dof_limited_results)),
                dtype=int
            )

            for ii in sample_idx:
                r = dof_limited_results[ii]
                log_line(
                    f"MAXDOF SAMPLE: "
                    f"cell={r['cell']}, "
                    f"z={r['zK']:.4f}, "
                    f"hK={r['hK']:.4e}, "
                    f"eta={r['eta']:.4e}, "
                    f"target={r['eta_target']:.4e}, "
                    f"ratio={r['eta_ratio']:.2e}, "
                    f"dofs={r['dofs']}"
                )

        
        # Print a report
        log_line(GREEN % (
            f"sweep {sweep}: "
            f"{counts['in']} in, "
            f"{counts['out']} out, "
            f"{counts['contour']} contour, "
            f"{counts['refine']} refined; "
            f"inner its avg {np.mean(its_hist):.2f}, "
            f"final cells avg {np.mean(cells_hist):.0f}, "
            f"final cells max {np.max(cells_hist):.0f}, "
            f"final dofs avg {np.mean(dofs_hist):.0f}, "
            f"final dofs max {np.max(dofs_hist):.0f}, "
            f"max_it limited {n_max_it}, "
            f"max_dofs limited {n_max_dofs}, "
            f"marked fraction avg "
            f"{np.mean(marked_hist) if marked_hist else 0:.2f}, "
            f"peak RSS across ranks {rss_max:.2f} GB"))

        # New triangle list
        triangles = np.asarray(new_triangles, dtype=complex)

        # Rank 0 alone writes logs, plots, and snapshots
        save_outer_sweep(sweep, records, triangles, field_samples, counts, its_hist, dofs_hist, marked_hist, cells_hist)
        plot_progress(sweep, records, triangles, field_samples)

    else:
        triangles = None

    # Outside the k loop and after rank-0 processing:
    # give every rank the next sweep's triangles
    triangles = WORLD.bcast(triangles, root=0)
    sweep += 1




# We hit max sweeps
if RANK==0 and len(triangles):

    log_line(RED % f"max_sweeps hit with {len(triangles)} cells left, "
                    "kept as contour")
    for t in triangles:
        records.append({"tri": t, "phi": np.nan, "R": np.nan, "cls": "contour"})
        
    # update counts
    counts["contour"] += len(triangles)

    # do a final save
    save_outer_sweep(sweep, records,[], field_samples, counts, [], [], [], [], filename=f"{OUT}/outer_sweep_final.npz",)


# VALIDATE AGAINST THE REFERENCE DISKS
##################################################################

if RANK == 0:
    if len(spec):
        mis = 0
        for r in records:
            pts = list(r["tri"]) + [r["tri"].mean()]
            if r["cls"] == "in" and any(exact_gamma(p) >= epsilon for p in pts):
                mis += 1
            if r["cls"] == "out" and any(exact_gamma(p) <= epsilon for p in pts):
                mis += 1
        log_line(GREEN % f"classified {len(records)} cells, {mis} "
                        "misclassified against the reference epsilon-disks")


# OUTPUT:
##################################################################

if RANK == 0:

    # OUTPUT 1: THE DWR-CORRECTED FIELD, VTK + PDF
    ##################################################################

    pts, vals = [], []
    for (px, py), v in field_samples.items():
        pts.append(complex(px, py))
        vals.append(v)
    pts = np.array(pts)
    residual_f = np.array(vals)
    triang = Triangulation(pts.real, pts.imag)
    log_line(GREEN % f"field assembled from {len(pts)} barycentre "
                    "evaluations (no extra solves)")

    _write_triangulated_vtk(f"{OUT}/maxwell2d_dwr_local_pseudospectra.vtk",
                            pts.real, pts.imag, triang.triangles, residual_f)
    _write_eigenvalues_vtk(f"{OUT}/maxwell2d_dwr_local_eigenvalues.vtk",
                        [complex(w) for w in spec if xmin <= w <= xmax])
    PETSc.Sys.Print(GREEN % "wrote maxwell2d_dwr_local VTK files")

    eigs = spec[in_view(spec)] if len(spec) else np.array([])
    vmin, vmax = float(residual_f.min()), float(residual_f.max())
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    cf = ax.tricontourf(triang, residual_f, levels=np.linspace(vmin, vmax, 25),
                        cmap=PV_MANA, vmin=vmin, vmax=vmax)
    ax.tricontour(triang, residual_f, levels=np.linspace(vmin, vmax, 9)[1:-1],
                colors="k", linewidths=0.35, alpha=0.4)
    if len(eigs):
        ax.scatter(eigs, 0*eigs, s=28, marker="o", facecolors="white",
                edgecolors="black", linewidths=1.2, zorder=5)
    ax.set_xlabel(r"$\operatorname{Re}(z)$")
    ax.set_ylabel(r"$\operatorname{Im}(z)$")
    s0 = (xmax - xmin) / Nx
    ax.set_xlim(xmin + s0/2, xmax - s0/2)
    ax.set_ylim(ymin + s0/2, ymax - s0/2)
    ax.set_aspect("equal")
    ax.tick_params(direction="in", which="both")
    cbar = fig.colorbar(cf, ax=ax, pad=0.02, extend="neither")
    cbar.set_label(r"$\sigma_{\min}(A - z)$")
    ax.set_title(rf"Folded Maxwell on the {DOMAIN}, {ELEMENT}$_{deg}$, "
                r"local DWR adaptivity", fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/pseudospectra_field.{ext}", dpi=300)
    plt.close(fig)

    # OUTPUT 2: THE OUTER-LOOP MESH, WITH ONLY THE CONTOUR GUESS
    ##################################################################

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.add_collection(PolyCollection(
        [np.column_stack([r["tri"].real, r["tri"].imag]) for r in records],
        facecolor="none", edgecolor="0.75", linewidth=0.2))
    contour_polys = [np.column_stack([r["tri"].real, r["tri"].imag])
                    for r in records if r["cls"] == "contour"]
    if contour_polys:
        ax.add_collection(PolyCollection(contour_polys, facecolor="#2e7d32",
                                        edgecolor="#2e7d32", linewidth=0.2))
    ax.plot(eigs, 0*eigs, 'ok', markersize=4)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\mathrm{Re}\, z$")
    ax.set_ylabel(r"$\mathrm{Im}\, z$")
    ax.set_title(rf"Outer-loop mesh and contour guess, $\epsilon = {epsilon}$, "
                rf"Maxwell on the {DOMAIN} (local DWR)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/pseudospectra_contour.pdf")
    plt.close(fig)

    # OUTPUT 3: THE ADAPTED INNER MESHES
    ##################################################################
    # if showcase:
    #     keys = sorted(showcase)[:6]
    #     ncol = min(3, len(keys))
    #     nrow = int(np.ceil(len(keys) / ncol))
    #     fig, axes = plt.subplots(nrow, ncol, figsize=(4*ncol, 4*nrow),
    #                              squeeze=False)
    #     for ax, key in zip(axes.ravel(), keys):
    #         zK, coords, cells, nits = showcase[key]
    #         tri = Triangulation(coords[:, 0], coords[:, 1], cells)
    #         ax.triplot(tri, linewidth=0.35, color="0.25")
    #         ax.set_aspect("equal")
    #         ax.set_title(rf"$z = {zK:.2f}$: {len(cells)} cells, {nits} its",
    #                      fontsize=9)
    #         ax.tick_params(labelsize=7)
    #     for ax in axes.ravel()[len(keys):]:
    #         ax.axis("off")
    #     fig.suptitle(f"Adapted inner meshes (local DWR marking), {DOMAIN}",
    #                  fontsize=11)
    #     fig.tight_layout()
    #     fig.savefig(f"{OUT}/inner_meshes.pdf")
    #     plt.close(fig)
    #     PETSc.Sys.Print(GREEN % f"wrote inner_meshes.pdf with {len(keys)} examples")

    log_line(GREEN % f"all output in {OUT}/")