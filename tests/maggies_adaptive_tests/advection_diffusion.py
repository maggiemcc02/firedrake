from firedrake import *
import numpy as np
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.tri import Triangulation
from matplotlib.colors import LinearSegmentedColormap
from ufl import conj, avg, dx, ds, dS
import os
import gc
import sys
import resource

# import umbertos_adaptivefoldedeigensolver as _maggie
import firedrake.umbertos_adaptivefoldedeigensolver as _maggie
from firedrake.umbertos_adaptivefoldedeigensolver import (
    GoalAdaptiveFoldedEigenSolver, residual, both,
    mixed_inner, mixed_weighted_inner)
from finat.ufl import BrokenElement, FiniteElement

# Pseudospectra of the 1D advection operator with Maggie's FULL inner loop:
# unlike pseudospectra_poisson_dwr.py, which uses only her global DWR
# estimate and escalates through uniform meshes, this driver runs her
# SOLVE -> ESTIMATE -> MARK -> REFINE cycle with genuinely LOCAL mesh
# adaptivity: her cell-wise indicators (bubble/cone projections), her
# max-marking, and hand-built non-uniform interval meshes in place of the
# netgen refine path (netgen has no 1D meshes; a 1D mesh is just a sorted
# vertex list, so we split the marked cells ourselves).

# Two 1D substitutions, both documented below: the FacetBubble element does
# not exist on intervals (facets are points), so the facet-residual space
# becomes broken P1 -- one dof per cell endpoint, exactly the bubble-on-a-
# facet role -- with the cones identically one in CG1; and warm starts go
# through cross-mesh interpolation instead of the netgen transfer manager.


# ADDED PARALLEL STUFF
##########################################################################
from mpi4py import MPI
WORLD = MPI.COMM_WORLD # distributes z values
RANK = WORLD.rank 
NPROCS = WORLD.size
# Every rank solves its assigned z values independently and serially.
# So, we need a communicator
LOCAL_COMM = MPI.COMM_SELF


# OUTPUT
##########################################################################
OUT = os.environ.get("OUTDIR","advection_output/")
if RANK == 0:
    os.makedirs(OUT, exist_ok=True)
WORLD.barrier()

_PROGRESS = f"{OUT}/progress.txt"
if RANK == 0:
    open(_PROGRESS, "w").close()
WORLD.barrier()

def log_line(msg):
    if RANK == 0:
        PETSc.Sys.Print(msg)
        with open(_PROGRESS, "a") as f:
            f.write(str(msg) + "\n")


# SETUP
##################################################################

# epsilon = 0.20   # pseudospectral level, spec(A) = the integers
# diam_tol = 0.01  # stop refining a z-cell below this diameter
# max_sweeps = 10   # max number of outer loop iterations 
# xmin, xmax, ymin, ymax = -0.5, 4.5, -1.0, 1.0 # Setting z grid bounds
# Nx, Ny = 20, 8    # initial z-grid, chosen so the cells are isotropic
# N0 = 32          # initial (coarse) interval mesh for every inner solve
# deg = 2           # Going to solve the folded problem with CG2 for both u and sigma
# epsilon = 0.20
# diam_tol = 0.01
max_sweeps = 10
# xmin, xmax = -15.0, 15.0
# ymin, ymax = -10.0, 10.0
# Nx, Ny = 15, 10
nu_value = 0.015
xmin, xmax = -6.0, 6.0
ymin, ymax = -5.0, 5.0
Nx, Ny = 24, 20
N0 = 16
deg = 2
epsilon = 0.20
diam_tol = 0.01

# 1D CELL INDICATORS (fork of her _compute_residual_indicators)
##################################################################

def _compute_residual_indicators_1d(F, z_err, options):
    from firedrake.assemble import assemble
    from firedrake.solving import solve

    # Recover mixed test space
    v, = F.arguments()
    V = v.function_space()
    mesh = V.mesh().unique()
    cell = mesh.ufl_cell()
    variant = "integral"

    # Construct interior bubbles
    # Interior bubbles, as in her code (degree dim+1 = 2 on intervals)
    B = FunctionSpace(mesh, "B", 2, variant=variant)
    bubbles = Function(B).assign(1)

    # DG space for cell residuals 
    DG_spaces = []
    for i, Vi in enumerate(V.subspaces):
        d = Vi.ufl_element().degree() + options.cell_residual_extra_degree[i]
        DG_spaces.append(FunctionSpace(mesh, "DG", d, variant=variant))
    DG = MixedFunctionSpace(DG_spaces)
    # Compute cell residuals 
    uc, vc = TrialFunction(DG), TestFunction(DG)
    ac = mixed_weighted_inner(uc, vc, bubbles) * dx
    Lc = residual(F, bubbles*vc)
    Rcell = Function(DG)
    solve(ac == Lc, Rcell, solver_parameters=options.sp_cell)

    # 1D substitution: FacetBubble does not exist on intervals; facets are
    # points, so broken P1 (one dof per cell endpoint) plays the facet-
    # bubble role and the cones are identically one
    cones = Function(FunctionSpace(mesh, "CG", 1)).assign(1)
    Q = MixedFunctionSpace(
        [FunctionSpace(mesh, BrokenElement(FiniteElement("Lagrange",
                                                         cell=cell, degree=1)))
         for Vi in V.subspaces])
    Qtest, Qtrial = TestFunction(Q), TrialFunction(Q)
    Lf = residual(F, Qtest) - mixed_inner(Rcell, Qtest)*dx
    facet_mass = mixed_weighted_inner(Qtrial, Qtest, 1/cones, first_arg=True)
    af = both(facet_mass) * dS + facet_mass * ds
    Rhat = Function(Q)
    # Compute facet residuals (broken P1, one dof per cell endpoint)
    solve(af == Lf, Rhat, solver_parameters=options.sp_facet)

    # Compute an indicator on each cell. 
    DG0 = FunctionSpace(mesh, "DG", degree=0)
    test = TestFunction(DG0)
    cell_weight = mixed_inner(Rcell, z_err)
    facet_weight = mixed_weighted_inner(Rhat, z_err, 1/cones, first_arg=True)
    eta_cell = assemble(inner(cell_weight, test)*dx
                        + inner(avg(facet_weight), both(test))*dS
                        + inner(facet_weight, test)*ds)
    # Take abs(indicator)
    with eta_cell.dat.vec as evec:
        evec.abs()
    return eta_cell

# Override maggies localization with this for 1D
_maggie._compute_residual_indicators = _compute_residual_indicators_1d

# THE FOLDED DIRAC PROBLEM AND HAND-BUILT 1D REFINEMENT
##################################################################

z = Constant(0)   # spectral parameter, shared by every mesh's forms

def make_problem(mesh, fold):
    V0 = FunctionSpace(mesh, "CG", deg)     # potential u
    V1 = FunctionSpace(mesh, "CG", deg)     # flux sigma ~ u'
    Z = MixedFunctionSpace([V0, V1])
    bc = [DirichletBC(Z.sub(0), 0, "on_boundary")]   # no BC on sigma

    nu = Constant(0.015)
    j = Constant(1j)
    U = TrialFunction(Z)
    (u, sigma) = split(U)
    V = TestFunction(Z)
    (v, tau) = split(V)

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


    M = inner(u, v)*dx + inner(sigma, tau)*dx

    if fold == 1:
        A = A1
    elif fold == 2:
        A = A2
    else:
        raise ValueError("fold must be 1 or 2")

    problem = LinearEigenproblem(A, M, bcs=bc, bc_shift=1e6, restrict=False)
    problem._original_A = A
    problem._original_M = M
    problem._original_bcs = bc
    return problem

# Take list of vertices and form a 1D mesh 
def interval_mesh_from_vertices(xs, comm=LOCAL_COMM):
    m = IntervalMesh(len(xs) - 1, 1.0, comm=comm) # make [0, 1] mesh 
    c = np.real(m.coordinates.dat.data_ro) # make desired coords real (they have 0 imag part)
    k = np.rint(c * (len(xs) - 1)).astype(int) # get mesh indices
    m.coordinates.dat.data[:] = xs[k] # plug in our desired coordinates 
    return m

# Updates to adaptive code for 1D
class GoalAdaptive1DFoldedEigenSolver(GoalAdaptiveFoldedEigenSolver):
    """Her solver with the netgen MARK->REFINE tail replaced by hand-built
    interval refinement; everything up to the markers is hers verbatim."""

    def __init__(self, *args, fold, **kwargs):
        self.fold = fold
        super().__init__(*args, **kwargs)

    # 1D refinement via inserting midpoints
    def refine_problem(self, markers, coef_map=None):
        # Pull the current mesh (unique to avoid mixed space issues)
        mesh = self.problem.output_space.mesh().unique()
        # Put the midpoints into an array 
        DG0 = FunctionSpace(mesh, "DG", 0)
        mids = np.real(Function(DG0).interpolate(
            SpatialCoordinate(mesh)[0]).dat.data_ro) # interpolate onto constant on each cell gives midpoints
        marked = np.zeros(len(mids), dtype=bool) # make a boolean array to store marked cells 
        for m in markers.subfunctions: # Mark the relevant cells
            marked |= (np.real(m.dat.data_ro) > 0.5)
        # record how aggresive the marking is 
        self.marked_fractions = getattr(self, "marked_fractions", [])
        self.marked_fractions.append(marked.sum() / len(marked))
        # Read old vertices and insert midpoints for the marked cells
        old_xs = np.sort(np.real(mesh.coordinates.dat.data_ro))
        new_xs = np.sort(np.concatenate([old_xs, mids[marked]]))
        # new_mesh = interval_mesh_from_vertices(new_xs)
        new_mesh = interval_mesh_from_vertices(new_xs,comm=LOCAL_COMM)
        # No hierarchy registration: the transfer manager is unused (warm
        # starts go through cross-mesh interpolation) and registering the
        # adapted mesh trips MixedFunctionSpace.set_hierarchy downstream
        #self.problem = make_problem(new_mesh) # rebuild problem on new mesh
        self.problem = make_problem(new_mesh, fold=self.fold)


    # No mesh hierarchy. 
    # Instead, pot-refinement will do the initial guess work for us
    def post_refinement(self):
        # pull the new space 
        V_new = self.problem.output_space
        # Interpolate componenentwise into the new space for inital guess
        initial_space = []
        try:
            for old_u in self.vecs:
                new_u = Function(V_new)
                for old_sub, new_sub in zip(old_u.subfunctions,
                                            new_u.subfunctions):
                    new_sub.interpolate(old_sub)
                initial_space.append(new_u)
        except Exception:
            PETSc.Sys.Print(RED % ("WARNING - Issues with interpolating to new mesh; setting empty initial space"))
            initial_space = []
        self.initial_space = tuple(initial_space)

# Same m_form as before 
def m_form(a, b):
    # As in parallel_outerloop_maxwell.py: the L2 inner product
    if hasattr(a, "subfunctions") and len(a.subfunctions) > 1:
        return assemble(sum(inner(ai, bi) * dx
                            for ai, bi in zip(a.subfunctions, b.subfunctions)))
    else:
        return assemble(inner(a, b) * dx)

# Solver parameters for inner loop
sp = {
    "goal_adaptive": {
        "tolerance": 1.0e-5,
        "max_it": 500, 
        "max_dofs": 20000,
        "dorfler_alpha": 0.5,
        "primal_extra_degree": (1, 1),
        "dual_extra_degree": (1, 1),
        "cell_residual_extra_degree": (1, 1),
        "facet_residual_extra_degree": (1, 1),
        "self_adjoint": True,
        "nev": 5,
        "verbose": True,
    },
    "eps_type": "krylovschur",
    "eps_tol": 1.0e-8,
    "st_type": "sinvert",
    "eps_target": 0,
}

# Enriched-problem cache for the (permanent) base problem only; adapted
# meshes are transient, so caching them would only leak memory
base_problem_1 = make_problem(IntervalMesh(N0, 1.0, comm=LOCAL_COMM), fold = 1) # set the standard base problem to reuse
base_problem_2 = make_problem(IntervalMesh(N0, 1.0, comm=LOCAL_COMM), fold = 2) # set the standard base problem to reuse
_reconstruct_orig = _maggie._reconstruct_eig_degree # save the original reconstruction function
_base_enriched = {} # where we will save enriched versions of base problem to be reused

# def _reconstruct_memo(problem, extra_degree): # new reconstructtion
#     if problem is base_problem: # if its the base problem then
#         key = tuple(extra_degree)
#         if key not in _base_enriched: # check if we have already made desired enrichment 
#             _base_enriched[key] = _reconstruct_orig(problem, extra_degree) # if not, make it and save it
#         return _base_enriched[key] # return the enriched problem
#     return _reconstruct_orig(problem, extra_degree) # if not base problem, do standard enrichment
# _maggie._reconstruct_eig_degree = _reconstruct_memo # patch-over original reconstructor


def _reconstruct_memo(problem, extra_degree):

    # Identify which permanent base problem this is
    if problem is base_problem_1:
        fold = 1
    elif problem is base_problem_2:
        fold = 2
    else:
        # Adapted problem: do not cache
        return _reconstruct_orig(problem, extra_degree)

    # Need fold in the key because the two enriched problems have
    # different bilinear forms
    key = (fold, tuple(extra_degree))

    if key not in _base_enriched:
        _base_enriched[key] = _reconstruct_orig(problem, extra_degree)

    return _base_enriched[key]
_maggie._reconstruct_eig_degree = _reconstruct_memo


_ncalls = [0] # where we will save number of calls for dwr_solve

# def dwr_solve(zval, hK):
#     _ncalls[0] += 1
#     if _ncalls[0] % 200 == 0: # every 200 calls, throw out cyclic garbage to avoid memory leaks
#         gc.collect()   # SLEPc factorisations and adapted-mesh spaces sit in
#                        # cyclic garbage that Python's GC heuristics undercount
#     # Call the inner loop 
#     z.assign(zval)
#     solver = GoalAdaptive1DFoldedEigenSolver(
#         base_problem, m_form=m_form, initial_space=(), target=0.0,
#         epsilon=epsilon, diam_cond=hK**2, diameter=hK,
#         imag_tol=1e-12, mult_tol=1e-2, solver_parameters=sp)
#     solver.solve()

#     # Pull the desired phi (uncorrected)
#     # Maggie change - I think this could hide issues
#     # phi = float(np.real(solver.matts_phi))
#     # if not np.isfinite(phi): # if nan, try to recompute
#     #     print(RED % f"WARNING - the phi was saved as NAN or INF!!")
#     #     phi = float(np.sqrt(max(np.real(solver._lam_h), 0.0)))
#     # corr = float(np.real(solver.corrected_phi)) # pull the corrected phi
#     # if not np.isfinite(corr): # if nan then fallback on phi
#     #     print(RED % f"WARNING - the corrected phi was saved as NAN or INF!!")
#     #     corr = phi
#     # eta = float(abs(solver.signed_error)) # pull |error est|
#     # phi = solver.matts_phi
#     # if not np.isfinite(phi): 
#     #     print(RED % f"WARNING - the phi was saved as NAN or INF!!")
#     #     # phi = float(np.sqrt(max((solver._lam_h), 0.0)))
#     # corr = solver.corrected_phi # pull the corrected phi
#     # if not np.isfinite(corr): 
#     #     print(RED % f"WARNING - the corrected phi was saved as NAN or INF!!")
#     #     # corr = phi
#     # eta = abs(solver.signed_error) # pull |error est|
#     # if not np.isfinite(eta): 
#     #     print(RED % f"WARNING - the signed error estimate was saved as NAN or INF!!")

#     phi = solver.matts_phi
#     corr = solver.corrected_phi
#     eta = abs(solver.signed_error)

#     status = {
#         "phi_nonfinite": not np.isfinite(phi),
#         "corr_nonfinite": not np.isfinite(corr),
#         "eta_nonfinite": not np.isfinite(eta),}

#     return phi, corr, eta, solver, status



def dwr_solve(zval, hK, fold):

    _ncalls[0] += 1
    if _ncalls[0] % 200 == 0: # every 200 calls, throw out cyclic garbage to avoid memory leaks
        gc.collect()   # SLEPc factorisations and adapted-mesh spaces sit in
                       # cyclic garbage that Python's GC heuristics undercount

    if fold == 1:
        base_problem = base_problem_1
    elif fold == 2:
        base_problem = base_problem_2
    else:
        raise ValueError("fold must be 1 or 2")

    z.assign(zval)

    solver = GoalAdaptive1DFoldedEigenSolver(
        base_problem,
        fold=fold,
        m_form=m_form,
        initial_space=(),
        target=0.0,
        epsilon=epsilon,
        diam_cond=hK**2 / 2,
        diameter=hK,
        imag_tol=1e-12,
        mult_tol=1e-2,
        solver_parameters=sp
    )

    solver.solve()

    phi = solver.matts_phi
    corr = solver.corrected_phi
    eta = abs(solver.signed_error)

    status = {
    "phi_nonfinite": not np.isfinite(phi),
    "corr_nonfinite": not np.isfinite(corr),
    "eta_nonfinite": not np.isfinite(eta),}

    return phi, corr, eta, solver, status



# STARTING HERMITAIN TEST
##################################################################
z.assign(14.333 - 3.333j)

for name, problem in [
    ("fold 1", base_problem_1),
    ("fold 2", base_problem_2),
]:
    A_mat = assemble(
        problem._original_A,
        mat_type="aij"
    ).petscmat

    M_mat = assemble(
        problem._original_M,
        mat_type="aij"
    ).petscmat

    PETSc.Sys.Print(
        name,
        "A Hermitian:", A_mat.isHermitian(tol=1e-10),
        "M Hermitian:", M_mat.isHermitian(tol=1e-10),
    )



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

def refine(triangle):
    a, b, c = triangle
    ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
    return [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]

# exact spectrum
nu_value = 0.015
positive_spec = np.array([ np.sqrt(1/(4*nu_value) + nu_value*(n*np.pi)**2) for n in range(1, 100 + 1) ]) 
spec = np.sort(np.concatenate([ -positive_spec, [0.0], positive_spec ]))

# def exact_gamma(zval):
#     return np.min(np.abs(zval - spec))

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

# PER-SWEEP PROGRESS SNAPSHOTS
##################################################################

def plot_progress(sweep, records, active, field_samples):
    eigs = spec[(spec >= xmin) & (spec <= xmax)]

    # FIRST PLOT IS OUTER SWEEP STATE (triangles coloured by classification)
    fig, ax = plt.subplots(figsize=(11, 4.9))
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
    ax.set_title(rf"Sweep {sweep}: classification state, $\epsilon = {epsilon}$")
    fig.tight_layout()
    fig.savefig(f"{OUT}/outer_state_sweep{sweep}.pdf")
    plt.close(fig)

    # SECOND PLOT IS DWR-CORRECTED FIELD (tricontourf of the field samples)
    if len(field_samples) > 3:
        p = np.array([complex(px, py) for (px, py) in field_samples])
        vv = np.array(list(field_samples.values()))
        tr = Triangulation(p.real, p.imag)
        fig, ax = plt.subplots(figsize=(11, 4.9), constrained_layout=True)
        vmn, vmx = float(vv.min()), float(vv.max())
        cf = ax.tricontourf(tr, vv, levels=np.linspace(vmn, vmx, 25),
                            cmap=PV_MANA)
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




# # Chatgpt help - save data per sweep so we can remake the plots later
def save_outer_sweep(sweep,records,active,field_samples,counts,results,filename=None):

    completed_tri = np.asarray(
        [[[p.real, p.imag] for p in r["tri"]] for r in records],
        dtype=float
    ).reshape((-1, 3, 2))

    completed_cls = np.asarray(
        [r["cls"] for r in records],
        dtype="U8"
    )

    active_tri = np.asarray(
        [[[p.real, p.imag] for p in t] for t in active],
        dtype=float
    ).reshape((-1, 3, 2))

    field_points = np.asarray(
        [[x, y] for x, y in field_samples.keys()],
        dtype=float
    ).reshape((-1, 2))

    field_values = np.asarray(list(field_samples.values()))

    if filename is None:
        filename = f"{OUT}/outer_sweep_{sweep:03d}.npz"

    np.savez_compressed(

        filename,

        # Final/classified outer cells accumulated so far
        completed_triangles=completed_tri,
        completed_classes=completed_cls,
        completed_phi=np.asarray([r["phi"] for r in records]),
        completed_R=np.asarray([r["R"] for r in records]),

        # Still-active outer cells
        active_triangles=active_tri,

        # Corrected Phi field
        field_points=field_points,
        field_values=field_values,

        # Classification counts
        counts=np.asarray([
            counts["in"],
            counts["out"],
            counts["contour"],
            counts["refine"]
        ]),

        # ----------------------
        # THIS SWEEP: BOTH FOLDS
        # ----------------------

        cells=np.asarray([r["cell"] for r in results]),

        phi=np.asarray([r["phi"] for r in results]),
        corr=np.asarray([r["corr"] for r in results]),

        phi1=np.asarray([r["phi1"] for r in results]),
        phi2=np.asarray([r["phi2"] for r in results]),

        corr1=np.asarray([r["corr1"] for r in results]),
        corr2=np.asarray([r["corr2"] for r in results]),

        eta1=np.asarray([r["eta1"] for r in results]),
        eta2=np.asarray([r["eta2"] for r in results]),

        iterations1=np.asarray([r["iterations1"] for r in results]),
        iterations2=np.asarray([r["iterations2"] for r in results]),

        dofs1=np.asarray([r["dofs1"] for r in results]),
        dofs2=np.asarray([r["dofs2"] for r in results]),

        final_cells1=np.asarray([r["cells1"] for r in results]),
        final_cells2=np.asarray([r["cells2"] for r in results]),

        max_it_limited1=np.asarray(
            [r["max_it_limited1"] for r in results]
        ),
        max_it_limited2=np.asarray(
            [r["max_it_limited2"] for r in results]
        ),

        dof_limited1=np.asarray(
            [r["dof_limited1"] for r in results]
        ),
        dof_limited2=np.asarray(
            [r["dof_limited2"] for r in results]
        ),

        active_fold=np.asarray(
            [r["active_fold"] for r in results]
        ),

        active_fold_corr=np.asarray(
            [r["active_fold_corr"] for r in results]
        ),

        # Flattened marking fractions. These are not one-per-z.
        marked_fractions1=np.asarray([
            x
            for r in results
            for x in r["marked_fractions1"]
        ]),

        marked_fractions2=np.asarray([
            x
            for r in results
            for x in r["marked_fractions2"]
        ]),
    )




# THE OUTER LOOP
##################################################################

triangles = initial_triangulation(xmin, ymin, xmax, ymax, Nx, Ny) # Create the initial z grid 
records = [] # where we store classified triangles and their phi, R, and class
field_samples = {} # where we store the DWR-corrected field samples at barycentres
showcase = {}     # representative adapted meshes for the inner-loops 
sweep = 0 # sweep counter

while len(triangles) > 0 and sweep < max_sweeps: # while we have unclassified triangles and we have not hit max sweeps

    # compute barycentres and diameters of the triangles for the acive triangles
    barycentres = get_barycentres(triangles)
    diameters = get_diameters(triangles)

    # Progress print
    log_line(RED % (
        f"---- [SWEEP {sweep}: "
        f"{len(triangles)} active cells on {NPROCS} ranks] ----"))

    # Prepare for the sweep
    # new_triangles = [] # stores triangles for next sweep (child triangles of refined triangles)
    # counts = {"in": 0, "out": 0, "contour": 0, "refine": 0} # counts of how many triangles are classified as in, out, contour, or refine
    # its_hist, dofs_hist, marked_hist = [], [], [] # histories of inner loop iterations, dofs, and marked fractions

    local_results = []

    for k in range(RANK, len(triangles), NPROCS):

        zK = barycentres[k]
        hK = diameters[k]

        # phi, corr, eta, solver, status = dwr_solve(zK, hK)
        # final_mesh = solver.problem.output_space.mesh().unique()
        phi1, corr1, eta1, solver1, status1 = dwr_solve(zK, hK, fold=1)
        phi2, corr2, eta2, solver2, status2 = dwr_solve(zK, hK, fold=2)
        phi = min(phi1, phi2)
        corr = min(corr1, corr2)
        final_mesh_1 = solver1.problem.output_space.mesh().unique()
        final_mesh_2 = solver2.problem.output_space.mesh().unique()


        # # check if we hit maxits
        inner_hit_maxit_1 = (getattr(solver1, "termination_reason", None) == "max_it_reached")
        inner_hit_maxit_2 = (getattr(solver2, "termination_reason", None) == "max_it_reached")
        # # Check our dof
        dof_limited_1 = (getattr(solver1, "termination_reason", None)== "max_dofs_reached")
        dof_limited_2 = (getattr(solver2, "termination_reason", None)== "max_dofs_reached")

        # Save results for rank
        local_results.append({
            "cell": int(k),

            # Combined Colbrook quantities
            "phi": float(np.real(phi)),
            "corr": float(np.real(corr)),

            # Fold 1
            "phi1": float(np.real(phi1)),
            "corr1": float(np.real(corr1)),
            "eta1": float(np.real(eta1)),
            "iterations1": len(solver1.Ndofs_vec),
            "dofs1": solver1.Ndofs_vec[-1],
            "cells1": final_mesh_1.num_cells(),
            "max_it_limited1": inner_hit_maxit_1,
            "dof_limited1": dof_limited_1,
            "phi_nonfinite1": status1["phi_nonfinite"],
            "corr_nonfinite1": status1["corr_nonfinite"],
            "eta_nonfinite1": status1["eta_nonfinite"],
            "marked_fractions1": [
                float(x) for x in getattr(solver1, "marked_fractions", [])
            ],

            # Fold 2
            "phi2": float(np.real(phi2)),
            "corr2": float(np.real(corr2)),
            "eta2": float(np.real(eta2)),
            "iterations2": len(solver2.Ndofs_vec),
            "dofs2": solver2.Ndofs_vec[-1],
            "cells2": final_mesh_2.num_cells(),
            "max_it_limited2": inner_hit_maxit_2,
            "dof_limited2": dof_limited_2,
            "phi_nonfinite2": status2["phi_nonfinite"],
            "corr_nonfinite2": status2["corr_nonfinite"],
            "eta_nonfinite2": status2["eta_nonfinite"],
            "marked_fractions2": [
                float(x) for x in getattr(solver2, "marked_fractions", [])
            ],
            # Useful diagnostic: which fold controls Phi?
            "active_fold": 1 if phi1 <= phi2 else 2,
            "active_fold_corr": 1 if corr1 <= corr2 else 2, })




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
        results = [result for rank_results in gathered for result in rank_results]
        results.sort(key=lambda result: result["cell"])


        # its_hist_1 = [result["iterations solver 1"] for result in results]
        # its_hist_2 = [result["iterations solver 2"] for result in results]
        # dofs_hist_1 = [result["dofs solver 1"] for result in results]
        # dofs_hist_2 = [result["dofs solver 2"] for result in results]
        # #cells_hist = [result["cells"] for result in results]
        # marked_hist_1 = [fraction for result in results for fraction in result["marked_fractions solver 1"]]
        # marked_hist_2 = [fraction for result in results for fraction in result["marked_fractions solver 2"]]
        its_hist_1 = [r["iterations1"] for r in results]
        its_hist_2 = [r["iterations2"] for r in results]
        dofs_hist_1 = [r["dofs1"] for r in results]
        dofs_hist_2 = [r["dofs2"] for r in results]
        cells_hist_1 = [r["cells1"] for r in results]
        cells_hist_2 = [r["cells2"] for r in results]
        marked_hist_1 = [frac for r in results for frac in r["marked_fractions1"]]
        marked_hist_2 = [frac for r in results for frac in r["marked_fractions2"]]



        # Classify time!!
        new_triangles = []
        counts = {"in": 0, "out": 0,"contour": 0,"refine": 0,}

         # Classify every completed z-cell
        for result in results:


            # Set k and zk
            k = result["cell"]
            zK = barycentres[k]


            # Check inner loop max its and dof
            if result["max_it_limited1"]:
                log_line(RED % (
                    f"WARNING: fold 1 hit max_it at cell {k}, "
                    f"z = {zK:.3f}, dofs = {result['dofs1']}"
                ))

            if result["max_it_limited2"]:
                log_line(RED % (
                    f"WARNING: fold 2 hit max_it at cell {k}, "
                    f"z = {zK:.3f}, dofs = {result['dofs2']}"
                ))

            if result["dof_limited1"]:
                log_line(RED % (
                    f"WARNING: fold 1 hit max_dofs at cell {k}, "
                    f"z = {zK:.3f}, dofs = {result['dofs1']}"
                ))

            if result["dof_limited2"]:
                log_line(RED % (
                    f"WARNING: fold 2 hit max_dofs at cell {k}, "
                    f"z = {zK:.3f}, dofs = {result['dofs2']}"
                ))

            for fold in (1, 2):

                if result[f"phi_nonfinite{fold}"]:
                    log_line(RED % (
                        f"WARNING: fold {fold} phi is NaN/Inf at "
                        f"cell {k}, z = {zK:.3f}"
                    ))
                    WORLD.Abort(1)

                if result[f"corr_nonfinite{fold}"]:
                    log_line(RED % (
                        f"WARNING: fold {fold} corrected phi is NaN/Inf at "
                        f"cell {k}, z = {zK:.3f}"
                    ))
                    WORLD.Abort(1)

                if result[f"eta_nonfinite{fold}"]:
                    log_line(RED % (
                        f"WARNING: fold {fold} DWR error is NaN/Inf at "
                        f"cell {k}, z = {zK:.3f}"
                    ))
                    WORLD.Abort(1)
                        


            # Get results for this cell
            triangle = triangles[k]
            hK = diameters[k]
            phi = result["phi"]
            corr = result["corr"]
            eta1 = result["eta1"]
            eta2 = result["eta2"]


            # save correction
            field_samples[(round(zK.real, 8), round(zK.imag, 8))] = corr

            # Refinement radius
            R = hK + np.sqrt(eta1 + eta2)

            # Classify
            if phi + R < epsilon:
                cls = "in"
            elif phi - R > epsilon:
                cls = "out"
            elif hK < diam_tol:
                cls = "contour"
            else: # still active (refine)
                cls = "refine"
                new_triangles.extend(refine(triangle))

            # Update classification counts
            counts[cls] += 1
            # Save classified triangles
            if cls != "refine":records.append({"tri": triangle,"phi": phi,"R": R,"cls": cls,})

        
        # # Print a report
        n_fold1_active = sum(r["active_fold"] == 1 for r in results)
        n_fold2_active = sum(r["active_fold"] == 2 for r in results)
        log_line(GREEN % (
            f"sweep {sweep}: "
            f"{counts['in']} in, "
            f"{counts['out']} out, "
            f"{counts['contour']} contour, "
            f"{counts['refine']} refined; "

            f"fold 1: its avg {np.mean(its_hist_1):.2f}, "
            f"dofs avg {np.mean(dofs_hist_1):.0f}, "
            f"dofs max {np.max(dofs_hist_1):.0f}, "
            f"cells avg {np.mean(cells_hist_1):.0f}, "
            f"marked avg "
            f"{np.mean(marked_hist_1) if marked_hist_1 else 0:.2f}; "

            f"fold 2: its avg {np.mean(its_hist_2):.2f}, "
            f"dofs avg {np.mean(dofs_hist_2):.0f}, "
            f"dofs max {np.max(dofs_hist_2):.0f}, "
            f"cells avg {np.mean(cells_hist_2):.0f}, "
            f"marked avg "
            f"{np.mean(marked_hist_2) if marked_hist_2 else 0:.2f}; "

            f"Phi minimum: fold 1 {n_fold1_active}/{len(results)}, "
            f"fold 2 {n_fold2_active}/{len(results)}; "

            f"peak RSS across ranks {rss_max:.2f} GB"))

        # New triangle list
        triangles = np.asarray(new_triangles, dtype=complex)
        # Rank 0 alone writes logs, plots, and snapshots
        save_outer_sweep(sweep,records,triangles,field_samples,counts,results)
        plot_progress(sweep, records, triangles, field_samples)

    else:
        triangles = None

    # Outside the k loop and after rank-0 processing:
    # give every rank the next sweep's triangles
    triangles = WORLD.bcast(triangles, root=0)
    sweep += 1

# If we reached max sweeps and still have triangles left, classify them as contour (yikes)
# We hit max sweeps
if RANK==0 and len(triangles):

    log_line(RED % f"max_sweeps hit with {len(triangles)} cells left, "
                    "kept as contour")
    for t in triangles:
        records.append({"tri": t, "phi": np.nan, "R": np.nan, "cls": "contour"})    
    # update counts
    counts["contour"] += len(triangles)
    # do a final save
    save_outer_sweep(sweep,records,[],field_samples,counts,[],filename=f"{OUT}/outer_sweep_final.npz")



# OUTPUT:
##################################################################

if RANK == 0:

    # OUTPUT 1: THE DWR-CORRECTED FIELD (barycentres only), VTK + PDF
    ##################################################################

    # The data stored in field_samples is of the form 
    # field_samples[(Re(zK), Im(zK)] = Phi_{corr}(zK)
    # Here, we plot the final z -> Phi_{corr}(z) contour plot
    # We also save VTK output 
    # note - the key is zK so a central child cell will overwrite its parent in field_samples

    # convert dictionary to arrays
    pts, vals = [], []
    for (px, py), phi in field_samples.items():
        pts.append(complex(px, py))
        vals.append(phi)
    pts = np.array(pts)
    residual_f = np.array(vals)
    triang = Triangulation(pts.real, pts.imag) # construct  triangulation where barycentres are vertices!! (not z grid)
    tri_indices = triang.triangles # integer array for plotting triangles
    log_line(GREEN % (
        f"field assembled from {len(pts)} "
        "barycentre evaluations"))
    # VTK OUTPUT:
    # Write (Re(zK), Im(zK), 0), triangulation indices, and the Phi_corr
    _write_triangulated_vtk(
        f"{OUT}/dwr_local_pseudospectra.vtk",
        pts.real, pts.imag, tri_indices, residual_f)
    # Write the eact eigenvalies
    _write_eigenvalues_vtk(
        f"{OUT}/dwr_local_eigenvalues.vtk",
        [complex(n) for n in spec if xmin <= n <= xmax])
    log_line(GREEN % "wrote dwr_local VTK files")


    # PDF AND PNG OUTPUT
    # Set colour range
    vmin, vmax = float(residual_f.min()), float(residual_f.max())
    levels = np.linspace(vmin, vmax, 25)
    # Draw filled contour for Phi_{corr}
    fig, ax = plt.subplots(figsize=(11, 4.9), constrained_layout=True)
    cf = ax.tricontourf(triang, residual_f, levels=levels,
                        cmap=PV_MANA, vmin=vmin, vmax=vmax)
    # Add in 7 black contours for Phi_{corr}
    iso_levels = np.linspace(vmin, vmax, 9)[1:-1]
    ax.tricontour(triang, residual_f, levels=iso_levels,
                colors="k", linewidths=0.35, alpha=0.4)
    # Overlay the exact eigenvalues
    eigs = spec[(spec >= xmin) & (spec <= xmax)]
    ax.scatter(eigs, 0*eigs, s=28, marker="o", facecolors="white",
            edgecolors="black", linewidths=1.2, zorder=5)
    # Create labels etc
    ax.set_xlabel(r"$\operatorname{Re}(z)$")
    ax.set_ylabel(r"$\operatorname{Im}(z)$")
    s0 = (xmax - xmin) / Nx # trim edges of domain, computed info starts at barycentre
    ax.set_xlim(xmin + s0/2, xmax - s0/2)
    ax.set_ylim(ymin + s0/2, ymax - s0/2)
    ax.set_aspect("equal")
    ax.tick_params(direction="in", which="both")
    cbar = fig.colorbar(cf, ax=ax, pad=0.02, extend="neither")
    cbar.set_label(r"$\sigma_{\min}(A - z)$")
    ax.set_title(r"Dirac splitting of $-\mathrm{d}^2/\mathrm{d}x^2$, CG$_2$, "
                r"local DWR adaptivity", fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/pseudospectra_field.{ext}",
                    dpi=300)
    plt.close(fig)

    # OUTPUT 2: THE OUTER-LOOP MESH, WITH ONLY THE CONTOUR GUESS
    ##################################################################

    fig, ax = plt.subplots(figsize=(11, 4.9))
    all_polys = [np.column_stack([r["tri"].real, r["tri"].imag]) for r in records]
    ax.add_collection(PolyCollection(all_polys, facecolor="none",
                                    edgecolor="0.75", linewidth=0.2))
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
                r"1D Poisson (local DWR)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/pseudospectra_contour.pdf")
    plt.close(fig)

    # OUTPUT 3: THE ADAPTED INNER MESHES
    ##################################################################

    # Save the representative inner loop meshes for each 'region'
    if showcase:
        fig, axes = plt.subplots(len(showcase), 1,
                                figsize=(11, 1.6*len(showcase) + 1),
                                sharex=True) # one subplot for representative mesh
        axes = np.atleast_1d(axes) # deal with case when len(showcase) = 1
        for ax, key in zip(axes, sorted(showcase)): # process the keys in order
            zK, xs, nits = showcase[key]
            h = np.diff(xs)
            ax.step(xs[:-1], h, where="post", linewidth=1.2) # plot h(x) vs x, take the right hj to represnt xj
            ax.set_ylabel(r"$h(x)$")
            ax.set_ylim(0, None)
            ax.set_title(rf"$z = {zK:.3f}$: {len(xs)-1} cells after {nits} "
                        rf"inner iterations", fontsize=9)
        axes[-1].set_xlabel(r"$x$")
        fig.suptitle("Adapted inner meshes (local DWR marking)", fontsize=11)
        fig.tight_layout()
        fig.savefig(f"{OUT}/inner_meshes.pdf")
        plt.close(fig)
        log_line(
        GREEN % (
            f"wrote inner_meshes.pdf with "
            f"{len(showcase)} examples"))