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
import resource
from netgen.occ import *

# import umbertos_adaptivefoldedeigensolver as _maggie
import firedrake.umbertos_adaptivefoldedeigensolver as _maggie
from firedrake.umbertos_adaptivefoldedeigensolver import (
    GoalAdaptiveFoldedEigenSolver, residual, both,
    mixed_inner, mixed_weighted_inner)
from finat.ufl import BrokenElement, FiniteElement



epsilon = 0.25    # pseudospectral level
diam_tol = 0.005  # stop refining a z-cell below this diameter
max_sweeps = 10   # max number of outer loop iterations 
xmin, xmax, ymin, ymax = 2, 4, -1.0, 1.0 # Setting z grid bounds
Nx, Ny = 8, 8    # initial z-grid, chosen so the cells are isotropic
N0 = 32           # initial (coarse) interval mesh for every inner solve
deg = 1           # Going to solve the folded problem with CG1 for both mixed spaces



# THE FOLDED MAXWELL PROBLEM
##################################################################

z = Constant(0)   # spectral parameter, shared by every mesh's forms

def make_problem(mesh):


    # Spaces
    complex_unit = Constant(1j)
    V0 = VectorFunctionSpace(mesh, "CG", deg, dim=2)
    V1 = FunctionSpace(mesh, "CG", deg)
    V = MixedFunctionSpace([V0, V1])
    u = TrialFunction(V)
    (E, H) = split(u)
    v = TestFunction(V)
    (F, G) = split(v)

    # Bcs
    bcs = [
    # E_x = 0 on horizontal edges
    DirichletBC(V.sub(0).sub(0), 0, (bottom, top)),
    # E_y = 0 on vertical edges
    DirichletBC(V.sub(0).sub(1), 0, (left, right)),]

    # 2D curls 
    # Scalar rot
    def rot_s(E):
        return E[1].dx(0) - E[0].dx(1)
    # Vector rot
    def rot_v(H):
        return as_vector([H.dx(1), -H.dx(0)])

    
    A = (
        # A^* M^{-1} A terms
        inner(rot_s(E), rot_s(F))*dx
        + inner(rot_v(H), rot_v(G))*dx
        # -2 z A terms
        - (conj(z) + z) * inner(complex_unit*rot_v(H), F)*dx
        + (conj(z) + z) * inner(complex_unit*rot_s(E), G)*dx
        # |z|^2 M terms
        + conj(z) * z * inner(E, F)*dx
        + conj(z) * z * inner(H, G)*dx
        )

    # M form
    M = inner(E, F)*dx + inner(H, G)*dx

    problem = LinearEigenproblem(A, M, bcs=bcs, bc_shift=1e6, restrict=False)
    problem._original_A = A
    problem._original_M = M
    problem._original_bcs = bcs
    return problem


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
        "max_it": 8,           # genuine inner adaptivity this time
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


# Starting mesh
square = WorkPlane().Rectangle(pi, pi).Face()
# Maximum element size
initial_mesh_size = np.pi / N0
# Set the boundary labels
square.edges.Min(X).name = "left" # x=0
square.edges.Max(X).name = "right" # x=pi
square.edges.Min(Y).name = "bottom" # y=0
square.edges.Max(Y).name = "top" # y=pi
# Create the 2D mesh
geo = OCCGeometry(square, dim=2)
ngm = geo.GenerateMesh(maxh=initial_mesh_size)
# Take the named boundaries and +1 to get netgen labelling
names = ngm.GetRegionNames(codim=1)
left = names.index("left") + 1
right = names.index("right") + 1
bottom = names.index("bottom") + 1
top = names.index("top") + 1
# PARALLEL CHANGE 2: (Codex)
# Give every MPI rank its own independent serial Firedrake mesh.
starting_mesh = Mesh(ngm)

# Enriched-problem cache for the (permanent) base problem only; adapted
# meshes are transient, so caching them would only leak memory
base_problem = make_problem(starting_mesh) # set the standard base problem to reuse
_reconstruct_orig = _maggie._reconstruct_eig_degree # save the original reconstruction function
_base_enriched = {} # where we will save enriched versions of base problem to be reused
def _reconstruct_memo(problem, extra_degree): # new reconstructtion
    if problem is base_problem: # if its the base problem then
        key = tuple(extra_degree)
        if key not in _base_enriched: # check if we have already made desired enrichment 
            _base_enriched[key] = _reconstruct_orig(problem, extra_degree) # if not, make it and save it
        return _base_enriched[key] # return the enriched problem
    return _reconstruct_orig(problem, extra_degree) # if not base problem, do standard enrichment
_maggie._reconstruct_eig_degree = _reconstruct_memo # patch-over original reconstructor

_ncalls = [0] # where we will save number of calls for dwr_solve

def dwr_solve(zval, hK):
    _ncalls[0] += 1
    if _ncalls[0] % 200 == 0: # every 200 calls, throw out cyclic garbage to avoid memory leaks
        gc.collect()   # SLEPc factorisations and adapted-mesh spaces sit in
                       # cyclic garbage that Python's GC heuristics undercount
    # Call the inner loop 
    z.assign(zval)
    solver = GoalAdaptiveFoldedEigenSolver(
        base_problem, m_form=m_form, initial_space=(), target=0.0,
        epsilon=epsilon, diam_cond=hK**2, diameter=hK,
        imag_tol=1e-12, mult_tol=1e-2, solver_parameters=sp)
    solver.solve()

    # Pull the desired phi (uncorrected)
    # Maggie change - I think this could hide issues
    # phi = float(np.real(solver.matts_phi))
    # if not np.isfinite(phi): # if nan, try to recompute
    #     print(RED % f"WARNING - the phi was saved as NAN or INF!!")
    #     phi = float(np.sqrt(max(np.real(solver._lam_h), 0.0)))
    # corr = float(np.real(solver.corrected_phi)) # pull the corrected phi
    # if not np.isfinite(corr): # if nan then fallback on phi
    #     print(RED % f"WARNING - the corrected phi was saved as NAN or INF!!")
    #     corr = phi
    # eta = float(abs(solver.signed_error)) # pull |error est|
    phi = solver.matts_phi
    if not np.isfinite(phi): 
        print(RED % f"WARNING - the phi was saved as NAN or INF!!")
        # phi = float(np.sqrt(max((solver._lam_h), 0.0)))
    corr = solver.corrected_phi # pull the corrected phi
    if not np.isfinite(corr): 
        print(RED % f"WARNING - the corrected phi was saved as NAN or INF!!")
        # corr = phi
    eta = abs(solver.signed_error) # pull |error est|
    if not np.isfinite(eta): 
        print(RED % f"WARNING - the signed error estimate was saved as NAN or INF!!")

    return phi, corr, eta, solver

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


# Actual spectrum:
# exact_omega_squared = np.array([n**2 + m**2 for n in range(10) for m in range(10) if (n**2 + m**2 <= 4**2 and n**2 + m**2 >= 2**2) ])
# exact_omega = np.sqrt(exact_omega_squared)
# spec = np.sort(exact_omega)
# def exact_gamma(zval):
#     return np.min(np.abs(zval - spec))
omega_squared = sorted({
    n*n + m*m
    for n in range(10)
    for m in range(10)
    if xmin**2 <= n*n + m*m <= xmax**2})
spec = np.sqrt(np.array(omega_squared, dtype=float))
def exact_gamma(zval):
    return float(np.min(np.abs(zval - spec)))

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
    fig.savefig(f"umberto_maxwell_fullrun_output/maxwell/outer_state_sweep{sweep}.pdf")
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
        fig.savefig(f"umberto_maxwell_fullrun_output/maxwell/field_sweep{sweep}.pdf")
        plt.close(fig)

# SELF-TEST
##################################################################

output_dir = "umberto_maxwell_fullrun_output/maxwell"
os.makedirs(output_dir, exist_ok=True)

# Some starting sanity tests
dev = 0.0
corr_err = 0.0
for zval in [2.4 + 0.3j, 2.7 - 0.6j, 3.5 + 0.4j, 3.8 + 0.2j]: # Test four points in the plane 
    phi_c, corr_c, _, _ = dwr_solve(zval, 0.35) # run inner loop on complex number, diam = 0.35
    phi_r, _, _, _ = dwr_solve(zval.real, 0.35) # run inner loop on real part of complex number, diam = 0.35
    dev = max(dev, abs((phi_c**2 - zval.imag**2) - phi_r**2)) # check if Phi(z+iy)^2 - y^2 - Phi(z)^2 is small. Keep biggest (worst) result
    corr_err = max(corr_err, abs(corr_c - exact_gamma(zval))) # check if |Phi_corrected - dist(z, Z)| is small. Keep biggest (worst) result
print(GREEN % f"self-test max |Phi(x+iy)^2 - y^2 - Phi(x)^2| = {dev}")
print(GREEN % f"self-test max |corrected_phi - dist(z, Z)|  = {corr_err}")



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
    print(RED % f"---- [SWEEP {sweep}: {len(triangles)} active cells] ----")

    # Prepare for the sweep
    new_triangles = [] # stores triangles for next sweep (child triangles of refined triangles)
    counts = {"in": 0, "out": 0, "contour": 0, "refine": 0} # counts of how many triangles are classified as in, out, contour, or refine
    its_hist, dofs_hist, marked_hist = [], [], [] # histories of inner loop iterations, dofs, and marked fractions


    # The sweep over the current triangles
    for k in range(len(triangles)):

        # The Inner Loop
        zK, hK = barycentres[k], diameters[k] # get the barycentre and diameter of the k-th triangle
        phi, corr, eta, solver = dwr_solve(zK, hK) # run the inner loop until error <= hk^2
        R = hK + np.sqrt(eta) # Refinement criteria 

        # Save inner loop results for this triangle
        field_samples[(round(zK.real, 8), round(zK.imag, 8))] = corr # save the DWR-corrected Phi at the barycentre of the triangle
        its_hist.append(len(solver.Ndofs_vec)) # save the number of inner iterations for this triangle
        dofs_hist.append(solver.Ndofs_vec[-1]) # save the final number of dofs for this triangle
        # marked_hist.extend(getattr(solver, "marked_fractions", [])) # save the marked fractions for this triangle

        # Classify the cell
        if phi + R < epsilon: # in the pseudospectrum
            cls = "in"
        elif phi - R > epsilon: # outside the pseudospectrum
            cls = "out"
        elif hK < diam_tol: # too small to refine further, classify as contour
            cls = "contour"
        else: # needs refinement
            cls = "refine"
            new_triangles.extend(refine(triangles[k]))
        # Incremenets the counter for the classification of this triangle
        counts[cls] += 1
        # If the triangle is not to be refined, it is classified fully and can be recorded for the final output
        if cls != "refine": 
            records.append({"tri": triangles[k], "phi": phi, "R": R, "cls": cls})

        # # Keep one adapted mesh per region of interest for the showcase
        # if len(solver.Ndofs_vec) > 1:
        #     key = int(np.round(zK.real)) # A region is the integer part of the real part for the barycentres 
        #     # If we have not saved a mesh for this region yet, 
        #     # # or if the current mesh has more dofs than the saved one, save it for this region
        #     if key not in showcase or len(solver.Ndofs_vec) > showcase[key][2]: 
        #         mesh = solver.problem.output_space.mesh().unique()
        #         xs = np.sort(np.real(mesh.coordinates.dat.data_ro)) # pull the final mesh
                # showcase[key] = (zK, xs, len(solver.Ndofs_vec)) # save the barycentre, the final mesh, and the number of inner iterations for this triangle

        # Report on the inner loop
        print(BLUE % (f"[sweep {sweep}] z = {zK:.3f}: Phi = {phi:.4f}, "
                      f"corr = {corr:.4f}, eta = {eta:.2e}, "
                      f"its = {len(solver.Ndofs_vec)}, "
                      f"dofs = {solver.Ndofs_vec[-1]}, h = {hK:.3f} -> {cls}"))

    # Memory checker for a Mac (need 1e6 for linux??)
    #rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576 # change to linux
    # Report on the outer loop
    print(GREEN % (f"sweep {sweep}: {counts['in']} in, {counts['out']} out, "
                   f"{counts['contour']} contour, {counts['refine']} refined; "
                   f"inner its avg {np.mean(its_hist):.2f}, "
                   f"final dofs avg {np.mean(dofs_hist):.0f}, "
                   #f"marked fraction avg "
                   #f"{np.mean(marked_hist) if marked_hist else 0:.2f}, "
                   f"peak RSS {rss:.2f} GB"))
    # Update triangles with the new refined triangles for the next sweep
    triangles = np.array(new_triangles, dtype=complex)
    # Plot progress for this sweep (2 plots)
    plot_progress(sweep, records, triangles, field_samples)
    sweep += 1 # update sweep counter

# If we reached max sweeps and still have triangles left, classify them as contour (yikes)
if len(triangles):
    print(RED % f"max_sweeps hit with {len(triangles)} cells left, kept as contour")
    for t in triangles:
        records.append({"tri": t, "phi": np.nan, "R": np.nan, "cls": "contour"})

# VALIDATE AGAINST THE EXACT DISKS
##################################################################

mis = 0 # counts mis-classifications 
for r in records: # for each classified triangle 
    pts = list(r["tri"]) + [r["tri"].mean()] # pull z1, z2, z3, and zK
    # if the triangle was classified as in, make sure that at least gamma(z_i) < eps
    if r["cls"] == "in" and any(exact_gamma(p) >= epsilon for p in pts): 
        mis += 1
    # if the triangle was classified as out, make sure that at least gamma(z_i) > eps
    if r["cls"] == "out" and any(exact_gamma(p) <= epsilon for p in pts):
        mis += 1
print(GREEN % f"classified {len(records)} cells, {mis} misclassified "
      f"against the exact epsilon-disks")

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
print(GREEN % f"field assembled from {len(pts)} barycentre evaluations "
      "(no extra solves)") # we interpolate between values we computed to make the plots

# VTK OUTPUT:
# Write (Re(zK), Im(zK), 0), triangulation indices, and the Phi_corr
_write_triangulated_vtk(
    "umberto_maxwell_fullrun_output/maxwell/dwr_local_pseudospectra.vtk",
    pts.real, pts.imag, tri_indices, residual_f)
# Write the eact eigenvalies
_write_eigenvalues_vtk(
    "umberto_maxwell_fullrun_output/maxwell/dwr_local_eigenvalues.vtk",
    [complex(n) for n in spec if xmin <= n <= xmax])
print(GREEN % "wrote maxwell_dwr_local VTK files")

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
ax.set_title(r"Mixed 2D Maxwell Example, CG$_1$, "
             r"local DWR adaptivity", fontsize=10)
for ext in ("png", "pdf"):
    fig.savefig(f"umberto_maxwell_fullrun_output/maxwell/pseudospectra_field.{ext}",
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
             r"2D Mixed Maxwell (local DWR)")
fig.tight_layout()
fig.savefig("umberto_maxwell_fullrun_output/maxwell/pseudospectra_contour.pdf")
plt.close(fig)

# # OUTPUT 3: THE ADAPTED INNER MESHES
# ##################################################################

# # Save the representative inner loop meshes for each 'region'
# if showcase:
#     fig, axes = plt.subplots(len(showcase), 1,
#                              figsize=(11, 1.6*len(showcase) + 1),
#                              sharex=True) # one subplot for representative mesh
#     axes = np.atleast_1d(axes) # deal with case when len(showcase) = 1
#     for ax, key in zip(axes, sorted(showcase)): # process the keys in order
#         zK, xs, nits = showcase[key]
#         h = np.diff(xs)
#         ax.step(xs[:-1], h, where="post", linewidth=1.2) # plot h(x) vs x, take the right hj to represnt xj
#         ax.set_ylabel(r"$h(x)$")
#         ax.set_ylim(0, None)
#         ax.set_title(rf"$z = {zK:.3f}$: {len(xs)-1} cells after {nits} "
#                      rf"inner iterations", fontsize=9)
#     axes[-1].set_xlabel(r"$x$")
#     fig.suptitle("Adapted inner meshes (local DWR marking)", fontsize=11)
#     fig.tight_layout()
#     fig.savefig("umberto_fullrun_output/1D_poisson/inner_meshes.pdf")
#     plt.close(fig)
#     print(GREEN % f"wrote inner_meshes.pdf with {len(showcase)} examples")