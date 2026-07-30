from firedrake import *
from netgen.occ import *
import numpy as np
import sys
from firedrake.outerloop_adaptivefoldedeigensolver import GoalAdaptiveFoldedEigenSolver
from z_grid_helper import *
from ufl import conj
import os
import matplotlib.pyplot as plt

# PARALLEL CHANGE 1: (Codex)
# COMM_WORLD distributes z-points. Each individual Firedrake solve will use
# COMM_SELF, so it remains serial.
world = COMM_WORLD
rank = world.rank
nprocs = world.size




# FORMULATE THE FOLDED EIGENPROBLEM
###################################################################

# Mesh and spaces


# Set the complex unit 
complex_unit = Constant(1j)

# Define initial mesh (netgen) ---------------------
N = 32
square = WorkPlane().Rectangle(pi, pi).Face()
# Maximum element size
initial_mesh_size = np.pi / N
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
mesh = Mesh(ngm, comm=COMM_SELF)


# The Mixed space
V0 = VectorFunctionSpace(mesh, "CG", 1, dim=2)
V1 = FunctionSpace(mesh, "CG", 1)
V = MixedFunctionSpace([V0, V1])
u = TrialFunction(V)
(E, H) = split(u)
v = TestFunction(V)
(F, G) = split(v)
# # The BC's (zero tangential trace)
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


# A  form

z = Constant(0) # placeholder for z
# A = (
#       # A^* M^{-1} A terms
#       inner(rot_s(E), rot_s(F))*dx
#     + inner(rot_v(H), rot_v(G))*dx
#       # -2 z A terms
#     - 2 * z * inner(complex_unit*rot_v(H), F)*dx
#     + 2 * z * inner(complex_unit*rot_s(E), G)*dx
#       # |z|^2 M terms
#     + conj(z) * z * inner(E, F)*dx
#     + conj(z) * z * inner(H, G)*dx
#     )
# A = (
#     inner(rot_s(E), rot_s(F))*dx
#     + inner(rot_v(H), rot_v(G))*dx
#     # complex conj terms
#     - conj(z) * inner(complex_unit * rot_v(H), F)*dx
#     + conj(z) * inner(complex_unit * rot_s(E), G)*dx
#     # z terms
#     - z * inner(E, complex_unit * rot_v(G))*dx
#     + z * inner(H, complex_unit * rot_s(F))*dx
#     # |z|^2 terms
#     + 
#     + conj(z) * z * inner(E, F) * dx
#     + conj(z) * z * inner(H, G) * dx
#     )
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

# Define the m form used for norms, etc
# Here, m is the L2 inner product (folded operator)
def m_form(u, v): 
    if hasattr(u, "subfunctions") and len(u.subfunctions) > 1: # then it is a mixed function
        return assemble(sum(inner(ui, vi) * dx for ui, vi in zip(u.subfunctions, v.subfunctions)))
    else:
        return assemble(inner(u, v) * dx)
        

# Not too sure how to set parameters 
solver_parameters = {
    # Options for your adaptive eigensolver
    "goal_adaptive": {
        "tolerance": 1.0e-5,
        "max_it": 50,
        "dorfler_alpha": 0.5,
        "primal_extra_degree": (1, 1),
        "dual_extra_degree": (1, 1),
        "cell_residual_extra_degree": (1, 1),
        "facet_residual_extra_degree": (1, 1),
        "self_adjoint": True,
        "nev": 5,
        "verbose": False, # do not print adaptivity prints
    },

    # Options for the inner SLEPc eigensolves
    "eps_type": "krylovschur",
    "eps_tol": 1.0e-8,
    "st_type": "sinvert",
    #"st_ksp_type": "preonly",
    #"st_pc_type": "lu",
    # "eps_monitor": None, # take out this monitor for now
    "eps_smallest_magnitude": None,
    #"eps_target_real": None,
    "eps_target": 0
}


# Set the linear eigenproblem
problem = LinearEigenproblem(A, M, bcs)



def make_output_callback(adaptive_tracker):

    def my_output(self, it: int):

        lambda_h = self._lam_h
        lambda_p = self._lam_p
        error_lambda = self.signed_error
        eta_h = abs(error_lambda)
        phi_h = sqrt(lambda_h)
        phi_enriched = np.sqrt(lambda_p)

        # correction
        corrected = lambda_h + error_lambda
        if corrected >= 0:
            corrected_phi = np.sqrt(corrected)
        else:
            corrected_phi = float("nan")
    

        adaptive_tracker.append({
            "iteration": it,
            "error_lambda": error_lambda,
            "eta_h": eta_h,
            "phi_h": phi_h,
            "phi_corr": corrected_phi,
            "phi_enriched": phi_enriched,})

    return my_output



# THE DESIRED SPECTRAL VALUES
##################################################################
exact_omega_squared = np.array([n**2 + m**2 for n in range(10) for m in range(10) if (n**2 + m**2 <= 4**2 and n**2 + m**2 >= 2**2) ])
exact_omega = np.sqrt(exact_omega_squared)
spec_vals = np.sort(exact_omega)


# CONSTRUCT THE INITIAL TRIANGULATION
##################################################################
N = 5
xmin = 2.0
xmax = 4.0
ymin = -1.0
ymax = 1.0
triangles = initial_triangulation(xmin, ymin, xmax, ymax, N)
marked = np.zeros(len(triangles), dtype=bool)
in_pseudospec = []
out_pseudospec = []
on_contour = []
max_z_its = 10
diam_tol = 1e-1
# smallest_eigvals = []
# phi_vals = []
# enriched_phi_vals = []
# DWR_phi_vals = []
# DWR_errors = []
# DWR_etas = []
epsilon = 1e-1
z_guesses = ()
z_its = -1


while len(triangles) > 0 and  z_its < max_z_its:

    # Update iteration count
    z_its += 1

    # Compute barycentres and diameters
    barycentres = get_barycentres(triangles)
    diameters = get_diameters(triangles)

    # Print the update, only in rank 0 (Codex)
    if rank == 0:
        print(RED% (f"---------------- [OUTER SWEEP {z_its}: " f"{len(triangles)} CELLS ON {nprocs} RANKS] " "----------------"))

    # # Reset marking
    # marked = np.zeros(len(triangles), dtype=bool)
    # # Reset new_triangles
    # new_triangles = []
    # print(RED % f"-- -- -- -- -- -- -- -- -- -- -- -- -- -- [FOR OUTER SWEEP NUMBER {z_its}] -- -- -- -- -- -- -- -- -- -- -- -- -- --")

    # Each rank stores the scalar results from its assigned z-points here. (codex)
    local_results = []

     # PARALLEL CHANGE 4:
    # This replaces: for k in range(len(triangles))
    num_tri = len(triangles)
    for k in range(rank, num_tri, nprocs):

    # # Iterate over the triangles (their barycentres and diameters)
    # num_tri = len(triangles)
    # for k in range(num_tri):

        # Set the triangle info
        #triangle = triangles[k]
        zK = barycentres[k]
        hK = diameters[k]
        diam_cond = hK**2 # we want |eta| <= hK^2

        # Set z for the folded operator problems
        z.assign(zK)
        # error_ests = [] # new list to save errors
        print(BLUE % f"---------------------------- [FOR rank {rank}, z = {z}, Sweep Number {z_its}, {k}th triangle (there are {num_tri})] ----------------------------")

        # Call the Adaptive eigensolver
        adaptive_tracker = []
        solver = GoalAdaptiveFoldedEigenSolver(problem, m_form = m_form, initial_space = z_guesses, target=0.0,\
                 epsilon = epsilon, diam_cond = diam_cond, diameter = hK, imag_tol = 1e-12, mult_tol = 1e-2,\
                  solver_parameters=solver_parameters, post_iteration_callback=make_output_callback(adaptive_tracker))
        solver.solve()

        # Print results
        n_adaptive_solves = len(solver.Ndofs_vec)
        n_refinements = max(0, n_adaptive_solves - 1)
        print(
            f"[rank {rank}] Finished cell {k}, z={zK}: "
            f"{n_refinements} mesh adaptations "
            f"({n_adaptive_solves} adaptive solves)",
            flush=True,)

        # Save results (Codex)
        # lambda_h, u_h = solver.get_eigenpair()
        # u_p = solver._u_p
        # zval_real = float(z.values()[0].real)
        # zval_imag = float(z.values()[0].imag)
        # z_dir = (f"parallel_outerloop_output/sweep-{z_its}/cell-{k}"f"-z-real-{zval_real:.6f}-imag-{zval_imag:.6f}")
        # primal_dir = f"{z_dir}/primal"
        # enriched_dir = f"{z_dir}/enriched"
        # os.makedirs(primal_dir, exist_ok=True)
        # os.makedirs(enriched_dir, exist_ok=True)
        # # pull uh
        # Eout, Hout = u_h.subfunctions
        # Eout.rename("E_primal_final")
        # Hout.rename("H_primal_final")
        # # pull up
        # Eout_p, Hout_p = u_p.subfunctions
        # Eout_p.rename("E_enriched_final")
        # Hout_p.rename("H_enriched_final")
        # # save each
        # primal_comm = u_h.function_space().mesh().comm
        # enriched_comm = u_p.function_space().mesh().comm
        # VTKFile(f"{primal_dir}/primal_final.pvd",comm=primal_comm,).write(Eout, Hout)
        # VTKFile(f"{enriched_dir}/enriched_final.pvd",comm=enriched_comm,).write(Eout_p, Hout_p)



        # Save the local results (codex)
        # Do not refine the outer triangle here. Return the values to rank 0.
        lambda_h, u_h = solver.get_eigenpair()
        error_lambda = solver.signed_error
        eta_h = abs(error_lambda)
        phi_h = solver.matts_phi
        phi_corr = solver.corrected_phi
        phi_enriched = solver.enriched_phi
        local_results.append({
            "cell": int(k),
            "lambda_h": complex(lambda_h),
            "error_lambda": complex(error_lambda),
            "eta_h": float(np.real(eta_h)),
            "phi_h": float(np.real(phi_h)),
            "phi_corr": float(np.real(phi_corr)),
            "phi_enriched": float(np.real(phi_enriched)),
            "n_refinements": int(n_refinements),
            "inner_converged": bool(solver.inner_converged),
            "termination_reason": solver.termination_reason,
            #"adaptive_tracker": adaptive_tracker,
        })


    # PARALLEL CHANGE 5: (codex)
    # Wait for the complete sweep and collect all z-results on rank 0.
    gathered_results = world.gather(local_results, root=0)

    if rank == 0: # create nested list of results and flatten in
        results = []
        for rank_results in gathered_results:
            for result in rank_results:
                results.append(result)
        results.sort(key=lambda result: result["cell"]) # sort by cell so that result[0] is for cell k=0?

    
        # Create empty marking and new triangle list
        marked = np.zeros(len(triangles), dtype=bool)
        new_triangles = []

        # Classify each cell
        for result in results:

            # Slice needed info
            k = result["cell"]
            lambda_h = result["lambda_h"]
            error_lambda = result["error_lambda"]
            eta_h = result["eta_h"]
            phi_h = result["phi_h"]
            phi_corr = result["phi_corr"]
            phi_enriched = result["phi_enriched"]
            n_refinements = result["n_refinements"]
            inner_converged = result["inner_converged"]
            termination_reason = result["termination_reason"]
            #adaptive_tracker = result["adaptive_tracker"]

            # Get triangle info
            triangle = triangles[k]
            zK = barycentres[k]
            hK = diameters[k]
            R = hK + np.sqrt(eta_h)
 
            print(
                GREEN % f"[sweep {z_its}, cell {k}]\n"
                + GREEN % "Data:\n"
                + f"z={zK},\n"
                + f"We performed {n_refinements} refinements;\n"
                + f"Did inner adaptivity converge? {inner_converged};\n"
                + f"Termination reason for inner adaptation: {termination_reason};\n"
                + f"lambda_h={lambda_h},\n"
                + f"phi_h={phi_h},\n"
                + f"eta_h={eta_h},\n"
                + f"phi_corr={phi_corr}",
                flush=True,
            )
            # # print(GREEN % f"Results at z = {zK}")
            # print(GREEN% ("The lower-degree solve gives "f"|dist(z, spectrum)| <= {phi_h}"))
            # print(GREEN% f"The lower-degree eigenvalue is {lambda_h}")
            # print(GREEN% f"The DWR eigenvalue error estimate is {error_lambda}")

            print(GREEN % f"Classifying the cell:")
            # Check our outer refinement criteria
            if phi_h + hK < epsilon:
                in_pseudospec.append(triangle)
                print('The cell is inside the epsilon-pseudospectrm (phi + diameter < epsilon)')
            elif phi_h + R < epsilon:
                    in_pseudospec.append(triangle)
                    print('The cell is inside the epsilon-pseudospectrm (phi + R < epsilon)')
            elif phi_h - R > epsilon:
                    out_pseudospec.append(triangle)
                    print('The cell is outside the epsilon-pseudospectrum (phi - R > epsilon)')
            elif hK < diam_tol:
                    on_contour.append(triangle)
                    print('The cell is too small to refine, so it is considered to be on the epsilon-pseudospectrum contour')
            elif np.abs(phi_h - epsilon) <= R:
                    marked[k] = True
                    print('The cell is to be refined')
                    a = triangle[0]
                    b = triangle[1]
                    c = triangle[2]

                    ab = (a + b) / 2
                    bc = (b + c) / 2
                    ca = (c + a) / 2

                    new_triangles.append([a, ab, ca])
                    new_triangles.append([ab, b, bc])
                    new_triangles.append([ca, bc, c])
                    new_triangles.append([ab, bc, ca])
            print()

        
        # Only rank 0 writes the combined outer-loop plot. (codex)
        outer_dir = "parallel_outerloop_output"
        os.makedirs(outer_dir, exist_ok=True)
        outer_file = (f"{outer_dir}/outer_state_level_{z_its}.pdf")
        z_grid_plot(
            triangles,
            file_dir=outer_file,
            in_pseudospec=in_pseudospec,
            out_pseudospec=out_pseudospec,
            on_contour=on_contour,
            marked=marked,
            spec_vals=spec_vals,
            epsilon=epsilon,
        )
        triangles = np.array(new_triangles, dtype=complex)

    else:
        triangles = None

    # PARALLEL CHANGE 6: (codex)
    # Give every rank the new active triangles for the next sweep.
    triangles = world.bcast(triangles, root=0) # sends the triangles list from rank 0 to all ranks













        
        # print(GREEN % f"Results at z = {z} (Sweep number {z_its})") 
        # print(GREEN % f"At z = {zK} the lower-degree solve gives|dist(z, spectrum)| <= {phi_h}") # Patrick's output choice
        # print(GREEN % f"The min eigenvalue from lower degree solve is: {lambda_h}")
        # print(GREEN % f"The error estimate, eta = rho/1-sigma, is: {error_lambda}")
        # print(GREEN % f"The 'improved' bound is |dist(z, spectrum)| <= {phi_corr}")
        # print(GREEN % f"The bound from the enriched solve |dist(z, spectrum)| <= {phi_enriched}")


        # # Compute R = hK + np.sqrt{|eta_a|}
        # R = hK + np.sqrt(eta_h)

        # # Check the outer refinement condition

        # if (phi_h + R) < epsilon: # inside the pseudospec
        #     print(GREEN % f"Since Phi + R = {phi_h + R} < epsilon = {epsilon}, this triangle is in the pseudospectrum")
        #     in_pseudospec.append(triangle)
        # elif (phi_h - R) > epsilon:
        #     print(GREEN % f"Since Phi - R = {phi_h - R} > epsilon = {epsilon}, this triangle is outside the pseudospectrum")
        #     out_pseudospec.append(triangle)
        # elif hK < diam_tol:
        #     print(GREEN % f"Since diameter = {hK} <= diameter tolerance = {diam_tol}, we will leave this triangle alone")
        #     print(GREEN % f"Instead, we will say its on the pseudospectral contour!")
        #     on_contour.append(triangle)
        # elif np.abs(phi_h - epsilon) <= R:
        #     print(GREEN % f"Since |Phi - eps| = {np.abs(phi_h - epsilon)} <= R = {R} this triangle is marked for refinement")
        #     marked[k] = True
        #     # refine the cell
        #     a = triangle[0]
        #     b = triangle[1]
        #     c = triangle[2]
        #     # Midpoints of its three edges
        #     ab = (a + b) / 2
        #     bc = (b + c) / 2
        #     ca = (c + a) / 2
        #     # Four child triangles
        #     child_1 = [a,  ab, ca]
        #     child_2 = [ab, b,  bc]
        #     child_3 = [ca, bc, c]
        #     child_4 = [ab, bc, ca]
        #     # Save the children in the new active triangulation
        #     new_triangles.append(child_1)
        #     new_triangles.append(child_2)
        #     new_triangles.append(child_3)
        #     new_triangles.append(child_4)
        #     print(GREEN % f"The triangle has been refined into four child triangles!")
        # print()

    
    
    # # Before moving on, plot the current state of affairs
    # outer_dir= f"first_outer_loop_output/"
    # os.makedirs(outer_dir, exist_ok=True)
    # outer_dir= f"first_outer_loop_output/outer_state_level_{z_its}.pdf"
    # z_grid_plot(triangles, file_dir = outer_dir,\
    #             in_pseudospec = in_pseudospec, out_pseudospec = out_pseudospec, on_contour = on_contour, marked = marked,\
    #              spec_vals = spec_vals, epsilon = epsilon)

 
    # # Before moving on, update the triangles list
    # triangles = np.array(new_triangles, dtype=complex)