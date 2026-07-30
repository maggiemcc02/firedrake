from firedrake import *
from netgen.occ import *
import numpy as np
import sys
from firedrake.outerloop_adaptivefoldedeigensolver import GoalAdaptiveFoldedEigenSolver
from z_grid_helper import *
from ufl import conj
import os
import matplotlib.pyplot as plt




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
# Create the final mesh
mesh = Mesh(ngm)



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
        "max_it": 100,
        "dorfler_alpha": 0.5,
        "primal_extra_degree": (1, 1),
        "dual_extra_degree": (1, 1),
        "cell_residual_extra_degree": (1, 1),
        "facet_residual_extra_degree": (1, 1),
        "self_adjoint": True,
        "nev": 5,
        "verbose": True,
    },

    # Options for the inner SLEPc eigensolves
    "eps_type": "krylovschur",
    "eps_tol": 1.0e-8,
    "st_type": "sinvert",
    #"st_ksp_type": "preonly",
    #"st_pc_type": "lu",
    "eps_monitor": None,
    "eps_smallest_magnitude": None,
    "eps_target_real": None,
    "eps_target": 0
}


# Set the linear eigenproblem
problem = LinearEigenproblem(A, M, bcs)


# Set my desired output
def my_output(self, it: int):


    print("Saving user's desired output ...")
    print("None to save")

    # # Create the directory
    # zval_real = float(z.values()[0].real)
    # zval_imag = float(z.values()[0].imag)
    # z_dir= f"first_outer_loop_output/z-real-{zval_real:.6f}-imag-{zval_imag:.6f}"
    # primal_dir = f"{z_dir}/primal"
    # enriched_dir = f"{z_dir}/enriched"
    # os.makedirs(z_dir, exist_ok=True)
    # os.makedirs(primal_dir, exist_ok=True)
    # os.makedirs(enriched_dir, exist_ok=True)

    # # Pull u_h and chosen u_p
    # u_h, u_p = self._u_h, self._u_p

    # # Save the current (primal) solution
    # Eout, Hout = u_h.subfunctions
    # Eout.rename(f"E_primal_{it=}")
    # Hout.rename(f"H_primal_{it=}")
    # VTKFile(f"{primal_dir}/primal_it_{it}.pvd").write(Eout, Hout)

    # # Save the current chosen enriched solution
    # Eout_p, Hout_p = u_p.subfunctions
    # Eout_p.rename(f"E_enriched_{it=}")
    # Hout_p.rename(f"H_enriched_{it=}")
    # VTKFile(f"{enriched_dir}/enriched_it_{it}.pvd").write(Eout_p, Hout_p)

    # # Also save the error estimate at the current z
    # error_ests.append(self.eta_h)

    # print("Done saving user's desired output ...")



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
max_z_its = 1000
diam_tol = 1e-2
smallest_eigvals = []
phi_vals = []
enriched_phi_vals = []
DWR_phi_vals = []
DWR_errors = []
DWR_etas = []
epsilon = 1e-1
z_guesses = ()
z_its = -1


while len(triangles) > 0 and  z_its < max_z_its:

    # Update iteration count
    z_its += 1

    # Compute barycentres and diameters
    barycentres = get_barycentres(triangles)
    diameters = get_diameters(triangles)

    # Reset marking
    marked = np.zeros(len(triangles), dtype=bool)

    # Reset new_triangles
    new_triangles = []

    print(RED % f"-- -- -- -- -- -- -- -- -- -- -- -- -- -- [FOR OUTER SWEEP NUMBER {z_its}] -- -- -- -- -- -- -- -- -- -- -- -- -- --")

    # Iterate over the triangles (their barycentres and diameters)
    num_tri = len(triangles)
    for k in range(num_tri):


        # Set the triangle
        triangle = triangles[k]
        zK = barycentres[k]
        hK = diameters[k]
        diam_cond = hK**2 # we want |eta| <= hK^2

        # Set z for the folded operator problems
        z.assign(zK)
        error_ests = [] # new list to save errors
        print(GREEN % f"---------------------------- [FOR z = {z} (Sweep Number {z_its}, {k}th triangle (there are {num_tri})] ----------------------------")

        # Call the Adaptive eigensolver
        solver = GoalAdaptiveFoldedEigenSolver(problem, m_form = m_form, initial_space = z_guesses, target=0.0, epsilon = 1e-2, diam_cond = diam_cond, imag_tol = 1e-12, mult_tol = 1e-2, solver_parameters=solver_parameters, post_iteration_callback = my_output)
        solver.solve()

        # Pull the final results 
        lambda_h, uh = solver.get_eigenpair()
        error_lambda = solver.signed_error
        eta_h = abs(error_lambda)
        phi_h = solver.matts_phi
        phi_corr = solver.corrected_phi
        phi_enriched = solver.enriched_phi
        eigfuncs = solver.vecs_p # pull the enriched u_h's as initial guess for next z

    
        # save to lists
        # smallest_eigvals.append(lambda_h)
        # phi_vals.append(phi_h)
        # DWR_phi_vals.append(phi_corr)
        # DWR_errors.append(error_lambda)
        # DWR_etas.append(eta_h)
        # enriched_phi_vals.append(phi_enriched)

        
        print(GREEN % f"Results at z = {z} (Sweep number {z_its})") 
        print(GREEN % f"At z = {zK} the lower-degree solve gives|dist(z, spectrum)| <= {phi_h}") # Patrick's output choice
        print(GREEN % f"The min eigenvalue from lower degree solve is: {lambda_h}")
        print(GREEN % f"The error estimate, eta = rho/1-sigma, is: {error_lambda}")
        print(GREEN % f"The 'improved' bound is |dist(z, spectrum)| <= {phi_corr}")
        print(GREEN % f"The bound from the enriched solve |dist(z, spectrum)| <= {phi_enriched}")


        # Compute R = hK + np.sqrt{|eta_a|}
        R = hK + np.sqrt(eta_h)

        # Check the outer refinement condition

        if (phi_h + R) < epsilon: # inside the pseudospec
            print(GREEN % f"Since Phi + R = {phi_h + R} < epsilon = {epsilon}, this triangle is in the pseudospectrum")
            in_pseudospec.append(triangle)
        elif (phi_h - R) > epsilon:
            print(GREEN % f"Since Phi - R = {phi_h - R} > epsilon = {epsilon}, this triangle is outside the pseudospectrum")
            out_pseudospec.append(triangle)
        elif hK < diam_tol:
            print(GREEN % f"Since diameter = {hK} <= diameter tolerance = {diam_tol}, we will leave this triangle alone")
            print(GREEN % f"Instead, we will say its on the pseudospectral contour!")
            on_contour.append(triangle)
        elif np.abs(phi_h - epsilon) <= R:
            print(GREEN % f"Since |Phi - eps| = {np.abs(phi_h - epsilon)} <= R = {R} this triangle is marked for refinement")
            marked[k] = True
            # refine the cell
            a = triangle[0]
            b = triangle[1]
            c = triangle[2]
            # Midpoints of its three edges
            ab = (a + b) / 2
            bc = (b + c) / 2
            ca = (c + a) / 2
            # Four child triangles
            child_1 = [a,  ab, ca]
            child_2 = [ab, b,  bc]
            child_3 = [ca, bc, c]
            child_4 = [ab, bc, ca]
            # Save the children in the new active triangulation
            new_triangles.append(child_1)
            new_triangles.append(child_2)
            new_triangles.append(child_3)
            new_triangles.append(child_4)
            print(GREEN % f"The triangle has been refined into four child triangles!")
        print()

    
    
    # Before moving on, plot the current state of affairs
    outer_dir= f"first_outer_loop_output/"
    os.makedirs(outer_dir, exist_ok=True)
    outer_dir= f"first_outer_loop_output/outer_state_level_{z_its}.pdf"
    z_grid_plot(triangles, file_dir = outer_dir,\
                in_pseudospec = in_pseudospec, out_pseudospec = out_pseudospec, on_contour = on_contour, marked = marked,\
                 spec_vals = spec_vals, epsilon = epsilon)

 
    # Before moving on, update the triangles list
    triangles = np.array(new_triangles, dtype=complex)
            