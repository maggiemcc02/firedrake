from firedrake import *
from netgen.occ import *
import numpy as np
import sys
from firedrake.maggies_adaptivefoldedeigensolver import GoalAdaptiveFoldedEigensolver
from ufl import conj
import os
import matplotlib.pyplot as plt


# THINGS TO SORT
###################################################################

# A is not hermitian it seems in the error estimate - complex error estimate
# How to save outputs


# FORMULATE THE FOLDED EIGENPROBLEM
###################################################################

# Mesh and spaces
#mesh = Mesh(unit_square.GenerateMesh(maxh=1/nx))

# Set the complex unit 
complex_unit = Constant(1j)

# Define initial mesh ---------------------
# N = 32
# mesh = SquareMesh(N, N, pi, quadrilateral=False)
N = 32
initial_mesh_size = np.pi / N
L = float(np.pi)
rect = WorkPlane().MoveTo(0, 0).Rectangle(L, L).Face()
geo = OCCGeometry(rect, dim=2)
ngm = geo.GenerateMesh(maxh=initial_mesh_size)
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
bcs = [DirichletBC(V.sub(0).sub(0), 0, (3, 4)),
          DirichletBC(V.sub(0).sub(1), 0, (1, 2))]


# 2D curls 
# Scalar rot
def rot_s(E):
    return E[1].dx(0) - E[0].dx(1)
# Vector rot
def rot_v(H):
    return as_vector([H.dx(1), -H.dx(0)])


# A  form

z = Constant(0) # placeholder for z
A = (
      # A^* M^{-1} A terms
      inner(rot_s(E), rot_s(F))*dx
    + inner(rot_v(H), rot_v(G))*dx
      # -2 z A terms
    - 2 * z * inner(complex_unit*rot_v(H), F)*dx
    + 2 * z * inner(complex_unit*rot_s(E), G)*dx
      # |z|^2 M terms
    + conj(z) * z * inner(E, F)*dx
    + conj(z) * z * inner(H, G)*dx
    )

# M form
M = inner(E, F)*dx + inner(H, G)*dx


# Maggie change - target eigenvalue and folder to first pair 

# solver_parameters = {
#     "max_iterations": 10,
#     "output_dir": "output/folded_maxwell",
#     "self_adjoint": True,
#     "manual_indicators": False,
#     "dual_extra_degree": 1,
#     "use_adjoint_residual": True,
#     "primal_low_method": "interpolate",
#     "dual_low_method": "interpolate",
#     #"uniform_refinement": True
#     #"use_adjoint_residual": True
# }

# solver_parameters["goal_adaptive"] = {
#     "tolerance": 1e-4,
#     "max_it": 100,
#     "dorfler_alpha": 0.5,
#     "use_adjoint_residual": True,
#     "dual_low_method": "solve",
#     "primal_low_method": "solve",
#     "dual_extra_degree": 1,
# }


# Not too sure how to set parameters - chat helped?
solver_parameters = {
    # Options for your adaptive eigensolver
    "goal_adaptive": {
        "tolerance": 1.0e-5,
        "max_it": 10,
        "dorfler_alpha": 0.5,
        "primal_extra_degree": 1,
        "dual_extra_degree": 1,
        "self_adjoint": True,
        "nev": 5,
        "verbose": True,
    },

    # Options for the inner SLEPc eigensolves
    "eps_type": "krylovschur",
    "eps_tol": 1.0e-8,
    "st_type": "sinvert",
    "st_ksp_type": "preonly",
    "st_pc_type": "lu",
    "eps_monitor": None,
    "eps_smallest_magnitude": None,
    "eps_target_real": None,
}


# Set the linear eigenproblem
problem = LinearEigenproblem(A, M, bcs)


# PseudoSpectral Contour
epsilon = 0.01
target = 0 # set slepc target eigenvalue 0


# LOOP OVER COMPLEX GRID(N)
##################################################################
h = 0.02 # grid spacing
n = 1/h # n for Grid(n)

grid = np.append(np.arange(3.5, 4.0, h),[4.0]) # Patrick makes this a list
smallest_eigvals = []
phi_vals = []
# enriched_phi_vals = []
DWR_phi_vals = []
DWR_errors = []


for curr_z in grid:


    # Set z
    z.assign(curr_z)

    # print(BLUE % f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - [FOR z = {z}] - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
    print(BLUE % f"---------------------------- [FOR z = {z}] ----------------------------")

    # Call the Adaptive eigensolver
    solver = GoalAdaptiveFoldedEigensolver(problem, target=0.0, solver_parameters=solver_parameters)
    solver.solve()

    # Pull the final results 
    lambda_h = solver._lam_h
    error_lambda = solver.signed_error
    phi_h = solver.matts_phi
    phi_corr = solver.corrected_phi
    # phi_enriched = solver.final_enriched_phi
    
    # save to lists
    smallest_eigvals.append(lambda_h)
    phi_vals.append(phi_h)
    DWR_phi_vals.append(phi_corr)
    DWR_errors.append(error_lambda)
    # enriched_phi_vals.append(phi_enriched)

    print(BLUE % f"At z = {curr_z} the lower-degree solve gives|dist(z, spectrum)| <= {phi_h}") # Patrick's output choice
    print(BLUE % f"The min eigenvalue from lower degree solve is: {lambda_h}")
    print(BLUE % f"The error estimate, eta = rho/1-sigma, is: {error_lambda}")
    print(BLUE % f"The 'improved' bound is |dist(z, spectrum)| <= {phi_corr}")
    print()



# Use the results to plot the phi plot

# desired eigenvalues
exact_omega_squared = np.array([n**2 + m**2 for n in range(10) for m in range(10) if n**2 + m**2 <= 4**2])
exact_omega = np.sqrt(exact_omega_squared)
exact_omega = np.sort(exact_omega)
# midpoints between neighbouring omega values
midpoints = 0.5 * (exact_omega[1:] + exact_omega[:-1])

# the plot
z_dir= f"output/z-{float(curr_z):.6f}"
os.makedirs(z_dir, exist_ok=True)
plt.plot(grid, phi_vals, linewidth=2, label = r"$\Phi_n(z, A)$")
plt.plot(exact_omega, 0*exact_omega, 'ok', markersize=5, label = r'exact $\omega$')
# plt.plot(grid, enriched_phi_vals, label = r"Enriched $\Phi_n(z, A)$")
plt.plot(grid, DWR_phi_vals, linestyle = "--",label = r"DWR corrected $\Phi_n(z, A)$")
plt.xlabel(r"$z$")
plt.title(rf"Approximations of $\Phi_n(z, A)$ and $\omega$ ($N = {N}$)")
plt.legend()
plt.savefig(f"{z_dir}/phi_plot_{N=}.pdf")
plt.close()