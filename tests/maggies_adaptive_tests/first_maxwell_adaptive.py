from firedrake import *
from netgen.occ import *
import numpy as np
import sys
from firedrake.maggies_current_adaptivefoldedeigensolver import GoalAdaptiveFoldedEigenSolver
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
        "max_it": 5,
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

    # Create the directory
    zval = float(z.values()[0].real)
    z_dir= f"output/z-{float(z.values()[0].real):.6f}"
    primal_dir = f"{z_dir}/primal"
    enriched_dir = f"{z_dir}/enriched"
    os.makedirs(z_dir, exist_ok=True)
    os.makedirs(primal_dir, exist_ok=True)
    os.makedirs(enriched_dir, exist_ok=True)

    # Pull u_h and chosen u_p
    u_h, u_p = self._u_h, self._u_p

    # Save the current (primal) solution
    Eout, Hout = u_h.subfunctions
    Eout.rename(f"E_primal_{it=}")
    Hout.rename(f"H_primal_{it=}")
    VTKFile(f"{primal_dir}/primal_it_{it}.pvd").write(Eout, Hout, time=zval)

    # Save the current chosen enriched solution
    Eout_p, Hout_p = u_p.subfunctions
    Eout_p.rename(f"E_enriched_{it=}")
    Hout_p.rename(f"H_enriched_{it=}")
    VTKFile(f"{enriched_dir}/enriched_it_{it}.pvd").write(Eout_p, Hout_p, time=zval)

    # Also save the error estimate at the current z
    error_ests.append(self.eta_h)

    print("Done saving user's desired output ...")





# LOOP OVER COMPLEX GRID(N)
##################################################################
h = 0.02 # grid spacing
n = 1/h # n for Grid(n)

grid = np.append(np.arange(1.5, 4.0, h),[4.0]) # Patrick makes this a list
smallest_eigvals = []
phi_vals = []
enriched_phi_vals = []
DWR_phi_vals = []
DWR_errors = []
z_guesses = ()

for curr_z in grid:


    # Set z
    z.assign(curr_z)
    error_ests = [] # new list to save errors
    print(BLUE % f"---------------------------- [FOR z = {z}] ----------------------------")

    # Call the Adaptive eigensolver
    solver = GoalAdaptiveFoldedEigenSolver(problem, m_form = m_form, initial_space = z_guesses, target=0.0,epsilon = 1e-2, imag_tol = 1e-12, mult_tol = 1e-2, solver_parameters=solver_parameters, post_iteration_callback = my_output)
    solver.solve()

    # Pull the final results 
    lambda_h, uh = solver.get_eigenpair()
    error_lambda = solver.signed_error
    phi_h = solver.matts_phi
    phi_corr = solver.corrected_phi
    phi_enriched = solver.enriched_phi
    eigfuncs = solver.vecs 
    
    # save to lists
    smallest_eigvals.append(lambda_h)
    phi_vals.append(phi_h)
    DWR_phi_vals.append(phi_corr)
    DWR_errors.append(error_lambda)
    enriched_phi_vals.append(phi_enriched)

    print(BLUE % f"At z = {curr_z} the lower-degree solve gives|dist(z, spectrum)| <= {phi_h}") # Patrick's output choice
    print(BLUE % f"The min eigenvalue from lower degree solve is: {lambda_h}")
    print(BLUE % f"The error estimate, eta = rho/1-sigma, is: {error_lambda}")
    print(BLUE % f"The 'improved' bound is |dist(z, spectrum)| <= {phi_corr}")
    print(BLUE % f"The 'bound from the enriched solve |dist(z, spectrum)| <= {phi_enriched}")
    print()

    # Save the error plot at the current z 
    z_dir= f"output/"
    os.makedirs(z_dir, exist_ok=True)
    plt.plot([i for i in range(len(error_ests))], error_ests)
    plt.xlabel(r"mesh")
    plt.title(rf"Error estimate over each mesh for $z = {curr_z}$")
    plt.savefig(f"{z_dir}/error_plot_for_z={float(z)}.pdf")
    plt.close()

    # Use the final found eigfuncs as initial guesses for next z solve
    print(BLUE % f'Updating initial guesses for next z value')
    z_guesses = []
    for old_u in eigfuncs: # iterate over eigfuncs on last adapted mesh
        new_u = Function(V)
        solver.atm.inject(old_u, new_u) # inject from fine to coarse mesh?
        z_guesses.append(new_u)
    z_guesses = tuple(z_guesses) 






# Use the results to plot the phi plot

# desired eigenvalues
exact_omega_squared = np.array([n**2 + m**2 for n in range(10) for m in range(10) if n**2 + m**2 <= 4**2])
exact_omega = np.sqrt(exact_omega_squared)
exact_omega = np.sort(exact_omega)
# midpoints between neighbouring omega values
midpoints = 0.5 * (exact_omega[1:] + exact_omega[:-1])

# the plot
z_dir= f"output/"
os.makedirs(z_dir, exist_ok=True)
plt.plot(grid, phi_vals, linewidth=2, label = r"$\Phi_n(z, A)$")
plt.plot(exact_omega, 0*exact_omega, 'ok', markersize=5, label = r'exact $\omega$')
plt.plot(grid, enriched_phi_vals, label = r"Enriched $\Phi_n(z, A)$")
plt.plot(grid, DWR_phi_vals, linestyle = "--",label = r"DWR corrected $\Phi_n(z, A)$")
plt.xlabel(r"$z$")
plt.title(rf"Approximations of $\Phi_n(z, A)$ and $\omega$ ($N = {N}$)")
plt.legend()
plt.savefig(f"{z_dir}/phi_plot_{N=}.pdf")
plt.close()