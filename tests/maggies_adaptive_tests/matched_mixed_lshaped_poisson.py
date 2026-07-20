from firedrake import *
from netgen.occ import *
import numpy as np
import sys
from firedrake.maggies_current_adaptivefoldedeigensolver import GoalAdaptiveFoldedEigenSolver
from ufl import conj
import os
import matplotlib.pyplot as plt


# Set the L-Shaped mesh (Patrick's code from Oxford 2026)
# Make 2D rectangle from (0, 0) to (1, 2)
rect1 = WorkPlane(Axes((0,0,0), n=Z, h=X)).Rectangle(1,2).Face()
# Make 2D rectangle from (0, 1) to (2, 2)
rect2 = WorkPlane(Axes((0,1,0), n=Z, h=X)).Rectangle(2,1).Face()
L = rect1 + rect2
geo = OCCGeometry(L, dim=2)
ngmesh = geo.GenerateMesh(maxh=0.1)
mesh = Mesh(ngmesh)

# Set space - lowest order RT and DG0
U_space = FunctionSpace(mesh, "CG", 1)
Sigma = VectorFunctionSpace(mesh, "DG", 0, dim=2)
V = Sigma * U_space
sigma, u = TrialFunctions(V)
tau, v = TestFunctions(V)

# Eigenproblem
# Take standard poisson: - lapace(u) = \lambda u
# Variational form: (grad(u), grad(v)) = \lambda (u, v)
# Let sigma = - grad(u): - (sigma, grad(v) = \lambda (u,v)
# Two equations are - (sigma, grad(v)) = \lambda (u, v) and (sigma, v) + (grad(u), v) = 0
# Add them up
A = -(inner(sigma, tau) + inner(grad(u), tau) + inner(sigma, grad(v))) * dx
M = inner(u, v) * dx
bc = DirichletBC(V.sub(1), 0, "on_boundary") # zero dirichlet bc for u component (second subspace)
problem = LinearEigenproblem(A, M, bc)

# set the m_form
# WE WILL DO L2 ALIGNMENT AND M-NORMALIZATION
# even tho it is a mixed problem, the m inner product is (u, v) (only in second equation)
def m_form(U, V): 

    sigma, u = U.subfunctions
    tau, v = V.subfunctions
    return assemble(inner(u, v) * dx)
    # if hasattr(u, "subfunctions") and len(u.subfunctions) > 1: # then it is a mixed function
    #     return assemble(sum(inner(ui, vi) * dx for ui, vi in zip(u.subfunctions, v.subfunctions)))
    # else:
    #     return assemble(inner(u, v) * dx)

# parameters
# Not too sure how to set parameters 
solver_parameters = {
    # Options for your adaptive eigensolver
    "goal_adaptive": {
        "tolerance": 1.0e-5,
        "max_it": 15,
        "dorfler_alpha": 0.5,
        "primal_extra_degree": (1,1),
        "dual_extra_degree": (1,1),
        "cell_residual_extra_degree": (1,1),
        "facet_residual_extra_degree": (1,1),
        "self_adjoint": True,
        "nev": 10,
        "verbose": True,
    },

    # Options for the inner SLEPc eigensolves
    "eps_type": "krylovschur",
    "eps_tol": 1.0e-8,
    "st_type": "sinvert",
    #"st_ksp_type": "preonly",
    #"st_pc_type": "lu",
    "eps_monitor": None,
    #"eps_smallest_magnitude": None,
    "eps_smallest_real": None,
    #"eps_target_real": None,
    "eps_target": 9
}

# Set my desired output
def my_output(self, it: int):

    print("Saving user's desired output ...")
    print("Saving user's desired output ...")

    z_dir = f"debug_lshaped_output/mixed_logg_local"
    primal_dir = f"{z_dir}/primal"
    enriched_dir = f"{z_dir}/enriched"
    os.makedirs(z_dir, exist_ok=True)
    os.makedirs(primal_dir, exist_ok=True)
    os.makedirs(enriched_dir, exist_ok=True)

    # Pull u_h and chosen u_p
    u_h, u_p = self._u_h, self._u_p

    # Save the current (primal) solution
    Sout, Uout = u_h.subfunctions
    Uout.rename(f"u_primal_{it=}")
    Sout.rename(f"sigma_primal_{it=}")
    VTKFile(f"{primal_dir}/primal_it_{it}.pvd").write(Sout, Uout)

    # Save the current chosen enriched solution
    Sout_p, Uout_p = u_p.subfunctions
    Uout_p.rename(f"u_enriched_{it=}")
    Sout_p.rename(f"sigma_enriched_{it=}")
    VTKFile(f"{enriched_dir}/enriched_it_{it}.pvd").write(Sout_p, Uout_p)

    # Also save the error estimates
    etah_ests.append(self.eta_h)
    eta_ests.append(self.eta)
    dofs_full.append(u_h.function_space().dim())
    dofs_u.append(Uout.function_space().dim())

    print("Done saving user's desired output ...")


# Solver
etah_ests = []
eta_ests = []
dofs_full = []
dofs_u = []
solver = GoalAdaptiveFoldedEigenSolver(problem, m_form = m_form, initial_space = (),\
 target=9.0, epsilon = 1e-2, imag_tol = 1e-12, mult_tol = 1e-2, \
 solver_parameters=solver_parameters, post_iteration_callback = my_output, exact_eigenvalue = 9.6397238440219)
solver.solve()



# Pull the final results and plot 

# Plot error vs. DOFS and approx slope
z_dir = f"debug_lshaped_output/mixed_logg_local"
os.makedirs(z_dir, exist_ok=True)
slope_h, intercept = np.polyfit(np.log10(dofs_full), np.log10(etah_ests), 1)
slope, intercept = np.polyfit(np.log10(dofs_full), np.log10(etah_ests), 1)
plt.loglog(dofs_full, etah_ests, label = rf"Error estimate with slope {slope_h}")
plt.loglog(dofs_full, eta_ests, label = rf"Actual error with slope {slope}" )
plt.title('Comparing Error Estimates')
plt.xlabel('dof')
plt.ylabel('error')
plt.legend()
plt.savefig(f"{z_dir}/fulldof_error_plot.pdf")
plt.close()

z_dir = f"debug_lshaped_output/mixed_logg_local"
os.makedirs(z_dir, exist_ok=True)
slope_h, intercept = np.polyfit(np.log10(dofs_u), np.log10(etah_ests), 1)
slope, intercept = np.polyfit(np.log10(dofs_u), np.log10(etah_ests), 1)
plt.loglog(dofs_u, etah_ests, label = rf"Error estimate with slope {slope_h}")
plt.loglog(dofs_u, eta_ests, label = rf"Actual error with slope {slope}" )
plt.title('Comparing Error Estimates')
plt.xlabel('dof')
plt.ylabel('error')
plt.legend()
plt.savefig(f"{z_dir}/dofu_error_plot.pdf")
plt.close()

# Effectivity plot
plt.plot(dofs_u[:-1], solver.eff1_vec, label = rf"$|\eta_h| / |\eta|$")
plt.plot(dofs_u[:-1], solver.eff2_vec, label = rf"$\sum |\eta_K| / |\eta|$" )
plt.title('Effectivity Indices')
plt.xlabel('dof')
plt.ylabel('effectivity')
plt.legend()
plt.savefig(f"{z_dir}/effectivity_plot.pdf")
plt.close()


print()
print(BLUE % f'_'*50)
print(GREEN % f'CONVERGENCE DIAGNOTICS')
print(BLUE % f'_'*50)

# Add in iter-by-iter convergence rate calculations
for i in range(1, len(dofs_full)):
    rate_h_full = np.log10(etah_ests[i]/etah_ests[i-1]) / np.log10(dofs_full[i]/dofs_full[i-1])
    rate_full = np.log10(eta_ests[i]/eta_ests[i-1]) / np.log10(dofs_full[i]/dofs_full[i-1])
    rate_h_u = np.log10(etah_ests[i]/etah_ests[i-1]) / np.log10(dofs_u[i]/dofs_u[i-1])
    rate_u = np.log10(eta_ests[i]/eta_ests[i-1]) / np.log10(dofs_u[i]/dofs_u[i-1])
    print(GREEN % f'From iteration {i-1} to iteration {i} the (full) dofs changed from {dofs_full[i-1]} to {dofs_full[i]}')
    print(GREEN % f'The exact error changed from {eta_ests[i-1]} to {eta_ests[i]}')
    print(GREEN % f'The error estimate changed from {etah_ests[i-1]} to {etah_ests[i]}')
    print(GREEN % f'The rate of change for the exact error is {rate_full}')
    print(GREEN % f'The rate of change for the error estimate is {rate_h_full}')
    print(GREEN % f'From iteration {i-1} to iteration {i} the (u) dofs changed from {dofs_u[i-1]} to {dofs_u[i]}')
    print(GREEN % f'The exact error changed from {eta_ests[i-1]} to {eta_ests[i]}')
    print(GREEN % f'The error estimate changed from {etah_ests[i-1]} to {etah_ests[i]}')
    print(GREEN % f'The rate of change for the exact error is {rate_u}')
    print(GREEN % f'The rate of change for the error estimate is {rate_h_u}')
    print()