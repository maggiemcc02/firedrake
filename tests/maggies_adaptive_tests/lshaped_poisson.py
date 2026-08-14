from firedrake import *
from netgen.occ import *
import numpy as np
import sys
#from firedrake.maggies_current_adaptivefoldedeigensolver import GoalAdaptiveFoldedEigenSolver
from firedrake.innerloop_adaptivefoldedeigensolver import GoalAdaptiveFoldedEigenSolver
# from firedrake.maggies_current_adaptivefoldedeigensolver import GoalAdaptiveFoldedEigenSolver
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
ngmesh = geo.GenerateMesh(maxh=0.05)
mesh = Mesh(ngmesh)

# Set space
V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)

# Eigenproblem
A = inner(grad(u), grad(v))*dx
M = inner(u, v)*dx
bc = DirichletBC(V, 0, "on_boundary")
problem = LinearEigenproblem(A, M, bc)

# set the m_form
def m_form(u, v): 
    if hasattr(u, "subfunctions") and len(u.subfunctions) > 1: # then it is a mixed function
        return assemble(sum(inner(ui, vi) * dx for ui, vi in zip(u.subfunctions, v.subfunctions)))
    else:
        return assemble(inner(u, v) * dx)

# parameters
# Not too sure how to set parameters 
solver_parameters = {
    # Options for your adaptive eigensolver
    "goal_adaptive": {
        "tolerance": 1.0e-5,
        "max_it": 50,
        "dorfler_alpha": 0.50,
        "primal_extra_degree": (1,),
        "dual_extra_degree": (1,),
        "cell_residual_extra_degree": (1,),
        "facet_residual_extra_degree": (1,),
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
    "eps_target": 9.0
}

# Set my desired output
def my_output(self, it: int):

    print("Saving user's desired output ...")

    # Create the directory
    z_dir= f"effectivity_poisson_output"
    primal_dir = f"{z_dir}/primal"
    enriched_dir = f"{z_dir}/enriched"
    os.makedirs(z_dir, exist_ok=True)
    os.makedirs(primal_dir, exist_ok=True)
    os.makedirs(enriched_dir, exist_ok=True)

    # Pull u_h and chosen u_p
    u_h, u_p = self._u_h, self._u_p

    # Save the current (primal) solution
    u_h.rename(f"uh_{it=}")
    VTKFile(f"{primal_dir}/it_{it}.pvd").write(u_h)

    # Save the current chosen enriched solution
    u_p.rename(f"up_{it=}")
    VTKFile(f"{enriched_dir}/it_{it}.pvd").write(u_p)

    # Also save the error estimate
    etah_ests.append(self.eta_h)
    eta_ests.append(self.eta)
    dofs.append(u_h.function_space().dim())

    print("Done saving user's desired output ...")


# Solver
etah_ests = []
eta_ests = []
dofs = []
solver = GoalAdaptiveFoldedEigenSolver(problem, m_form = m_form, initial_space = (),\
 target=9.0, epsilon = 1e-2, imag_tol = 1e-12, mult_tol = 1e-2, \
 solver_parameters=solver_parameters, post_iteration_callback = my_output, exact_eigenvalue = 9.6397238440219)
solver.solve()



# Pull the final results and plot 

# Plot error vs. DOFS and approx slope
z_dir= f"effectivity_poisson_output"
os.makedirs(z_dir, exist_ok=True)
slope_h, intercept = np.polyfit(np.log10(dofs), np.log10(etah_ests), 1)
slope, intercept = np.polyfit(np.log10(dofs), np.log10(eta_ests), 1)
plt.loglog(dofs, etah_ests, label = rf"Error estimate with slope {slope_h}")
plt.loglog(dofs, eta_ests, label = rf"Actual error with slope {slope}" )
plt.loglog(dofs, [1/i for i in dofs], label = r"$N^{-1}$")
plt.title('Comparing Error Estimates')
plt.xlabel('dof')
plt.ylabel('error')
plt.legend()
plt.savefig(f"{z_dir}/error_plot.pdf")
plt.close()

# Effectivity plot
plt.plot(dofs[:-1], solver.eff1_vec, label = rf"$|\eta_h| / |\eta|$")
plt.plot(dofs[:-1], solver.eff2_vec, label = rf"$\sum |\eta_K| / |\eta|$" )
plt.plot(dofs[:-1], solver.eff3_vec, label = rf"$\sum |\eta_K| / |\eta_h|$" )
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
for i in range(1, len(dofs)):
    rate_h = np.log10(etah_ests[i]/etah_ests[i-1]) / np.log10(dofs[i]/dofs[i-1])
    rate = np.log10(eta_ests[i]/eta_ests[i-1]) / np.log10(dofs[i]/dofs[i-1])
    print(GREEN % f'From iteration {i-1} to iteration {i} the dofs changed from {dofs[i-1]} to {dofs[i]}')
    print(GREEN % f'The exact error changed from {eta_ests[i-1]} to {eta_ests[i]}')
    print(GREEN % f'The error estimate changed from {etah_ests[i-1]} to {etah_ests[i]}')
    print(GREEN % f'The rate of change for the exact error is {rate}')
    print(GREEN % f'The rate of change for the error estimate is {rate_h}')
    print()
    






