from firedrake import *
from netgen.occ import *
from netgen.geom2d import SplineGeometry
import numpy as np
import sys
from firedrake.maggies_current_adaptivefoldedeigensolver import GoalAdaptiveFoldedEigenSolver
from ufl import conj
import os
import matplotlib.pyplot as plt


# Set the pacman domain (Umberto's spectral course)
geo = SplineGeometry()
pnts = [(0, 0), (1, 0), (1, 1),
        (0, 1), (-1, 1), (-1, 0),
        (-1, -1), (0, -1)]
p1, p2, p3, p4, p5, p6, p7, p8 = [geo.AppendPoint(*pnt) for pnt in pnts]
curves = [[["line", p1, p2], "line"],
          [["spline3", p2, p3, p4], "curve"],
          [["spline3", p4, p5, p6], "curve"],
          [["spline3", p6, p7, p8], "curve"],
          [["line", p8, p1], "line"]]
[geo.Append(c, bc=bc) for c, bc in curves]
ngmesh = geo.GenerateMesh(maxh=0.1)
mesh = Mesh(ngmesh)

# Set space - lowest order RT and DG0
rt = FiniteElement("Raviart-Thomas", triangle, 1) 
Sigma = FunctionSpace(mesh, rt)
W = FunctionSpace(mesh, "DG", 0)
V = Sigma * W
sigma, u = TrialFunctions(V)
tau, v = TestFunctions(V)

# Eigenproblem
A = -(inner(sigma, tau) + inner(u, div(tau)) + inner(div(sigma), v)) * dx
M = inner(u, v)*dx
problem = LinearEigenproblem(A, M)

# set the m_form
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
        "max_it": 10,
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
    "eps_target": 11
}

# Set my desired output
def my_output(self, it: int):

    print("Saving user's desired output ...")
    print("Saving user's desired output ...")

    z_dir= f"mixed_pacman_output"
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
    dofs.append(u_h.function_space().dim())

    print("Done saving user's desired output ...")


# Solver
etah_ests = []
eta_ests = []
dofs = []
solver = GoalAdaptiveFoldedEigenSolver(problem, m_form = m_form, initial_space = (),\
 target=11, epsilon = 1e-2, imag_tol = 1e-12, mult_tol = 1e-2, \
 solver_parameters=solver_parameters, post_iteration_callback = my_output, exact_eigenvalue = 3.375610652693620492628**2)
solver.solve()



# Pull the final results and plot 

# Plot error vs. DOFS and approx slope
z_dir= f"mixed_pacman_output"
os.makedirs(z_dir, exist_ok=True)
slope_h, intercept = np.polyfit(np.log10(dofs), np.log10(etah_ests), 1)
slope, intercept = np.polyfit(np.log10(dofs), np.log10(etah_ests), 1)
plt.loglog(dofs, etah_ests, label = rf"Error estimate with slope {slope_h}")
plt.loglog(dofs, eta_ests, label = rf"Actual error with slope {slope}" )
plt.title('Comparing Error Estimates')
plt.xlabel('dof')
plt.ylabel('error')
plt.legend()
plt.savefig(f"{z_dir}/mixed_pacman_error_plot.pdf")
plt.close()

# Effectivity plot
plt.plot(dofs[:-1], solver.eff1_vec, label = rf"$|\eta_h| / |\eta|$")
plt.plot(dofs[:-1], solver.eff2_vec, label = rf"$\sum |\eta_K| / |\eta|$" )
plt.title('Effectivity Indices')
plt.xlabel('dof')
plt.ylabel('effectivity')
plt.legend()
plt.savefig(f"{z_dir}/mixed_pacman_effectivity_plot.pdf")
plt.close()