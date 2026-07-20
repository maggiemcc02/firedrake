from firedrake import *
from netgen.occ import *
import numpy as np
import sys
from firedrake.maggies_current_adaptivefoldedeigensolver import GoalAdaptiveFoldedEigenSolver
from ufl import conj
import os
import matplotlib.pyplot as plt


# Uniform convergence test!!


# base mesh
# Create the mesh
rect1 = WorkPlane(Axes((0,0,0), n=Z, h=X)).Rectangle(1,2).Face()
rect2 = WorkPlane(Axes((0,1,0), n=Z, h=X)).Rectangle(2,1).Face()
L = rect1 + rect2
geo = OCCGeometry(L, dim=2)
ngmesh = geo.GenerateMesh(maxh=0.1)
base_mesh = Mesh(ngmesh)

# Mesh hierarchy
mesh_hierarchy = MeshHierarchy(base_mesh, 5)

# solver params
sp = {"eps_gen_hermitian": None,
"eps_type": "krylovschur",
"eps_tol": 1.0e-8,
"st_type": "sinvert",
#"st_ksp_type": "preonly",
#"st_pc_type": "lu",
"eps_monitor": None,
#"eps_smallest_magnitude": None,
"eps_smallest_real": None,
#"eps_target_real": None,
"eps_target": 9}





# Solver
errs = []
dofs = []

for level, mesh in enumerate(mesh_hierarchy):

    print(BLUE % f"For mesh level {level}")


    # Create the space and the problem
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    # Eigenproblem
    A = inner(grad(u), grad(v))*dx
    M = inner(u, v)*dx
    bc = DirichletBC(V, 0, "on_boundary")
    problem = LinearEigenproblem(A, M, bc)
    print(f"DOFS = {V.dim()}")

    # Make solver
    solver = LinearEigensolver(problem, n_evals=10, solver_parameters=sp)

    # Solve it 
    nconv = solver.solve()
    lam_h = solver.eigenvalue(0)
    uh = solver.eigenfunction(0)[0]

    print(f'Eigenvalue = {lam_h}')

    # Compute the error
    err = np.abs(9.6397238440219 - lam_h)
    errs.append(err)
    dofs.append(uh.function_space().dim())

    print(f'Error = {err}')
    print()



# Pull the final results and plot 

# Plot error vs. DOFS and approx slope
z_dir= f"debug_lshaped_output/uniform_results"
os.makedirs(z_dir, exist_ok=True)
slope, intercept = np.polyfit(np.log10(dofs), np.log10(errs), 1)
plt.loglog(dofs, errs, label = rf"Actual error with slope {slope}" )
plt.loglog(dofs, [1/i for i in dofs], label = r"$N^{-1}$")
plt.loglog(dofs, [i**(-2/3) for i in dofs], label = r"$N^{-2/3}$")
plt.title('Comparing Error Estimates')
plt.xlabel('dof')
plt.ylabel('error')
plt.legend()
plt.savefig(f"{z_dir}/error_plot.pdf")
plt.close()



print()
print(BLUE % f'_'*50)
print(GREEN % f'CONVERGENCE DIAGNOTICS')
print(BLUE % f'_'*50)

# Add in iter-by-iter convergence rate calculations
for i in range(1, len(dofs)):
    rate = np.log10(errs[i]/errs[i-1]) / np.log10(dofs[i]/dofs[i-1])
    print(GREEN % f'From iteration {i-1} to iteration {i} the dofs changed from {dofs[i-1]} to {dofs[i]}')
    print(GREEN % f'The exact error changed from {errs[i-1]} to {errs[i]}')
    print(GREEN % f'The rate of change for the exact error is {rate}')
    print()