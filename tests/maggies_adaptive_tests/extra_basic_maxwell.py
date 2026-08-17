from firedrake import *
from netgen.occ import *
import numpy as np
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import os


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


# Decide if you want edge elements or Lagrange elements to discretize the space for E
hcurl = False
if hcurl:
    V0 = FunctionSpace(mesh, "N1curl", 2)
else:
    V0 = VectorFunctionSpace(mesh, "CG", 1, dim=2)

# The space for H is CG1
V1 = FunctionSpace(mesh, "CG", 1)

# We need a mixed function space here: (space for E) x (space for H)
Z = MixedFunctionSpace([V0, V1])

# Set zero tangential trace conditions for E:
# Edge elements only force continuity in tangential component so we just set zero conditions on boundary.
# If using Lagrange elements, we need so explicily set the tangential trace to zero.
# if hcurl:
#     bc = [DirichletBC(Z.sub(0), Constant((0, 0)), "on_boundary")]
# else:
#     bc = [DirichletBC(Z.sub(0).sub(0), 0, (3, 4)),
#           DirichletBC(Z.sub(0).sub(1), 0, (1, 2))]
# # The BC's (zero tangential trace)
bcs = [
    # E_x = 0 on horizontal edges
    DirichletBC(Z.sub(0).sub(0), 0, (bottom, top)),
    # E_y = 0 on vertical edges
    DirichletBC(Z.sub(0).sub(1), 0, (left, right)),]

# Scalar rot
def rot_s(E):
    return E[1].dx(0) - E[0].dx(1)

# Vector rot
def rot_v(H):
    return as_vector([H.dx(1), -H.dx(0)])

# Set the complex unit as j
j = Constant(1j)

# Test and trial functions
u = TrialFunction(Z)
(E, H) = split(u)
v = TestFunction(Z)
(F, G) = split(v)

# Placeholder for eigenvalue
z = Constant(0)

# Set Ln(z) = <(A-zI)u, (A-zI)v>
a = (
      # A^* M^{-1} A terms
      inner(rot_s(E), rot_s(F))*dx
    + inner(rot_v(H), rot_v(G))*dx
      # -2 z A terms
    - 2 * z * inner(j*rot_v(H), F)*dx
    + 2 * z * inner(j*rot_s(E), G)*dx
      # |z|^2 M terms
    + conj(z) * z * inner(E, F)*dx
    + conj(z) * z * inner(H, G)*dx
    )

# Set Gn = <u, v>
b = inner(E, F)*dx + inner(H, G)*dx

# Set the generalized eigenproblem Ln(z)x = lambda Gn x on restructed space
problem = LinearEigenproblem(a, b, bcs=bcs, restrict=True)

# Solver parameters and subsequent solver
sp = {"eps_gen_hermitian": None,  # solver parameters, passed to SLEPc
      "eps_type": "krylovschur",
      "eps_monitor": None,
      "eps_smallest_magnitude": None,
      #"eps_view": None,  # uncomment to see the solver
      "eps_target": 0,
      "eps_target_real": None,
      "st_type": "sinvert",
      }
solver = LinearEigensolver(problem, n_evals=1, solver_parameters=sp) # ask for one eigenpair


# Set an empty function on the restricted space to hold our eigenfunction initial guess
cache_guess = Function(problem.restricted_space)

# Set xs = Grid(n) and ys to hold Phi_n
h = 0.01
xs = list(np.arange(0.01, 4, h)) + [4]
h = 0.02 # grid spacing
xs = np.append(np.arange(2.0, 4.0, h),[4.0]) 
ys = []
# The loop for Matt's algorithm
for x in xs:
    z.assign(x) # set z as current grid point
    solver.solve() # solver eiegenproblem
    ys.append(sqrt(solver.eigenvalue(0))) # save Phi_n = sqrt{ lambda_min }
    print(BLUE % f"x = {x}: |dist(x, spectrum)| <= {ys[-1]}")


    # Cycle eigenspace from one solve to another. Krylov-Schur only takes
    # one initial guess as input, so we just need to cycle one vector
    with cache_guess.dat.vec_wo as vec_r:
        solver.es.getEigenvector(0, vec_r) # place the eigenfunction for lambda_min in vec_r
        solver.es.setInitialSpace(vec_r) # use the eigenfunction (in vec_r) as initial guess for next iter (next grid point)



# # Exact results
# exact_omega_squared = np.array([n**2 + m**2 for n in range(10) for m in range(10) if n**2 + m**2 <= 4**2])
# exact_omega = np.unique(np.sqrt(exact_omega_squared)) # take our duplicates
# exact_omega = np.sort(exact_omega) # sort them
# exact_dist = np.min(np.abs(np.subtract.outer(xs, exact_omega)), axis=1)
# error = np.abs(np.array(ys) - exact_dist)
# Plotting routine
# plt.plot(xs, ys, linewidth=2, label = r'$\Phi_n(z, A)')
# #nn = np.array([n**2 + m**2 for n in range(10) for m in range(10) if n**2 + m**2 < 4**2])
# #plt.plot(np.sqrt(nn), 0*nn, 'ok', markersize=5, label = r'exact $\omega$')
# plt.plot(exact_omega, 0*exact_omega, 'ok', markersize=5, label = r'exact $\omega$')
# plt.plot(xs, exact_dist, linestyle = '--', color = 'grey', label = r"Exact $\operatorname{dist}(z, \operatorname{Sp}(A))$")
# os.makedirs("basic_maxwell_output/", exist_ok=True)
# plt.xlabel(r'$z$')
# plt.legend()
# #plt.title(rf"Approximation of $\mathrm{{dist}}(x, \text{{spectrum}})$ ($N = {N}$)")
# plt.savefig(f"basic_maxwell_output/dist_plot_{N=}.pdf", dpi=300, bbox_inches="tight" )
# # plt.show()

# # Triplot the mesh
# nmarkers = len(mesh.exterior_facets.unique_markers)
# fig, ax = plt.subplots(figsize=(5, 5))
# triplot(mesh, axes=ax, \
#         interior_kw={"edgecolors": "black","linewidths": 0.7}, \
#         boundary_kw={"colors": ["black"] * nmarkers,"linewidths": 1.2},)
# ax.set_aspect("equal")
# ax.set_axis_off()
# plt.savefig(f"basic_maxwell_output/mesh_plot_{N=}.pdf", dpi=300, bbox_inches="tight" )

# Save results for later plotting
np.savez(
    f"finerzgrid_folded_maxwell_output/netgen_basic_data_{N=}.npz",
    grid=np.array(xs),
    phi_vals=np.array(ys),
    #exact_omega=exact_omega,
    #exact_dist=exact_dist,
    #error=error,
    dofs=problem.restricted_space.dim(),
    N=N,
)