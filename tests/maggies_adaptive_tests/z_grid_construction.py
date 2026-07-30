# In this file, we will store helper functions for the triangulation of the complex plane
import numpy as np
import matplotlib.pyplot as plt

# Constructing a triangulation on a square

def initial_triangulation(x_min, y_min, x_max, y_max, N):

    x = np.linspace(x_min, x_max, N + 1)
    y = np.linspace(y_min, y_max, N + 1)
    triangles = []
    for j in range(N):
        for i in range(N):

            z00 = x[i]     + 1j * y[j]
            z10 = x[i + 1] + 1j * y[j]
            z01 = x[i]     + 1j * y[j + 1]
            z11 = x[i + 1] + 1j * y[j + 1]

            triangles.append([z00, z10, z11])
            triangles.append([z00, z11, z01])
    triangles = np.array(triangles, dtype=complex)
    return triangles

# Computing the barycentres

def get_barycentres(triangles):
    return triangles.mean(axis=1)



# Computing the diameters


def get_diameters(triangles):
    diameters = []
    for triangle in triangles:
        edge_01 = np.abs(triangle[0] - triangle[1])
        edge_12 = np.abs(triangle[1] - triangle[2])
        edge_20 = np.abs(triangle[2] - triangle[0])
        diameter = max(edge_01, edge_12, edge_20)
        diameters.append(diameter)
    diameters = np.array(diameters)
    return diameters


# Plotting Help
def z_grid_plot(triangles, file_dir, in_pseudospec = None, out_pseudospec = None, marked = None, spec_vals = None, eps = None):

    fig, ax = plt.subplots(figsize=(7, 7))

    # Iterate over the triangles and plot
    for k, triangle in enumerate(triangles):

        x = triangle.real
        y = triangle.imag

        if marked is not None: # if marked, show it
            if marked[k]:
                ax.fill(x, y, color="red", alpha=0.5)
        
        if in_pseudospec is not None: # if in the pseudospec, show it
            if in_pseudospec[k]:
                ax.fill(x, y, color="green", alpha=0.5)
        
        # if out_pseudospec is not None: # if not in the pseudospec, leave empty
        #     if out_pseudospec[k]:
        #         ax.fill(x, y, color="g", alpha=0.5)
        
        closed_triangle = np.append(triangle, triangle[0])

        ax.plot(closed_triangle.real, closed_triangle.imag, color="black", linewidth=0.5,)

    
    # Add in eigenvalues if given
    ax.scatter(eigenvalues.real, eigenvalues.imag, color="green", s=40, label="Eigenvalues", zorder=3,)

    # Plot epsilon balls around the eigenvalues
    for lam in eigenvalues:
        circle = plt.Circle((lam.real, lam.imag),epsilon, fill=False, color="green",linewidth=1.2,)
        ax.add_patch(circle)

    # Set figure
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\operatorname{Re}(z)$")
    ax.set_ylabel(r"$\operatorname{Im}(z)$")
    ax.legend()

    # Save it
    fig.savefig(file_dir, dpi=300, bbox_inches="tight")

