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
def z_grid_plot(triangles, file_dir, in_pseudospec = None, out_pseudospec = None, on_contour = None, marked = None, spec_vals = None, epsilon = None):

    fig, ax = plt.subplots(figsize=(7, 7))

    # Iterate over the triangles and plot
    for k, triangle in enumerate(triangles):

        x = triangle.real
        y = triangle.imag

        if marked is not None: # if marked, show it
            if marked[k]:
                ax.fill(x, y, color="red", alpha=0.5)
        
        
        closed_triangle = np.append(triangle, triangle[0])

        ax.plot(closed_triangle.real, closed_triangle.imag, color="black", linewidth=0.5,)

    # Shade the triangles in the pseudospectra
    for triangle in in_pseudospec:
        x = triangle.real
        y = triangle.imag
        ax.fill(x, y, color="green", alpha=0.5, label = "in pseudospectrum")

    for triangle in on_contour:
        x = triangle.real
        y = triangle.imag
        ax.fill(x, y, color="green", alpha=0.75, label = "on contour")

    for triangle in out_pseudospec:
        x = triangle.real
        y = triangle.imag
        ax.fill(x, y, color="lightgrey", alpha=0.5, label = "outside the pseudospectrum")


    # Add in eigenvalues if given
    ax.scatter(spec_vals.real, spec_vals.imag, color="green", s=40, label="Spectral Values", zorder=3,)

    # Plot epsilon balls around the eigenvalues
    for lam in spec_vals:
        circle = plt.Circle((lam.real, lam.imag),epsilon, fill=False, color="green",linewidth=1.2,)
        ax.add_patch(circle)

    # Set figure
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\operatorname{Re}(z)$")
    ax.set_ylabel(r"$\operatorname{Im}(z)$")
    ax.legend()

    # Save it
    fig.savefig(file_dir, dpi=300, bbox_inches="tight")

