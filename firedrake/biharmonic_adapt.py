"""Boundary conditions for the initial clamped Hermite biharmonic test."""

from functools import cached_property

import finat
import numpy as np
from ufl import as_ufl
from ufl.classes import Zero

from firedrake.bcs import DirichletBC
from firedrake.functionspaceimpl import WithGeometryBase


class ClampedHermiteBC(DirichletBC):
    """Impose zero value and zero derivative at both interval endpoints.
        Chat gpt made this script for me. 

    Parameters
    ----------
    V
        A scalar, one-dimensional Hermite function space.
    g
        Literal zero boundary data, also accepted as a UFL Zero.
    sub_domain
        The full boundary, specified as "on_boundary". Its normalized
        singleton tuple is accepted when Firedrake reconstructs the BC.
    """

    def __init__(self, V: WithGeometryBase,
                 g: int | float | complex | Zero = 0,
                 sub_domain: str | tuple[str, ...] = "on_boundary") -> None:
        if (V.mesh().topological_dimension != 1 or V.value_shape != ()
                or not isinstance(V.finat_element, finat.Hermite)): # we require 1D mesh, scalar function, and Hermite elements
            raise ValueError("ClampedHermiteBC requires a scalar 1D Hermite space.")
        if not isinstance(as_ufl(g), Zero): # for now, we take zero bcs, we chck if it is a literal zero.
            raise ValueError("This BC currently supports only literal zero data.")
        if sub_domain not in ("on_boundary", ("on_boundary",)): # make sure we are on boundary 
            raise ValueError("This BC currently clamps the full boundary only.")

        super().__init__(V, g, sub_domain) # runs parent DirichletBC constructor

    @cached_property
    def nodes(self) -> np.ndarray:
        """Return both value and derivative nodes at the boundary."""
        return self.function_space().boundary_nodes("on_boundary") # return both the point evals and derivative evals
