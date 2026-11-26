""" General Imports """
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import petsctools
import ufl
from finat.ufl import BrokenElement, FiniteElement
from petsctools.options import DefaultOptionSet, get_default_options
from ufl import avg, dS, ds, dx, inner, replace

from firedrake.assemble import assemble
from firedrake.cython import dmcommon
from firedrake.cython import mgimpl as impl
from firedrake.utils import IntType
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace, TensorFunctionSpace
from firedrake.logging import RED, warning
from firedrake.mesh import Mesh, DISTRIBUTION_PARAMETERS_NOOP
from firedrake.netgen import _snap_to_netgen, _curve_netgen_mesh
from firedrake.petsc import PETSc
from firedrake.solving_utils import _SNESContext
from firedrake.ufl_expr import TestFunction, TrialFunction, derivative
from firedrake.variational_solver import (LinearVariationalProblem,
                                          LinearVariationalSolver,
                                          NonlinearVariationalSolver)


from functools import cached_property

import finat
import numpy as np
from ufl import as_ufl
from ufl import FacetNormal, dot, grad
from ufl.classes import Zero

from firedrake.bcs import DirichletBC
from firedrake.functionspaceimpl import WithGeometryBase


""" Needed tools from adapt.py """

from firedrake.adapt import (
    GoalEstimate,
    _replace_arguments,
    _both,
    _dorfler_mark,
)


#####################################################################################
# First things first, I need homogeneous clamped boundary conditions for 1D Hermite:
#####################################################################################


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





#################################################################################
# Now, my C1 Localisation, built upon Pablo's C0 localisation (Rognes and Logg)
#################################################################################


def _C1_residual_indicators(F, dual_error, residual_degree, options_prefix, return_diagnostics=False):
    """Compute one dual-weighted residual error indicator per cell.

    The residual of the primal solution ``u_h`` is written as a sum of
    cell and facet integrals

    .. math::

        F(u_h; v) = \\sum_K (R_K, v)_K 
                        + (R_{\\partial K}^{0}, v)_{\\partial K} 
                        + (R_{\\partial K}^{1}, {\\partial_n v} )_{\\partial K},

    where ``R_K`` is a polynomial on the cell ``K`` and ``R_dK^{0, 1}`` are
    polynomials on each facet of ``K``. This representation is computed
    from the form ``F``, so the strong form of the equation is not needed.
    Three sets of local problems give the residuals. Test functions that are
    a cell bubble^2 times a polynomial vanish and have vanishing flux on all facets, 
    so testing ``F`` against them gives ``R_K``. 
    Test functions that are a facet (bubbles)*(facet bubble) times a polynomial 
    then give ``R_dK^{1}`` from the remainder
    ``F(u_h; v) - (R_K, v)_K``. Test functions that are a facet bubble  
    times a polynomial then give ``R_dK^{0}`` from the remainder
    ``F(u_h; v) - (R_K, v)_K - (R_{dK}^{1}, d_n v)_{dK}.``
    The indicator of a cell ``K`` is

    .. math::

        \\eta_K = \\left| (R_K, z - z_h)_K
        + (R_{\\partial K}^{0}, z - z_h)_{\\partial K} 
        + (R_{\\partial K}^{1}, \\partial_n{z - z_h})_{\\partial K} \\right|,

    where the contribution of an interior facet is the average of the
    contributions from its two sides.

    Parameters
    ----------
    F
        The residual form of the primal problem.
    dual_error
        The dual error representative ``z - z_h``.
    residual_degree
        The number of degrees that the localization spaces add to the primal
        space.
    options_prefix
        The options prefix of the solver that calls the marker.
    return_diagnostics
        Returns dictionary of results, more-so for testing purposes. 

    Returns
    -------
    A DG0 `~firedrake.function.Function` that holds one error indicator per
    cell.

    References
    ----------
    Rognes, M. E. and Logg, A., 2013: "Automated goal-oriented error control
    I: Stationary variational problems". https://doi.org/10.1137/10081962X

    Carstensen, C., Gallistl, D., and Gedicke, J. (2019). Residual-based
     a posteriori error analysis for symmetric mixed Arnold–Winther FEM. 
     Numerische Mathematik, 142(2), 205–234. doi:10.1007/s00211-019-01029-7.
    """
    v, = F.arguments()
    V = v.function_space()
    mesh = V.mesh().unique()
    dim = mesh.topological_dimension
    degree = V.ufl_element().degree() + residual_degree
    variant = "integral"

    cell_prefix = options_prefix + "goal_cell_"
    facet_prefix = options_prefix + "goal_facet_"


    # First Local Problem
    ##########################

    # Testing F against cell bubbles isolates the cell residual.
    bubble_space = FunctionSpace(mesh, "B", dim + 1, variant=variant)
    bubble = Function(bubble_space).assign(1)
    bubble_sq = bubble**2

    if V.value_shape == ():
        cell_space = FunctionSpace(mesh, "DG", degree, variant=variant)
    else:
        cell_space = TensorFunctionSpace(mesh, "DG", degree,
                                         shape=V.value_shape, variant=variant)
    cell_trial = TrialFunction(cell_space)
    cell_test = TestFunction(cell_space)
    cell_residual = Function(cell_space)
    cell_lhs = inner(cell_trial, bubble_sq * cell_test) * dx
    cell_rhs = _replace_arguments(F, bubble_sq * cell_test)

    # Testing cancellation:
    # from firedrake import SpatialCoordinate
    # from firedrake import div, grad
    # def lap(w):
    #     return div(grad(w))
    # x = SpatialCoordinate(mesh)[0]
    # f = 8400*x**3 - 7200*x**2 + 1200*x + 24
    # u_e = x**2 * (1 - x)**2 * (1 + 10*x**3)
    # uh = u_e - dual_error
    # cell_rhs = inner(lap(lap(uh)) - f, bubble_sq * cell_test)*dx

    cell_problem = LinearVariationalProblem(cell_lhs, cell_rhs, cell_residual)
    cell_solver = LinearVariationalSolver(cell_problem, options_prefix=cell_prefix)
    cell_solver.solve()


    # Second local problem 
    ############################
    cell = mesh.ufl_cell()

    # Maggie change - add in 1D patch
    if dim == 1: # we need a different facet bubble space
            cone = Function(FunctionSpace(mesh, "CG", 1)).assign(1)
            element = BrokenElement(FiniteElement("Lagrange", cell=cell, degree=1))

    else:
            # Testing the remainder against facet bubbles isolates the facet residual.
            cone_space = FunctionSpace(mesh, "FB", dim, variant=variant)
            cone = Function(cone_space).assign(1)
            element = BrokenElement(FiniteElement("FB", cell=cell,
                                                degree=degree + dim, variant=variant))
    if V.value_shape == ():
        facet_space = FunctionSpace(mesh, element)
    else:
        facet_space = TensorFunctionSpace(mesh, element, shape=V.value_shape)

    
    facet_trial_1 = TrialFunction(facet_space)

    facet_test_1 = (-bubble) * TestFunction(facet_space)

    n_f = FacetNormal(mesh)

    dn_facet_test_1 = dot(grad(facet_test_1), n_f)

    facet_residual_hat_1 = Function(facet_space)

    facet_rhs_1 = _replace_arguments(F, facet_test_1) - inner(cell_residual, facet_test_1) * dx

    facet_lhs_1 = (_both(inner(facet_trial_1 / cone, dn_facet_test_1)) * dS
                 + inner(facet_trial_1 / cone, dn_facet_test_1) * ds)

    first_facet_problem = LinearVariationalProblem(facet_lhs_1, facet_rhs_1, facet_residual_hat_1)
    first_facet_solver = LinearVariationalSolver(first_facet_problem, options_prefix=facet_prefix)
    first_facet_solver.solve()
    facet_residual_1 = facet_residual_hat_1 / cone



    # Third Local Problem
    ###################################

    # Testing the remainder against facet bubbles isolates the facet residual.
    
    facet_trial_0 = TrialFunction(facet_space)

    facet_test_0 = TestFunction(facet_space)

    dn_facet_test_0 = dot(grad(facet_test_0), n_f)

    facet_residual_hat_0 = Function(facet_space)

    facet_rhs_0 = (_replace_arguments(F, facet_test_0) 
                 - inner(cell_residual, facet_test_0) * dx 
                 - _both(inner(facet_residual_1, dn_facet_test_0)) * dS
                 - inner(facet_residual_1, dn_facet_test_0) * ds)

    facet_lhs_0 = (_both(inner(facet_trial_0 / cone, facet_test_0)) * dS
                 + inner(facet_trial_0 / cone, facet_test_0) * ds)

    second_facet_problem = LinearVariationalProblem(facet_lhs_0, facet_rhs_0, facet_residual_hat_0)

    second_facet_solver = LinearVariationalSolver(second_facet_problem, options_prefix=facet_prefix)

    second_facet_solver.solve()

    facet_residual_0 = facet_residual_hat_0 / cone



    # Local Error Indicators
    ################################################



    indicator_space = FunctionSpace(mesh, "DG", 0)

    indicator_test = TestFunction(indicator_space)

    dn_dual_error = dot(grad(dual_error), n_f)

    cell_term = inner(cell_residual, dual_error)
    facet_value_term = inner(facet_residual_0, dual_error)
    facet_normal_term = inner(facet_residual_1, dn_dual_error)

    # indicators = assemble(
    #     inner(inner(cell_residual, dual_error), indicator_test) * dx
    #     + inner(avg(inner(facet_residual, dual_error)), _both(indicator_test)) * dS
    #     + inner(inner(facet_residual, dual_error), indicator_test) * ds
    # )
    primal_indicators = assemble(
        inner(cell_term, indicator_test) * dx
        + inner(avg(facet_value_term + facet_normal_term), _both(indicator_test)) * dS
        + inner(facet_value_term + facet_normal_term, indicator_test) * ds)


    # My primal = dual combined error indicator:

    # dual indicators are conjuate of primal
    dual_indicators = primal_indicators.copy(deepcopy=True)
    with dual_indicators.dat.vec_rw as vec:
        vec.conjugate()

    # take their average = Re(primal)
    combined_indicators = assemble(0.5 * (primal_indicators + dual_indicators))


    # absolute value is our indicator
    with combined_indicators.dat.vec_rw as vec:
        vec.abs()

    if return_diagnostics:
        return {
            "signed_primal_indicators": primal_indicators,
            "signed_dual_indicators": dual_indicators,
            "combined_indicators": combined_indicators,
            "cell_residual": cell_residual,
            "facet_residual_1": facet_residual_1,
            "facet_residual_0": facet_residual_0,
        }
    else:
        return combined_indicators


    # with indicators.dat.vec as vec:
    #     vec.abs()
    # return indicators




###################################################################
# Now the new marking strategy based on this biharmonic example
###################################################################

class Biharmonic1DMarker:
    """Mark cells for goal-oriented adaptive refinement.

    The marker estimates the error ``J(u) - J(u_h)`` in a scalar goal
    functional ``J`` with the dual-weighted residual method. The estimate
    is computed from the residual form, so it does not need a derivation
    for each problem. 
    
    However, we write this specific method for the 1D biharmpnic problem 
    (or for any clamped C^1 problem with Hermite elements and with J(u) = (f, u))

    Pass the marker as the ``marking_callback`` of a
    `NonlinearVariationalSolver` and set ``snes_adapt_sequence``. On each
    mesh of the sequence, the marker adds a `GoalEstimate` to
    ``estimates`` and returns the cells to refine. When the estimate meets
    the tolerances, the marker returns `None`, and the adaptation stops.

    Parameters
    ----------
    goal_functional
        A scalar UFL 0-form that depends on the solution of the problem.
    exact_solution
        An optional UFL expression for the exact solution, which gives the
        true error and the effectivity index of each estimate.

    Attributes
    ----------
    goal_functional
        The goal functional on the most recently marked mesh. The marker
        moves it to each new mesh.
    exact_solution
        The exact solution on the most recently marked mesh, or `None`.
    estimates
        A `GoalEstimate` for each marked mesh, from the coarsest to the finest.

    Notes
    -----
    The marker reads these options from the options prefix of the solver
    that calls it:

    ``goal_atol`` (default 1e-50) and ``goal_rtol`` (default 0)
        Adaptation stops once
        ``|error_estimate| < max(goal_atol, goal_rtol * |J(u_h)|)``. With the
        default tolerances, adaptation runs for the whole
        ``snes_adapt_sequence``.
    ``goal_marking_fraction`` (default 0.5)
        The marker refines each cell whose error indicator is at least this
        fraction of the largest error indicator. Smaller values refine more
        cells at each step.
    ``goal_enrichment_degree`` (default 1) and ``goal_residual_degree`` (default 1)
        The increases in polynomial degree that the error estimate uses.
        Larger values give a more accurate estimate at a higher cost.
    ``goal_monitor`` (default off)
        Print each estimate.

    The marker solves the problem again in a space of higher degree,
    with the options of the solver and the ``goal_enriched_`` options, which
    take precedence. The ``goal_cell_`` and ``goal_facet_`` options configure
    the local solves that distribute the estimate over the cells. The error
    estimate requires solves with the transpose of the Jacobian, so the
    preconditioners of the solver and of the ``goal_enriched_`` solver must
    implement ``applyTranspose``.

    Examples
    --------
    Refine at most three times, or until the estimate is below 0.1% of the
    goal::

        marker = GoalOrientedMarker(u * dx)
        solver = NonlinearVariationalSolver(
            problem,
            marking_callback=marker,
            solver_parameters={"snes_adapt_sequence": 3,
                               "adaptor_criterion": "refine",
                               "goal_rtol": 1e-3},
        )
        u_adapted = solver.solve()
        for estimate in marker.estimates:
            print(estimate.num_dofs, estimate.error_estimate)

    References
    ----------
    Rognes, M. E. and Logg, A., 2013: "Automated goal-oriented error control
    I: Stationary variational problems". https://doi.org/10.1137/10081962X
    """

    def __init__(self, goal_functional: ufl.Form,
                 exact_solution: ufl.classes.Expr | None = None):
        if not isinstance(goal_functional, ufl.Form) or goal_functional.arguments():
            raise ValueError("goal_functional must be a 0-form")
        self.goal_functional = goal_functional
        self.exact_solution = exact_solution
        self.estimates: list[GoalEstimate] = []

    def _estimate_error(self, problem, current_solution: Function,
                        dual_low: Function, dual_error: ufl.classes.Expr,
                        options_prefix: str) -> GoalEstimate:
        """Estimate ``J(u) - J(u_h)`` and print the estimate if ``goal_monitor`` is set.

        Parameters
        ----------
        problem
            The variational problem on the current mesh.
        current_solution
            The primal solution ``u_h``.
        dual_low
            The dual solution ``z_h`` in the primal space.
        dual_error
            The dual error representative ``z - z_h``.
        options_prefix
            The PETSc options prefix of the active solver.

        Returns
        -------
        The error estimate on the current mesh.
        """
        goal = assemble(self.goal_functional)
        true_error = None
        if self.exact_solution is not None:
            exact_goal = assemble(replace(self.goal_functional,
                                          {current_solution: self.exact_solution}))
            true_error = exact_goal - goal
        num_dofs = current_solution.function_space().dim()
        discretisation_error = assemble(_replace_arguments(problem.F, -dual_error))
        solver_error = assemble(_replace_arguments(problem.F, -dual_low))
        estimate = GoalEstimate(
            num_dofs=num_dofs,
            goal=goal,
            discretisation_error=discretisation_error,
            solver_error=solver_error,
            true_error=true_error,
        )

        if abs(estimate.solver_error) > abs(estimate.discretisation_error):
            warning(RED % ("DWR: the solver error exceeds the discretisation error. "
                           "Tighten the solver tolerances."))

        if PETSc.Options(options_prefix).getBool("goal_monitor", False):
            report = [("degrees of freedom", estimate.num_dofs),
                      ("goal J(u_h)", estimate.goal),
                      ("discretisation error rho(u_h; z-z_h)", estimate.discretisation_error),
                      ("solver error rho(u_h; z_h)", estimate.solver_error),
                      ("error estimate eta", estimate.error_estimate),
                      ("true error J(u) - J(u_h)", estimate.true_error),
                      ("effectivity index", estimate.effectivity_index)]
            for label, value in report:
                if value is not None:
                    PETSc.Sys.Print(f"    DWR {label:<38s}{value: 15.8e}",
                                    comm=current_solution.comm)
        return estimate

    
    # Maggie will change the call method to account for the biharmonic bcs and C1 localisation!!
    ############################################################################################
    
    def __call__(self, ctx: _SNESContext, current_solution: Function) -> Function | None:
        """Estimate the error on the current mesh and mark the cells to refine.

        Parameters
        ----------
        ctx
            The `_SNESContext` of the solver on the current mesh.
        current_solution
            The primal solution ``u_h`` on the current mesh.

        Returns
        -------
        A DG0 `~firedrake.function.Function` that is 1 on the cells to refine,
        or `None` if the estimate meets the tolerances.
        """

        # Changes:
        # - No dual solves needed!
        # - Call the C1 localisation. 
        

        from firedrake.mg.ufl_utils import refine

        problem = ctx._problem
        V = current_solution.function_space()


        if self.goal_functional.ufl_domain() is not V.mesh():
            # The solver has refined the mesh since the previous call.
            mapping = ctx._coefficient_mapping
            self.goal_functional = refine(self.goal_functional, refine, coefficient_mapping=mapping)
            self.exact_solution = refine(self.exact_solution, refine, coefficient_mapping=mapping)

        
        # Each refined context has a prefix for its level. The SNES has the
        # same prefix on all levels, so read the options from the SNES.
        prefix = ctx.snes.getOptionsPrefix() or ""
        options = PETSc.Options(prefix)
        enrichment_degree = options.getInt("goal_enrichment_degree", 1)
        residual_degree = options.getInt("goal_residual_degree", 1)
        marking_fraction = options.getReal("goal_marking_fraction", 0.5)
        high_degree = V.ufl_element().degree() + enrichment_degree
        high_space = V.reconstruct(degree=high_degree)

        # I do not need a dual solution. 
        # dual_low = Function(V, name="dual_low")
        # goal_derivative = derivative(self.goal_functional, current_solution)
        # rhs = assemble(goal_derivative, bcs=problem.bcs)
        # ctx.solve_jacobian(rhs, dual_low, transpose=True)

        # Enriched Primal solution:
        #################################


        primal_high = Function(high_space, name="primal_high") # create uh^+
        # Maggie change - cant interpolate hermite?
        # primal_high.interpolate(current_solution)
        high_problem = problem.rediscretise(u=primal_high) # rediscretise for enriched problem

        # Use nullspace info from the original problem for the enriched solve
        # Some PDE's only determine a solution up to an arbitrary constant, 
        # if u solves the problem, so does u+c.
        # Here, we tell the solver about that extra freedom
        # Overall, these next three lines check the original solver for nullspace info.
        # If it finds it, it transfers that info to the enriched problem.

        # Has the original solver given a nullspace?  If so, transfer it to the enriched space.
        nullspace = None if ctx._nullspace is None else ctx._nullspace.rediscretise(high_space)
        # Has the original solver supplied a transpose nullspace? If so, transfer it to the enriched space
        transpose_nullspace = None if ctx._nullspace_T is None else ctx._nullspace_T.rediscretise(high_space)
        # Check original solver for a `near_nullspace`
        near_nullspace = None if ctx._near_nullspace is None else ctx._near_nullspace.rediscretise(high_space)

        # The goal_enriched_ solver takes the options of the outer solver as
        # defaults, but drops the snes_adapt_ options.
        parameters = get_default_options(DefaultOptionSet(prefix, ("snes_grid_", "snes_adapt_")))

        # Create the enriched solver
        high_solver = NonlinearVariationalSolver(
            high_problem,
            options_prefix=prefix + "goal_enriched_", # give solver its option names
            solver_parameters=parameters,
            nullspace=nullspace,
            transpose_nullspace=transpose_nullspace,
            near_nullspace=near_nullspace,
        )
        high_solver_dm = high_solver.snes.getDM() # store of reference to discretisation?
        high_solver.solve() # solve for primal_high
        assert high_solver.snes.getDM() == high_solver_dm # check that the DM did not change.

        # I do not need dual solution
        # goal_high = replace(self.goal_functional, {current_solution: primal_high})
        # dual_high = Function(high_space, name="dual_high")
        # rhs_high = assemble(derivative(goal_high, primal_high), bcs=high_problem.bcs)
        # high_solver._ctx.solve_jacobian(rhs_high, dual_high, transpose=True)

        
        # dual_error = dual_high - dual_low
        dual_low = current_solution
        dual_error = primal_high - current_solution
        
        
        estimate = self._estimate_error(problem, current_solution, dual_low, dual_error, prefix)
        self.estimates.append(estimate)


        atol = options.getReal("goal_atol", 1e-50)
        rtol = options.getReal("goal_rtol", 0.0)

        if abs(estimate.error_estimate) < max(atol, rtol * abs(estimate.goal)):
            # Returning None tells PETSc to stop adapting.
            return None

        # Use my C1 localisation
        indicators = _C1_residual_indicators(problem.F, dual_error, residual_degree, prefix)


        return _dorfler_mark(indicators, marking_fraction)