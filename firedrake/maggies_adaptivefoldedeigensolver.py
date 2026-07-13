# Here we make the skeleton for an adaptive eigensolver based on old eigensolver in Pablo's code (see commented code at end)
# I am attempting to make an updated eigensolver that matches the current state of Pablo's code

# Pablo's current imports:

import numbers
import numpy as np
from dataclasses import dataclass
from typing import ClassVar

from petsctools import OptionsManager

from firedrake.assemble import assemble
from firedrake import split
from firedrake.petsc import PETSc
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace, TensorFunctionSpace, MixedFunctionSpace
from firedrake.ufl_expr import TestFunction, TrialFunction, derivative, action, adjoint
from firedrake.variational_solver import (NonlinearVariationalProblem, NonlinearVariationalSolver,
                                          LinearVariationalProblem, LinearVariationalSolver)
from firedrake.solving import solve
from firedrake.output import VTKFile
from firedrake.logging import RED, BLUE, GREEN

from firedrake.preconditioners.pmg import PMGPC
from firedrake.mg.utils import get_level
from firedrake.mg.ufl_utils import refine
from firedrake.mg.adaptive_hierarchy import AdaptiveMeshHierarchy
from firedrake.mg.adaptive_transfer_manager import AdaptiveTransferManager

from finat.ufl import BrokenElement, FiniteElement
from ufl import avg, dx, ds, dS, inner, replace
import ufl



# Earlier stuff but add in eigensolver
__all__ = ["GoalAdaptiveSolverBase",
           "SteadyGoalAdaptiveSolver",
           "GoalAdaptiveNonlinearVariationalSolver",
           "vtk_output_callback",
           "GoalAdaptiveEigensolver"]


# Pablo's new initialization

# Notes on how it compares to the old:
# Adds in a tolerance 
# max_iterations (old) now max_it (new)
# no longer given an output_dir or run_name for saving output to a specific location (maybe this is elsewhere)
# Old code claims that use_adjoint_residual, primal_old_method, dual_old_method are not used by the eigensolver class
# New code uses @dataclass(frozen=True) to generate the constructor. The old code had a constructor (init)
# Solver parameters for cell and facet solves are the same choices

@dataclass(frozen=True)
class GoalAdaptiveOptions:
    """Options for goal-adaptive solvers.

    Parameters
    ----------
    tolerance
        Terminate the adaptive loop when ``|eta_h| < tolerance``.
    max_it
        Maximum number of SOLVE–ESTIMATE–MARK–REFINE cycles.  The loop also
        terminates early if the error estimate falls below the requested tolerance.
        Defaults to ``10``.
    dorfler_alpha
        Threshold parameter for Dörfler (bulk) marking: cells whose local error
        indicator exceeds ``dorfler_alpha * max_indicator`` are marked for
        refinement.  Must lie in ``(0, 1]``.  Larger values mark fewer cells and
        produce more targeted (but potentially slower-converging) refinement.
        Defaults to ``0.5``.
    primal_extra_degree
        Extra polynomial degree used when solving the primal problem in an
        enriched space (only relevant when ``use_adjoint_residual=True``).
        The enriched-space degree is ``degree + primal_extra_degree``.
        Defaults to ``1``.
    dual_extra_degree
        Extra polynomial degree used when solving the dual (adjoint) problem
        in an enriched space.  The enriched-space degree is
        ``degree + dual_extra_degree``.  Defaults to ``1``.
    cell_residual_extra_degree
        Extra polynomial degree for the DG space used to represent cell
        residuals during error indicator computation.  The DG space degree is
        ``degree + cell_residual_extra_degree``.  Defaults to ``1``.
    facet_residual_extra_degree
        Extra polynomial degree for the broken facet-bubble space used to
        represent facet residuals during error indicator computation.
        The space degree is ``degree + facet_residual_extra_degree``.
        Defaults to ``1``.
    use_adjoint_residual
        If ``True``, both the primal residual
        :math:`\\rho(u_h; z - z_h)` and the adjoint residual
        :math:`\\rho^*(z_h; u - u_h)` are used to form the error estimate
        :math:`\\frac{1}{2}(\\rho + \\rho^*)`, which gives a remainder term
        that is cubic in the errors z - z_h and u - u_h.
        This requires four PDE solves per iteration (primal and dual each at two degrees).
        If ``False`` (the default), only the primal residual is used, requiring
        only two PDE solves per iteration (primal at degree :math:`p` and dual
        at degree :math:`p + \\text{dual\\_extra\\_degree}`). This means
        that the remainder term is instead quadratic in the errors
        z - z_h and u - u_h.
        Defaults to ``False``.
    primal_low_method
        How to obtain the low-degree primal solution when
        ``use_adjoint_residual=True``.  Options:

        * ``"interpolate"`` (default) – nodally interpolate the enriched-space
          solution into the base space.
        * ``"project"`` – :math:`L^2`-project the enriched-space solution into
          the base space.
        * ``"solve"`` – solve the primal problem independently in the base
          space; the enriched-space solution is used only for the error estimate.

    dual_low_method
        How to obtain the low-degree dual solution used in the error estimate.
        Options:

        * ``"interpolate"`` (default) – nodally interpolate the enriched-space
          dual solution into the base space.
        * ``"project"`` – :math:`L^2`-project the enriched-space dual solution
          into the base space.
        * ``"solve"`` – solve the dual problem independently in the base space;
          this is the most expensive option but produces the most accurate solver-error
          estimate.

    verbose
        If ``True`` (the default), print progress information at each
        iteration via :func:`PETSc.Sys.Print`.
    """

    # dataclass will build the desired init with the below info? 
    tolerance: float = 1e-4
    max_it: int = 10
    dorfler_alpha: float = 0.5
    primal_extra_degree: int = 1
    dual_extra_degree: int = 1
    cell_residual_extra_degree: int = 1
    facet_residual_extra_degree: int = 1
    use_adjoint_residual: bool = False
    primal_low_method: str = "interpolate"
    dual_low_method: str = "interpolate"
    verbose: bool = True

    # Solver parameters for cell/facet bubble projections
    # These are the same choices as in the old code (pre deletion of eigensolver)
    sp_cell: ClassVar[dict] = {
        "mat_type": "matfree",
        "snes_type": "ksponly",
        "ksp_type": "cg",
        "pc_type": "jacobi",
    }
    sp_facet: ClassVar[dict] = {
        "snes_type": "ksponly",
        "ksp_type": "cg",
        "pc_type": "jacobi",
    }





# The old code has special eigenoptions which we will need again
# We will use an updated form using Pablo's dataclass approach
class GoalAdaptiveEigenOptions(GoalAdaptiveOptions):
    """Options for :class:`GoalAdaptiveEigensolver`.

    Extends :class:`GoalAdaptiveOptions` with eigenproblem-specific parameters.

    Parameters
    ----------
    self_adjoint
        If ``True``, the eigenproblem is self-adjoint so the dual eigenproblem
        equals the primal; only one set of eigensolves is performed per
        iteration.  Defaults to ``False``.
    nev
        Number of eigenvalue/eigenvector pairs to compute at each solve.
        The solver picks the one best correlated with the current eigenfunction
        estimate via :func:`match_best`.  Defaults to ``5``.

    Notes
    -----
    The options ``use_adjoint_residual``, ``primal_low_method``, and
    ``dual_low_method`` inherited from :class:`GoalAdaptiveOptions` are not
    used by :class:`GoalAdaptiveEigensolver`.
    """

    def __init__(self, *args, self_adjoint: bool = False, nev: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.self_adjoint = self_adjoint
        self.nev = nev


# The extra options needed for an eigensolver
@dataclass(frozen=True)
class GoalAdaptiveEigenOptions(GoalAdaptiveOptions):
    """Options for :class:`GoalAdaptiveEigensolver`.

    Extends :class:`GoalAdaptiveOptions` with eigenproblem-specific options.

    Parameters
    ----------
    self_adjoint
        Whether the eigenproblem is self-adjoint. If ``True``, the dual
        eigenproblem agrees with the primal eigenproblem. Defaults to ``False``.

    nev
        Number of eigenpairs requested from each eigensolve. Defaults to ``5``.
        The solver constructs (via least squares problem) the 
        eigenfunction in the span of the eigenbasis that is
        best correlated with the current eigenfunction estimate.
        This is done via :func:`match_best`.  
        Defaults to ``5``.

    Notes
    -----
    The options ``use_adjoint_residual``, ``primal_low_method``, and
    ``dual_low_method`` inherited from :class:`GoalAdaptiveOptions` are not
    used by :class:`GoalAdaptiveEigensolver`.
    """

    self_adjoint: bool = False
    nev: int = 5


# Pablo's new solver base:
    

class GoalAdaptiveSolverBase:
    """Base class for goal-adaptive solvers.

    Owns the SOLVE→ESTIMATE→MARK→REFINE loop and the Dörfler marking
    strategy.  Subclasses must implement :meth:`solve_and_estimate`,
    :meth:`compute_error_indicators`, and :meth:`refine_problem`.

    Parameters
    ----------
    goal_adaptive_options
        A ``GoalAdaptiveOptions`` instance, or a dict of keyword
        arguments to construct one.  Defaults to ``{}``.
    exact_goal
        Exact value of the goal functional (or eigenvalue).  Optional; used
        by subclasses to compute efficiency indices.
    """

  

    def __init__(self, base_mesh,
                 goal_adaptive_options: "GoalAdaptiveOptions | dict | None" = None,
                 exact_goal=None,
                 post_iteration_callback=None,
                 ):



        # Some noted changes from older code:
        # no seperate tolerance argument in this code. Now it is in self.options.tolerance
        # Stopping criteria is if abs(eta_h) < self.options.tolerance:
        # So use self.options.tolerance and not self.tolerance
        # 
        # Also max_iterations is now self.options.max_it
        #
        # The new base needs the initial mesh because it now constructs and manages the adaptive mesh hierarchy itself. Thus, we need a base_mesh input
        # I should keep this in mind - my eigensolver constructor may need to pass the base mesh (most likely vis V.mesh())
        #
        # A big change is that in this new code, this base constructs the mesh hierarchy.
        # The old code has calls like AdaptiveMeshHierarchy(...) and AdaptiveTransferManager(...)
        # The new code base has amh = AdaptiveMeshHierarchy(mesh) and later calls self.amh = amh, self.base_levels = len(amh), self.atm = AdaptiveTransferManager()
        # So the eigensolver should not create amh or atm, it should inherit them
        #
        # The new base supports mesh sequences: meshes = {*base_mesh, base_mesh}
        # Thus, it can now support problems with a single mesh and problems made from several component meshes

        if goal_adaptive_options is None:
            goal_adaptive_options = {}
        self.options = self._make_options(goal_adaptive_options)
        self.goal_exact = exact_goal
        self.post_iteration_callback = post_iteration_callback

        self.Ndofs_vec: list[int] = []
        self.eta_vec: list[float] = []
        self.etah_vec: list[float] = []

        # Set up an AdaptiveMeshHierarchy for every mesh of the problem.
        # For a MeshSequenceGeometry, iterating over it yields the component meshes;
        # we also need a separate AMH for the sequence geometry itself.
        meshes = {*base_mesh, base_mesh}  # component meshes + the sequence geometry (same object for a regular mesh)
        for mesh in meshes:
            mh, level = get_level(mesh)
            if mh is None:
                amh = AdaptiveMeshHierarchy(mesh)
            else:
                amh = AdaptiveMeshHierarchy(mh[0])
                for m in mh[1:level+1]:
                    amh.add_mesh(m)

        amh, _ = get_level(base_mesh)
        self.amh = amh
        self.base_levels = len(amh)
        self.atm = AdaptiveTransferManager()



    # Old code assumed the options were in a dictionary
    # New code allows for as options object?
    # My eigensolver code should overwrite this to take the extra arguments (below)
    def _make_options(self, d):
        """Construct a ``GoalAdaptiveOptions`` from a dict or pass through as-is.  Override in subclasses."""
        if isinstance(d, GoalAdaptiveOptions):
            return d
        # FIXME
        return GoalAdaptiveOptions(**d["goal_adaptive"])

    
    # The new code adds the following, which returns the most recently stored predicted error estimate
    def get_error_estimate(self):
        return self.etah_vec[-1]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------




    # Solve is the same as in the older code
    def solve(self):
        """Run the adaptive SOLVE→ESTIMATE→MARK→REFINE loop to convergence."""
        # Runs the standard loop for adaptivity where step() executes the steps of a given iteration
        for it in range(self.options.max_it):
            try:
                self.step(it=it)
            except StopIteration:
                break


    # 22. Maggie change - I seperate refinement criteria so I can overwrite it for the folded operator
    def should_refine(self, it, eta_h, eta):
        """ Return true when refinement is needed """
        return abs(eta_h) >= self.options.tolerance


    
    
    # Some changes since older code:
    # write_solution was replaced with a call back hook via post_iteration
    
    def step(self, it):
        """Execute one SOLVE→ESTIMATE→MARK→REFINE cycle.

        Parameters
        ----------
        it
            Current iteration index (mesh level).

        Raises
        ------
        StopIteration
            Raised (instead of returning) when the error estimate falls below
            ``tolerance`` or when ``it`` reaches ``max_it - 1``.
            :meth:`solve` catches this automatically; callers driving the loop
            manually with :meth:`step` must handle it themselves.
        """

        # One iteration of the adaptive loop!!!


        self.print(f"---------------------------- [MESH LEVEL {it}] ----------------------------")


        # SOLVE + ESTIMATE
        eta_h, eta = self.solve_and_estimate() # defined in the subclass
        self.post_iteration(it) # user given actions
        
        # Stopping criteria
        # if abs(eta_h) < self.options.tolerance: # tolerance stopping criteria 
        #     self.print("Error estimate below tolerance, finished.")
        #     raise StopIteration
        # elif it == self.options.max_it - 1: # max iter stopping criteria
        #     self.print(f"Maximum iteration ({self.options.max_it}) reached. Exiting.")
        #     raise StopIteration
        if not self.should_refine(it, eta_h, eta): # tolerance stopping criteria 
            self.print("We have decided not to adapt - we are done!.")
            raise StopIteration
        elif it == self.options.max_it - 1: # max iter stopping criteria
            self.print(f"Maximum iteration ({self.options.max_it}) reached. Exiting.")
            raise StopIteration


        
        # MARK
        self.print("Computing local refinement indicators eta_K ...")
        eta_cell = self.compute_error_indicators() # defined in the subclass - local indicators
        self.compute_efficiency_indices(eta_cell, eta_h, eta) # effectivity indices
        markers = self.set_adaptive_cell_markers(eta_cell) # call the marking routine - in base class


        # REFINE
        self.print("Transferring problem to new mesh ...")
        self.refine_problem(markers) # refine the problem (base class)

    # Allows the user to design, say output.
    # Pretty much, after each iteration, this will run any user-supplied code.
    # For example, the user could want to save the solution, print output, save plots, etc
    # This replaces the old write solution toolgi
    def post_iteration(self, it: int):
        """Hook called after SOLVE+ESTIMATE, before convergence check.
           Invokes the user-supplied ``post_iteration_callback``, if any."""
        if self.post_iteration_callback is not None:
            self.post_iteration_callback(self, it)

    # ------------------------------------------------------------------
    # Common machinery (mark + refine + efficiency)
    # ------------------------------------------------------------------


    # Not change to set_adaptive_cell_markers
    # note - maximum marking not dorfler marking!!!

    def set_adaptive_cell_markers(self, eta_cell):
        """Mark cells for refinement using Dörfler marking.

        Parameters
        ----------
        eta_cell
            Cell-wise error indicators (DG0 Function).

        Returns
        -------
        Function
            A DG0 Function with value 1 on cells selected for refinement.
        """
        # NOTE this is not quite Dorfler marking
        # For Dorfler marking we need to implement a parallel sort
        markers = Function(eta_cell.function_space())
        for m, e in zip(markers.subfunctions, eta_cell.subfunctions):
            with e.dat.vec_ro as evec:
                _, emax = evec.max()
            threshold = self.options.dorfler_alpha * emax
            m.dat.data_wo[e.dat.data_ro > threshold] = 1
        return markers

    
    # In the old code, this was implemented in subclasses not in this base class
    # Thus, every solver had to have its own refinement method
    # Thus, the new version of the eigensolver should inherit this 
    # The question is whether refine(self.problem, ...) will know how to handle an eigenproblem - I think it does (bult in and in old code)

    def refine_problem(self, markers, coef_map=None):
        """Refine the mesh and reconstruct the problem on the new mesh.

        Parameters
        ----------
        markers
            DG0 Function with value 1 on cells to refine.
        """

        # Loop over markers 
        for marker in markers.subfunctions:
            mesh = marker.function_space().mesh()
            # refine marked elements
            new_mesh = mesh.refine_marked_elements(marker)
            # add it to its repsective hierarchy
            amh, _ = get_level(mesh)
            amh.add_mesh(new_mesh)

        # Reconstruct MeshSequence with the refined meshes
        # Reconstruct the problem on the refined mesh
        mesh = self.amh[-1]
        if len(mesh) > 1:
            new_mesh = type(mesh)([get_level(m)[0][-1] for m in mesh])
            self.amh.add_mesh(new_mesh)
        if coef_map is None:
            coef_map = {}
        self.problem = refine(self.problem, refine, coefficient_mapping=coef_map)


    # Old code computed several diagnostics on the error
    # In particular, it computed eff1, eff2, and eff3 like in Joe's old code
    # This code leaves it up to the subclasses to decide what their measure of efficiency it
    # This means the generic base no longer assumes that all goal-adaptive solvers should measure efficiency in exactly the same way.

    def compute_efficiency_indices(self, eta_cell, eta_h, eta):
        """Hook called after marking.  Default no-op; override in subclasses."""
        pass

    
    # Same as before 
    # Chat: This uses PETSc-aware printing. In parallel, it avoids every MPI process printing the same message.

    def print(self, *args, **kwargs):
        if self.options.verbose:
            PETSc.Sys.Print(*args, **kwargs)

    # ------------------------------------------------------------------
    # Abstract interface for subclasses
    # ------------------------------------------------------------------


    # Same as before - needs to be defined in the subclass

    def solve_and_estimate(self):
        """Solve the PDE(s) and compute a global error estimate.

        Must store any state needed by :meth:`compute_error_indicators`.

        Returns
        -------
        tuple[float, float | None]
            ``(eta_h, eta)`` where ``eta_h`` is the error estimate and ``eta``
            is the true error if an exact solution/goal was supplied, else
            ``None``.
        """
        raise NotImplementedError

    
    # Same as before - needs to be defined in the subclass 

    def compute_error_indicators(self):
        """Compute cell-wise error indicators using stored solver state.

        Returns
        -------
        Function
            A DG0 Function of cell-wise indicators (absolute values).
        """
        raise NotImplementedError



# Not in the old code
# In the old code, the effectivity index computations were set in the base class
# Makes more sense to seperate out - not every problem has a scalar goal value

class SteadyGoalAdaptiveSolver(GoalAdaptiveSolverBase):
    """Intermediate base for steady-state goal-adaptive solvers.

    Adds the effectivity-index tracking that is specific to steady problems
    (where the notion of a scalar goal value and a true error make sense).
    The efficiency vectors are populated by :meth:`compute_efficiency_indices`,
    which is called automatically from :meth:`~GoalAdaptiveSolverBase.step`.

    Attributes
    ----------
    eta_cell_sum_vec : list[float]
        Sum of all cell indicators at each refinement level.
    eff1_vec : list[float]
        Effectivity index ``|eta_h / eta|`` at each level (only when
        ``exact_goal`` is provided).
    eff2_vec : list[float]
        Localisation efficiency ``|sum(eta_K) / eta|`` (only when
        ``exact_goal`` is provided).
    eff3_vec : list[float]
        Ratio ``sum(eta_K) / eta_h`` when no exact goal is available.
    """

    def __init__(self, base_mesh, goal_adaptive_options=None, exact_goal=None, post_iteration_callback=None):
        super().__init__(base_mesh, goal_adaptive_options, exact_goal, post_iteration_callback)
        self.eta_cell_sum_vec: list[float] = []
        self.eff1_vec: list[float] = []
        self.eff2_vec: list[float] = []
        self.eff3_vec: list[float] = []

    def compute_efficiency_indices(self, eta_cell, eta_h, eta):
        """Compute and log effectivity and localisation efficiency indices."""
        with eta_cell.dat.vec as evec:
            eta_cell_total = abs(evec.sum())

        self.eta_cell_sum_vec.append(eta_cell_total)
        self.print(f'{"Sum of refinement indicators:":40s}{eta_cell_total: 15.12e}')

        if eta is not None:
            eff1 = abs(eta_h / eta)
            eff2 = abs(eta_cell_total / eta)
            self.eff1_vec.append(eff1)
            self.eff2_vec.append(eff2)
            self.print(f'{"Effectivity index:":40s}{eff1: 15.12f}')
            self.print(f'{"Localisation efficiency:":40s}{eff2: 15.12f}')
        else:
            eff3 = eta_cell_total / abs(eta_h)
            self.eff3_vec.append(eff3)
            self.print(f'{"Localisation efficiency:":40s}{eff3: 15.12f}')



# The newer nonlinear variational code
# Note the inheritance changes - inherits from the steady solver and options manager for PETSc

class GoalAdaptiveNonlinearVariationalSolver(SteadyGoalAdaptiveSolver, OptionsManager):
    """Solves a nonlinear variational problem to minimise the error in a
    user-specified goal functional by adaptively refining the mesh using the
    dual-weighted residual (DWR) error estimate.

    All options — both goal-adaptive loop parameters and PETSc solver
    parameters for the inner primal/dual solves — are passed through a
    single ``solver_parameters`` dictionary.  Goal-adaptive parameters are
    distinguished by a ``"goal_adaptive"`` namespace key (which after
    flattening becomes a ``goal_adaptive_`` prefix), e.g.::

        solver_parameters = {
            "goal_adaptive": {
                "tolerance": 1e-4,
                "max_it": 8,
                "dual_low_method": "interpolate",
                "verbose": False,
            },
            "snes_type": "ksponly",
            "ksp_type": "preonly",
            "pc_type": "lu",
        }

    Parameters
    ----------
    problem
        The variational problem defined on the initial (coarse) mesh.
    goal_functional
        The goal functional — a zero-form in terms of the primal solution.
    solver_parameters
        Unified parameter dictionary.  Keys prefixed by ``goal_adaptive_``
        (or nested under a ``"goal_adaptive"`` sub-dict) configure the
        adaptive loop (see ``GoalAdaptiveOptions``); all other keys are
        passed to the inner :class:`~.NonlinearVariationalSolver` /
        :class:`~.LinearVariationalSolver`.
    options_prefix
        PETSc options prefix, forwarded to ``petsctools.OptionsManager``.
        Allows command-line overrides, e.g.
        ``-mysolve_snes_type ksponly``.
    primal_solver_kwargs
        Extra keyword arguments for the primal :class:`~.NonlinearVariationalSolver`.
    dual_solver_kwargs
        Extra keyword arguments for the dual :class:`~.LinearVariationalSolver`.
    exact_solution
        Exact primal solution (UFL expression or list/tuple for mixed spaces).
        Used to compute the true error for efficiency indices.
    exact_goal
        Exact scalar value of the goal functional.  Used to compute the true
        error when an analytic formula is available.
    post_iteration_callback
        Optional callable ``callback(solver, it)`` invoked after each
        SOLVE+ESTIMATE step (before convergence check and refinement).
        Use this for visualisation or post-processing at each mesh level.
        See :func:`vtk_output_callback` for a ready-made VTK writer.
    """

    _GOAL_PREFIX = "goal_adaptive_" 

    # The asterix means subsequent arguments must be passed by name
    # tolerance now lives in solver parameters
    
    def __init__(self,
                 problem: NonlinearVariationalProblem,
                 goal_functional: ufl.BaseForm,
                 *,
                 solver_parameters: dict | None = None,
                 options_prefix: str | None = None,
                 primal_solver_kwargs: dict | None = None,
                 dual_solver_kwargs: dict | None = None,
                 exact_solution: ufl.classes.Expr | None = None,
                 exact_goal: ufl.classes.Expr | None = None,
                 post_iteration_callback=None,
                 ):
        
        # No test or trials in the goal functional - just a scalar result
        if not (isinstance(goal_functional, ufl.BaseForm) and len(goal_functional.arguments()) == 0):
            raise ValueError("goal_functional must be a 0-form")

        # Initialize the constructors of both parents
        if options_prefix is None:
            options_prefix = ""
        base_mesh = problem.u.function_space().mesh() # base mesh obtained automatically! 
        # Uses the base to set up the mesh hierarchy! Old code did this itself
        SteadyGoalAdaptiveSolver.__init__(self, base_mesh, solver_parameters,
                                          exact_goal=exact_goal,
                                          post_iteration_callback=post_iteration_callback)

        # Seperate primal and dual solver parameters are removed
        # primal: primal_options_prefix = self.options_prefix + "primal_"
        # dual: dual_options_prefix = self.options_prefix + "dual_"
        OptionsManager.__init__(self, solver_parameters, options_prefix)

        # Sets the adaptive machinary
        self.problem = problem
        self.goal_functional = goal_functional
        self.primal_solver_kwargs = primal_solver_kwargs or {}
        self.dual_solver_kwargs = dual_solver_kwargs or {}

        # hmmm 
        # Old code has self.u_exact = as_mixed(exact_solution) if isinstance(exact_solution, (tuple, list)) else exact_solution
        # Does this new code support mixed solutions?
        self.u_exact = exact_solution
        
        self.u_high = None
        # Internal state set by solve_and_estimate, used by compute_error_indicators
        self._u_err = None
        self._z_lo = None
        self._z_err = None

    # A new helper - gives one definition of the best currently available solution
    def _current_solution(self):
        """Return the solution function on the current (finest) mesh."""
        return self.u_high if self.u_high is not None else self.problem.u


    

    # New code onlu returns the solution and not the error estimate
    # Instead, the error can be found via solver.get_error_estimate()
    # New code does not reset u_high=None    
    def solve(self):
        """Run the adaptive loop and return the final solution and error estimate.

        Each call continues from the current mesh and solution; history vectors
        (``Ndofs_vec``, ``etah_vec``, etc.) are appended rather than reset.

        Returns
        -------
        tuple[Function, float]
            ``(u_out, error_estimate)`` where ``u_out`` is the solution on the
            finest mesh reached and ``error_estimate`` is the final ``|eta_h|``.
        """
        super().solve()
        return self._current_solution()

    
    # One iter of the adaptive loop
    # Uses current solution helper to output the best available current solution
    # Old code used to also output the error

    def step(self, it=None):
        """Compute one SOLVE→ESTIMATE→MARK→REFINE step and return the current solution.

        Useful for users who want to inspect or post-process the solution at
        each mesh level without running the full :meth:`solve` loop.

        Parameters
        ----------
        it
            Mesh level index.  If ``None``, inferred from the current mesh
            hierarchy level.

        Returns
        -------
        tuple[Function, float]
            ``(u_out, eta_h)`` — the solution and error estimate on the current
            mesh.  Only returned when refinement was performed.

        Raises
        ------
        StopIteration
            When the error estimate is below ``tolerance`` or the maximum
            iteration count is reached.  Callers must catch this::

                for it in range(solver.options.max_it):
                    try:
                        u, eta = solver.step(it)
                    except StopIteration:
                        break
        """
        if it is None:
            V = self.problem.u.function_space()
            _, it = get_level(V.mesh())
        super().step(it)
        return self._current_solution()

    
    
    
    # Same as old code 
    # Solve the primal and dual problems
    # Compute the global error estimate
    # CHECK - z_err vs z_lo vs u_err -> uh^+ - Ih uh^+ or uh^+ - uh
    def solve_and_estimate(self):
        """Solve primal and dual, compute global error estimate."""
        u_err = self.solve_primal()
        z_lo, z_err = self.solve_dual()
        self._u_err = u_err
        self._z_lo = z_lo
        self._z_err = z_err
        eta_h, eta = self.estimate_error(u_err, z_lo, z_err)
        return eta_h, eta

    
    
    def solve_primal(self):
        """Solve the primal problem and return the primal error representative.

        When ``use_adjoint_residual=False`` (the default), the primal problem is
        solved once in the base space ``V``.

        When ``use_adjoint_residual=True``, the problem is also solved in an
        enriched space of degree ``degree + primal_extra_degree``.  The
        low-degree approximation is obtained via
        ``primal_low_method`` (interpolate / project / solve), and the
        difference ``u_high - u`` is returned as the primal error representative.

        Returns
        -------
        Function or None
            The primal error representative ``u_high - u_h``, or ``None`` when
            ``use_adjoint_residual=False``.
        """
        F = self.problem.F
        u = self.problem.u
        bcs = self.problem.bcs
        V = self.problem.u.function_space()
        self.Ndofs_vec.append(V.dim())
        primal_options_prefix = self.options_prefix + "primal_" # this new prefix methodology

        
        # Solve the primal problem
        def solve_uh():
            self.print(f'Solving primal (degree: {V.ufl_element().degree()}, dofs: {V.dim()}) ...')
            solver = NonlinearVariationalSolver(self.problem, solver_parameters=self.parameters,
                                                options_prefix=primal_options_prefix,
                                                **self.primal_solver_kwargs)
            solver.set_transfer_manager(self.atm)
            solver.solve()
            self.primal_solver = solver

        
        # If we are going to compute the residual using the adjoint problem ...
        if self.options.use_adjoint_residual:
            if self.options.primal_low_method == "solve":
                solve_uh()

            # Now solve in higher-order space
            high_degree = V.ufl_element().degree() + self.options.primal_extra_degree # Does this account for mixed problems?
            V_high = reconstruct_degree(V, high_degree) # construct enriched space
            u_high = Function(V_high, name="high_order_solution")
            u_high.interpolate(u) # setting u as initial guess?

            v_old, = F.arguments()
            v_high = v_old.reconstruct(function_space=V_high)
            F_high = replace(F, {v_old: v_high, u: u_high})
            bcs_high = [bc.reconstruct(V=V_high, indices=bc._indices) for bc in bcs]
            problem_high = NonlinearVariationalProblem(F_high, u_high, bcs_high) # create the enriched problem

            self.print(f"Solving primal with higher order for error estimate (degree: {high_degree}, dofs: {V_high.dim()}) ...")
            # solve the enriched problem
            solver = NonlinearVariationalSolver(problem_high, solver_parameters=self.parameters,
                                                options_prefix=primal_options_prefix,
                                                **self.primal_solver_kwargs)
            solver.set_transfer_manager(self.atm)
            solver.solve()
            self.primal_solver = solver

            self.u_high = u_high

            
            # Ways to obtain uh : 
            # "solve" - independently solve the nonlinear problem in V
            # "project" - solve only in V_high, then L²-project u_high into V
            # "interpolate" - solve only in V_high, then interpolate u_high into V
            
            if self.options.primal_low_method == "solve":
                pass
            elif self.options.primal_low_method == "project":
                u.project(u_high)
            elif self.options.primal_low_method == "interpolate":
                u.interpolate(u_high)
            else:
                raise ValueError(f"Unrecognised primal_low_method {self.options.primal_low_method}")
            
            
            # u_err is uh^+ - uh 
            # I want uh but I also want u_err = uh^+ - Ih uh^+ 
            # So I will need to reassess this!!

            u_err = u_high - u

        # Dont need primal error for primal residual 
        else:
            solve_uh()
            u_err = None
        return u_err



    
    
    def solve_dual(self):
        """Solve the dual (adjoint) problem and return the dual solutions.

        The dual problem is always solved in an enriched space of degree
        ``degree + dual_extra_degree``.  A low-degree approximation ``z_lo`` is
        obtained via ``dual_low_method`` (interpolate / project / solve).

        Returns
        -------
        tuple[Function, Function]
            ``(z_lo, z_err)`` where ``z_lo`` is the low-degree dual solution
            (in the same space as the primal ``u``) and ``z_err = z - z_lo``
            is the dual error representative used to weight the residuals.
        """

        def solve_zh(z, linearise_at_high=True):
            bcs = self.problem.bcs
            J = self.goal_functional
            F = self.problem.F
            u = self.problem.u
            dual_options_prefix = self.options_prefix + "dual_"

            Z = z.function_space()
            self.print(f"Solving dual (degree: {Z.ufl_element().degree()}, dofs: {Z.dim()}) ...")

            Fz = residual(F, TestFunction(Z))
            dF = derivative(Fz, u, TrialFunction(Z))
            dJ = derivative(J, u, TestFunction(Z))
            a = adjoint(dF)

            if linearise_at_high and self.u_high is not None:
                a = replace(a, {u: self.u_high})
                dJ = replace(dJ, {u: self.u_high})

            bcs_dual = [bc.reconstruct(V=Z, indices=bc._indices, g=0) for bc in bcs]
            problem = LinearVariationalProblem(a, dJ, z, bcs_dual)
            solver = LinearVariationalSolver(problem, solver_parameters=self.parameters,
                                             options_prefix=dual_options_prefix,
                                             **self.dual_solver_kwargs)
            solver.set_transfer_manager(self.atm)
            solver.solve()

        # Higher-order dual solution
        V = self.problem.u.function_space()
        dual_degree = V.ufl_element().degree() + self.options.dual_extra_degree
        V_dual = reconstruct_degree(V, dual_degree)
        z = Function(V_dual, name="dual_high_order_solution")
        solve_zh(z, linearise_at_high=True)

        # Lower-order dual solution
        z_lo = Function(V, name="dual_low_order_solution")
        if self.options.dual_low_method == "solve":
            z_lo.interpolate(z)
            # Linearise at the low-order primal (self.problem.u), not u_high,
            # so the low-order adjoint matches the derivation of the cubic DWR estimate.
            solve_zh(z_lo, linearise_at_high=False)
        elif self.options.dual_low_method == "project":
            z_lo.project(z)
        elif self.options.dual_low_method == "interpolate":
            z_lo.interpolate(z)
        else:
            raise ValueError(f"Unrecognised dual_low_method {self.options.dual_low_method}")
        z_err = z - z_lo
        self.z = z
        return z_lo, z_err

    
    
    
    
    def estimate_error(self, u_err, z_lo, z_err):
        """Compute the global DWR error estimate for the goal functional.

        Computes the primal residual :math:`\\rho(u_h; z - z_h)` and, when
        ``use_adjoint_residual=True``, also the adjoint residual
        :math:`\\rho^*(z_h; u - u_h)`, combining them as
        :math:`\\frac{1}{2}(\\rho + \\rho^*)`.  Also estimates the solver error
        :math:`\\rho(u_h; z_h)`.

        Parameters
        ----------
        u_err
            Primal error representative ``u_high - u_h`` (or ``None`` when
            ``use_adjoint_residual=False``).
        z_lo
            Low-degree dual solution in the base space.
        z_err
            Dual error representative ``z - z_lo``.

        Returns
        -------
        tuple[float, float | None]
            ``(eta_h, eta)`` — the error estimate and the true error
            ``J(u) - J(u_h)`` (or ``None`` if no exact value was supplied).
        """


        J = self.goal_functional
        F = self.problem.F
        u = self.problem.u

        # Primal contribution to error estimator
        primal_err = assemble(residual(F, -z_err))

        # Dual contribution to error estimator
        if self.options.use_adjoint_residual:
            Z = z_lo.function_space()
            dF = derivative(F, u, TrialFunction(Z))
            dJ = derivative(J, u)
            G = action(adjoint(dF), z_lo) - dJ

            dual_err = assemble(residual(G, -u_err))
            discretisation_error = 0.5 * (primal_err + dual_err)
        else:
            discretisation_error = primal_err

        # Estimate of solver error
        solver_error = assemble(residual(F, -z_lo))

        if abs(solver_error) > abs(discretisation_error):
            self.print(RED % 'Warning: solver error estimate greater than discretisation error estimate, refine solver tolerances')

        # Final error estimate
        eta_h = discretisation_error + solver_error
        self.etah_vec.append(eta_h)

        Juh = assemble(J)
        self.print(f'{"Computed goal J(uh):":40s}{Juh:15.12f}')
        self.Juh = Juh
        if self.goal_exact is not None:
            if isinstance(self.goal_exact, numbers.Real):
                Ju = self.goal_exact
            else:
                Ju = assemble(self.goal_exact)
        elif self.u_exact is not None:
            Ju = assemble(replace(J, {u: self.u_exact}))
        else:
            Ju = None

        if Ju is not None:
            eta = Ju - Juh
            self.eta_vec.append(eta)
            self.print(f'{"Exact goal J(u):":40s}{Ju: 15.12f}')
            self.print(f'{"True error, J(u) - J(u_h):":40s}{eta: 15.12e}')
        else:
            eta = None

        if self.options.use_adjoint_residual:
            self.print(f'{"Primal error, rho(u_h; z-z_h):":40s}{primal_err: 15.12e}')
            self.print(f'{"Dual error,  rho*(z_h; u-u_h):":40s}{dual_err: 15.12e}')
            self.print(f'{"Difference":40s}{abs(primal_err-dual_err):19.12e}')
            self.print(f'{"Discretisation error, 0.5(rho + rho*)":40s}{discretisation_error: 15.12e}')
        else:
            self.print(f'{"Discretisation error, rho(u_h; z-z_h)":40s}{discretisation_error: 15.12e}')
        self.print(f'{"Solver error, rho(u_h; z_h):":40s}{solver_error: 15.12e}')
        self.print(f'{"Final error estimate:":40s}{eta_h: 15.12e}')
        return eta_h, eta

    
    
    
    def compute_error_indicators(self):
        """Compute cell-wise DWR error indicators via bubble/cone projections.

        Projects the primal residual :math:`F(u_h; \\cdot)` onto cell-bubble
        and facet-bubble spaces, then weights by the dual error ``z_err``.
        When ``use_adjoint_residual=True``, the adjoint residual is also
        projected and weighted by ``u_err``, and the two contributions are
        averaged.

        Returns
        -------
        Function
            DG0 Function of absolute-value cell-wise indicators :math:`\\eta_K`.
        """
        J = self.goal_functional
        F = self.problem.F
        u = self.problem.u
        u_err = self._u_err # need to check what this is!!
        z_lo = self._z_lo
        z_err = self._z_err
        V = u.function_space()

        mesh = V.mesh().unique()
        dim = mesh.topological_dimension
        cell = mesh.ufl_cell()
        variant = "integral"

        # Might need to change for mixed spaces!!
        degree = V.ufl_element().degree()
        cell_residual_degree = degree + self.options.cell_residual_extra_degree
        facet_residual_degree = degree + self.options.facet_residual_extra_degree


        # ------------------------------- Primal residual -------------------------------
        # Cell bubbles
        B = FunctionSpace(mesh, "B", dim+1, variant=variant)
        bubbles = Function(B).assign(1)

        # Might need to change for mixed spaces!!
        # DG space on cell interiors
        if V.value_shape == ():
            DG = FunctionSpace(mesh, "DG", cell_residual_degree, variant=variant)
        else:
            DG = TensorFunctionSpace(mesh, "DG", cell_residual_degree, variant=variant, shape=V.value_shape)


        # Might need problem on each subspace of a mixed space
        uc = TrialFunction(DG)
        vc = TestFunction(DG)
        ac = inner(uc, bubbles*vc)*dx
        Lc = residual(F, bubbles*vc)
        # solve for Rcell
        Rcell = Function(DG)
        solve(ac == Lc, Rcell, solver_parameters=self.options.sp_cell)

        # Facet bubbles
        FB = FunctionSpace(mesh, "FB", dim, variant=variant)
        cones = Function(FB).assign(1)

        # Broken facet bubble space
        # Might need to change for mixed spaces!!
        el = BrokenElement(FiniteElement("FB", cell=cell, degree=facet_residual_degree+dim, variant=variant))
        if V.value_shape == ():
            Q = FunctionSpace(mesh, el)
        else:
            Q = TensorFunctionSpace(mesh, el, shape=V.value_shape)

        # Might need problem on each subspace of a mixed space
        Qtest = TestFunction(Q)
        Qtrial = TrialFunction(Q)
        Lf = residual(F, Qtest) - inner(Rcell, Qtest)*dx
        af = both(inner(Qtrial/cones, Qtest))*dS + inner(Qtrial/cones, Qtest)*ds
        # solve for Rfacet
        Rhat = Function(Q)
        solve(af == Lf, Rhat, solver_parameters=self.options.sp_facet)
        Rfacet = Rhat/cones

        # Primal error indicators
        DG0 = FunctionSpace(mesh, "DG", degree=0)
        test = TestFunction(DG0)

        # Will inner sum over spaces?
        eta_primal = assemble(
            inner(inner(Rcell, z_err), test)*dx +
            + inner(avg(inner(Rfacet, z_err)), both(test))*dS +
            + inner(inner(Rfacet, z_err), test)*ds
        )
        with eta_primal.dat.vec as evec:
            evec.abs()

        # ------------------------------- Adjoint residual -------------------------------
        if self.options.use_adjoint_residual:
            # r*(v) = J'(u)[v] - A'_u(u)[v, z] since F = A(u;v) - L(v)
            dF = derivative(F, u, TrialFunction(V))
            dJ = derivative(J, u, TestFunction(V))
            rstar = action(adjoint(dF), z_lo) - dJ

            # dual: project r* -> Rcell*, Rfacet*
            Lc_star = residual(rstar, bubbles*vc)
            Rcell_star = Function(DG)
            solve(ac == Lc_star, Rcell_star, solver_parameters=self.options.sp_cell)

            Lf_star = residual(rstar, Qtest) - inner(Rcell_star, Qtest)*dx
            Rhat_star = Function(Q)
            solve(af == Lf_star, Rhat_star, solver_parameters=self.options.sp_facet)
            Rfacet_star = Rhat_star/cones

            eta_dual = assemble(
                inner(inner(Rcell_star, u_err), test)*dx
                + inner(avg(inner(Rfacet_star, u_err)), both(test))*dS
                + inner(inner(Rfacet_star, u_err), test)*ds
            )
            with eta_dual.dat.vec as evec:
                evec.abs()
            eta_cell = assemble(0.5*(eta_primal + eta_dual))

            with eta_primal.dat.vec as evec:
                self.eta_primal_total = abs(evec.sum())
            with eta_dual.dat.vec as evec:
                self.eta_dual_total = abs(evec.sum())
            self.print(f'{"Sum of primal refinement indicators:":40s}{self.eta_primal_total: 15.12e}')
            self.print(f'{"Sum of dual refinement indicators:":40s}{self.eta_dual_total: 15.12e}')
        else:
            eta_cell = eta_primal

        return eta_cell

    
    # Simplified from older code by delegating the task to the base class
    def refine_problem(self, markers):
        """Adaptively refine the mesh and rediscretise the problem on the refined mesh"""
        coef_map = {}
        super().refine_problem(markers, coef_map=coef_map)

        self.goal_functional = refine(self.goal_functional, refine, coefficient_mapping=coef_map)
        if self.u_exact is not None:
            self.u_exact = refine(self.u_exact, refine, coefficient_mapping=coef_map)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def vtk_output_callback(output_dir="./output", run_name="default"):
    """Return a ``post_iteration_callback`` that writes primal and dual solutions to VTK.

    Usage::

        from firedrake import *
        solver = GoalAdaptiveNonlinearVariationalSolver(
            problem, goal_functional,
            solver_parameters={...},
            post_iteration_callback=vtk_output_callback(
                output_dir="./output", run_name="myproblem"
            ),
        )

    Parameters
    ----------
    output_dir
        Directory in which to write the VTK files.
    run_name
        Label prepended to filenames:
        ``<output_dir>/<run_name>/<run_name>_solution_<it>.pvd`` and
        ``<output_dir>/<run_name>/<run_name>_dual_solution_<it>.pvd``.

    Returns
    -------
    callable
        A function ``callback(solver, it)`` suitable for passing to
        ``post_iteration_callback``.
    """
    def _callback(solver, it):
        prefix = f"{output_dir}/{run_name}/{run_name}"
        comm = solver.problem.u.function_space().mesh().comm
        solver.print("Writing (primal) solution ...")
        VTKFile(f"{prefix}_solution_{it}.pvd", comm=comm).write(*solver.problem.u.subfunctions)
        solver.print("Writing (dual) solution ...")
        VTKFile(f"{prefix}_dual_solution_{it}.pvd", comm=comm).write(*solver.z.subfunctions)
    return _callback





























class GoalAdaptiveFoldedEigensolver(SteadyGoalAdaptiveSolver, OptionsManager):
    """Solves an eigenvalue problem adaptively to minimise the error in a
    target eigenvalue of a folded operator.  The goal functional is :math:`J = \\lambda`.

    At each iteration the solver:

    1. Solves the primal (and, if non-self-adjoint, dual) eigenproblem at both
       degree ``p`` and ``p + dual_extra_degree``.
    2. Estimates the error in the eigenvalue via the dual-weighted residual
       formula of Larson & Bengzon.
    3. Computes cell-wise indicators using the same bubble/cone projection as
       :class:`GoalAdaptiveNonlinearVariationalSolver`.
    4. Refines the mesh by Dörfler marking.

    Parameters
    ----------
    problem
        A :class:`~.LinearEigenproblem` on the initial mesh.
    target
        Target eigenvalue used by SLEPc for spectral targeting
        (``eps_target``).
    epsilon
        the desired pseudospectral contour
    tolerance
        Terminate when ``|eta_h| < tolerance``.
    goal_adaptive_options
        Dictionary passed to :class:`GoalAdaptiveEigenOptions`.
        Key extra entries: ``"self_adjoint"`` (bool) and ``"nev"`` (int).
    solver_parameters
        SLEPc solver parameters forwarded to :class:`~.LinearEigensolver`.
        Do not set ``eps_target`` here; it is set from ``target``.
    exact_eigenvalue
        Exact eigenvalue, if known, for computing efficiency indices.
    """

    # def __init__(self,
    #              problem,
    #              target: float,
    #              tolerance: float,
    #              goal_adaptive_options: dict | None = None,
    #              solver_parameters: dict | None = None,
    #              exact_eigenvalue: float | None = None,
    #              ):
    #     super().__init__(tolerance, goal_adaptive_options, exact_goal=exact_eigenvalue)
    #     self.problem = problem
    #     self.target = target
    #     self.sp = solver_parameters or {}

    #     # Set up AdaptiveMeshHierarchy
    #     mesh = problem.output_space.mesh()
    #     mh, level = get_level(mesh)
    #     if mh is None:
    #         AdaptiveMeshHierarchy(mesh)
    #     else:
    #         amh = AdaptiveMeshHierarchy(mh[0])
    #         for m in mh[1:level+1]:
    #             amh.add_mesh(m)

    #     self.atm = AdaptiveTransferManager()
    #     self._lam_h = None

    
    # # Old options handler:
    # # def _make_options(self, d):
    # #     return GoalAdaptiveEigenOptions(**d)


    # New init (made by chat to match the nonlinar solver approach)


    # 1. Maggie change - New init (made by chat to match the nonlinar solver approach). Also add in epsilon for pseudospectrum

    _GOAL_PREFIX = "goal_adaptive_"

    def __init__(
        self,
        problem,
        target: float | complex,
        *,
        epsilon: float = 1.0e-2,
        solver_parameters: dict | None = None,
        options_prefix: str | None = None,
        exact_eigenvalue: float | complex | None = None,
        post_iteration_callback=None,):
        
        if solver_parameters is None:
            solver_parameters = {}

        if options_prefix is None:
            options_prefix = ""

        # The new adaptive base creates the mesh hierarchy, transfer manager,
        # history vectors, tolerance options and effectivity vectors.
        base_mesh = problem.output_space.mesh()

        SteadyGoalAdaptiveSolver.__init__(
            self,
            base_mesh,
            solver_parameters,
            exact_goal=exact_eigenvalue,
            post_iteration_callback=post_iteration_callback,)

        # Handles the PETSc/SLEPc parameters and options prefix.
        OptionsManager.__init__(
            self,
            solver_parameters,
            options_prefix,)

        # Eigenproblem data.
        self.problem = problem
        self.target = target
        self.epsilon = epsilon
        self._lam_h = None
        self._u_h = None
        self._lam_p = None
        self._u_p = None
        self._z_h = None
        self._z_p = None
        self._z_err = None

        # 2. Maggie Change - extra for Matt's algorithm 
        self.signed_error = None
        self.corrected_eigenvalue = None

    # 3. Maggie Change - new make options (created with chat)
    def _make_options(self, data):
    
        # If its already in format of EigenOptions do nothing
        if isinstance(data, GoalAdaptiveEigenOptions):
            return data

        # If given no inputs, set the default
        if data is None:
            return GoalAdaptiveEigenOptions()

        #  If a dictionary, extract the adaptive part
        if isinstance(data, dict):
            eigen_data = data.get("goal_adaptive", {})
            return GoalAdaptiveEigenOptions(**eigen_data)

        # If we dont know what to do with it raise an error 
        raise TypeError("Expected GoalAdaptiveEigenOptions or a solver-parameter dictionary.")


    # 4. Maggie change - instead of current solution in nonlinear variational problem, return current eigenpair
    # Set what we mean by the current eigenpair
    # Could be changed to return the enriched result
    def _current_eigenpair(self):
        """Return the most recently computed low-order eigenpair."""
        if self._lam_h is None or self._u_h is None:
            raise RuntimeError("No eigenpair has been computed yet.")
        return self._lam_h, self._u_h

    # Solve setup
    # def solve(self):
    #     """Run the adaptive loop and return the eigenvalue and error estimate.
    #     Returns
    #     -------
    #     tuple[float, float]
    #         ``(lam_h, error_estimate)`` on the finest mesh.
    #     """
    #     super().solve()
    #     # return self._lam_h, self.etah_vec[-1]
    #     return self._lam_h



    # Stopping criteria for folded operators
    def should_refine(self, it, eta_h, eta):
        """ The stopping criteria we want the base to use to decide when to refine """

        lam_h = self._lam_h    

        # Compute and Correct Matt's phi value
        self.matts_phi = np.sqrt(lam_h)
        corrected = lam_h + self.signed_error

        if corrected >= 0:
            self.corrected_phi = np.sqrt(corrected)
        else:
            self.corrected_phi = float("nan")

        self.print(f'Matts phi value is {self.matts_phi}')
        self.print(f'The (error) corrected phi value is {self.corrected_phi}')

        # Check if L2 alignement failed 
        if self.check_up:
            self.print(GREEN % f'The L2 aligned enriched soln was very small (in norm) - red flag!!')
            self.print(GREEN % f'As a result, we will adapt the eigensolve')
            return True 
        
        # Check pseudospectral info
        gap_spec = self.matts_phi  - self.epsilon 
        gap_correct = self.corrected_phi - self.epsilon
        if gap_spec * gap_correct < 0:
            self.print(GREEN % f'The corrected phi and original phi disagree')
            self.print(GREEN % f'As a result, we will adapt the eigensolve')
            return True 
        
        # Otherwise, dont adapt
        else:
            self.print(GREEN % f'The corrected and original phi match')
            self.print(GREEN % f'No need to do an adaptive solve')
            self.print(GREEN % f'But I will adapt anyways for now')
            return True


    # 5. Maggie change - Rewrite solve and step like in nonlinear code
    def solve(self):
        """Run the adaptive loop and return the final computed eigenpair."""
        super().solve()
        return self._current_eigenpair()

    
    # A step method
    def step(self, it=None):
        """Run one adaptive eigenvalue iteration and return the current eigenpair."""
        if it is None:
            V = self.problem.output_space
            _, it = get_level(V.mesh())
        super().step(it)
        return self._current_eigenpair()
        #  if it is None:
        #     V = self.problem.u.function_space()
        #     _, it = get_level(V.mesh())
        # super().step(it)
        # return self._current_solution()


    def solve_and_estimate(self):

        """Solve the eigenproblem at two polynomial degrees, match eigenfunctions,
        and compute a global error estimate for the eigenvalue."""


        opts = self.options
        problem = self.problem
        V = problem.output_space
        self.Ndofs_vec.append(V.dim())
        self.print(f"Solving eigenproblem (degree: {V.ufl_element().degree()}, dofs: {V.dim()}) ...")

        #sp_target = dict(self.sp)
        sp_target = dict(self.parameters) # needed change?
        if opts.self_adjoint: # if self adjoint then tell the solver
            sp_target.setdefault("eps_gen_hermitian", None)
        sp_target["eps_target"] = self.target

        # Solve at degree p
        lams, vecs = _solve_eigs(problem, opts.nev, sp_target)
        self._lam_h = lams[0]
        self._u_h = vecs[0]
        self.print(f'{"Computed eigenvalue:":40s}{self._lam_h:15.12f}')

        # 6. Maggie change - adding in some checks
        # Some checks for a self adjoint problem
        if opts.self_adjoint:
            # Check that lam_h is real 
            if abs(self._lam_h.imag) > 1.0e-8:
                print(RED % f"Warning: computed folded eigenvalue has nontrivial imaginary part: {self._lam_h}")
                print(RED % f"We will take the real part")
                self._lam_h = self._lam_h.real
            # Check that lam_h is nonnegative
            # neg_tol = 1e-8
            # if self._lam_h < 0:
            #     print(RED % f"WARNING - the min eigenval is negative: lambda_min = {self._lam_h}")
            #     if abs(self._lam_h) <= neg_tol:
            #         print(RED % f"Since the eigenvalue is small and negative, we set lambda_min = 0")
            #         self._lam_h = 0.0
            #     else:
            #         raise ValueError(RED % f"The eigenvalue is significantly negative and thats an issue so we quit!")



        # Solve at degree p + dual_extra_degree (enriched primal)
        # dual_extra_degree or primal_extra_degree??
        high_problem = _reconstruct_eig_degree(problem, opts.dual_extra_degree) # Might need to change for mixed spaces
        self.print(f"Solving enriched eigenproblem (dofs: {high_problem.output_space.dim()}) ...")
        lams_p, vecs_p = _solve_eigs(high_problem, opts.nev, sp_target)
        self._lam_p = lams_p[0]


        
        # 7. Maggie Change - checking multiplicity
        # Check multiplicity based on lower degree solve
        mult_lower = 1
        tolerance = 0.01 
        for i in range(1, len(vecs)):
            if abs(lams[i] - self._lam_h) <= tolerance:
                mult_lower += 1
        self.print(f"Lower degree solve multiplicity: {mult_lower}")


        # check multiplicity based on enriched solve
        self.mult = 1
        tolerance = 0.01 
        for i in range(1, len(vecs_p)):
            if abs(lams_p[i] - lams_p[0]) <= tolerance:
                self.mult += 1
        self.print(f"Enriched solve multiplicity: {self.mult}")
        self.enriched_cluster = vecs_p[:self.mult] 

        # Compare multiplicities
        if mult_lower != self.mult:
            self.print(RED % f'Warning - the lower degree and enriched multiplicities dont match')
        
        # 8. Maggie change - L2 matching
        # I will need to update this matching!!
        # self._lam_p, self._u_p = match_best(self._u_h, vecs_p, lams_p)
        self.print("Matching eigenfunctions with L2 projection (Maggie Change)...")
        if is_mixed_space(V): # Assumes mixed space has 2 subspaces for now (Maxwell example)
            # Set the enriched bases (mixed space) for the L2 projection code
            self.enriched_basis_E = [self.enriched_cluster[i].subfunctions[0] for i in range(self.mult)]
            self.enriched_basis_H = [self.enriched_cluster[i].subfunctions[1] for i in range(self.mult)]
            # L2 enriched matching
            self._u_p = match_best_mixed(self._u_h, self.enriched_basis_E, self.enriched_basis_H, self.mult, high_problem.output_space) # is this the right call to V_high?
        # If not, just do L2 projection without any splitting:
        else:
            self._u_p = match_best_notmixed(self._u_h, self.enriched_cluster, self.mult, high_problem.output_space)


        # Check the size of the enriched result
        check_nrm = m_norm(self._u_p)
        self.check_up = False
        if check_nrm < 1e-12:
            print(RED % f'Since the L2 aligned enriched vector is small (in norm) we will want to adapt!!')
            self.check_up = True


        if opts.self_adjoint: # if self adjoint we dont need to dual problem
            self._z_h = self._u_h
            self._z_p = self._u_p


        else: # if not need adjoint problem # Maggie never touched this because folded operator is self adjoint
            # Adjoint eigenproblem at degree p
            adj_problem = _make_adjoint_eig_problem(problem)
            self.print(f"Solving adjoint eigenproblem (dofs: {adj_problem.output_space.dim()}) ...")
            lamz, zs = _solve_eigs(adj_problem, opts.nev, sp_target)
            _, self._z_h = match_best(self._u_h, zs, lamz)

            # Adjoint at degree p + dual_extra_degree
            adj_high = _reconstruct_eig_degree(adj_problem, opts.dual_extra_degree)
            self.print(f"Solving enriched adjoint eigenproblem (dofs: {adj_high.output_space.dim()}) ...")
            lamzp, zsp = _solve_eigs(adj_high, opts.nev, sp_target)
            _, self._z_p = match_best(self._u_h, zsp, lamzp)

        # 9. Maggie change - Making z_err a function so we can split it later
        # Pablo: Dual error representative (UFL expression, may span two spaces)
        # I think this is wrong!!
        # My change to make it a function
        self._z_err = Function(self._z_p.function_space())
        self._z_err.interpolate(self._z_p - self._z_h)
        # self._z_err = self._z_p - self._z_h

        # Compute the errors
        eta_h, eta = self._estimate_eigenvalue_error()
        return eta_h, eta


    

    def _estimate_eigenvalue_error(self):

        """Compute global error estimate for the eigenvalue (Larson–Bengzon formula)."""
        from firedrake.assemble import assemble


        u_h, u_p = self._u_h, self._u_p
        z_h, z_p = self._z_h, self._z_p
        lam_h = self._lam_h
        A = self.problem._original_A
        M = self.problem._original_M

        # Low-order primal representative in the base space
        phi_h = Function(u_h.function_space())
        phi_h.interpolate(u_p)
        e = u_p - phi_h     # primal enrichment error (UFL expression)
        e_sigma = u_p - u_h # for the remainder term

        # 10. Maggie change - m_norm in error
        if self.options.self_adjoint:
            #sigma_h = 0.5 * float(assemble(inner(e_sigma, e_sigma) * dx))
            sigma_h = 0.5 * m_norm(e_sigma)**2 # Maggie change for now
            rhs = assemble(replace_both_args(A, u_h, e)
                                 - lam_h * replace_both_args(M, u_h, e)) # took float out

        else: # Maggie never touched this because folded operator is self adjoint
            e_adj = z_p - z_h
            sigma_h = 0.5 * float(assemble(inner(e_sigma, e_adj) * dx))
            rhs = 0.5 * (
                float(assemble(replace_both_args(A, u_h, e_adj)))
                - lam_h * float(assemble(replace_both_args(M, u_h, e_adj)))
                + float(assemble(replace_both_args(adjoint(A), z_h, e)))
                - lam_h * float(assemble(replace_both_args(M, z_h, e)))
            )

        # 11. Maggie change - signed error, phi_value, and its correction.
        denom = 1.0 - sigma_h
        self.signed_error = rhs/denom if abs(denom) > 1e-14 else float("nan")

        # a bunch of checks:
        m_hh = assemble(replace_both_args(M, u_h, u_h))
        m_pp = assemble(replace_both_args(M, u_p, u_p))
        m_hp = assemble(replace_both_args(M, u_h, u_p))
        print("m(u_h, u_h) =", m_hh)
        print("m(u_p, u_p) =", m_pp)
        print("m(u_h, u_p) =", m_hp)
        print('top of estimate =', rhs)
        print('bottom of estimate =', denom)
        print('extra checks')
        a_hp = assemble(replace_both_args(A, u_h, u_p))
        a_ph = assemble(replace_both_args(A, u_p, u_h))
        m_hp = assemble(replace_both_args(M, u_h, u_p))
        m_ph = assemble(replace_both_args(M, u_p, u_h))
        print("a(u_h, u_p) =", a_hp)
        print("conj(a(u_p, u_h)) =", np.conj(a_ph))
        print("A Hermitian defect =", a_hp - np.conj(a_ph))
        print("m(u_h, u_p) =", m_hp)
        print("conj(m(u_p, u_h)) =", np.conj(m_ph))
        print("M Hermitian defect =", m_hp - np.conj(m_ph))
        a_pp = assemble(replace_both_args(A, u_p, u_p))
        m_pp = assemble(replace_both_args(M, u_p, u_p))
        lam_p_rayleigh = a_pp / m_pp
        print("Rayleigh quotient of matched u_p =", lam_p_rayleigh)
        print('even more checks')
        u_h_high = Function(u_p.function_space())
        for uh_high_i, uh_i in zip(
            u_h_high.subfunctions,
            u_h.subfunctions,
        ):
            uh_high_i.interpolate(uh_i)

        a_hp = assemble(
            replace_both_args(A, u_h_high, u_p)
        )

        a_ph = assemble(
            replace_both_args(A, u_p, u_h_high)
        )

        m_hp = assemble(
            replace_both_args(M, u_h_high, u_p)
        )

        m_ph = assemble(
            replace_both_args(M, u_p, u_h_high)
        )

        print("A Hermitian defect:", a_hp - np.conj(a_ph))
        print("M Hermitian defect:", m_hp - np.conj(m_ph))

        # Maggie change 11.5 - Check if imaginary
        if abs(self.signed_error.imag) > 1.0e-10:
                self.print(RED % f"Error estimate has a nontrivial imaginary part and we will take real part: {self.signed_error}")
                self.signed_error = self.signed_error.real 


        eta_h = abs(rhs / denom) if abs(denom) > 1e-14 else float("nan")
        self.etah_vec.append(eta_h)
        self.print(f'{"Predicted magnitude of error:":40s}{eta_h: 15.12e}')
        self.print(f'{"Predicted signed error:":40s}{self.signed_error: 15.12e}')

        
        # # Correct Matt's phi value
        # self.matts_phi = sqrt(lam_h)
        # corrected = lam_h + self.signed_error
        # if corrected >= 0:
        #     self.corrected_phi = sqrt(corrected)
        # else:
        #     self.corrected_phi = float("nan")
        # print(f'Matts phi value is {self.matts_phi}')
        # print(f'The (error) corrected phi value is {self.corrected_phi}')

        eta = None
        if self.goal_exact is not None:
            eta = abs(self.goal_exact - lam_h)
            self.eta_vec.append(eta)
            self.print(f'{"Exact eigenvalue:":40s}{self.goal_exact: 15.12f}')
            self.print(f'{"True error:":40s}{eta: 15.12e}')

        return eta_h, eta


    def compute_error_indicators(self):
        """Compute cell/facet residual indicators for the eigenvalue problem."""
        A = self.problem._original_A
        M = self.problem._original_M
        u_h = self._u_h
        lam_h = self._lam_h
        z_err = self._z_err
        # Residual linear form: A(u_h, v) - lam_h * M(u_h, v)
        _, trial = A.arguments()
        F_eig = replace(A, {trial: u_h}) - lam_h * replace(M, {trial: u_h})
        return _compute_residual_indicators(F_eig, z_err, self.options)

    

    # I will use the base refinement for now!

    # def refine_problem(self, markers):
    #     """Refine the mesh and reconstruct the :class:`~.LinearEigenproblem`."""
    #     mesh = markers.function_space().mesh()
    #     new_mesh = mesh.refine_marked_elements(markers)
    #     amh, _ = get_level(mesh)
    #     amh.add_mesh(new_mesh)
    #     coef_map = {}
    #     self.problem = refine(self.problem, refine, coefficient_mapping=coef_map)


    # def write_solution(self, it):
    #     ws = self.options.write_solution
    #     if ws is False:
    #         return
    #     elif ws is True:
    #         should_write = True
    #     elif isinstance(ws, Integral):
    #         should_write = (it % ws == 0)
    #     else:
    #         raise ValueError(f"write_solution must be False, True, or a positive integer, got {ws!r}")
    #     if should_write:
    #         output_dir = self.options.output_dir
    #         run_name = self.options.run_name
    #         self.print("Writing (primal) eigenfunction ...")
    #         VTKFile(f"{output_dir}/{run_name}/{run_name}_eigenfunction_{it}.pvd"
    #                 ).write(*self._u_h.subfunctions)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def residual(F, test):
    """Replace the test function argument of a linear Form."""
    v, = F.arguments()
    return replace(F, {v: test})


def both(u):
    """Add u on both sides of a facet."""
    return u("+") + u("-")


def as_mixed(exprs):
    """Flatten a list of ufl.Expr objects into a vector."""
    return as_vector([e[idx] for e in exprs for idx in np.ndindex(e.ufl_shape)])


# # 23. Maggie change - passing mesh to reconstructor (needed for mixed spaces)
# def reconstruct_degree(V, degree, mesh):
#     """Reconstruct a FunctionSpace with a different polynomial degree."""
#     element = PMGPC.reconstruct_degree(V.ufl_element(), degree)
#     return V.reconstruct(mesh=mesh,element=element,)

# Pablos version
def reconstruct_degree(V, degree):
    """Reconstruct a FunctionSpace with a different polynomial degree."""
    return V.reconstruct(element=PMGPC.reconstruct_degree(V.ufl_element(), degree))


def reconstruct_bcs(bcs, V):
    """Reconstruct a list of BCs on a different FunctionSpace."""
    return [bc.reconstruct(V=V, indices=bc._indices) for bc in bcs]


def replace_both_args(bilinear_form, trial_coeff, test_coeff):
    """Substitute both arguments of a bilinear form with coefficients.

    Parameters
    ----------
    bilinear_form
        A UFL bilinear form with two arguments.
    trial_coeff
        Coefficient to substitute for the trial (second) argument.
    test_coeff
        Coefficient to substitute for the test (first) argument.

    Returns
    -------
    ufl.Form
        A 0-form (scalar integral).
    """
    test, trial = bilinear_form.arguments()
    return replace(bilinear_form, {test: test_coeff, trial: trial_coeff})


def l2_normalize(f):
    """L2-normalise a :class:`~.Function` in place and return it."""
    from firedrake.assemble import assemble
    nrm = float(assemble(inner(f, f) * dx)) ** 0.5
    if nrm > 0:
        f.assign(f / nrm)
    return f


# 12. Maggie change - Adding in mixed versions of inner products and norms
# DO I NEED TO SPLIT THE INNER PRODUCT?
def is_mixed_function(f):
    return hasattr(f, "subfunctions") and len(f.subfunctions) > 1

def is_mixed_space(V):
    return hasattr(V, "subspaces") and len(V.subspaces) > 0

def mixed_inner(a, b):
    return sum(inner(ai, bi) for ai, bi in zip(split(a), split(b)))

def mixed_weighted_inner(a, b, weight):
    return sum(inner(ai, weight * bi) for ai, bi in zip(split(a), split(b)))

# Here, m is the L2 inner product (folded operator)

def m_inner(u, v):
    from firedrake.assemble import assemble
    if is_mixed_function(u):
        return assemble(sum(inner(ui, vi) * dx for ui, vi in zip(u.subfunctions, v.subfunctions)))
    else:
        return assemble(inner(u, v) * dx)

def m_norm(u):
    val = m_inner(u, u)
    return abs(val) ** 0.5

def m_normalize(f):
    nrm = m_norm(f)
    if nrm < 1e-12:
        print(RED % f"Warning - when m-normalizing, m norm is really small - {nrm}")
        
    else:
        if is_mixed_function(f):
            for fi in f.subfunctions:
                fi.assign(fi / nrm)
        else:
            f.assign(f / nrm)
    return f


# 13. Maggie change - BIGGEST change 
# Joe finds best match for uh in the computed enriched eigenbasis 
# Instead, we are going to compute the best candidate in the span of the basis using
# the L2 projection of uh onto the enriched eigenbasis.


def match_best_mixed(target, E_candidates, H_candidates, mult, V_high):

    from firedrake.assemble import assemble

    # matching for a mixed problem
    # target: Function; candidates: list[Function]; lambdas: list[...] or None
    # mult: computed multiplicity from size of enriched cluster

    # Create the Least squares problem:

    # Assemble the K matrix 
    K = np.zeros((mult, mult), dtype=complex)
    for i in range(mult):
        for j in range(mult):
            # Double check Kij vs Kji indexing - I think it matters for complex numbers (conjugate in inner product)
            K[i, j] = assemble(inner(E_candidates[j], E_candidates[i])*dx) + assemble(inner(H_candidates[j], H_candidates[i])*dx)

    # Assemble the F vector
    F = np.zeros((mult, 1), dtype=complex)
    target_E, target_H = target.subfunctions
    for i in range(mult):
        F[i, 0] = assemble(inner(target_E, E_candidates[i])*dx) + assemble(inner(target_H, H_candidates[i])*dx)

    # Solve the least squares numpy problem
    X = np.linalg.solve(K, F)

    # Reassemble the needed enriched basis function
    chosen_enriched = Function(V_high)
    for i in range(mult):
        basis_coeff = complex(X[i,0])
        chosen_enriched.subfunctions[0].interpolate(chosen_enriched.subfunctions[0] + basis_coeff * E_candidates[i])
        chosen_enriched.subfunctions[1].interpolate(chosen_enriched.subfunctions[1] + basis_coeff * H_candidates[i])

    # normalize chosen enriched result
    print('Normalizing the least squares solution')
    chosen_enriched.assign(m_normalize(chosen_enriched))
    return chosen_enriched

def match_best_notmixed(target, candidates, mult, V_high):

    # matching for a non-mixed problem
    # target: Function; candidates: list[Function]; lambdas: list[...] or None
    # mult: computed multiplicity from size of enriched cluster

    # Create the Least squares problem:

    # Assemble the K matrix 
    K = np.zeros((mult, mult), dtype=complex)
    for i in range(mult):
        for j in range(mult):
            # Double check Kij vs Kji indexing - I think it matters for complex numbers (conjugate in inner product)
            K[i, j] = assemble(inner(candidates[j], candidates[i])*dx)


    # Assemble the F vector
    F = np.zeros((mult, 1), dtype=complex)
    for i in range(mult):
        F[i, 0] = assemble(inner(target, candidates[i])*dx) 

    # Solve the least squares numpy problem
    X = np.linalg.solve(K, F)

    # Reassemble the needed enriched basis function
    chosen_enriched = Function(V_high)
    for i in range(mult):
        basis_coeff = complex(X[i,0])
        chosen_enriched.interpolate(chosen_enriched + basis_coeff * candidates[i])

    # normalize chosen enriched result
    print('Normalizing the least squares solution')
    chosen_enriched.assign(m_normalize(chosen_enriched))

    return chosen_enriched


# def match_best(target, candidates, lambdas=None):
#     """Return the candidate best correlated with ``target`` in L2.

#     Parameters
#     ----------
#     target
#         Reference :class:`~.Function`.
#     candidates
#         List of candidate :class:`~.Function` objects.
#     lambdas
#         List of associated eigenvalues (optional).

#     Returns
#     -------
#     tuple
#         ``(lam, aligned)`` where ``lam`` is the eigenvalue of the best match
#         (or its index if ``lambdas`` is ``None``) and ``aligned`` is a
#         phase/sign-aligned copy of the best candidate in the same space.
#     """
#     from firedrake.assemble import assemble
#     nt = float(assemble(inner(target, target) * dx)) ** 0.5
#     scores = []
#     for i, w in enumerate(candidates):
#         nw = float(assemble(inner(w, w) * dx)) ** 0.5
#         if nw == 0.0:
#             continue
#         c = complex(assemble(inner(target, w) * dx))
#         scores.append((abs(c) / (nt * nw), i, c))

#     if not scores:
#         raise RuntimeError("No nonzero candidate found in match_best.")

#     scores.sort(key=lambda t: t[0], reverse=True)
#     _, best_i, best_c = scores[0]

#     aligned = candidates[best_i].copy(deepcopy=True)
#     if best_c != 0:
#         phase = best_c.conjugate() / abs(best_c)
#         aligned.assign(phase * aligned)

#     lam = lambdas[best_i] if lambdas is not None else best_i
#     return lam, aligned


# Maggie change - m normalize
def _solve_eigs(problem, nev, solver_parameters):
    """Solve a :class:`~.LinearEigenproblem` and return L2-normalised eigenpairs.

    Parameters
    ----------
    problem
        :class:`~.LinearEigenproblem` to solve.
    nev
        Number of eigenvalue/eigenvector pairs to request.
    solver_parameters
        Dictionary of SLEPc solver parameters.

    Returns
    -------
    tuple[list, list]
        ``(eigenvalues, eigenfunctions)`` — both lists have length
        ``min(nconv, nev)``.
    """
    from firedrake import LinearEigensolver
    es = LinearEigensolver(problem, n_evals=nev, solver_parameters=solver_parameters)
    nconv = es.solve()
    lams, vecs = [], []
    for i in range(min(nconv, nev)):
        lams.append(es.eigenvalue(i))
        vr, _ = es.eigenfunction(i)
        # change to m normalize
        # vecs.append(l2_normalize(vr))
        vecs.append(m_normalize(vr))
    return lams, vecs


# 15. Maggie change - Construct higher degree space subspace by subspace
def _reconstruct_eig_degree(problem, extra_degree):
    """Return a new :class:`~.LinearEigenproblem` on a space of degree+extra_degree.

    Uses the original unrestricted forms stored on ``problem._original_A/M/bcs``
    to avoid double-restricting on the reconstructed problem.
    """
    from firedrake import LinearEigenproblem
    A = problem._original_A
    M = problem._original_M
    bcs = problem._original_bcs
    v, u = A.arguments()
    V = u.function_space()

    # Will change by accounting for Mixed spaces 
    # old code:
    # high_degree = V.ufl_element().degree() + extra_degree
    # V_high = reconstruct_degree(V, high_degree)
    # Build higher-order mixed space component by component
    # high_spaces = []
    # meshes = V.mesh()
    # # Need to consider all subspaces (mixed formulation)
    # # reconstruct_degree did not seem to work
    # for i, V_i in enumerate(V.subspaces):
    #     mesh_i = meshes[i]
    #     element_i = V_i.ufl_element()
    #     degree_i = element_i.degree()
    #     high_degree = degree_i + extra_degree
    #     Space_i = reconstruct_degree(V_i, high_degree, mesh = mesh_i)
    #     high_spaces.append(Space_i)
    # V_high = MixedFunctionSpace(high_spaces)
    high_spaces = []
    meshes = V.mesh()
    for i, V_i in enumerate(V.subspaces):
        mesh_i = meshes[i]
        element_i = V_i.ufl_element()
        degree_i = element_i.degree()
        high_element_i = PMGPC.reconstruct_degree(element_i, degree_i + extra_degree)
        high_spaces.append(FunctionSpace(mesh_i, high_element_i))
    V_high = MixedFunctionSpace(high_spaces)

    # new trial and test
    u_high = TrialFunction(V_high)
    v_high = TestFunction(V_high)
    A_high = replace(A, {v: v_high, u: u_high}) # Is this swapped?
    M_high = replace(M, {M.arguments()[0]: v_high, M.arguments()[1]: u_high}) # Is this swapped?

    # Bc's
    bcs_high = [bc.reconstruct(V=V_high, indices=bc._indices) for bc in bcs]

    # New problem
    return LinearEigenproblem(A_high, M_high, bcs=bcs_high,
                              bc_shift=problem.bc_shift, restrict=problem.restrict)


# Maggie leaving this alone because folded problem is self adjoint
def _make_adjoint_eig_problem(problem):
    """Return the adjoint of a :class:`~.LinearEigenproblem`.

    For a self-adjoint problem this is identical to the primal.
    """
    from firedrake import LinearEigenproblem
    A_adj = adjoint(problem._original_A)
    return LinearEigenproblem(A_adj, problem._original_M,
                              bcs=problem._original_bcs,
                              bc_shift=problem.bc_shift, restrict=problem.restrict)





def _compute_residual_indicators(F, z_err, options):
    """Compute DG0 cell-wise error indicators via bubble/cone projections.

    This implements the primal-residual indicator

    .. math::

        \\eta_K = \\int_K R_{\\text{cell}} \\cdot z_{\\text{err}}\\,\\mathrm{d}x
                  + \\int_{\\partial K} R_{\\text{facet}} \\cdot z_{\\text{err}}\\,\\mathrm{d}s

    for the residual form ``F`` against the dual error ``z_err``.

    Parameters
    ----------
    F
        Residual as a linear form (test function as its sole argument).
    z_err
        Dual error representative ``z_p - z_h`` (UFL expression or Function).
    options
        :class:`GoalAdaptiveOptions` (or subclass) instance for degree
        parameters and solver parameters.

    Returns
    -------
    Function
        DG0 Function of absolute-value cell indicators.
    """
    from firedrake.assemble import assemble

    v, = F.arguments()
    V = v.function_space()
    mesh = V.mesh().unique()
    dim = mesh.topological_dimension
    cell = mesh.ufl_cell()
    variant = "integral"

    # Will change for mixed spaces
    # Old code:
    # degree = V.ufl_element().degree()
    # cell_residual_degree = degree + options.cell_residual_extra_degree
    # facet_residual_degree = degree + options.facet_residual_extra_degree

    # 16. Maggie change - taking degree as max degree of mixed space
    # getting degree not so simple for mixed space?
    # Here I take the max degree of the mixed space
    # Select max degree of subspaces as our degree 
    if is_mixed_space(V):
        degrees = []
        for V_i in V.subspaces:
            d = V_i.ufl_element().degree()
            if isinstance(d, tuple):
                d = max(d)
            degrees.append(int(d))
        degree = max(degrees)
    else:
        d = element.degree()
        if isinstance(d, tuple):
            d = max(d)
        degree = int(d)

    # Then as before
    cell_residual_degree = degree + options.cell_residual_extra_degree
    facet_residual_degree = degree + options.facet_residual_extra_degree


    # Bubble functions
    B = FunctionSpace(mesh, "B", dim+1, variant=variant)
    bubbles = Function(B).assign(1)

    
    # Accounting for mixed spaces when forming DG spaces
    # Old code:
    # if V.value_shape == ():
    #     DG = FunctionSpace(mesh, "DG", cell_residual_degree, variant=variant)
    # else:
    #     DG = TensorFunctionSpace(mesh, "DG", cell_residual_degree, variant=variant, shape=V.value_shape)

    # 17. Maggie change - DG for mixed spaces
    if is_mixed_space(V):
        DG_spaces = []
        for Vi in V.subspaces:
            if Vi.value_shape == ():
                DG_spaces.append(FunctionSpace(mesh, "DG", cell_residual_degree, variant=variant))
            else:
                DG_spaces.append(TensorFunctionSpace(mesh, "DG", cell_residual_degree, variant=variant, shape=Vi.value_shape))
        DG = MixedFunctionSpace(DG_spaces)
    elif V.value_shape == ():
        DG = FunctionSpace(mesh, "DG", cell_residual_degree, variant=variant)
    else:
        DG = TensorFunctionSpace(mesh, "DG", cell_residual_degree, variant=variant, shape=V.value_shape)
                

    uc = TrialFunction(DG)
    vc = TestFunction(DG)

    # 18. Maggie change -
    # New code for ac if mixed space
    # Old code:
    # ac = inner(uc, bubbles*vc)*dx
    if is_mixed_space(DG):
        ac = mixed_weighted_inner(uc, vc, bubbles) * dx
    else:
        ac = inner(uc, bubbles * vc) * dx
    
    Lc = residual(F, bubbles*vc)
    Rcell = Function(DG)
    solve(ac == Lc, Rcell, solver_parameters=options.sp_cell)

    FB = FunctionSpace(mesh, "FB", dim, variant=variant)
    cones = Function(FB).assign(1)

    # 19. Maggie change - DG for mixed spaces
    # Another mixed change 
    # old code:
    # el = BrokenElement(FiniteElement("FB", cell=cell, degree=facet_residual_degree+dim, variant=variant))
    # if V.value_shape == ():
    #     Q = FunctionSpace(mesh, el)
    # else:
    #     Q = TensorFunctionSpace(mesh, el, shape=V.value_shape)
    # New code:
    el = BrokenElement(FiniteElement("FB", cell=cell, degree=facet_residual_degree+dim, variant=variant))
    if is_mixed_space(V):
        Q_spaces = []
        for Vi in V.subspaces:
            if Vi.value_shape == ():
                Q_spaces.append(FunctionSpace(mesh, el))
            else:
                Q_spaces.append(TensorFunctionSpace(mesh, el, shape=Vi.value_shape))
        Q = MixedFunctionSpace(Q_spaces)
    elif V.value_shape == ():
        Q = FunctionSpace(mesh, el)
    else:
        Q = TensorFunctionSpace(mesh, el, shape=V.value_shape)

    Qtest = TestFunction(Q)
    Qtrial = TrialFunction(Q)

    # 20. Maggie change - for mixed spaces
    # Another mixed change:
    # Old code:
    # Lf = residual(F, Qtest) - inner(Rcell, Qtest)*dx
    # af = both(inner(Qtrial/cones, Qtest))*dS + inner(Qtrial/cones, Qtest)*ds
    # New code:
    if is_mixed_space(V):
            Lf = residual(F, Qtest) - mixed_inner(Rcell, Qtest)*dx
            facet_mass = mixed_weighted_inner(Qtrial, Qtest, 1/cones)
            af = both(facet_mass) * dS + facet_mass * ds
    else:
        Lf = residual(F, Qtest) - inner(Rcell, Qtest)*dx
        af = both(inner(Qtrial/cones, Qtest))*dS + inner(Qtrial/cones, Qtest)*ds

    Rhat = Function(Q)
    solve(af == Lf, Rhat, solver_parameters=options.sp_facet)
    Rfacet = Rhat/cones
    
    
    # Build cell-wise indicators
    DG0 = FunctionSpace(mesh, "DG", degree=0)
    test = TestFunction(DG0)

    # 21. Maggie change - for mixed spaces
    # Another change based on Mixed spaces
    # Old code:
    # eta_cell = assemble(
    #     inner(inner(Rcell, z_err), test)*dx
    #     + inner(avg(inner(Rfacet, z_err)), both(test))*dS
    #     + inner(inner(Rfacet, z_err), test)*ds
    # )
    # New code
    if is_mixed_space(V):
        zerr_components = split(z_err)
        Rcell_components = split(Rcell)
        Rhat_components = split(Rhat)
        cell_weight = sum(inner(Rcell_i, zerr_i) for Rcell_i, zerr_i in zip(Rcell_components, zerr_components))
        facet_weight = sum(inner(Rhat_i / cones, zerr_i) for Rhat_i, zerr_i in zip(Rhat_components, zerr_components))

    else:
        cell_weight = inner(Rcell, z_err)
        facet_weight = inner(Rfacet, z_err)
    
    # Then eta is 
    eta_cell = assemble(inner(cell_weight, test)*dx + inner(avg(facet_weight), both(test))*dS + inner(facet_weight, test)*ds )

    # Return abs value array of etas 
    with eta_cell.dat.vec as evec:
        evec.abs()
    return eta_cell

























# Pablo's deleted code:

# class GoalAdaptiveEigensolver(GoalAdaptiveSolverBase):
#     """Solves an eigenvalue problem adaptively to minimise the error in a
#     target eigenvalue.  The goal functional is :math:`J = \\lambda`.

#     At each iteration the solver:

#     1. Solves the primal (and, if non-self-adjoint, dual) eigenproblem at both
#        degree ``p`` and ``p + dual_extra_degree``.
#     2. Estimates the error in the eigenvalue via the dual-weighted residual
#        formula of Larson & Bengzon.
#     3. Computes cell-wise indicators using the same bubble/cone projection as
#        :class:`GoalAdaptiveNonlinearVariationalSolver`.
#     4. Refines the mesh by Dörfler marking.

#     Parameters
#     ----------
#     problem
#         A :class:`~.LinearEigenproblem` on the initial mesh.
#     target
#         Target eigenvalue used by SLEPc for spectral targeting
#         (``eps_target``).
#     tolerance
#         Terminate when ``|eta_h| < tolerance``.
#     goal_adaptive_options
#         Dictionary passed to :class:`GoalAdaptiveEigenOptions`.
#         Key extra entries: ``"self_adjoint"`` (bool) and ``"nev"`` (int).
#     solver_parameters
#         SLEPc solver parameters forwarded to :class:`~.LinearEigensolver`.
#         Do not set ``eps_target`` here; it is set from ``target``.
#     exact_eigenvalue
#         Exact eigenvalue, if known, for computing efficiency indices.
#     """

#     def __init__(self,
#                  problem,
#                  target: float,
#                  tolerance: float,
#                  goal_adaptive_options: dict | None = None,
#                  solver_parameters: dict | None = None,
#                  exact_eigenvalue: float | None = None,
#                  ):
#         super().__init__(tolerance, goal_adaptive_options, exact_goal=exact_eigenvalue)
#         self.problem = problem
#         self.target = target
#         self.sp = solver_parameters or {}

#         # Set up AdaptiveMeshHierarchy
#         mesh = problem.output_space.mesh()
#         mh, level = get_level(mesh)
#         if mh is None:
#             AdaptiveMeshHierarchy(mesh)
#         else:
#             amh = AdaptiveMeshHierarchy(mh[0])
#             for m in mh[1:level+1]:
#                 amh.add_mesh(m)

#         self.atm = AdaptiveTransferManager()
#         self._lam_h = None

#     def _make_options(self, d):
#         return GoalAdaptiveEigenOptions(**d)

#     def solve(self):
#         """Run the adaptive loop and return the eigenvalue and error estimate.

#         Returns
#         -------
#         tuple[float, float]
#             ``(lam_h, error_estimate)`` on the finest mesh.
#         """
#         super().solve()
#         return self._lam_h, self.etah_vec[-1]

#     def solve_and_estimate(self):
#         """Solve the eigenproblem at two polynomial degrees, match eigenfunctions,
#         and compute a global error estimate for the eigenvalue."""
#         opts = self.options
#         problem = self.problem
#         V = problem.output_space
#         self.Ndofs_vec.append(V.dim())
#         self.print(f"Solving eigenproblem (degree: {V.ufl_element().degree()}, dofs: {V.dim()}) ...")

#         sp_target = dict(self.sp)
#         if opts.self_adjoint:
#             sp_target.setdefault("eps_gen_hermitian", None)
#         sp_target["eps_target"] = self.target

#         # Solve at degree p
#         lams, vecs = _solve_eigs(problem, opts.nev, sp_target)
#         self._lam_h = lams[0]
#         self._u_h = vecs[0]
#         self.print(f'{"Computed eigenvalue:":40s}{self._lam_h:15.12f}')

#         # Solve at degree p + dual_extra_degree (enriched primal)
#         high_problem = _reconstruct_eig_degree(problem, opts.dual_extra_degree)
#         self.print(f"Solving enriched eigenproblem (dofs: {high_problem.output_space.dim()}) ...")
#         lams_p, vecs_p = _solve_eigs(high_problem, opts.nev, sp_target)
#         self._lam_p, self._u_p = match_best(self._u_h, vecs_p, lams_p)

#         if opts.self_adjoint:
#             self._z_h = self._u_h
#             self._z_p = self._u_p
#         else:
#             # Adjoint eigenproblem at degree p
#             adj_problem = _make_adjoint_eig_problem(problem)
#             self.print(f"Solving adjoint eigenproblem (dofs: {adj_problem.output_space.dim()}) ...")
#             lamz, zs = _solve_eigs(adj_problem, opts.nev, sp_target)
#             _, self._z_h = match_best(self._u_h, zs, lamz)

#             # Adjoint at degree p + dual_extra_degree
#             adj_high = _reconstruct_eig_degree(adj_problem, opts.dual_extra_degree)
#             self.print(f"Solving enriched adjoint eigenproblem (dofs: {adj_high.output_space.dim()}) ...")
#             lamzp, zsp = _solve_eigs(adj_high, opts.nev, sp_target)
#             _, self._z_p = match_best(self._u_h, zsp, lamzp)

#         # Dual error representative (UFL expression, may span two spaces)
#         self._z_err = self._z_p - self._z_h

#         eta_h, eta = self._estimate_eigenvalue_error()
#         return eta_h, eta

#     def _estimate_eigenvalue_error(self):
#         """Compute global error estimate for the eigenvalue (Larson–Bengzon formula)."""
#         from firedrake.assemble import assemble
#         u_h, u_p = self._u_h, self._u_p
#         z_h, z_p = self._z_h, self._z_p
#         lam_h = self._lam_h
#         A = self.problem._original_A
#         M = self.problem._original_M

#         # Low-order primal representative in the base space
#         phi_h = Function(u_h.function_space())
#         phi_h.interpolate(u_p)
#         e = u_p - phi_h     # primal enrichment error (UFL expression)
#         e_sigma = u_p - u_h

#         if self.options.self_adjoint:
#             sigma_h = 0.5 * float(assemble(inner(e_sigma, e_sigma) * dx))
#             rhs = float(assemble(replace_both_args(A, u_h, e)
#                                  - lam_h * replace_both_args(M, u_h, e)))
#         else:
#             e_adj = z_p - z_h
#             sigma_h = 0.5 * float(assemble(inner(e_sigma, e_adj) * dx))
#             rhs = 0.5 * (
#                 float(assemble(replace_both_args(A, u_h, e_adj)))
#                 - lam_h * float(assemble(replace_both_args(M, u_h, e_adj)))
#                 + float(assemble(replace_both_args(adjoint(A), z_h, e)))
#                 - lam_h * float(assemble(replace_both_args(M, z_h, e)))
#             )

#         denom = 1.0 - sigma_h
#         eta_h = abs(rhs / denom) if abs(denom) > 1e-14 else float("nan")
#         self.etah_vec.append(eta_h)
#         self.print(f'{"Predicted error:":40s}{eta_h: 15.12e}')

#         eta = None
#         if self.goal_exact is not None:
#             eta = abs(self.goal_exact - lam_h)
#             self.eta_vec.append(eta)
#             self.print(f'{"Exact eigenvalue:":40s}{self.goal_exact: 15.12f}')
#             self.print(f'{"True error:":40s}{eta: 15.12e}')

#         return eta_h, eta

#     def compute_error_indicators(self):
#         """Compute cell/facet residual indicators for the eigenvalue problem."""
#         A = self.problem._original_A
#         M = self.problem._original_M
#         u_h = self._u_h
#         lam_h = self._lam_h
#         z_err = self._z_err
#         # Residual linear form: A(u_h, v) - lam_h * M(u_h, v)
#         _, trial = A.arguments()
#         F_eig = replace(A, {trial: u_h}) - lam_h * replace(M, {trial: u_h})
#         return _compute_residual_indicators(F_eig, z_err, self.options)

#     def refine_problem(self, markers):
#         """Refine the mesh and reconstruct the :class:`~.LinearEigenproblem`."""
#         mesh = markers.function_space().mesh()
#         new_mesh = mesh.refine_marked_elements(markers)
#         amh, _ = get_level(mesh)
#         amh.add_mesh(new_mesh)
#         coef_map = {}
#         self.problem = refine(self.problem, refine, coefficient_mapping=coef_map)

#     def write_solution(self, it):
#         ws = self.options.write_solution
#         if ws is False:
#             return
#         elif ws is True:
#             should_write = True
#         elif isinstance(ws, Integral):
#             should_write = (it % ws == 0)
#         else:
#             raise ValueError(f"write_solution must be False, True, or a positive integer, got {ws!r}")
#         if should_write:
#             output_dir = self.options.output_dir
#             run_name = self.options.run_name
#             self.print("Writing (primal) eigenfunction ...")
#             VTKFile(f"{output_dir}/{run_name}/{run_name}_eigenfunction_{it}.pvd"
#                     ).write(*self._u_h.subfunctions)


# # ---------------------------------------------------------------------------
# # Module-level helpers
# # ---------------------------------------------------------------------------

# def residual(F, test):
#     """Replace the test function argument of a linear Form."""
#     v, = F.arguments()
#     return replace(F, {v: test})


# def both(u):
#     """Add u on both sides of a facet."""
#     return u("+") + u("-")


# def as_mixed(exprs):
#     """Flatten a list of ufl.Expr objects into a vector."""
#     return as_vector([e[idx] for e in exprs for idx in np.ndindex(e.ufl_shape)])


# def reconstruct_degree(V, degree):
#     """Reconstruct a FunctionSpace with a different polynomial degree."""
#     return V.reconstruct(element=PMGPC.reconstruct_degree(V.ufl_element(), degree))


# def reconstruct_bcs(bcs, V):
#     """Reconstruct a list of BCs on a different FunctionSpace."""
#     return [bc.reconstruct(V=V, indices=bc._indices) for bc in bcs]


# def replace_both_args(bilinear_form, trial_coeff, test_coeff):
#     """Substitute both arguments of a bilinear form with coefficients.

#     Parameters
#     ----------
#     bilinear_form
#         A UFL bilinear form with two arguments.
#     trial_coeff
#         Coefficient to substitute for the trial (second) argument.
#     test_coeff
#         Coefficient to substitute for the test (first) argument.

#     Returns
#     -------
#     ufl.Form
#         A 0-form (scalar integral).
#     """
#     test, trial = bilinear_form.arguments()
#     return replace(bilinear_form, {test: test_coeff, trial: trial_coeff})


# def l2_normalize(f):
#     """L2-normalise a :class:`~.Function` in place and return it."""
#     from firedrake.assemble import assemble
#     nrm = float(assemble(inner(f, f) * dx)) ** 0.5
#     if nrm > 0:
#         f.assign(f / nrm)
#     return f


# def match_best(target, candidates, lambdas=None):
#     """Return the candidate best correlated with ``target`` in L2.

#     Parameters
#     ----------
#     target
#         Reference :class:`~.Function`.
#     candidates
#         List of candidate :class:`~.Function` objects.
#     lambdas
#         List of associated eigenvalues (optional).

#     Returns
#     -------
#     tuple
#         ``(lam, aligned)`` where ``lam`` is the eigenvalue of the best match
#         (or its index if ``lambdas`` is ``None``) and ``aligned`` is a
#         phase/sign-aligned copy of the best candidate in the same space.
#     """
#     from firedrake.assemble import assemble
#     nt = float(assemble(inner(target, target) * dx)) ** 0.5
#     scores = []
#     for i, w in enumerate(candidates):
#         nw = float(assemble(inner(w, w) * dx)) ** 0.5
#         if nw == 0.0:
#             continue
#         c = complex(assemble(inner(target, w) * dx))
#         scores.append((abs(c) / (nt * nw), i, c))

#     if not scores:
#         raise RuntimeError("No nonzero candidate found in match_best.")

#     scores.sort(key=lambda t: t[0], reverse=True)
#     _, best_i, best_c = scores[0]

#     aligned = candidates[best_i].copy(deepcopy=True)
#     if best_c != 0:
#         phase = best_c.conjugate() / abs(best_c)
#         aligned.assign(phase * aligned)

#     lam = lambdas[best_i] if lambdas is not None else best_i
#     return lam, aligned


# def _solve_eigs(problem, nev, solver_parameters):
#     """Solve a :class:`~.LinearEigenproblem` and return L2-normalised eigenpairs.

#     Parameters
#     ----------
#     problem
#         :class:`~.LinearEigenproblem` to solve.
#     nev
#         Number of eigenvalue/eigenvector pairs to request.
#     solver_parameters
#         Dictionary of SLEPc solver parameters.

#     Returns
#     -------
#     tuple[list, list]
#         ``(eigenvalues, eigenfunctions)`` — both lists have length
#         ``min(nconv, nev)``.
#     """
#     from firedrake import LinearEigensolver
#     es = LinearEigensolver(problem, n_evals=nev, solver_parameters=solver_parameters)
#     nconv = es.solve()
#     lams, vecs = [], []
#     for i in range(min(nconv, nev)):
#         lams.append(es.eigenvalue(i))
#         vr, _ = es.eigenfunction(i)
#         vecs.append(l2_normalize(vr))
#     return lams, vecs


# def _reconstruct_eig_degree(problem, extra_degree):
#     """Return a new :class:`~.LinearEigenproblem` on a space of degree+extra_degree.

#     Uses the original unrestricted forms stored on ``problem._original_A/M/bcs``
#     to avoid double-restricting on the reconstructed problem.
#     """
#     from firedrake import LinearEigenproblem
#     A = problem._original_A
#     M = problem._original_M
#     bcs = problem._original_bcs
#     v, u = A.arguments()
#     V = u.function_space()
#     high_degree = V.ufl_element().degree() + extra_degree
#     V_high = reconstruct_degree(V, high_degree)
#     u_high = TrialFunction(V_high)
#     v_high = TestFunction(V_high)
#     A_high = replace(A, {v: v_high, u: u_high})
#     M_high = replace(M, {M.arguments()[0]: v_high, M.arguments()[1]: u_high})
#     bcs_high = [bc.reconstruct(V=V_high, indices=bc._indices) for bc in bcs]
#     return LinearEigenproblem(A_high, M_high, bcs=bcs_high,
#                               bc_shift=problem.bc_shift, restrict=problem.restrict)


# def _make_adjoint_eig_problem(problem):
#     """Return the adjoint of a :class:`~.LinearEigenproblem`.

#     For a self-adjoint problem this is identical to the primal.
#     """
#     from firedrake import LinearEigenproblem
#     A_adj = adjoint(problem._original_A)
#     return LinearEigenproblem(A_adj, problem._original_M,
#                               bcs=problem._original_bcs,
#                               bc_shift=problem.bc_shift, restrict=problem.restrict)


# def _compute_residual_indicators(F, z_err, options):
#     """Compute DG0 cell-wise error indicators via bubble/cone projections.

#     This implements the primal-residual indicator

#     .. math::

#         \\eta_K = \\int_K R_{\\text{cell}} \\cdot z_{\\text{err}}\\,\\mathrm{d}x
#                   + \\int_{\\partial K} R_{\\text{facet}} \\cdot z_{\\text{err}}\\,\\mathrm{d}s

#     for the residual form ``F`` against the dual error ``z_err``.

#     Parameters
#     ----------
#     F
#         Residual as a linear form (test function as its sole argument).
#     z_err
#         Dual error representative ``z_p - z_h`` (UFL expression or Function).
#     options
#         :class:`GoalAdaptiveOptions` (or subclass) instance for degree
#         parameters and solver parameters.

#     Returns
#     -------
#     Function
#         DG0 Function of absolute-value cell indicators.
#     """
#     from firedrake.assemble import assemble

#     v, = F.arguments()
#     V = v.function_space()
#     mesh = V.mesh().unique()
#     dim = mesh.topological_dimension
#     cell = mesh.ufl_cell()
#     variant = "integral"
#     degree = V.ufl_element().degree()
#     cell_residual_degree = degree + options.cell_residual_extra_degree
#     facet_residual_degree = degree + options.facet_residual_extra_degree

#     B = FunctionSpace(mesh, "B", dim+1, variant=variant)
#     bubbles = Function(B).assign(1)

#     if V.value_shape == ():
#         DG = FunctionSpace(mesh, "DG", cell_residual_degree, variant=variant)
#     else:
#         DG = TensorFunctionSpace(mesh, "DG", cell_residual_degree, variant=variant, shape=V.value_shape)
#     uc = TrialFunction(DG)
#     vc = TestFunction(DG)
#     ac = inner(uc, bubbles*vc)*dx
#     Lc = residual(F, bubbles*vc)
#     Rcell = Function(DG)
#     solve(ac == Lc, Rcell, solver_parameters=options.sp_cell)

#     FB = FunctionSpace(mesh, "FB", dim, variant=variant)
#     cones = Function(FB).assign(1)
#     el = BrokenElement(FiniteElement("FB", cell=cell, degree=facet_residual_degree+dim, variant=variant))
#     if V.value_shape == ():
#         Q = FunctionSpace(mesh, el)
#     else:
#         Q = TensorFunctionSpace(mesh, el, shape=V.value_shape)
#     Qtest = TestFunction(Q)
#     Qtrial = TrialFunction(Q)
#     Lf = residual(F, Qtest) - inner(Rcell, Qtest)*dx
#     af = both(inner(Qtrial/cones, Qtest))*dS + inner(Qtrial/cones, Qtest)*ds
#     Rhat = Function(Q)
#     solve(af == Lf, Rhat, solver_parameters=options.sp_facet)
#     Rfacet = Rhat/cones

#     DG0 = FunctionSpace(mesh, "DG", degree=0)
#     test = TestFunction(DG0)
#     eta_cell = assemble(
#         inner(inner(Rcell, z_err), test)*dx
#         + inner(avg(inner(Rfacet, z_err)), both(test))*dS
#         + inner(inner(Rfacet, z_err), test)*ds
#     )
#     with eta_cell.dat.vec as evec:
#         evec.abs()
#     return eta_cell

