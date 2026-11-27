# My eigensolver, based on where it ended the summer.
# Now I want to update it for recent Firedrake changes 
# and I want make it more like Pablo's goal adaptive PR.



import numbers
import numpy as np
from dataclasses import dataclass
from typing import ClassVar

from petsctools import OptionsManager

from firedrake import warning

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
from firedrake import Constant

# Keep temporarily; its old reconstruct_degree calls need updating next.
from firedrake.preconditioners.pmg import PMGPC
from firedrake.mg.utils import get_level
from firedrake.mg.ufl_utils import refine

# try:
#     from firedrake.mg.ufl_utils import refine
# except ImportError:
#     def refine(*args, **kwargs):
#         raise NotImplementedError("mesh-adaptation path requires Maggie's Firedrake branch")
# from firedrake.utils import RealType
# from firedrake.mg.adaptive_hierarchy import MeshHierarchy
# from firedrake.mg.adaptive_transfer_manager import TransferManager
# Current implementations, with aliases matching your existing code.

from firedrake import MeshHierarchy, TransferManager

from finat.ufl import BrokenElement, FiniteElement
from ufl import avg, dx, ds, dS, inner, replace
import ufl



# Earlier stuff but add in eigensolver
__all__ = ["GoalAdaptiveSolverBase",
           "SteadyGoalAdaptiveSolver",
           # "GoalAdaptiveNonlinearVariationalSolver",
           "vtk_output_callback",
           "GoalAdaptiveFoldedEigensolver"]


# Pablo's new initialization

# Notes on how it compares to the old:
# Adds in a tolerance 
# max_iterations (old) now max_it (new)
# no longer given an output_dir or run_name for saving output to a specific location (maybe this is elsewhere)
# Old code claims that use_adjoint_residual, primal_old_method, dual_old_method are not used by the eigensolver class
# New code uses @dataclass(frozen=True) to generate the constructor. The old code had a constructor (init)
# Solver parameters for cell and facet solves are the same choices


# Maggie change MT 2026 
# One options class because I am no longer using the nonlinear variational solver. 
# I will just use the base class and the eigensolver class. 

@dataclass(frozen=True)
class GoalAdaptiveEigenOptions:

    """Options for goal-adaptive eigensolver.

    Parameters
    ----------
    max_it
        Maximum number of SOLVE–ESTIMATE–MARK–REFINE cycles.
    max_dofs
        Maximum number of dofs we will allows the meshes to have. This will 
        avoid long adaptive loops.
    dorfler_alpha
        Threshold parameter for Dörfler (bulk) marking: cells whose local error
        indicator exceeds ``dorfler_alpha * max_indicator`` are marked for
        refinement.  Must lie in ``(0, 1]``.  Larger values mark fewer cells and
        produce more targeted (but potentially slower-converging) refinement.
        Defaults to ``0.5``.
    primal_extra_degree
        Extra polynomial degree(s) used when solving the primal problem in an
        enriched space (only relevant when ``use_adjoint_residual=True``).
        The enriched-space degree is ``degree + primal_extra_degree``.
        Defaults to ``1``.
    cell_residual_extra_degree
        Extra polynomial degree(s) for the DG space used to represent cell
        residuals during error indicator computation.  The DG space degree is
        ``degree + cell_residual_extra_degree``.  Defaults to ``(1,)``.
    facet_residual_extra_degree
        Extra polynomial degree(s) for the broken facet-bubble space used to
        represent facet residuals during error indicator computation.
        The space degree is ``degree + facet_residual_extra_degree``.
        Defaults to ``(1,)``.
    verbose
        If ``True`` (the default), print progress information at each
        iteration via :func:`PETSc.Sys.Print`.
    nev
        Number of eigenpairs requested from each eigensolve. Defaults to ``5``.
    sp_cell
        Solver parameters for cell residual local problems.
    sp_facet
        Solver parameters for facet residual local problems.
    """

    max_it: int = 10
    max_dofs: int | None = None
    dorfler_alpha: float = 0.5
    verbose: bool = True
    primal_extra_degree: tuple[int, ...] = (1,)
    cell_residual_extra_degree: tuple[int, ...] = (1,)
    facet_residual_extra_degree: tuple[int, ...] = (1,)
    nev: int = 5 # default number of eigenpairs requested from each eigensolve. 

    # mult_tol: float = 0.01 # Maggie addition - tolerance for computing cluster size.

    # Solver parameters for cell/facet bubble projections
    # These are the same choices as in the old code (pre deletion of eigensolver)
    # I am not sure of the motivation behind these choices, but I will keep them the same for now.
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


    

class GoalAdaptiveSolverBase:
    """Base class for goal-adaptive eigensolvers.

    Owns the SOLVE→ESTIMATE→MARK→REFINE loop and the Dörfler(Max) marking
    strategy.  Subclasses must implement :meth:`solve_and_estimate`,
    :meth:`compute_error_indicators`, and :meth:`refine_problem`.

    Parameters
    ----------
    goal_adaptive_eigenoptions
        A ``GoalAdaptiveEigenOptions`` instance, or a dict of keyword
        arguments to construct one.  Defaults to ``{}``.
    exact_goal
        Exact value of the goal functional (or eigenvalue).  Optional; used
        by subclasses to compute efficiency indices.
    """

  

    def __init__(self, base_mesh,
                 goal_adaptive_eigenoptions: "GoalAdaptiveEigenOptions | dict | None" = None,
                 exact_goal=None,
                 post_iteration_callback=None,
                 ):


        if goal_adaptive_eigenoptions is None:
            goal_adaptive_eigenoptions = {}
        self.options = self._make_options(goal_adaptive_eigenoptions)
        self.goal_exact = exact_goal
        self.post_iteration_callback = post_iteration_callback

        self.Ndofs_vec: list[int] = []
        self.eta_vec: list[float] = []
        self.etah_vec: list[float] = []

        # Set up an MeshHierarchy for every mesh of the problem.
        # For a MeshSequenceGeometry, iterating over it yields the component meshes;
        # we also need a separate AMH for the sequence geometry itself.
        # meshes = {*base_mesh, base_mesh}  # component meshes + the sequence geometry (same object for a regular mesh)
        # for mesh in meshes:
        #     mh, level = get_level(mesh)
        #     if mh is None:
        #         amh = MeshHierarchy(mesh)
        #     else:
        #         amh = MeshHierarchy(mh[0])
        #         for m in mh[1:level+1]:
        #             amh.add_mesh(m)

        # amh, _ = get_level(base_mesh)
        # self.amh = amh
        # self.base_levels = len(amh)
        # self.atm = TransferManager()

        # Set up each distinct physical mesh once.
        for mesh in dict.fromkeys(base_mesh):
            mh, level = get_level(mesh)

            if mh is None:
                mh = MeshHierarchy(mesh)
                level = 0

            if mh.refinements_per_level != 1:
                raise ValueError(
                    "Adaptive refinement requires refinements_per_level=1."
                )

            if level != len(mh) - 1:
                raise ValueError(
                    "Start adaptation from the finest mesh in its hierarchy."
                )

        # Derive the sequence hierarchy from its component hierarchies.
        if isinstance(base_mesh, MeshSequenceGeometry):
            base_mesh.set_hierarchy()

        self.amh, _ = get_level(base_mesh)
        self.base_levels = len(self.amh)
        self.atm = TransferManager()


    def _make_options(self, d):
        """Construct a ``GoalAdaptiveEigenOptions`` from a dict or pass through as-is.  Override in subclasses."""
        if isinstance(d, GoalAdaptiveEigenOptions):
            return d
        return GoalAdaptiveEigenOptions(**d["goal_adaptive"])


    
    # def get_error_estimate(self):
    #     return self.etah_vec[-1]
    def get_error_estimate(self):
        """Return the most recently stored error-estimate magnitude."""
        if not self.etah_vec:
            raise RuntimeError("No error estimate has been computed yet.")
        return self.etah_vec[-1]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------


    def solve(self):
        """Run the adaptive SOLVE→ESTIMATE→MARK→REFINE loop to convergence."""
        # Runs the standard loop for adaptivity where step() executes the steps of a given iteration
        for it in range(self.options.max_it):
            try:
                self.step(it=it)
            except StopIteration:
                break


    # # Maggie change - I seperated refinement criteria so I can overwrite it for the folded operator
    # def should_refine(self, it, eta_h, eta):
    #     """ Return true when refinement is needed """
    #     return abs(eta_h) >= self.options.tolerance
    def should_refine(self, it, eta_h, eta):
        """Return whether refinement is needed; implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement their refinement criterion.")


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


        self.print(BLUE % f"---------------------------- [MESH LEVEL {it}] ----------------------------")


        # 1. SOLVE + ESTIMATE
        self.eta_h, self.eta = self.solve_and_estimate() # defined in the subclass
        self.post_iteration(it) # user given actions
        
        
        # Maggie change - stopping criteria moved outside
        if not self.should_refine(it, self.eta_h, self.eta): # tolerance stopping criteria 
            self.print("We have decided not to adapt - we are done!.")
            self.inner_converged = True
            self.termination_reason = "inner_target_reached"
            raise StopIteration

        elif it == self.options.max_it - 1: # max iter stopping criteria
            self.print(f"Maximum iteration ({self.options.max_it}) reached. Exiting.")
            self.inner_converged = False
            self.termination_reason = "max_it_reached"
            raise StopIteration
        
        # 2. MARK
        self.print("Computing local refinement indicators eta_K ...")
        eta_cell = self.compute_error_indicators() # defined in the subclass - local indicators
        self.compute_efficiency_indices(eta_cell, self.eta_h, self.eta) # effectivity indices
        markers = self.set_adaptive_cell_markers(eta_cell) # call the marking routine - in base class
        marker_values = markers.dat.data_ro
        max_imag = float(np.max(np.abs(marker_values.imag))) # check size of imaginary part
        self.print(f"Maximum imaginary part of cell markers is: {max_imag:.3e}")
        if max_imag > self.imag_tol:
            raise ValueError(f"Cell markers have a nontrivial nonzero imaginary part: {max_imag:.3e}")
        else:
            self.print("Thus, do not worry about the throwing out imaginary part warning to follow:")
  


        # 3. REFINE
        # ATTEMPT REFINEMENT
        self.print("Preparing a candidate refined mesh ...")

        if not self.refine_problem(markers): # attempt the transfer
            self.inner_converged = False
            self.termination_reason = "max_dofs_reached"
            raise StopIteration

        # The candidate was accepted: transfer eigenfunctions as initial guesses.
        self.print("Transferring problem to new mesh ...")
        self.post_refinement()

        # self.print("Transferring problem to new mesh ...")
        # self.refine_problem(markers) # refine the problem (base class)

        # # CHECK MAX DOF:
        # new_dof = self.problem.output_space.dim()
        # if (self.options.max_dofs is not None and new_dof >= self.options.max_dofs):
        #     self.print(f"Maximum DOF limit ({self.options.max_dofs}) reached on refined mesh: "f"{new_dof} DOFs. Exiting.")
        #     self.inner_converged = False
        #     self.termination_reason = "max_dofs_reached"
        #     raise StopIteration

        # Allow subclasses to transfer solutions/eigenfunctions.
        # self.post_refinement()

    # Pretty much, after each iteration, this will run any user-supplied code.
    # For example, the user could want to save the solution, print output, save plots, etc
    def post_iteration(self, it : int):
        """Hook called after SOLVE+ESTIMATE, before convergence check.
           Invokes the user-supplied ``post_iteration_callback``, if any."""
        if self.post_iteration_callback is not None:
            self.post_iteration_callback(self, it)
    
    # Allow the subclasss to extend found solutions on coarser mesh to adapted mesh as initial guess
    # def post_refinement(self):
    #     """Hook called after SOLVE+ESTIMATE+MARK+REFINE. 
    #     Invokes the subclass's ``post_refinement``, if any."""
    #     if self.post_refinement is not None:
    #         self.post_refinement(self)

    def post_refinement(self):
        """Hook for subclasses to transfer solutions after accepted refinement."""
        pass


    # ------------------------------------------------------------------
    # Common machinery (mark + refine + efficiency)
    # ------------------------------------------------------------------


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
        # note - this is not quite Dorfler marking
        # For Dorfler marking we need to implement a parallel sort

        # if not 0 < self.options.dorfler_alpha <= 1:
        #     raise ValueError("dorfler_alpha must lie in (0, 1]")
        

        # markers = Function(eta_cell.function_space())
        
        # for m, e in zip(markers.subfunctions, eta_cell.subfunctions):
        #     with e.dat.vec_ro as evec:
        #         _, emax = evec.max()
        #     threshold = self.options.dorfler_alpha * emax
        #     m.dat.data_wo[e.dat.data_ro > threshold] = 1
        # return markers

        fraction = self.options.dorfler_alpha

        if not 0 < fraction <= 1:
            raise ValueError("dorfler_alpha must lie in (0, 1]")

        # Global maximum across MPI processes.
        with eta_cell.dat.vec_ro as vec:
            _, eta_max = vec.max()

        markers = Function(eta_cell.function_space())

        if eta_max > 0:
            markers.dat.data_wo[:] = (eta_cell.dat.data_ro.real >= fraction * eta_max)

        return markers

        
        


    
    # In the old code, this was implemented in subclasses not in this base class
    # Thus, every solver had to have its own refinement method
    # Thus, the new version of the eigensolver should inherit this 
    # The question is whether refine(self.problem, ...) will know how to handle an eigenproblem - I think it does (bult in and in old code)

    # def refine_problem(self, markers, coef_map=None):
    #     """Refine the mesh and reconstruct the problem on the new mesh.

    #     Parameters
    #     ----------
    #     markers
    #         DG0 Function with value 1 on cells to refine.
    #     """

    #     # Loop over markers 
    #     for marker in markers.subfunctions:
    #         mesh = marker.function_space().mesh()
    #         # refine marked elements
    #         new_mesh = mesh.refine_marked_elements(marker)
    #         # add it to its repsective hierarchy
    #         amh, _ = get_level(mesh)
    #         amh.add_mesh(new_mesh)

    #     # Reconstruct MeshSequence with the refined meshes
    #     # Reconstruct the problem on the refined mesh
    #     mesh = self.amh[-1]
    #     if len(mesh) > 1:
    #         new_mesh = type(mesh)([get_level(m)[0][-1] for m in mesh])
    #         self.amh.add_mesh(new_mesh)
    #     if coef_map is None:
    #         coef_map = {}
    #     self.problem = refine(self.problem, refine, coefficient_mapping=coef_map)

    
    
    def refine_problem(self, markers, coef_map=None):
        """Return True after refinement, or False if the DOF limit rejects it.
           Assumes each component shares the same mesh."""
    
        V = self.problem.output_space
        old_domain = V.mesh()
        old_mesh = old_domain.unique()

        # 1. Create a candidate mesh without extending the hierarchy.
        new_mesh = old_mesh.refine_marked_elements(markers)

        # Preserve the domain structure for mixed function spaces.
        if isinstance(old_domain, MeshSequenceGeometry):
            candidate_domain = MeshSequenceGeometry(
                [new_mesh for _ in old_domain],
                set_hierarchy=False,
            )
        else:
            candidate_domain = new_mesh

        # 2. Construct a space only to count its DOFs.
        # No eigenproblem is solved here.
        candidate_space = V.reconstruct(
            mesh=candidate_domain,
            name="dof_check_space",
        )
        candidate_dofs = candidate_space.dim()

        # 3. Reject the candidate before changing the active problem/hierarchy.
        limit = self.options.max_dofs
        if limit is not None and candidate_dofs >= limit:
            self.print(
                f"Candidate mesh has {candidate_dofs} DOFs; "
                f"limit is {limit}. Keeping the last solved problem."
            )
            return False

        # 4. Accept the candidate and extend the hierarchy.
        mh, _ = get_level(old_mesh)
        mh.add_mesh(new_mesh)

        if isinstance(old_domain, MeshSequenceGeometry):
            old_domain.set_hierarchy()

        self.amh, _ = get_level(old_domain)

        # 5. Reconstruct the problem on the accepted mesh.
        if coef_map is None:
            coef_map = {}

        self.problem = refine(
            self.problem,
            refine,
            coefficient_mapping=coef_map,
        )
        return True



    def compute_efficiency_indices(self, eta_cell, eta_h, eta):
        """Hook called after marking.  Default no-op; override in subclasses."""
        pass

    def print(self, *args, **kwargs):
        if self.options.verbose:
            PETSc.Sys.Print(*args, **kwargs)



    # ------------------------------------------------------------------
    # Abstract interface for subclasses
    # ------------------------------------------------------------------


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
        Ratio ``sum(eta_K) / abs(eta_h)`` when no exact goal is available.
    """

    def __init__(self, base_mesh, goal_adaptive_eigenoptions=None, exact_goal=None, post_iteration_callback=None):
        super().__init__(base_mesh, goal_adaptive_eigenoptions, exact_goal, post_iteration_callback)
        self.eta_cell_sum_vec: list[float] = []
        self.eff1_vec: list[float] = []
        self.eff2_vec: list[float] = []
        self.eff3_vec: list[float] = []

    def compute_efficiency_indices(self, eta_cell, eta_h, eta):
        """Compute and log effectivity and localisation efficiency indices."""
        with eta_cell.dat.vec_ro as evec:
            eta_cell_total = abs(evec.sum())

        self.eta_cell_sum_vec.append(eta_cell_total)
        self.print(BLUE % f'{"Sum of refinement indicators:":40s}{eta_cell_total: 15.12e}')

        if eta is not None:
            eff1 = abs(eta_h / eta)
            eff2 = abs(eta_cell_total / eta)
            self.eff1_vec.append(eff1)
            self.eff2_vec.append(eff2)
            self.print(BLUE % f'{"Effectivity index:":40s}{eff1: 15.12f}')
            self.print(BLUE % f'{"Localisation efficiency:":40s}{eff2: 15.12f}')
        else:
            eff3 = eta_cell_total / abs(eta_h)
            self.eff3_vec.append(eff3)
            self.print(BLUE % f'{"(Approximate) Localisation efficiency, (sum of indicators / |eta_h|):":40s}{eff3: 15.12f}')




# My adaptive eigensolver, based on the old version but placed in Pablo's current code


class GoalAdaptiveFoldedEigenSolver(SteadyGoalAdaptiveSolver, OptionsManager):
    """Solves an eigenvalue problem adaptively to minimise the error in a
    target eigenvalue of a folded operator.  The goal functional is :math:`J = \\lambda`.

    At each iteration the solver:

    1. Solves the primal (and, if non-self-adjoint, dual) eigenproblem at both
       degree ``p`` and ``p + primal/dual_extra_degree``.
    2. Estimates the error in the eigenvalue via the dual-weighted residual
       formula of Larson & Bengzon.
    3. Computes cell-wise indicators using the same bubble/cone projection as
       :class:`GoalAdaptiveNonlinearVariationalSolver`.
    4. Refines the mesh using marking. Note that it is maximum and not Dorfler marking.

    Parameters
    ----------
    problem
        A :class:`~.LinearEigenproblem` on the initial mesh.
    m_form 
        The symmetric semi-definite sesquilinear form m(,) corresponding to
        M (usually the L 2 scalar product.
    initial_space
        Any initial guesses for the eigensolve on the initial mesh.
    target
        Target eigenvalue used by SLEPc for spectral targeting
        (``eps_target``).
    epsilon
        Desired epsilon-pseudospectral contour
    diam_cond
        The diameter condition for adaptivity
    diameter
        The diamater of the current cell in z grid
    mult_tol
        The tolerance used to compute a cluster size.
    imag_tol
        Tolerance for throwing out imaginary part of eigenvalue and error
    can_interp
        Can the finite element space use interpolation?
    solver_parameters
        SLEPc solver parameters forwarded to :class:`~.LinearEigensolver`.
        Do not set ``eps_target`` here; it is set from ``target``.
    exact_eigenvalue
        Exact eigenvalue, if known, for computing efficiency indices.
    """

    # 3. Maggie change - New init (made by chat to match the nonlinar solver approach). Also add in epsilon for pseudospectrum
    _GOAL_PREFIX = "goal_adaptive_"

    def __init__(
        self,
        problem,
        *,
        m_form = None, # m(,) form
        initial_space: tuple = (), # initial guess space, default empty
        target: float = 0, # set zero for folded operator
        epsilon: float = 1.0e-2, # default pseudospectral contour is 0.01
        diam_cond: float = 0.0, # The cell diameter condition of interest (for adaptivity)
        diameter: float = 0.0, # the cell diameter
        mult_tol: float = 1e-2, # default tol for cluster size is 0.01
        imag_tol: float = 1.0e-12, # defualt tol for throwing out imgainary part is 1e-12
        can_interp: bool = True,
        solver_parameters: dict | None = None,
        options_prefix: str | None = None,
        exact_eigenvalue: float | complex | None = None,
        post_iteration_callback = None ,):
        
        if solver_parameters is None:
            solver_parameters = {}

        if options_prefix is None:
            options_prefix = ""

        # The new adaptive base creates the mesh hierarchy, transfer manager,
        # history vectors, tolerance options and effectivity vectors.
        base_mesh = problem.output_space.mesh()

        SteadyGoalAdaptiveSolver.__init__(self, base_mesh, solver_parameters, exact_goal=exact_eigenvalue, post_iteration_callback=post_iteration_callback,)

        # Handles the PETSc/SLEPc parameters and options prefix.
        OptionsManager.__init__(self, solver_parameters, options_prefix,)

        # Eigenproblem data.
        self.problem = problem
        self.m_form = m_form
        self.initial_space = initial_space
        self.target = target
        self.epsilon = epsilon
        self.diam_cond = diam_cond
        self.diameter = diameter
        self.mult_tol = mult_tol
        self.imag_tol = imag_tol
        self.can_interp = can_interp
        self._lam_h = None
        self._u_h = None
        self._lam_p = None
        self._u_p = None
        self._z_h = None
        self._z_p = None
        self._z_err = None
        self.exact_eigenvalue = exact_eigenvalue
        # 4. Maggie Change - extra for Matt's algorithm 
        self.signed_error = None

    # 5. Maggie Change - new make options (created with chat)
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


    # 6. Maggie change - user can retrieve eigenpair
    def get_eigenpair(self):
        """Return the most recently computed low-order eigenpair to the user."""
        return self._lam_h, self._u_h


    # 7. Maggie change - instead of current solution in nonlinear variational problem, return current eigenpair. Meant for inside class not user.
    def _current_eigenpair(self):
        """Return the most recently computed low-order eigenpair. Meant for class use"""
        if self._lam_h is None or self._u_h is None:
            raise RuntimeError("No eigenpair has been computed yet.")
        return self._lam_h, self._u_h


    # 8. Maggie change - Stopping criteria for folded operators
    def should_refine(self, it, eta_h, eta):
        """ The stopping criteria we want the base to use to decide when to refine """

        lam_h = self._lam_h  
        lam_p = self._lam_p  
        diam_cond = self.diam_cond
        signed_err = self.signed_error

        # Compute and Correct Matt's phi value
        self.matts_phi = np.sqrt(lam_h)
        self.enriched_phi = np.sqrt(lam_p)    # Compute the enriched phi as well for comparison 
        corrected = lam_h + signed_err
        if corrected >= 0:
            self.corrected_phi = np.sqrt(corrected)
        else:
            self.corrected_phi = float("nan")
        
        self.print(BLUE % f'Eigenvalue before correction is {self._lam_h}')
        self.print(BLUE % f'Eigenvalue after correction is {corrected}')
        self.print(BLUE % f'Matts phi value is {self.matts_phi}')
        self.print(BLUE % f'The (error) corrected phi value is {self.corrected_phi}')

        # Check if L2 alignement failed 
        if self.check_up:
            self.print(RED % f'The L2 aligned enriched soln was very small (in norm) - red flag!!')
            self.print(RED % f'As a result, we will adapt the eigensolve')
            return True 

        # Check the diameter condition
        if eta_h > diam_cond:
            self.print(BLUE % f'The error estimate, {eta_h}, is > the cell diameter condition, {diam_cond}')
            self.print(BLUE % f'As a result, we will adapt the eigensolve')
            return True 
        
        # Otherwise, dont adapt
        else:
            self.print(BLUE % f'No refinement criteria were met')
            self.print(BLUE % f'Double check: eta_h = {eta_h}, diameter condition = {diam_cond}')
            self.print(BLUE % f'No need to do an adaptive solve')
            return False

    def solve(self):
        """Run the adaptive loop and return the final computed eigenpair."""
        super().solve()
        return self._current_eigenpair()

    def step(self, it=None):
        """Run one adaptive eigenvalue iteration and return the current eigenpair."""
        if it is None:
            V = self.problem.output_space
            _, it = get_level(V.mesh())
        super().step(it)
        return self._current_eigenpair()


    # 10. Maggie change - After refinement, use the solutions on the last mesh as initial guesses on the new mesh.
    def post_refinement(self):
        """ Extend current available eigenfunctions to adapted mesh as initial guess."""

        self.print("Updating initial guess space after refinement ...")
        V_new = self.problem.output_space
        initial_space = []

        # MAGGIE CHANGE FOR //
        for old_u in self.vecs:
            new_u = Function(V_new)
            for old_sub, new_sub in zip(old_u.subfunctions,new_u.subfunctions,):
                self.atm.prolong(old_sub, new_sub)
            initial_space.append(new_u)
        self.initial_space = tuple(initial_space) # set the initial space

    ###################################################################
    # START HERE MONDAY
    ###################################################################

    def solve_and_estimate(self):

        """Solve the eigenproblem at two polynomial degrees, match eigenfunctions,
        and compute a global error estimate for the eigenvalue."""


        # Setup
        opts = self.options
        problem = self.problem
        V = problem.output_space
        self.Ndofs_vec.append(V.dim())
        self.print(f"Solving eigenproblem (degree: {V.ufl_element().degree()}, dofs: {V.dim()}) ...")


        # Add in the eigenoptions
        sp_target = dict(self.parameters) # needed change?
        if opts.self_adjoint: # if self adjoint then tell the solver
            sp_target.setdefault("eps_gen_hermitian", None)
        sp_target["eps_target"] = self.target

            
        # 11. Maggie change - now we pass the m_form and guess_space to the solver
        # Solve at degree p
        lams, self.vecs = _solve_eigs(problem, opts.nev, sp_target, self.m_form, guess_space = self.initial_space, can_interp = self.can_interp)
        self._lam_h = lams[0]
        self._u_h = self.vecs[0]
        

        # 12. Maggie change - adding in some checks for a self adjoint problem
        if opts.self_adjoint:
            # Make sure the eigenvalue is real (small imaginary part in complex mode)
            if abs(self._lam_h.imag) > self.imag_tol:
                print(RED % f"Warning: computed folded eigenvalue has nontrivial imaginary part: {self._lam_h}")
                print(RED % f"We will take the real part")
                warning("Self-adjoint eigenproblem has complex eigenvalue with nontrivial imaginary part. We take its real part")
                self._lam_h = self._lam_h.real
            elif (abs(self._lam_h.imag) > 0 and abs(self._lam_h.imag) < self.imag_tol):
                print(RED % f"Warning: computed folded eigenvalue has a small imaginary part: {self._lam_h}")
                print(RED % f"We will take the real part")
                warning("Self-adjoint eigenproblem has complex eigenvalue with small imaginary part. We take its real part")
                self._lam_h = self._lam_h.real

        # 13 Maggie change - I think this shoukd be the extra primal degree? 
        # Pablo had: 
        # Solve at degree p + dual_extra_degree (enriched primal)
        # high_problem = _reconstruct_eig_degree(problem, opts.dual_extra_degree) # Might need to change for mixed spaces

        # Set enriched problem
        high_problem = _reconstruct_eig_degree(problem, opts.primal_extra_degree)


        # 14. Maggie change - Set uh's as initial guesses for the enriched solve
        enriched_guesses = []
        # if can_interp:
        #     for lower_eigfunc in self.vecs:
        #         lower_eigfunc_interp = Function(high_problem.output_space).interpolate(lower_eigfunc)
        #         enriched_guesses.append(lower_eigfunc_interp)
        #     enriched_guesses = tuple(enriched_guesses)
        # else:
        #     for lower_eigfunc in self.vecs:
        #         lower_eigfunc_interp = Function(high_problem.output_space).project(lower_eigfunc)
        #         enriched_guesses.append(lower_eigfunc_interp)
        #     enriched_guesses = tuple(enriched_guesses)

        enriched_guesses = []

        for lower_eigfunc in self.vecs:

            enriched_guess = Function(high_problem.output_space)

            if self.can_interp:
                enriched_guess.interpolate(lower_eigfunc)
            else:
                enriched_guess.project(lower_eigfunc)

            enriched_guesses.append(enriched_guess)

        enriched_guesses = tuple(enriched_guesses)



        # Solve
        self.print(f"Solving enriched eigenproblem (dofs: {high_problem.output_space.dim()}) ...")
        lams_p, self.vecs_p = _solve_eigs(high_problem, opts.nev, sp_target, self.m_form, guess_space = enriched_guesses, can_interp = self.can_interp)
        self._lam_p = lams_p[0]

        # Prints
        self.print(BLUE % f'Lower-degree eigenvalues {lams}')
        self.print(BLUE % f'Enriched eigenvalues: {lams_p}')
        self.print(BLUE % f'{"Computed eigenvalue of interest is:":40s}{self._lam_h:15.12f}')

        
        # 15. Maggie Change - checking cluster size for each solve and taking enriched multiplicity (should be better)

        # Check multiplicity based on lower degree solve
        mult_lower = 1
        for i in range(1, len(self.vecs)):
            if abs(lams[i] - self._lam_h) <= self.mult_tol:
                mult_lower += 1
        self.print(BLUE % f"Lower degree solve multiplicity: {mult_lower}")
        # check multiplicity based on enriched solve
        self.mult = 1 
        for i in range(1, len(self.vecs_p)):
            if abs(lams_p[i] - lams_p[0]) <= self.mult_tol:
                self.mult += 1
        self.print(BLUE % f"Enriched solve multiplicity: {self.mult}")
        # Choose the multiplicity based on enriched solve (should be better)
        self.enriched_cluster = self.vecs_p[:self.mult] 
        # Compare multiplicities
        if mult_lower != self.mult:
            print(RED % f'Warning - the lower degree and enriched multiplicities dont match')
            warning("Smallest lower- and higher- degree eigenvalues have different cluster sizes")
        
        # 16. Maggie change - L2 or m_form matching??? matching instead of Pablo/Joe's match best (phase/sign alignement)
        self.print("Matching eigenfunctions with L2 projection ...")
        self._u_p = match_best(self._u_h, self.enriched_cluster, self.mult, high_problem.output_space, self.m_form)


        # 17. Maggie change - check the size of the aligned enriched vector - if it seems to be zero vector, adapt.
        check_nrm = m_norm(self._u_p, self.m_form)
        self.check_up = False
        if check_nrm < 1e-12:
            print(RED % f'Since the L2 aligned enriched vector is small (in norm) we may want to adapt!!')
            self.check_up = True


        if opts.self_adjoint: # if self adjoint we dont need to dual problem
            self._z_h = self._u_h
            self._z_p = self._u_p


        # 18. Maggie change - folded problem is self adjoint so we dont need the adjoint/dual/
        # else: # if not need adjoint problem # Maggie never touched this because folded operator is self adjoint
            # # Adjoint eigenproblem at degree p
            # adj_problem = _make_adjoint_eig_problem(problem)
            # self.print(f"Solving adjoint eigenproblem (dofs: {adj_problem.output_space.dim()}) ...")
            # lamz, zs = _solve_eigs(adj_problem, opts.nev, sp_target)
            # _, self._z_h = match_best(self._u_h, zs, lamz)

            # # Adjoint at degree p + dual_extra_degree
            # adj_high = _reconstruct_eig_degree(adj_problem, opts.dual_extra_degree)
            # self.print(f"Solving enriched adjoint eigenproblem (dofs: {adj_high.output_space.dim()}) ...")
            # lamzp, zsp = _solve_eigs(adj_high, opts.nev, sp_target)
            # _, self._z_p = match_best(self._u_h, zsp, lamzp)

        # 19. Maggie change - Making z_err a function so we can split it later. Pablo had it as a ufl
        self._z_err = Function(self._z_p.function_space())
        if self.can_interp:
            self._z_err.interpolate(self._z_p - self._z_h)
        else:
            self._z_err.project(self._z_p - self._z_h)
        # self._z_err = Function(self._z_p.function_space())
        # self._z_err.interpolate(self._z_p - self._z_h) 
        # Dubugging - try something different
        #self._z_err = Function(self._z_p.function_space())
        #Ih_zp = Function(self._z_h.function_space()).interpolate(self._z_p)
        #self._z_err.interpolate(self._z_p - Ih_zp)


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

        # ------------------------------------------------------------
        # DEBUGGING: CHECK DISCRETE EIGENRELATIONS
        # ------------------------------------------------------------

        # raw_up = self.vecs_p[0]

        # raw_cross_res = (
        #     assemble(replace_both_args(A, u_h, raw_up))
        #     - self._lam_p * assemble(replace_both_args(M, u_h, raw_up))
        # )

        # matched_cross_res = (
        #     assemble(replace_both_args(A, u_h, u_p))
        #     - self._lam_p * assemble(replace_both_args(M, u_h, u_p))
        # )

        # low_self_res = (
        #     assemble(replace_both_args(A, u_h, u_h))
        #     - self._lam_h * assemble(replace_both_args(M, u_h, u_h))
        # )

        # print(RED % (
        #     "\nEIGENRELATION DEBUG\n"
        #     f"RAW CROSS EIG RES     = {raw_cross_res}\n"
        #     f"MATCHED CROSS EIG RES = {matched_cross_res}\n"
        #     f"LOW SELF RES          = {low_self_res}\n"
        # ))

        # 20. Maggie change - m_norm in error
        # ISSUE - For mixed spaces and inner product do I need to interpolate e_sigma???!! Interpolation not reliable !!??
        # But needed for mixed function buisness to work??

        # Low-order primal representative in the base space
        # phi_h = Function(u_h.function_space())
        # phi_h.interpolate(u_p) # Will interpolation always work? We may need to project.
        # # if self.can_interp:
        # #     phi_h.interpolate(u_p)
        # # else:
        # #     phi_h.project(u_p)
        # # MAGGIE RECENT CHANG 
        # # e = u_p - phi_h     # primal enrichment error (UFL expression)
        # e = u_p - u_h
        # # e_sigma = u_p - u_h # for the remainder term
        # e_sigma = Function(u_p.function_space()).interpolate(u_p - u_h) 

        # UPDATE - CANNOT ALWAYS INTERPOLATE
        e = u_p - u_h
        if self.can_interp:
            e_sigma = Function(u_p.function_space()).interpolate(u_p - u_h)
        else:
            e_sigma = Function(u_p.function_space()).project(u_p - u_h)

            



        if self.options.self_adjoint:
            #sigma_h = 0.5 * float(assemble(inner(e_sigma, e_sigma) * dx))
            sigma_h = 0.5 * m_norm(e_sigma, self.m_form)**2
            # SHIM 3 (U.Z.): assemble the two pieces separately (identical by
            # linearity).  In the original, the scalar lam_h is baked into the
            # form as a literal, so every new eigenvalue changes the form
            # signature and forces a full TSFC/PyOP2 kernel compilation
            # (~0.75 s) per estimate; split, both forms hit the kernel cache.
            rhs = (assemble(replace_both_args(A, u_h, e))
                   - lam_h * assemble(replace_both_args(M, u_h, e)))

        # 21. Maggie change - folded operator problem is self-adjoint
        # else: # Maggie never touched this because folded operator is self adjoint
        #     e_adj = z_p - z_h
        #     sigma_h = 0.5 * float(assemble(inner(e_sigma, e_adj) * dx))
        #     rhs = 0.5 * (
        #         float(assemble(replace_both_args(A, u_h, e_adj)))
        #         - lam_h * float(assemble(replace_both_args(M, u_h, e_adj)))
        #         + float(assemble(replace_both_args(adjoint(A), z_h, e)))
        #         - lam_h * float(assemble(replace_both_args(M, z_h, e)))
        #     )


        
        # a bunch of checks:
        # m_hh = assemble(replace_both_args(M, u_h, u_h))
        # m_pp = assemble(replace_both_args(M, u_p, u_p))
        # m_hp = assemble(replace_both_args(M, u_h, u_p))
        # self.print(BLUE % f"m(u_h, u_h) = {m_hh}")
        # self.print(BLUE % f"m(u_p, u_p) = {m_pp}")
        # self.print(BLUE % f"m(u_h, u_p) = {m_hp}")
        # self.print(BLUE % f'top of estimate = {rhs}')
        # self.print(BLUE % f'bottom of estimate = {1.0 - sigma_h}')



        # 22. Maggie change - signed error so that we can correct our eigenvalue
        denom = 1.0 - sigma_h
        self.signed_error = rhs/denom if abs(denom) > 1e-14 else float("nan")

        # # Debugging
        # discrete_gap = self._lam_p - self._lam_h
        # gap_defect = self.signed_error - discrete_gap
        # gap_defect_rel = abs(gap_defect) / max(abs(discrete_gap), 1e-14)

        # if gap_defect_rel > 1e-6:
        #     print(RED % (
        #         "\nDWR / EIGENVALUE-GAP CHECK\n"
        #         f"DWR estimate      = {self.signed_error}\n"
        #         f"lambda_p-lambda_h = {discrete_gap}\n"
        #         f"difference        = {gap_defect}\n"
        #         f"relative defect   = {gap_defect_rel:.3e}\n"
        #     ))

        # # Debugging
        # rhs_expected = (self._lam_p - self._lam_h) * m_hp
        # if abs(np.imag(self.signed_error)) > 1e-8:
        #     print(RED % (
        #         "\nCOMPLEX DWR DIAGNOSTIC\n"
        #         f"lambda_h       = {self._lam_h}\n"
        #         f"lambda_p       = {self._lam_p}\n"
        #         f"m(uh, up)           = {m_hp}\n"
        #         f"rhs            = {rhs}\n"
        #         f"rhs_expected   = {rhs_expected}\n"
        #         f"rhs difference = {rhs - rhs_expected}\n"
        #         f"sigma_h        = {sigma_h}\n"
        #         f"denom          = {denom}\n"
        #         f"signed_error   = {self.signed_error}\n"
        #     ))

        # 23. Maggie change - Check if complex and how complex
        if abs(self.signed_error.imag) > self.imag_tol:
                print(RED % f"Warning - Error estimate has a nontrivial imaginary part and we will take real part: {self.signed_error}")
                warning('Self-adjoint problem has DWR estimate with nontrivial imaginary part. We took its real part')
                self.signed_error = self.signed_error.real 
        if (abs(self.signed_error.imag) > 0 and abs(self.signed_error.imag) < self.imag_tol):
                print(RED % f"Warning - Error estimate has a trivial imaginary part and we will take real part: {self.signed_error}")
                warning('Self-adjoint problem has DWR estimate with a small imaginary part. We took its real part')
                self.signed_error = self.signed_error.real 

        # Check for small real part
        # if (abs(self.signed_error.real) > 0 and abs(self.signed_error.real) < self.imag_tol):
        #     print(RED % f"Warning - Error estimate has a trivial trivial real part: {self.signed_error}")
        #     warning('Self-adjoint problem has DWR estimate with a small real part. We took made it zero')
        #     self.signed_error = 0.0



        # Save error magnitude
        eta_h = abs(self.signed_error)
        self.etah_vec.append(eta_h)
        self.print(BLUE % f'{"Predicted magnitude of error:":40s}{eta_h: 15.12e}')
        self.print(BLUE % f'{"Predicted signed error:":40s}{self.signed_error: 15.12e}')


        # If given the exact eigenvalue, compute the exact error
        eta = None
        if self.goal_exact is not None:
            self.print('Computing exact error in eigenvalue where exact eigenvalue is given as', self.goal_exact)
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
        # SHIM 4 (U.Z.): wrap lam_h in a Constant -- as a raw scalar it is
        # baked into the residual form as a literal, so every marking step
        # (new eigenvalue) changes the form signature and forces a full
        # TSFC/PyOP2 recompilation of the cell/facet indicator kernels
        F_eig = replace(A, {trial: u_h}) - Constant(lam_h) * replace(M, {trial: u_h})
        return _compute_residual_indicators(F_eig, z_err, self.options)



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


def as_mixed(exprs): # Not sure if Pablo uses this - Eigen code doesnt
    """Flatten a list of ufl.Expr objects into a vector."""
    return as_vector([e[idx] for e in exprs for idx in np.ndindex(e.ufl_shape)])


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


# def l2_normalize(f):
#     """L2-normalise a :class:`~.Function` in place and return it."""
#     from firedrake.assemble import assemble
#     nrm = float(assemble(inner(f, f) * dx)) ** 0.5
#     if nrm > 0:
#         f.assign(f / nrm)
#     return f


# 24. Maggie change - User supplies m(,), checks for mixed spaces/functions, and added in mixed versions of inner products and norms - 
# inner(,) may do this for me already



def is_mixed_function(f):
    return hasattr(f, "subfunctions") and len(f.subfunctions) > 1

def is_mixed_space(V):
    return hasattr(V, "subspaces") and len(V.subspaces) > 1

def l2_inner(u, v):
    if is_mixed_function(u):
        return assemble(sum(inner(ui, vi) * dx for ui, vi in zip(u.subfunctions, v.subfunctions)))
    else:
        return assemble(inner(u, v) * dx)

def mixed_inner(u, v): # used in local indicators (seperate from m_form)
    return sum(inner(ui, vi) for ui, vi in zip(split(u), split(v)))

def mixed_weighted_inner(a, b, weight, first_arg = False):
    if first_arg:
        return sum(inner(weight * ai, bi) for ai, bi in zip(split(a), split(b)))
    else:
        return sum(inner(ai, weight * bi) for ai, bi in zip(split(a), split(b)))

# Now the user supplies the m_form
# def m_form(u, v): # used for m-inner product (eventually user-defined)
#     from firedrake.assemble import assemble
#     if is_mixed_function(u):
#         return assemble(sum(inner(ui, vi) * dx for ui, vi in zip(u.subfunctions, v.subfunctions)))
#     else:
#         return assemble(inner(u, v) * dx)

def m_norm(u, m_form):
    val = m_form(u, u)
    return abs(val) ** 0.5

def m_normalize(f, m_form):
    nrm = m_norm(f, m_form)
    if nrm < 1e-12:
        print(RED % f"Warning - when m-normalizing, m norm is really small - {nrm}")
        warning("When m-normalizing the input function has a small m-norm")
        
    else:
        if is_mixed_function(f):
            for fi in f.subfunctions:
                fi.assign(fi / nrm)
        else:
            f.assign(f / nrm)
    return f


# 25. Maggie change - BIGGEST change 
# Joe finds best match for uh in the computed enriched eigenbasis 
# Instead, we are going to compute the best candidate in the span of the basis using
# the L2 projection of uh onto the enriched eigenbasis.
# QUESTION - Form matrices using L2 inner product or mass inner product?? (think mixed poisson)


def match_best(target, candidates, mult, V_high, m_form):

    from firedrake.assemble import assemble

    # matching for a mixed problem
    # target: Function; candidates: list[Function]; lambdas: list[...] or None
    # mult: computed multiplicity from size of enriched cluster

    # Create the Least squares problem:

    # Assemble the K matrix 
    K = np.zeros((mult, mult), dtype=complex)
    for i in range(mult):
        for j in range(mult):
            K[i, j] = l2_inner(candidates[j], candidates[i]) # l2_inner should take care of mixed spaces


    # Assemble the F vector
    F = np.zeros((mult, 1), dtype=complex)
    for i in range(mult):
        F[i, 0] = l2_inner(target, candidates[i])

    # Solve the least squares numpy problem
    X = np.linalg.solve(K, F)

    # Reassemble the needed enriched basis function
    chosen_enriched = Function(V_high)
    for i, candidate in enumerate(candidates):
        basis_coeff = complex(X[i, 0])
        chosen_enriched.assign(chosen_enriched + basis_coeff * candidate)

    # normalize chosen enriched result w.r.t mass inner product
    #print('Normalizing the least squares solution ...')
    chosen_enriched.assign(m_normalize(chosen_enriched, m_form))
    return chosen_enriched


def _solve_eigs(problem, nev, solver_parameters, m_form, guess_space = (), can_interp = True):
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
    # Set solver
    from firedrake import LinearEigensolver
    es = LinearEigensolver(problem, n_evals=nev, solver_parameters=solver_parameters)


    #print(f"Received {len(guess_space)} Firedrake initial guesses")
    # 26. Maggie change - Set initial guess space (could be over adapted meshes or over outer z loop)
    # initial_vecs = []
    # for guess in guess_space:
    #     # Accounting for potential restricted spaces:
    #     if problem.restrict:
    #         slepc_guess = Function(problem.restricted_space).interpolate(guess)
    #     else:
    #         slepc_guess = guess
    #     with slepc_guess.dat.vec_ro as v:
    #         initial_vecs.append(v.copy())
    # if initial_vecs:
    #     #print("Calling EPS.setInitialSpace ...")
    #     es.es.setInitialSpace(initial_vecs)
    # else:
    #     print("No initial space supplied: cold eigensolve")


    # UPDATE - CANT ALWAYS INTERP
    initial_vecs = []
    for guess in guess_space:
        # Accounting for potential restricted spaces:
        if problem.restrict:
            slepc_guess = Function(problem.restricted_space)

            if can_interp:
                slepc_guess.interpolate(guess)
            else:
                slepc_guess.project(guess)
        else:
            slepc_guess = guess

        with slepc_guess.dat.vec_ro as v:
            initial_vecs.append(v.copy())
    if initial_vecs:
        #print("Calling EPS.setInitialSpace ...")
        es.es.setInitialSpace(initial_vecs)
    else:
        print("No initial space supplied: cold eigensolve")

    # Solve
    nconv = es.solve()
    lams, vecs = [], []

    #  27. Maggie change - m normalize instead of l2 normalize
    # In complex mode, whole eigenfunction lives in [0] 
    for i in range(min(nconv, nev)):
        lams.append(es.eigenvalue(i))
        v, _ = es.eigenfunction(i)
        vecs.append(m_normalize(v, m_form))
    
    return lams, vecs


def _reconstruct_eig_degree(problem, extra_degree):
    """Return a new :class:`~.LinearEigenproblem` on a space of degree+extra_degree.
    For mixed spaces, we expect extra degree to be a list 
    that tells us the extra degree to be added to each subspace.

    Uses the original unrestricted forms stored on ``problem._original_A/M/bcs``
    to avoid double-restricting on the reconstructed problem.
    """
    from firedrake import LinearEigenproblem
    A = problem._original_A
    M = problem._original_M
    bcs = problem._original_bcs
    v, u = A.arguments()
    V = u.function_space()


    # 28. Maggie change - build enriched space subspace by subspace where the user specifies the degree addition per subspace
    #print('Building enriched space ...')
    if is_mixed_space(V): # check if mixed space
        high_spaces = []
        meshes = V.mesh()
        if len(extra_degree) != len(V.subspaces): # check if user gave a degree per subspace
            raise ValueError(f"extra degrees for enriched space must contain one entry per subspace: expected {len(V.subspaces)} received {len(extra_degree)}.")
        for i, V_i in enumerate(V.subspaces): # create enriched subspaces
            #print('Raising subspace number', i+1, 'by degree', extra_degree[i])
            mesh_i = meshes[i]
            element_i = V_i.ufl_element()
            degree_i = element_i.degree()
            degree_plus_i = extra_degree[i]
            high_element_i = PMGPC.reconstruct_degree(element_i, degree_i + degree_plus_i)
            high_spaces.append(FunctionSpace(mesh_i, high_element_i))
        V_high = MixedFunctionSpace(high_spaces) # create full enriched space
    else: # Pablo's reconstruction for non-mixed problems
        high_degree = V.ufl_element().degree() + extra_degree[0]
        V_high = reconstruct_degree(V, high_degree)


    # New trial and test
    u_high = TrialFunction(V_high)
    v_high = TestFunction(V_high)
    A_high = replace(A, {v: v_high, u: u_high}) 
    M_high = replace(M, {M.arguments()[0]: v_high, M.arguments()[1]: u_high}) # Is this swapped?

    # Bc's
    bcs_high = [bc.reconstruct(V=V_high, indices=bc._indices) for bc in bcs]

    # New problem
    return LinearEigenproblem(A_high, M_high, bcs=bcs_high,
                              bc_shift=problem.bc_shift, restrict=problem.restrict)


# Maggie left this alone because folded problem is self adjoint
# def _make_adjoint_eig_problem(problem):
#     """Return the adjoint of a :class:`~.LinearEigenproblem`.

#     For a self-adjoint problem this is identical to the primal.
#     """
#     from firedrake import LinearEigenproblem
#     A_adj = adjoint(problem._original_A)
#     return LinearEigenproblem(A_adj, problem._original_M,
#                               bcs=problem._original_bcs,
#                               bc_shift=problem.bc_shift, restrict=problem.restrict)





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
        :class:`GoalAdaptiveEigenOptions` (or subclass) instance for degree
        parameters and solver parameters.

    Returns
    -------
    Function
        DG0 Function of absolute-value cell indicators.
    """
    from firedrake.assemble import assemble

    # Setup
    v, = F.arguments()
    V = v.function_space()
    mesh = V.mesh().unique()
    dim = mesh.topological_dimension
    cell = mesh.ufl_cell()
    variant = "integral"


    # Bubble functions
    B = FunctionSpace(mesh, "B", dim+1, variant=variant)
    bubbles = Function(B).assign(1)

    
    
    # 29. Maggie change - accounting for mixed spaces when forming the DG spaces
    # Old code:
    # degree = V.ufl_element().degree()
    # cell_residual_degree = degree + options.cell_residual_extra_degree
    # if V.value_shape == ():
    #     DG = FunctionSpace(mesh, "DG", cell_residual_degree, variant=variant)
    # else:
    #     DG = TensorFunctionSpace(mesh, "DG", cell_residual_degree, variant=variant, shape=V.value_shape)
    # New code:
    # If a mixed problem, we
    # Take the degree of each subspace and adds to it a user-supplied amount (user must specify addition per subspace, defualt is 1)
    # If not mixed, we use Pablo's previous code

    if is_mixed_space(V):
        DG_spaces = []
        # check to make sure the user gave us enough extra degrees
        if len(options.cell_residual_extra_degree) != len(V.subspaces):
            raise ValueError(f"cell_residual_extra_degree must contain one entry per subspace: expected {len(V.subspaces)} received {len(options.cell_residual_extra_degree)}.")
        for i, Vi in enumerate(V.subspaces): # iterate over the subspaces
            degree_i = Vi.ufl_element().degree() # Take subspace degree
            cell_residual_degree_i = degree_i + options.cell_residual_extra_degree[i] # Add user-supplied additional degree for cell residual
            # Form the DG subspaces
            if Vi.value_shape == (): 
                DG_spaces.append(FunctionSpace(mesh, "DG", cell_residual_degree_i, variant=variant))
            else: # inherit the shape of Vi (ex: if a vector space)
                DG_spaces.append(TensorFunctionSpace(mesh, "DG", cell_residual_degree_i, variant=variant, shape=Vi.value_shape))
        # Form the DG space 
        DG = MixedFunctionSpace(DG_spaces)
    # If not a mixed space use Pablo's old code
    elif V.value_shape == ():
        degree = V.ufl_element().degree()
        cell_residual_degree = degree + options.cell_residual_extra_degree[0] # for non-mixed space, the extra degree is just a tuple (d,)
        DG = FunctionSpace(mesh, "DG", cell_residual_degree, variant=variant)
    else:
        degree = V.ufl_element().degree()
        cell_residual_degree = degree + options.cell_residual_extra_degree[0]
        DG = TensorFunctionSpace(mesh, "DG", cell_residual_degree, variant=variant, shape=V.value_shape)
                

    uc = TrialFunction(DG)
    vc = TestFunction(DG)

    # 30. Maggie change - accounting for mixed space in construction of cell residual problem
    # mixed_weighted_inner multiplies bubbles by every vi in split(vc)
    if is_mixed_space(DG):
        ac = mixed_weighted_inner(uc, vc, bubbles) * dx
    else: # Pablo's code
        ac = inner(uc, bubbles * vc) * dx
    

    # Solve the cell problem
    Lc = residual(F, bubbles*vc)
    Rcell = Function(DG)
    solve(ac == Lc, Rcell, solver_parameters=options.sp_cell)

    FB = FunctionSpace(mesh, "FB", dim, variant=variant)
    cones = Function(FB).assign(1)

    # 31. Maggie change - analagous to the change for DG spaces (change 29)
    if is_mixed_space(V):
        Q_spaces = []
        # check that we have enough extra degrees      
        if len(options.facet_residual_extra_degree) != len(V.subspaces):
            raise ValueError(f"facet_residual_extra_degree must contain one entry per subspace: expected {len(V.subspaces)} received {len(options.facet_residual_extra_degree)}.")
        for i, Vi in enumerate(V.subspaces):
            degree_i = Vi.ufl_element().degree()
            facet_residual_degree_i = degree_i + options.facet_residual_extra_degree[i]
            el_i = BrokenElement(FiniteElement("FB", cell=cell, degree=facet_residual_degree_i + dim, variant=variant))
            if Vi.value_shape == ():
                Q_spaces.append(FunctionSpace(mesh, el_i))
            else:
                Q_spaces.append(TensorFunctionSpace(mesh, el_i, shape=Vi.value_shape))
        Q = MixedFunctionSpace(Q_spaces)
    elif V.value_shape == ():
        degree = V.ufl_element().degree()
        facet_residual_degree = degree + options.facet_residual_extra_degree[0]
        el = BrokenElement(FiniteElement("FB", cell=cell, degree=facet_residual_degree + dim, variant=variant))
        Q = FunctionSpace(mesh, el)
    else:
        degree = V.ufl_element().degree()
        facet_residual_degree = degree + options.facet_residual_extra_degree[0]
        el = BrokenElement(FiniteElement("FB", cell=cell, degree=facet_residual_degree + dim, variant=variant))
        Q = TensorFunctionSpace(mesh, el, shape=V.value_shape)

    Qtest = TestFunction(Q)
    Qtrial = TrialFunction(Q)

    # 32. Maggie change - accounting for mixed spaces for facet residual problem
    # mixed_weighted_inner(Qtrial, Qtest, 1/cones, first_arg = True) will multiply each split(Qtrial) by 1/cones
    if is_mixed_space(V):
            Lf = residual(F, Qtest) - mixed_inner(Rcell, Qtest)*dx
            facet_mass = mixed_weighted_inner(Qtrial, Qtest, 1/cones, first_arg = True)
            af = both(facet_mass) * dS + facet_mass * ds
    else: # Pablo's code
        Lf = residual(F, Qtest) - inner(Rcell, Qtest)*dx
        af = both(inner(Qtrial/cones, Qtest))*dS + inner(Qtrial/cones, Qtest)*ds

    # Solve the facet problem
    Rhat = Function(Q)
    solve(af == Lf, Rhat, solver_parameters=options.sp_facet)
    Rfacet = Rhat/cones
    
    # Build cell-wise indicators
    DG0 = FunctionSpace(mesh, "DG", degree=0)
    test = TestFunction(DG0)

    # 33. Maggie change - Computing cell-wise indicators but accounting for mixed spaces
    # Another change based on Mixed spaces
    # Old code:
    # eta_cell = assemble(
    #     inner(inner(Rcell, z_err), test)*dx
    #     + inner(avg(inner(Rfacet, z_err)), both(test))*dS
    #     + inner(inner(Rfacet, z_err), test)*ds
    # )
    if is_mixed_space(V):
        cell_weight = mixed_inner(Rcell, z_err)
        facet_weight = mixed_weighted_inner(Rhat, z_err, 1/cones, first_arg = True)

    else:
        cell_weight = inner(Rcell, z_err)
        facet_weight = inner(Rfacet, z_err)
    # Then eta is 
    eta_cell = assemble(inner(cell_weight, test)*dx + inner(avg(facet_weight), both(test))*dS + inner(facet_weight, test)*ds )

    # Return abs value array of etas 
    with eta_cell.dat.vec as evec:
        evec.abs()
    return eta_cell



