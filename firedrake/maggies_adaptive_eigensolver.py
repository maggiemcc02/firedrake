# Here we make the skeleton for an adaptive eigensolver based on old eigensolver in Pablo's code (see commented code at end)
# I am attempting to make an updated eigensolver that matches the current state of Pablo's code

# Pablo's current imports:

import numbers
from dataclasses import dataclass
from typing import ClassVar

from petsctools import OptionsManager

from firedrake.assemble import assemble
from firedrake.petsc import PETSc
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace, TensorFunctionSpace
from firedrake.ufl_expr import TestFunction, TrialFunction, derivative, action, adjoint
from firedrake.variational_solver import (NonlinearVariationalProblem, NonlinearVariationalSolver,
                                          LinearVariationalProblem, LinearVariationalSolver)
from firedrake.solving import solve
from firedrake.output import VTKFile
from firedrake.logging import RED

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
        for it in range(self.options.max_it):
            try:
                self.step(it=it)
            except StopIteration:
                break

    
    
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
        self.print(f"---------------------------- [MESH LEVEL {it}] ----------------------------")
        # SOLVE + ESTIMATE
        eta_h, eta = self.solve_and_estimate()
        self.post_iteration(it)
        if abs(eta_h) < self.options.tolerance:
            self.print("Error estimate below tolerance, finished.")
            raise StopIteration
        elif it == self.options.max_it - 1:
            self.print(f"Maximum iteration ({self.options.max_it}) reached. Exiting.")
            raise StopIteration
        # MARK
        self.print("Computing local refinement indicators eta_K ...")
        eta_cell = self.compute_error_indicators()
        self.compute_efficiency_indices(eta_cell, eta_h, eta)
        markers = self.set_adaptive_cell_markers(eta_cell)
        # REFINE
        self.print("Transferring problem to new mesh ...")
        self.refine_problem(markers)

    # Allows the user to design, say output.
    # Pretty much, after each iteration, this will run any user-supplied code.
    # For example, the user could want to save the solution, print output, save plots, etc
    # This replaces the old write solution tools
    def post_iteration(self, it: int):
        """Hook called after SOLVE+ESTIMATE, before convergence check.
           Invokes the user-supplied ``post_iteration_callback``, if any."""
        if self.post_iteration_callback is not None:
            self.post_iteration_callback(self, it)

    # ------------------------------------------------------------------
    # Common machinery (mark + refine + efficiency)
    # ------------------------------------------------------------------

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

    def refine_problem(self, markers, coef_map=None):
        """Refine the mesh and reconstruct the problem on the new mesh.

        Parameters
        ----------
        markers
            DG0 Function with value 1 on cells to refine.
        """
        for marker in markers.subfunctions:
            mesh = marker.function_space().mesh()
            new_mesh = mesh.refine_marked_elements(marker)
            amh, _ = get_level(mesh)
            amh.add_mesh(new_mesh)

        # Reconstruct MeshSequence with the refined meshes
        mesh = self.amh[-1]
        if len(mesh) > 1:
            new_mesh = type(mesh)([get_level(m)[0][-1] for m in mesh])
            self.amh.add_mesh(new_mesh)
        if coef_map is None:
            coef_map = {}
        self.problem = refine(self.problem, refine, coefficient_mapping=coef_map)

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

    def compute_error_indicators(self):
        """Compute cell-wise error indicators using stored solver state.

        Returns
        -------
        Function
            A DG0 Function of cell-wise indicators (absolute values).
        """
        raise NotImplementedError


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























class GoalAdaptiveEigensolver(GoalAdaptiveSolverBase):
    """Solves an eigenvalue problem adaptively to minimise the error in a
    target eigenvalue.  The goal functional is :math:`J = \\lambda`.

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

    def __init__(self,
                 problem,
                 target: float,
                 tolerance: float,
                 goal_adaptive_options: dict | None = None,
                 solver_parameters: dict | None = None,
                 exact_eigenvalue: float | None = None,
                 ):
        super().__init__(tolerance, goal_adaptive_options, exact_goal=exact_eigenvalue)
        self.problem = problem
        self.target = target
        self.sp = solver_parameters or {}

        # Set up AdaptiveMeshHierarchy
        mesh = problem.output_space.mesh()
        mh, level = get_level(mesh)
        if mh is None:
            AdaptiveMeshHierarchy(mesh)
        else:
            amh = AdaptiveMeshHierarchy(mh[0])
            for m in mh[1:level+1]:
                amh.add_mesh(m)

        self.atm = AdaptiveTransferManager()
        self._lam_h = None

    def _make_options(self, d):
        return GoalAdaptiveEigenOptions(**d)

    def solve(self):
        """Run the adaptive loop and return the eigenvalue and error estimate.

        Returns
        -------
        tuple[float, float]
            ``(lam_h, error_estimate)`` on the finest mesh.
        """
        super().solve()
        return self._lam_h, self.etah_vec[-1]

    def solve_and_estimate(self):
        """Solve the eigenproblem at two polynomial degrees, match eigenfunctions,
        and compute a global error estimate for the eigenvalue."""
        opts = self.options
        problem = self.problem
        V = problem.output_space
        self.Ndofs_vec.append(V.dim())
        self.print(f"Solving eigenproblem (degree: {V.ufl_element().degree()}, dofs: {V.dim()}) ...")

        sp_target = dict(self.sp)
        if opts.self_adjoint:
            sp_target.setdefault("eps_gen_hermitian", None)
        sp_target["eps_target"] = self.target

        # Solve at degree p
        lams, vecs = _solve_eigs(problem, opts.nev, sp_target)
        self._lam_h = lams[0]
        self._u_h = vecs[0]
        self.print(f'{"Computed eigenvalue:":40s}{self._lam_h:15.12f}')

        # Solve at degree p + dual_extra_degree (enriched primal)
        high_problem = _reconstruct_eig_degree(problem, opts.dual_extra_degree)
        self.print(f"Solving enriched eigenproblem (dofs: {high_problem.output_space.dim()}) ...")
        lams_p, vecs_p = _solve_eigs(high_problem, opts.nev, sp_target)
        self._lam_p, self._u_p = match_best(self._u_h, vecs_p, lams_p)

        if opts.self_adjoint:
            self._z_h = self._u_h
            self._z_p = self._u_p
        else:
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

        # Dual error representative (UFL expression, may span two spaces)
        self._z_err = self._z_p - self._z_h

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
        e_sigma = u_p - u_h

        if self.options.self_adjoint:
            sigma_h = 0.5 * float(assemble(inner(e_sigma, e_sigma) * dx))
            rhs = float(assemble(replace_both_args(A, u_h, e)
                                 - lam_h * replace_both_args(M, u_h, e)))
        else:
            e_adj = z_p - z_h
            sigma_h = 0.5 * float(assemble(inner(e_sigma, e_adj) * dx))
            rhs = 0.5 * (
                float(assemble(replace_both_args(A, u_h, e_adj)))
                - lam_h * float(assemble(replace_both_args(M, u_h, e_adj)))
                + float(assemble(replace_both_args(adjoint(A), z_h, e)))
                - lam_h * float(assemble(replace_both_args(M, z_h, e)))
            )

        denom = 1.0 - sigma_h
        eta_h = abs(rhs / denom) if abs(denom) > 1e-14 else float("nan")
        self.etah_vec.append(eta_h)
        self.print(f'{"Predicted error:":40s}{eta_h: 15.12e}')

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

    def refine_problem(self, markers):
        """Refine the mesh and reconstruct the :class:`~.LinearEigenproblem`."""
        mesh = markers.function_space().mesh()
        new_mesh = mesh.refine_marked_elements(markers)
        amh, _ = get_level(mesh)
        amh.add_mesh(new_mesh)
        coef_map = {}
        self.problem = refine(self.problem, refine, coefficient_mapping=coef_map)

    def write_solution(self, it):
        ws = self.options.write_solution
        if ws is False:
            return
        elif ws is True:
            should_write = True
        elif isinstance(ws, Integral):
            should_write = (it % ws == 0)
        else:
            raise ValueError(f"write_solution must be False, True, or a positive integer, got {ws!r}")
        if should_write:
            output_dir = self.options.output_dir
            run_name = self.options.run_name
            self.print("Writing (primal) eigenfunction ...")
            VTKFile(f"{output_dir}/{run_name}/{run_name}_eigenfunction_{it}.pvd"
                    ).write(*self._u_h.subfunctions)


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


def match_best(target, candidates, lambdas=None):
    """Return the candidate best correlated with ``target`` in L2.

    Parameters
    ----------
    target
        Reference :class:`~.Function`.
    candidates
        List of candidate :class:`~.Function` objects.
    lambdas
        List of associated eigenvalues (optional).

    Returns
    -------
    tuple
        ``(lam, aligned)`` where ``lam`` is the eigenvalue of the best match
        (or its index if ``lambdas`` is ``None``) and ``aligned`` is a
        phase/sign-aligned copy of the best candidate in the same space.
    """
    from firedrake.assemble import assemble
    nt = float(assemble(inner(target, target) * dx)) ** 0.5
    scores = []
    for i, w in enumerate(candidates):
        nw = float(assemble(inner(w, w) * dx)) ** 0.5
        if nw == 0.0:
            continue
        c = complex(assemble(inner(target, w) * dx))
        scores.append((abs(c) / (nt * nw), i, c))

    if not scores:
        raise RuntimeError("No nonzero candidate found in match_best.")

    scores.sort(key=lambda t: t[0], reverse=True)
    _, best_i, best_c = scores[0]

    aligned = candidates[best_i].copy(deepcopy=True)
    if best_c != 0:
        phase = best_c.conjugate() / abs(best_c)
        aligned.assign(phase * aligned)

    lam = lambdas[best_i] if lambdas is not None else best_i
    return lam, aligned


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
        vecs.append(l2_normalize(vr))
    return lams, vecs


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
    high_degree = V.ufl_element().degree() + extra_degree
    V_high = reconstruct_degree(V, high_degree)
    u_high = TrialFunction(V_high)
    v_high = TestFunction(V_high)
    A_high = replace(A, {v: v_high, u: u_high})
    M_high = replace(M, {M.arguments()[0]: v_high, M.arguments()[1]: u_high})
    bcs_high = [bc.reconstruct(V=V_high, indices=bc._indices) for bc in bcs]
    return LinearEigenproblem(A_high, M_high, bcs=bcs_high,
                              bc_shift=problem.bc_shift, restrict=problem.restrict)


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
    degree = V.ufl_element().degree()
    cell_residual_degree = degree + options.cell_residual_extra_degree
    facet_residual_degree = degree + options.facet_residual_extra_degree

    B = FunctionSpace(mesh, "B", dim+1, variant=variant)
    bubbles = Function(B).assign(1)

    if V.value_shape == ():
        DG = FunctionSpace(mesh, "DG", cell_residual_degree, variant=variant)
    else:
        DG = TensorFunctionSpace(mesh, "DG", cell_residual_degree, variant=variant, shape=V.value_shape)
    uc = TrialFunction(DG)
    vc = TestFunction(DG)
    ac = inner(uc, bubbles*vc)*dx
    Lc = residual(F, bubbles*vc)
    Rcell = Function(DG)
    solve(ac == Lc, Rcell, solver_parameters=options.sp_cell)

    FB = FunctionSpace(mesh, "FB", dim, variant=variant)
    cones = Function(FB).assign(1)
    el = BrokenElement(FiniteElement("FB", cell=cell, degree=facet_residual_degree+dim, variant=variant))
    if V.value_shape == ():
        Q = FunctionSpace(mesh, el)
    else:
        Q = TensorFunctionSpace(mesh, el, shape=V.value_shape)
    Qtest = TestFunction(Q)
    Qtrial = TrialFunction(Q)
    Lf = residual(F, Qtest) - inner(Rcell, Qtest)*dx
    af = both(inner(Qtrial/cones, Qtest))*dS + inner(Qtrial/cones, Qtest)*ds
    Rhat = Function(Q)
    solve(af == Lf, Rhat, solver_parameters=options.sp_facet)
    Rfacet = Rhat/cones

    DG0 = FunctionSpace(mesh, "DG", degree=0)
    test = TestFunction(DG0)
    eta_cell = assemble(
        inner(inner(Rcell, z_err), test)*dx
        + inner(avg(inner(Rfacet, z_err)), both(test))*dS
        + inner(inner(Rfacet, z_err), test)*ds
    )
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

