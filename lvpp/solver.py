"""Latent variable proximal point (LVPP) solver.

LVPP (Dokken, Farrell, Keith, Papadopoulos, & Surowiec (2025)) solves
variational problems with pointwise inequality constraints ``B u(x) in C(x)``
by giving each constrained unknown a latent variable ``psi`` and iterating the
mixed proximal system (2.7a)-(2.7b) of the paper.  At proximal iteration ``k``
one (semismooth) Newton solve of

    alpha_k <J'(u^k), v> + (psi^k, Bv) - (psi^{k-1}, Bv) = 0   for all v   (2.7a)
    (B u^k, w) - (grad R*(psi^k), w)                      = 0   for all w   (2.7b)

is done, starting from ``psi^0 = 0``, until the stopping rule fires.  Because
``grad R*(psi)`` lies in the interior of the admissible set for every ``psi``,
the reconstruction ``u_tilde = grad R*(psi^k)`` is feasible pointwise by
construction whatever alpha is, so convergence does not need ``alpha -> inf``.

This class is the loop: it owns the proximal iteration, the schedule, the
stopping rule, the SNES, and the diagnostics.  The structure -- spaces, weak
form, Jacobians -- lives in :mod:`lvpp.assembly`; how the Newton systems are
approximately factored lives in :mod:`lvpp.preconditioners`; the alpha
schedules and the stopping rules live in :mod:`lvpp.schedules`.

A user ``solver_parameters`` dict always wins over a preconditioner's
``parameters()``, which in turn wins over :data:`DEFAULT_SOLVER_PARAMETERS`, so
swapping preconditioners never silently overrides an explicit choice.
"""

import warnings

from firedrake import (ConvergenceError, NonlinearVariationalProblem,
                       NonlinearVariationalSolver, PETSc, assemble)

from .assembly import MixedSystem, build_spec
from .preconditioners import (DegeneracyFloor, DirectFactorization,
                              JacobianCorrection, SaddleView,
                              resolve_preconditioner)
from .schedules import (PrimalIncrement, describe, make_schedule)

__all__ = ["LVPP", "LVPPConvergenceError", "DEFAULT_SOLVER_PARAMETERS"]


DEFAULT_SOLVER_PARAMETERS = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "basic",
    "snes_rtol": 1e-10,
    "snes_atol": 1e-12,
    "snes_stol": 0.0,
    "snes_max_it": 200,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
"""Default SNES/KSP parameters; a preconditioner's ``parameters()`` and then a
user ``solver_parameters`` dict win on conflicting keys."""

_MAX_ALPHA_HALVINGS = 20
"""Retries (each halving alpha) allowed per proximal iteration when
``on_newton_failure="reduce_alpha"``.  The ProximalGalerkin reference example
allows up to 50; the overall budget is additionally bounded by
``max_consecutive_failures``."""


class LVPPConvergenceError(RuntimeError):
    """Raised when the LVPP proximal iteration fails to converge."""


class LVPP:
    """An LVPP object solves one variational problem with pointwise inequality
    constraints by the latent variable proximal point algorithm of Dokken,
    Farrell, Keith, Papadopoulos, & Surowiec (2025).

    Central notions behind this class:
      * Each constrained unknown ``u`` is paired with a latent variable
        ``psi`` on its own (often much lower-dimensional) space ``W``.  The
        admissible set is the image of a concave ``R*``, and the problem is
        solved through the mixed proximal system (2.7a)-(2.7b) of the paper: a
        primal row from the user's energy or residual, shifted by the proximal
        term ``(psi^k - psi^{k-1}, B v)``, and a state row ``(B u^k, w) -
        (grad R*(psi^k), w)`` that drives the latent variable to the point
        where its reconstruction matches the constrained unknown.
      * ``grad R*(psi)`` lies in the interior of the admissible set for every
        ``psi``, so the reconstruction ``u_tilde = grad R*(psi^k)`` is feasible
        pointwise by construction, and the iteration needs no
        ``alpha -> inf`` to produce an admissible answer.
      * The proximal parameter ``alpha_k`` follows a schedule from
        :mod:`lvpp.schedules`; the default double-exponential rule is eq.
        (3.8) of the paper, and the proximal iteration count it produces is
        mesh independent on the benchmark problems.
      * On a contact set the latent variable drifts to -infinity at the rate
        ``alpha * lambda`` per iteration, ``lambda`` being the multiplier
        there, so the entropy Jacobian rows degenerate once ``|psi|`` nears
        float saturation.  Terminate early enough (a moderate ``tol``), keep
        ``alpha_max`` modest for such problems, or stabilize Newton through a
        preconditioner-side correction (see
        :class:`~lvpp.preconditioners.DegeneracyFloor`); very tight bounds with
        strong forcing may also need ``on_newton_failure="reduce_alpha"``.

    Regarding the arguments: exactly one of ``energy`` (a scalar 0-form) and
    ``residual`` (weak 1-form(s)) is given, the latter either as one form per
    unknown, one single form whose test function lives on a MixedFunctionSpace
    with one subspace per unknown in the order of ``u``, or -- for a single
    unknown -- one form on its space.  ``u`` is the unknown or list of unknowns
    whose current values are the initial data for solve(), and the same
    Function must not appear twice.  ``bounds`` says, per unknown, ``None``
    (unconstrained), a ``(lower, upper)`` pair with either side possibly
    ``None`` (built into a :class:`~lvpp.constraints.BoxConstraint`), a
    :class:`~lvpp.constraints.Constraint`, or a list of constraints acting on
    that unknown; a bare pair or Constraint is accepted for one unknown.
    ``bcs`` holds essential boundary conditions on (subspaces of) the unknown
    spaces, and ``latent_spaces`` optionally overrides the
    constraint-provided latent space per unknown (``None`` entries use
    ``constraint.latent_space(V)``, a copy of ``V`` for box constraints).
    ``preconditioner`` says how the Newton system is approximately factored:
    ``None`` is direct LU/MUMPS, ``"schur"`` is the floorless Schur fieldsplit,
    a dict is raw PETSc options, and an object with ``parameters()`` is used as
    it stands (see :mod:`lvpp.preconditioners`).  ``solver_parameters``,
    ``form_compiler_parameters``, and ``options_prefix`` go to
    :class:`firedrake.NonlinearVariationalSolver`, with ``solver_parameters``
    merged over :data:`DEFAULT_SOLVER_PARAMETERS` and the preconditioner's
    options.  ``alpha_schedule`` and ``alpha_parameters`` select and tune the
    step-size rule, either a name from
    :data:`lvpp.schedules.ALPHA_RULE_DEFAULTS` or a callable
    ``f(k, alpha_prev, newton_its) -> alpha``, and ``stopping`` replaces the
    ``PrimalIncrement(tol)`` built from the ``tol`` of solve();
    ``increment_norm`` ("L2" or "H1") sets the norm in which the primal and
    latent increments are reported.
    ``on_newton_failure`` is "raise" or "reduce_alpha": the latter restores the
    previous iterate, halves alpha, and retries the same proximal iteration, at
    most ``_MAX_ALPHA_HALVINGS`` times per iteration and at most
    ``max_consecutive_failures`` consecutive failed iterations overall.

    The public API of the LVPP class is:

      solve():  run the proximal iteration and return self

      matrix():  assemble the true Jacobian ``J`` at the current iterate, with the Jp-only corrections left out

      install_monitor():  attach a KSP monitor to the outer Krylov solver

      energy(), feasibility(), complementarity(), dual_feasibility():  diagnostics of the current iterate, each vanishing or minimal at the solution

      close():  release the preconditioner's PETSc objects

    The properties ``snes``, ``ksp``, and ``pc`` expose the PETSc objects of
    the nonlinear solve, while ``constraints``, ``primal_spaces``,
    ``latent_spaces``, and ``bcs`` expose the assembled problem data.  After a
    solve the results sit on ``z`` (the mixed unknown) and on the live
    subfunction views ``u_out`` and ``psi_out``, with the pointwise feasible
    reconstructions on ``u_tilde``; ``alpha_constant`` is the live proximal
    parameter, ``drift`` holds the discrete multipliers, and ``history``
    together with ``alpha_history``, ``newton_iterations``,
    ``proximal_iterations``, ``alpha``, and ``stopping_description`` records
    the run.

    A typical obstacle problem, minimizing an energy subject to ``u >= psi``:

    .. code-block:: python3

      u = Function(V, name="u")
      solver = LVPP(energy=energy, u=u, bounds=(psi, None),
                    alpha_schedule="linear",
                    alpha_parameters={"alpha0": 2**-7, "c": sqrt(2), "C_max": 2**-3})
      solver.solve(tol=1e-8, max_proximal_iterations=100)
      u_tilde = solver.u_tilde[0]        # pointwise feasible reconstruction

    The earlier keyword spellings are also accepted: ``psi_spaces`` for
    ``latent_spaces``, ``alpha_rule`` for ``alpha_schedule``, and
    ``psi_floor`` / ``psi_floor_drift`` / ``psi_floor_operator`` /
    ``jacobian_regularization`` for a preconditioner-side correction (the first
    three select ``preconditioner=DegeneracyFloor(...)``).  Each emits a
    DeprecationWarning, and the scripts in ``lvpp/experiments/`` still call
    them.
    """

    def __init__(self, energy=None, residual=None, u=None, bounds=None, bcs=None,
                 latent_spaces=None, preconditioner=None,
                 alpha_schedule="double_exponential", alpha_parameters=None,
                 stopping=None,
                 solver_parameters=None, form_compiler_parameters=None,
                 options_prefix=None, increment_norm="L2",
                 on_newton_failure="raise", max_consecutive_failures=50,
                 verbose=True, name="lvpp",
                 # --- earlier spellings, also accepted (see the class docstring)
                 psi_spaces=None, alpha_rule=None,
                 psi_floor=0.0, psi_floor_drift=0.0, psi_floor_operator=False,
                 jacobian_regularization=None, **kwargs):
        if kwargs:
            raise TypeError(f"LVPP got unexpected keyword arguments {sorted(kwargs)}")
        if on_newton_failure not in ("raise", "reduce_alpha"):
            raise ValueError("on_newton_failure must be 'raise' or 'reduce_alpha'")

        # --- earlier spellings ----------------------------------------------
        if psi_spaces is not None:
            if latent_spaces is not None:
                raise TypeError("pass either psi_spaces= (deprecated) or latent_spaces=, not both")
            warnings.warn("psi_spaces= is deprecated; use latent_spaces=",
                          DeprecationWarning, stacklevel=2)
            latent_spaces = psi_spaces
        if alpha_rule is not None:
            warnings.warn("alpha_rule= is deprecated; use alpha_schedule=",
                          DeprecationWarning, stacklevel=2)
            alpha_schedule = alpha_rule
        floor_active = bool(psi_floor) or bool(psi_floor_drift) or bool(psi_floor_operator)
        if floor_active:
            if preconditioner is not None:
                raise ValueError(
                    "psi_floor* and preconditioner= are mutually exclusive; pass "
                    "DegeneracyFloor(...) as the preconditioner instead")
            warnings.warn(
                "psi_floor/psi_floor_drift/psi_floor_operator are deprecated; use "
                "preconditioner=DegeneracyFloor(...)",
                DeprecationWarning, stacklevel=2)
            preconditioner = DegeneracyFloor(
                constant=psi_floor, drift=psi_floor_drift, on_operator=psi_floor_operator)

        # --- structure -------------------------------------------------------
        spec = build_spec(energy=energy, residual=residual, u=u, bounds=bounds, bcs=bcs,
                          latent_spaces=latent_spaces)
        self.spec = spec
        self._system = MixedSystem(spec, increment_norm=increment_norm, name=name)
        self._name = name
        self._verbose = verbose
        self._on_newton_failure = on_newton_failure
        self._max_consecutive_failures = max_consecutive_failures
        sys = self._system
        n = spec.n_unknowns

        # --- preconditioner --------------------------------------------------
        if preconditioner is None:
            preconditioner = DirectFactorization()
        self.preconditioner = resolve_preconditioner(preconditioner)

        self._view = SaddleView(
            z=sys.z,
            alpha=sys.alpha,
            drift=tuple(sys.drift),
            constraint_unknowns=tuple(spec.constraint_unknowns),
            primal_spaces=tuple(spec.spaces),
            latent_spaces=tuple(spec.latent_spaces),
            bcs=tuple(sys.bcs),
            n_primal=sum(V.dim() for V in spec.spaces),
            mesh=spec.spaces[0].mesh(),
            jacobian=sys.J,
        )

        # --- Jacobians: J is exact; corrections go on Jp only ----------------
        J_eff = sys.J
        op_corr = self.preconditioner.operator_correction(self._view)
        if op_corr is not None:
            # DegeneracyFloor(on_operator=True): the floor lands on the
            # operator itself rather than on Jp, which reproduces the recorded
            # negative result; a jacobian_regularization passed alongside is
            # not applied in this branch.
            J_eff = sys.J + op_corr
            Jp = J_eff
        else:
            corrections = [self.preconditioner.jacobian_correction(self._view)]
            if jacobian_regularization is not None:
                corrections.append(JacobianCorrection(jacobian_regularization)(self._view))
            Jp = sys.jacobian_with(corrections)

        # --- SNES -------------------------------------------------------------
        problem = NonlinearVariationalProblem(
            sys.F, sys.z, bcs=sys.bcs, J=J_eff, Jp=Jp,
            form_compiler_parameters=form_compiler_parameters)
        params = dict(DEFAULT_SOLVER_PARAMETERS)
        params.update(self.preconditioner.parameters())
        if solver_parameters:
            params.update(solver_parameters)
        prefix = options_prefix if options_prefix is not None else f"{name}_"
        self._solver = NonlinearVariationalSolver(
            problem, options_prefix=prefix, solver_parameters=params)
        self.preconditioner.install(self._solver.snes, self._view)

        # --- schedule and stopping -------------------------------------------
        self._rule_fn, self.alpha_rule_description = _resolve_schedule(
            alpha_schedule, alpha_parameters)
        self._stopping = stopping
        self._stopping_factory = stopping is None

        # --- published state -------------------------------------------------
        self.z = sys.z
        self.u_out = list(sys.subfunctions[:n])
        self.psi_out = [sys.subfunctions[n + j] for j in range(spec.n_constraints)]
        self.psi_prev = sys.psi_prev       # proximal shift; current after solve()
        self.drift = sys.drift             # discrete multiplier; current after solve()
        self.alpha_constant = sys.alpha    # live proximal parameter
        self.u_tilde = None
        self.newton_iterations = []
        self.alpha_history = []
        self.proximal_iterations = 0
        self.alpha = None
        self.history = {}
        self.stopping_description = "default"

        # --- private handles onto the same state, read by the scripts in
        # --- lvpp/experiments/ ----------------------------------------------
        self._z = sys.z
        self._Z = sys.Z
        self._alpha = sys.alpha
        self._u = list(spec.u)
        self._spaces = list(spec.spaces)
        self._latent_spaces = list(spec.latent_spaces)
        self._constraint_unknowns = list(spec.constraint_unknowns)
        self._raw_constraints = list(spec.constraints)
        self._constraints = sys.constraints
        self._bcs = sys.bcs
        self._subfunctions = sys.subfunctions
        self._mapping = sys.mapping
        self._u_prev = sys.u_prev
        self._psi_prev = sys.psi_prev
        self._z_backup = sys.z_backup
        self._drift = sys.drift
        self._recon = sys.recon
        self._recon_prev = sys.recon_prev
        self._energy = spec.energy
        self._energy_form = sys.energy_form
        self._residual_forms = spec.residual_forms
        self._residual_mixed = spec.residual_is_mixed
        self._J = sys.J
        self._primal_inc_forms = sys.primal_increment_forms
        self._latent_inc_forms = sys.latent_increment_forms
        self._diag_feas_forms = sys.diag_feasibility_forms
        self._diag_comp_forms = sys.diag_complementarity_forms
        self._diag_dual_forms = sys.diag_dual_forms
        self._increment_norm = increment_norm

    # ---------------------------------------------------------------- handles

    @property
    def snes(self):
        """The PETSc SNES (public handle for monitors/PC surgery)."""
        return self._solver.snes

    @property
    def ksp(self):
        """The outer PETSc KSP."""
        return self._solver.snes.ksp

    @property
    def pc(self):
        """The outer PETSc PC."""
        return self._solver.snes.ksp.getPC()

    @property
    def constraints(self):
        """The raw (user-data) constraints, one per latent block."""
        return self._raw_constraints

    @property
    def primal_spaces(self):
        """One FunctionSpace per user unknown."""
        return self._spaces

    @property
    def latent_spaces(self):
        """One FunctionSpace per latent block."""
        return self._latent_spaces

    @property
    def bcs(self):
        """Essential BCs lifted onto the mixed space."""
        return self._bcs

    def matrix(self, mat_type="aij"):
        """Assemble the true Jacobian ``J`` of the mixed system (2.7a)-(2.7b) at
        the current iterate.  The preconditioner corrections land on ``Jp``
        only, so this returns the operator the residual actually
        differentiates, which is what the recorded scripts inspect."""
        return assemble(self._system.J, mat_type=mat_type, bcs=self._system.bcs)

    def install_monitor(self, fn):
        """Attach a KSP monitor to the outer Krylov solver."""
        self.ksp.setMonitor(fn)

    # ------------------------------------------------------------------ solve

    def solve(self, tol=1e-8, max_proximal_iterations=100, warm_start=False):
        """Run the proximal iteration until the stopping rule fires.

        Resets the state from the user's initial data -- the latent blocks to
        zero unless ``warm_start`` -- and then repeats the paper's loop: pick
        ``alpha_k`` from the schedule, back up the mixed iterate, take one
        Newton solve of (2.7a)-(2.7b), refresh the reconstructions and the
        diagnostics, capture the drift, and shift ``psi_prev`` and ``u_prev``.
        ``tol`` builds the default ``PrimalIncrement`` stopping rule when the
        constructor was given no ``stopping``.  Raises
        :class:`LVPPConvergenceError` if Newton keeps failing (see
        ``on_newton_failure``) or ``max_proximal_iterations`` is exhausted.

        Returns ``self``.  The results sit on ``self.z`` (the mixed unknown),
        ``self.u_out`` and ``self.psi_out`` (live subfunction views),
        ``self.u_tilde`` (the pointwise feasible reconstructions), and in
        ``self.history`` with the iteration counters.

        With ``warm_start=True`` the current latent block is kept and carried
        into the proximal shift ``psi_prev`` instead of being reset to zero.
        Seed the latent block beforehand through
        ``lvpp.psi_out[j].assign(...)``, for instance by prolonging it from a
        previous mesh level in an adaptive loop; the primal block is always
        reset from the user's ``u``.
        """
        self._pending_tol = tol
        self._max_proximal_iterations = max_proximal_iterations
        self._consecutive_failures = 0
        self._initialise(warm_start)
        history = self.history
        alpha_prev = 0.0
        newton_its_prev = 0
        k = 0
        for k in range(1, max_proximal_iterations + 1):
            alpha = float(self._rule_fn(k, alpha_prev, newton_its_prev))
            self._system.alpha.assign(alpha)
            self._system.z_backup.assign(self._system.z)
            alpha = self._newton_iteration(k, alpha)
            self._consecutive_failures = 0
            newton_its = self._system_newton_its()

            self._system.refresh_reconstructions()
            self._record_diagnostics(k, alpha, newton_its)
            converged = self._stopping.converged(k, alpha, alpha_prev, history)

            # capture the dual (multiplier) BEFORE shifting: the drift
            # (psi_prev - psi)/alpha persists on the contact set at the fixed
            # point and converges to the exact multiplier
            self._system.capture_drift()
            self._system.shift_previous()
            alpha_prev = alpha
            newton_its_prev = newton_its
            if converged:
                break
        else:
            raise LVPPConvergenceError(
                f"lvpp[{self._name}]: no convergence in {max_proximal_iterations} proximal "
                f"iterations (final primal increment "
                f"{history['primal_increment'][-1]:.3e} > tol {tol:.3e})")

        self.proximal_iterations = k
        self.alpha = alpha
        self.u_tilde = [self._system.reconstruct(j)
                        for j in range(self.spec.n_constraints)]
        return self

    # -------------------------------------------------------- loop internals

    def _initialise(self, warm_start):
        """Reset the mixed iterate and the history for a fresh solve.

        The primal blocks take the user's current ``u``, and ``u_prev`` is set
        to the same values so the first primal increment is measured from the
        initial data.  The latent blocks go to zero -- the paper's ``psi^0 =
        0`` -- unless ``warm_start``, in which case the seeded latent block is
        carried into the proximal shift so the first subproblem continues the
        previous level's solution.  The default stopping rule is built here
        from ``tol``.
        """
        spec = self.spec
        sys = self._system
        n = spec.n_unknowns
        subs = sys.subfunctions
        for i in range(n):
            subs[i].assign(spec.u[i])
        if warm_start:
            # the caller seeded the latent block (psi_out views); carry it
            # into the proximal shift so the first subproblem continues the
            # previous level's solution
            for j in range(spec.n_constraints):
                sys.psi_prev[j].assign(subs[n + j])
        else:
            for j in range(spec.n_constraints):
                subs[n + j].zero()
            for f in sys.psi_prev:
                f.zero()
        for i in range(n):
            sys.u_prev[i].assign(subs[i])

        history = {"alpha": [], "newton_iterations": [], "primal_increment": [],
                   "latent_increment": [], "energy": [], "feasibility": [],
                   "complementarity": [], "dual_feasibility": []}
        self.history = history
        self.alpha_history = history["alpha"]
        self.newton_iterations = history["newton_iterations"]
        if self._stopping_factory:
            self._stopping = PrimalIncrement(self._pending_tol)
            self.stopping_description = f"primal_increment(tol={self._pending_tol:g})"
        if self._verbose:
            PETSc.Sys.Print(f"lvpp[{self._name}] alpha rule: {self.alpha_rule_description}, "
                            f"stopping={self.stopping_description}, "
                            f"max_proximal_iterations={self._max_proximal_iterations}")

    def _newton_iteration(self, k, alpha):
        """One SNES solve of (2.7a)-(2.7b) with the ``reduce_alpha`` retry loop.

        On a ConvergenceError the previous iterate is restored, alpha is
        halved, and the same proximal iteration is attempted again, within the
        halving and consecutive-failure budgets.
        """
        halvings = 0
        while True:
            try:
                self._solver.solve()
                return alpha
            except ConvergenceError:
                self._consecutive_failures += 1
                if self._on_newton_failure != "reduce_alpha":
                    raise
                if self._consecutive_failures > self._max_consecutive_failures:
                    raise LVPPConvergenceError(
                        f"lvpp[{self._name}]: {self._consecutive_failures} consecutive "
                        f"Newton failures (limit {self._max_consecutive_failures})") from None
                halvings += 1
                if halvings > _MAX_ALPHA_HALVINGS:
                    raise LVPPConvergenceError(
                        f"lvpp[{self._name}]: more than {_MAX_ALPHA_HALVINGS} alpha "
                        f"halvings needed at proximal iteration {k}") from None
                self._system.z.assign(self._system.z_backup)
                alpha *= 0.5
                self._system.alpha.assign(alpha)
                if self._verbose:
                    PETSc.Sys.Print(
                        f"lvpp[{self._name}] {k:3d} Newton failed; retrying with "
                        f"alpha={alpha:.6e}")

    def _system_newton_its(self):
        """The Newton iteration count of the most recent SNES solve, which a
        NewtonAdaptive schedule consumes on the next proximal step."""
        return self._solver.snes.getIterationNumber()

    def _record_diagnostics(self, k, alpha, newton_its):
        """Append this iteration's diagnostics to the history and print the
        verbose line.

        The primal and latent increments are what the stopping rules inspect;
        the feasibility, complementarity, and dual-feasibility values are the
        VI diagnostics, all of which vanish at the solution.
        """
        sys = self._system
        history = self.history
        pinc = sys.primal_increment()
        linc = sys.latent_increment()
        e = sys.energy_value()
        feas = sys.feasibility_value()
        comp = sys.complementarity_value()
        dual = sys.dual_feasibility_value()
        history["alpha"].append(alpha)
        history["newton_iterations"].append(newton_its)
        history["primal_increment"].append(pinc)
        history["latent_increment"].append(linc)
        history["energy"].append(e)
        history["feasibility"].append(feas)
        history["complementarity"].append(comp)
        history["dual_feasibility"].append(dual)
        if self._verbose:
            PETSc.Sys.Print(
                f"lvpp[{self._name}] {k:3d} alpha={alpha:9.3e} |du|={pinc:9.3e} "
                f"|dpsi|={linc:9.3e} feas={feas:9.3e} comp={comp:9.3e} "
                f"dual={dual:9.3e} newton={newton_its}")

    # ------------------------------------------------- post-solve diagnostics

    def _require_solved(self):
        """Raise if solve() has not run yet, so the post-solve diagnostics
        cannot report uninitialised state."""
        if self.alpha is None:
            raise RuntimeError("call solve() before requesting diagnostics")

    def energy(self):
        """Energy of the current iterate (energy path only)."""
        self._require_solved()
        if self._system.energy_form is None:
            raise RuntimeError("energy diagnostics need energy=, not residual=")
        return self._system.energy_value()

    def feasibility(self):
        """Total integral primal constraint violation of the iterate."""
        self._require_solved()
        return self._system.feasibility_value()

    def complementarity(self):
        """Total complementarity measure (vanishes at the solution)."""
        self._require_solved()
        return self._system.complementarity_value()

    def dual_feasibility(self):
        """Total dual-variable progress measure (vanishes at the solution)."""
        self._require_solved()
        return self._system.dual_feasibility_value()

    def close(self):
        """Release the preconditioner's PETSc objects."""
        self.preconditioner.finalize()

    # ----------------------------------------------------------------- internals


def _resolve_schedule(alpha_schedule, alpha_parameters):
    """Resolve the user's schedule into a callable and its verbose-line
    description."""
    schedule = make_schedule(alpha_schedule, alpha_parameters)
    return schedule, describe(schedule)
