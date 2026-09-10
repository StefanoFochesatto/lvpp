"""Latent variable proximal point (LVPP) algorithm for variational inequalities.

LVPP (Dokken, Farrell, Keith, Papadopoulos, Surowiec, "The latent variable
proximal point algorithm for variational problems with inequality
constraints", arXiv:2503.05672) solves variational problems with pointwise
inequality constraints Bu(x) in C(x) by giving each constrained unknown a
latent variable psi and iterating the mixed proximal system (paper eq. 2.7).
At proximal iteration k one (semismooth) Newton solve of

    alpha_k <J'(u^k), v> + (psi^k, Bv) - (psi^{k-1}, Bv) = 0   for all v   (2.7a)
    (Bu^k, w) - (grad R*(psi^k), w)                      = 0   for all w   (2.7b)

is done, starting from psi^0 = 0, until ||u^k - u^{k-1}|| < tol.  The
reconstruction grad R*(psi^k) lies in the interior of C(x) for every psi, so
the bound-preserving output ``u_tilde`` is feasible pointwise by construction
for any alpha; convergence of the outer iteration does NOT need alpha -> inf.

The :class:`LVPP` class accepts either an ``energy`` (UFL 0-form; the weak
residual is formed with ``ufl.derivative``) or a ``residual`` (weak 1-form(s)),
any number of unknowns (only constrained ones get latent blocks), and
per-unknown constraints (:class:`lvpp.constraints.BoxConstraint` or any
subclass of :class:`lvpp.constraints.Constraint`).
"""

import math

import ufl
from ufl.algorithms import replace

from firedrake import (Constant, ConvergenceError, DirichletBC, Function,
                       MixedFunctionSpace, NonlinearVariationalProblem,
                       NonlinearVariationalSolver, PETSc, TestFunction,
                       TrialFunction, derivative, split)
from firedrake import assemble, grad, inner

from .constraints import BoxConstraint, Constraint

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
"""Default SNES/KSP parameters; a user ``solver_parameters`` dict wins on
conflicting keys."""

_MAX_ALPHA_HALVINGS = 5
"""Retries (each halving alpha) allowed per proximal iteration when
``on_newton_failure="reduce_alpha"``."""

_ALPHA_RULE_DEFAULTS = {
    "constant": {"C": 1.0},
    "geometric": {"C": 1.0, "r": 1.5},
    "linear": {"alpha0": 1.0, "c": 1.5, "C_max": 1e10},
    "double_exponential": {"C": 1.0, "r": 1.5, "q": 1.5, "alpha_max": 10.0},
    "newton_adaptive": {"alpha0": 1.0, "alpha_max": 1e6},
}


class LVPPConvergenceError(RuntimeError):
    """Raised when the LVPP proximal iteration fails to converge."""


_MAX_ALPHA_HALVINGS = 20
"""Retries (each halving alpha) allowed per proximal iteration when
``on_newton_failure="reduce_alpha"``.  The ProximalGalerkin reference example
allows up to 50; the overall budget is additionally bounded by
``max_consecutive_failures``."""


def _make_alpha_rule(rule, params=None):
    """Return ``(rule_fn, description)`` for an alpha schedule.

    ``rule`` is one of ``"constant"``, ``"geometric"``, ``"linear"``,
    ``"double_exponential"``, ``"newton_adaptive"``, or a callable
    ``f(k, alpha_prev, newton_its) -> alpha`` (params ignored).  ``params``
    overrides the per-rule defaults in ``_ALPHA_RULE_DEFAULTS``.
    """
    params = dict(params or {})
    if callable(rule):
        return rule, f"custom({getattr(rule, '__name__', 'callable')})"
    if rule not in _ALPHA_RULE_DEFAULTS:
        raise ValueError(
            f"alpha_rule {rule!r} not recognized; expected one of "
            f"{sorted(_ALPHA_RULE_DEFAULTS)} or a callable f(k, alpha_prev, newton_its)")
    p = {**_ALPHA_RULE_DEFAULTS[rule], **params}
    if rule == "constant":
        def rule_fn(k, alpha_prev, newton_its):
            return p["C"]
    elif rule == "geometric":
        logC = math.log(p["C"])
        logr = math.log(p["r"])

        def rule_fn(k, alpha_prev, newton_its):
            return math.exp(min(logC + k * logr, 700.0))  # overflow-guarded
    elif rule == "linear":
        def rule_fn(k, alpha_prev, newton_its):
            a = p["alpha0"] if k == 1 else p["c"] * alpha_prev
            return min(a, p["C_max"])
    elif rule == "double_exponential":
        # paper eq. (3.8), evaluated in log space: C*r^(q^k) overflows
        # Python floats around k ~ 19 for the default r = q = 1.5
        logC = math.log(p["C"])
        logq = math.log(p["q"])
        logr = math.log(p["r"])

        def rule_fn(k, alpha_prev, newton_its):
            y = k * logq
            if y > 700.0:
                qk_logr = float("inf") * math.copysign(1.0, logr)
            else:
                qk_logr = math.exp(y) * logr
            log_term = logC + qk_logr
            term = math.exp(log_term) if log_term < 700.0 else float("inf")
            return min(max(term - alpha_prev, p["C"]), p["alpha_max"])
    else:  # "newton_adaptive" (fracture heuristic)
        def rule_fn(k, alpha_prev, newton_its):
            if k == 1:
                return p["alpha0"]
            if newton_its <= 4:
                a = 2.0 * alpha_prev
            elif newton_its >= 10:
                a = 0.5 * alpha_prev
            else:
                a = alpha_prev
            return min(a, p["alpha_max"])
    desc = ", ".join(f"{key}={p[key]:g}" for key in sorted(p))
    return rule_fn, f"{rule}({desc})"


def _as_function_list(u, what):
    if isinstance(u, Function):
        return [u]
    if isinstance(u, (list, tuple)) and len(u) > 0 and all(isinstance(x, Function) for x in u):
        return list(u)
    raise TypeError(f"{what} must be a firedrake Function or a non-empty list of Functions")


def _as_form(f, what="residual"):
    if isinstance(f, ufl.Form):
        return f
    raise TypeError(f"{what} must be a ufl Form, got {type(f).__name__}")


def _normalize_residual(residual, spaces):
    """Return (per-unknown 1-forms, mixed_flag).

    ``mixed_flag`` marks the single-form-with-mixed-test case, which is
    rewritten wholesale (its mixed test argument becomes the solver's mixed
    test function, landing the user's rows on the unknown blocks).
    """
    n = len(spaces)
    if isinstance(residual, (list, tuple)):
        forms = [_as_form(F) for F in residual]
        if len(forms) != n:
            raise ValueError(
                f"residual list has {len(forms)} forms but the problem has {n} unknown(s); "
                "pass one 1-form per unknown")
        for i, F in enumerate(forms):
            if len(F.arguments()) != 1:
                raise ValueError(f"residual form {i} must be a 1-form")
            if F.arguments()[0].function_space() != spaces[i]:
                raise ValueError(
                    f"residual form {i}'s test function does not live on unknown {i}'s "
                    "function space")
        return forms, False
    F = _as_form(residual)
    if len(F.arguments()) != 1:
        raise ValueError("residual must be a 1-form, or a list of 1-forms, one per unknown")
    test = F.arguments()[0]
    tspace = test.function_space()
    # firedrake mixed spaces wrap to WithGeometry like any other space;
    # discriminate via the UFL element type (ufl moved MixedElement around
    # between versions, so match on the class name)
    if (n > 1 and type(tspace.ufl_element()).__name__ == "MixedElement"
            and len(tspace) == n):
        return [F], True
    if n == 1:
        return [F], False
    raise ValueError(
        "a single residual form's test function must live on the unknown's space (one "
        f"unknown) or on a MixedFunctionSpace with {n} subspaces; otherwise pass a list "
        "of 1-forms, one per unknown")


def _as_constraint_list(entry):
    """One per-unknown bounds entry -> a list of Constraint instances."""
    if entry is None:
        return []
    if isinstance(entry, Constraint):
        return [entry]
    if isinstance(entry, (list, tuple)):
        if len(entry) == 2 and not any(
                isinstance(e, (Constraint, list, tuple)) for e in entry):
            return [BoxConstraint(lower=entry[0], upper=entry[1])]
        out = []
        for e in entry:
            out.extend(_as_constraint_list(e))
        return out
    raise TypeError(
        f"bounds entry {entry!r} invalid; expected None, a Constraint, a "
        "(lower, upper) pair, or a list of these")


def _normalize_bounds(bounds, n):
    """Return n lists of Constraint instances (one list per unknown; several
    constraints may act on the same unknown)."""
    if bounds is None:
        return [[] for _ in range(n)]
    if isinstance(bounds, Constraint):
        if n != 1:
            raise TypeError(
                "bounds must be a list with one entry per unknown for multi-unknown problems")
        return [[bounds]]
    if not isinstance(bounds, (list, tuple)):
        raise TypeError(
            "bounds must be None, a Constraint, a (lower, upper) pair, or a list with one "
            "entry per unknown")
    entries = list(bounds)
    if n == 1 and len(entries) != 1:
        # a bare (lower, upper) pair, or several constraints on the one unknown
        entries = [entries]
    if len(entries) != n:
        raise TypeError(f"bounds has {len(entries)} entries; expected one per unknown ({n})")
    return [_as_constraint_list(e) for e in entries]


def _normalize_psi_spaces(psi_spaces, n):
    if psi_spaces is None:
        return [None] * n
    if isinstance(psi_spaces, (list, tuple)):
        if len(psi_spaces) != n:
            raise ValueError(
                f"psi_spaces has {len(psi_spaces)} entries; expected one per unknown ({n})")
        return list(psi_spaces)
    if n == 1:
        return [psi_spaces]
    raise TypeError("psi_spaces must be None or a list with one entry per unknown")


class LVPP:
    """Latent variable proximal point solver for one variational problem.

    Parameters
    ----------
    energy : ufl.Form
        Scalar 0-form in the user unknown(s).  Exactly one of ``energy`` and
        ``residual`` must be given.
    residual : ufl.Form or list of ufl.Form
        Weak 1-form(s).  Either one form per unknown (test function on that
        unknown's space), one single form whose test function lives on a
        MixedFunctionSpace with one subspace per unknown (in the same order as
        ``u``), or -- for a single unknown -- one form on its space.
    u : Function or list of Function
        Unknowns; their current values are the initial data for ``solve``.
        The same Function object must not appear twice.
    bounds : None | Constraint | (lower, upper) | list per unknown
        Per unknown: ``None`` (unconstrained), a ``(lower, upper)`` pair with
        either side possibly ``None`` (built into a :class:`BoxConstraint`),
        a :class:`lvpp.constraints.Constraint` instance, or a list of
        Constraint instances (several constraints may act on the same
        unknown).  A bare pair or Constraint is accepted when there is one
        unknown.
    bcs : DirichletBC or list
        Essential boundary conditions on (a subspace of) the unknown spaces.
    psi_spaces : list of FunctionSpace or None
        Optional per-unknown override of the latent space; ``None`` entries
        use ``constraint.latent_space(V)`` (a copy of V for box constraints).
    solver_parameters, form_compiler_parameters, options_prefix :
        Forwarded to :class:`firedrake.NonlinearVariationalSolver` /
        ``NonlinearVariationalProblem``.  ``solver_parameters`` is merged over
        :data:`DEFAULT_SOLVER_PARAMETERS`.
    alpha_rule : str or callable
        One of the names in ``_ALPHA_RULE_DEFAULTS`` or a callable
        ``f(k, alpha_prev, newton_its) -> alpha``.
    alpha_parameters : dict
        Overrides of the selected rule's default parameters.
    jacobian_regularization : callable(z, z_test, z_trial) -> 2-form or None
        Optional extra Jacobian terms (e.g. the paper's fracture "reps"
        regularization), added to the preconditioner Jacobian Jp (not the
        true Jacobian: the Newton direction stays exact).
    psi_floor : float
        Constant diagonal floor on the latent block of the preconditioner
        Jacobian (see the degeneracy note below).  0 disables.
    psi_floor_drift : float
        Field-proportional part of that floor: eps(x) grows with the local
        proximal drift alpha*|lambda(x)|, i.e. with the local contact force.
        0 disables.
    psi_floor_operator : bool
        If True, the floor is also added to the true Jacobian J (and Jp = J),
        i.e. the *operator* is regularized, not just the preconditioner.
        Measured (2026-09-04): this fails at every level including coarse —
        the proximal iteration needs exact Newton steps; kept only as a
        diagnostic switch.  Default False (Jp-only, the correct mode).

    increment_norm : "L2" or "H1"
        Norm for the primal/latent increments reported by ``solve``.
    on_newton_failure : "raise" or "reduce_alpha"
        Newton failure handling: re-raise, or restore the previous iterate,
        halve alpha, and retry the same proximal iteration (at most
        ``_MAX_ALPHA_HALVINGS`` times per iteration; at most
        ``max_consecutive_failures`` consecutive failed iterations overall).

    Note
    ----
    For solutions that sit exactly on a bound over extended regions the latent
    variable psi drifts to -infinity (logarithmically, at the rate
    ``alpha * lambda`` per proximal iteration) and the entropy Jacobian rows
    degenerate once |psi| nears float saturation.  Terminate early enough
    (moderate ``tol``), keep ``alpha_max`` modest for such problems, or
    stabilize Newton through ``jacobian_regularization``; very tight bounds
    with strong forcing may need ``on_newton_failure="reduce_alpha"``.
    max_consecutive_failures : int
    verbose : bool
        One PETSc.Sys.Print line per proximal iteration.
    name : str
        Used in log lines and as the default PETSc options prefix.
    """

    def __init__(self, energy=None, residual=None, u=None, bounds=None, bcs=None,
                 psi_spaces=None, solver_parameters=None, form_compiler_parameters=None,
                 options_prefix=None,
                 alpha_rule="double_exponential", alpha_parameters=None,
                 jacobian_regularization=None, psi_floor=0.0,
                 psi_floor_drift=0.0, psi_floor_operator=False,
                 increment_norm="L2",
                 on_newton_failure="raise", max_consecutive_failures=50,
                 verbose=True, name="lvpp", **kwargs):
        if kwargs:
            raise TypeError(f"LVPP got unexpected keyword arguments {sorted(kwargs)}")
        if (energy is None) == (residual is None):
            raise ValueError("exactly one of energy= or residual= must be given")
        if on_newton_failure not in ("raise", "reduce_alpha"):
            raise ValueError("on_newton_failure must be 'raise' or 'reduce_alpha'")
        if increment_norm not in ("L2", "H1"):
            raise ValueError(f"increment_norm must be 'L2' or 'H1', got {increment_norm!r}")

        # --- unknowns -------------------------------------------------------
        self._u = _as_function_list(u, "u")
        n = len(self._u)
        if len({id(ui) for ui in self._u}) != n:
            raise ValueError("the same Function was passed twice in u")
        self._spaces = [ui.function_space() for ui in self._u]

        self._energy = None if energy is None else _as_form(energy, "energy")
        if self._energy is not None and len(self._energy.arguments()) != 0:
            raise ValueError("energy must be a scalar 0-form (no test/trial functions)")
        self._residual_forms, self._residual_mixed = (
            _normalize_residual(residual, self._spaces) if residual is not None else (None, False))

        # --- constraints and latent spaces ----------------------------------
        constraint_lists = _normalize_bounds(bounds, n)  # one list per unknown
        # flattened in order: one latent block per constraint (several
        # constraints may act on the same unknown)
        self._raw_constraints = []
        self._constraint_unknowns = []
        for i, clist in enumerate(constraint_lists):
            for c in clist:
                self._raw_constraints.append(c)
                self._constraint_unknowns.append(i)
        psi_spaces = _normalize_psi_spaces(psi_spaces, n)
        self._latent_spaces = []
        for j, i in enumerate(self._constraint_unknowns):
            override = psi_spaces[i]
            c = self._raw_constraints[j]
            self._latent_spaces.append(
                override if override is not None else c.latent_space(self._spaces[i]))

        # --- mixed space, pieces, constraints bound to the iterate ----------
        self._Z = MixedFunctionSpace(self._spaces + self._latent_spaces)
        self._z = Function(self._Z, name=f"{name}:z")
        self._subfunctions = self._z.subfunctions  # live views; stay current
        self._mapping = {ui: split(self._z)[i] for i, ui in enumerate(self._u)}
        # QVI-type bounds (data referencing user unknowns) see the current
        # iterate through these remapped copies.
        self._constraints = [c.remap(self._mapping) for c in self._raw_constraints]
        # --- scratch state ---------------------------------------------------
        self._u_prev = [Function(V) for V in self._spaces]
        self._psi_prev = [Function(W) for W in self._latent_spaces]
        self._z_backup = Function(self._Z)
        # per-accepted-iterate dual lambda_j = (psi_prev - psi)/alpha: the
        # discrete multiplier (persists on the contact set at the fixed point
        # because psi itself drifts to -infinity there).  Created here so the
        # Jp floor form below can reference it; updated in place during solve.
        self._drift = [Function(W, name=f"{name}:drift_{j}")
                       for j, W in enumerate(self._latent_spaces)]

        # --- boundary conditions ---------------------------------------------
        self._bcs = []
        if bcs is not None:
            for bc in (bcs if isinstance(bcs, (list, tuple)) else [bcs]):
                for i, V in enumerate(self._spaces):
                    if bc.function_space() == V:
                        self._bcs.append(DirichletBC(self._Z.sub(i), bc.function_arg,
                                                     bc.sub_domain))
                        break
                else:
                    raise ValueError(
                        f"{bc!r} does not live on any unknown's function space")

        # --- mixed weak form (paper eq. 2.7) ---------------------------------
        self._alpha = Constant(1.0)
        zt = TestFunction(self._Z)
        if self._energy is not None:
            R = derivative(replace(self._energy, self._mapping), self._z, zt)
        elif self._residual_mixed:
            F_user = self._residual_forms[0]
            v_user = F_user.arguments()[0]
            R = replace(F_user, {v_user: zt, **self._mapping})
        else:
            R = 0
            for i, F_i in enumerate(self._residual_forms):
                v_i = F_i.arguments()[0]
                R = R + replace(F_i, {v_i: split(zt)[i],
                                      self._u[i]: split(self._z)[i]})
        F = self._alpha * R
        for j, i in enumerate(self._constraint_unknowns):
            c = self._constraints[j]
            V = self._spaces[i]
            psi = split(self._z)[n + j]
            dw = split(zt)[n + j]
            F = (F
                 + c.coupling_form(V, split(zt)[i], psi, self._psi_prev[j])
                 + c.state_form(V, split(self._z)[i], dw, psi))
        ztr = TrialFunction(self._Z)
        J = derivative(F, self._z, ztr)
        # The regularization is a PRECONDITIONER-side term: it floors the
        # latent-block diagonal (which degenerates as psi -> -inf on the
        # contact set) but must not perturb the Newton direction, so it goes
        # on Jp (the preconditioner Jacobian) only.  This also keeps the
        # matfree action path free of the added form, whose mixed-indexed
        # arguments mis-evaluate there.
        Jp = None
        if jacobian_regularization is not None:
            reg = jacobian_regularization(self._z, zt, ztr)
            if reg is not None:
                Jp = J + reg
        if psi_floor > 0.0 or psi_floor_drift > 0.0:
            # Degeneracy-aware floor for the latent-block diagonal of the
            # preconditioner.  D = diag(exp(psi)) loses its diagonal where the
            # solution sits exactly on the bound (psi -> -inf); the floor keeps
            # the block preconditioner nonsingular there.  psi_floor is a
            # constant; psi_floor_drift scales a field-proportional part
            #     eps(x) = psi_floor + psi_floor_drift * alpha * |lambda(x)|
            # with lambda the proximal drift (= the discrete multiplier,
            # updated in place per accepted proximal iterate), so the floor
            # tracks the local contact force.  Jp-only: the Newton direction
            # is unperturbed, and the floor is harmless where D is healthy.
            floor = 0
            for j, i in enumerate(self._constraint_unknowns):
                eps_x = psi_floor + psi_floor_drift * self._alpha \
                    * abs(self._drift[j])
                dxj = ufl.Measure("dx", domain=self._spaces[i].mesh())
                floor = floor + eps_x * ufl.inner(
                    split(ztr)[n + j], split(zt)[n + j]) * dxj
            if psi_floor_operator:
                # Operator-level regularization: keeps the *operator* itself
                # nonsingular so GMRES cannot break down on the degenerate
                # latent rows (D = diag(exp(psi)) -> 0).  Perturbs the Newton
                # path, not the root: F = 0 is unaffected.
                J = J + floor
                Jp = J
            else:
                Jp = (Jp if Jp is not None else J) + floor
        self._F = F
        self._J = J
        problem = NonlinearVariationalProblem(
            F, self._z, bcs=self._bcs, J=J, Jp=Jp,
            form_compiler_parameters=form_compiler_parameters)
        params = dict(DEFAULT_SOLVER_PARAMETERS)
        if solver_parameters:
            params.update(solver_parameters)
        prefix = options_prefix if options_prefix is not None else f"{name}_"
        self._solver = NonlinearVariationalSolver(
            problem, options_prefix=prefix, solver_parameters=params)

        # --- increment and diagnostic forms (static; values live) ------------
        self._increment_norm = increment_norm

        def _increment_sq(expr, V):
            e = inner(expr, expr)
            if increment_norm == "H1":
                e = e + inner(grad(expr), grad(expr))
            return e * ufl.Measure("dx", domain=V.mesh())

        self._primal_inc_forms = [
            _increment_sq(self._subfunctions[i] - self._u_prev[i], self._spaces[i])
            for i in range(n)]

        self._recon = []
        self._recon_prev = []
        self._latent_inc_forms = []
        self._diag_feas_forms = []
        self._diag_comp_forms = []
        self._diag_dual_forms = []
        for j, i in enumerate(self._constraint_unknowns):
            V = self._spaces[i]
            c_raw = self._raw_constraints[j]
            c = self._constraints[j]
            W_obs = c_raw.observable_space(V)
            r = Function(W_obs, name=f"{name}:recon_{j}")
            rp = Function(W_obs, name=f"{name}:recon_prev_{j}")
            self._recon.append(r)
            self._recon_prev.append(rp)
            self._latent_inc_forms.append(_increment_sq(r - rp, W_obs))
            u_piece = self._subfunctions[i]
            psi_piece = self._subfunctions[n + j]
            self._diag_feas_forms.append(c.feasibility_forms(V, u_piece))
            self._diag_comp_forms.append(
                c.complementarity_form(V, u_piece, psi_piece, self._psi_prev[j], self._alpha))
            self._diag_dual_forms.append(
                c.dual_feasibility_form(V, psi_piece, self._psi_prev[j], self._alpha))

        self._energy_form = (replace(self._energy, self._mapping)
                             if self._energy is not None else None)

        # --- alpha rule and outputs ------------------------------------------
        self._rule_fn, self.alpha_rule_description = _make_alpha_rule(
            alpha_rule, alpha_parameters)
        self._on_newton_failure = on_newton_failure
        self._max_consecutive_failures = max_consecutive_failures
        self._verbose = verbose
        self._name = name

        self.z = self._z
        self.u_out = list(self._subfunctions[:n])
        self.psi_out = [self._subfunctions[n + j]
                        for j in range(len(self._constraint_unknowns))]
        self.psi_prev = self._psi_prev  # proximal shift; current after solve()
        self.drift = self._drift  # discrete multiplier; current after solve()
        self.u_tilde = None
        self.newton_iterations = []
        self.alpha_history = []
        self.proximal_iterations = 0
        self.alpha = None
        self.history = {}

    # ------------------------------------------------------------------ solve

    def solve(self, tol=1e-8, max_proximal_iterations=100, warm_start=False):
        """Run the proximal iteration until ``||u^k - u^{k-1}|| < tol``.

        Resets the state from the user's initial data (latent variables zero
        unless ``warm_start``), then iterates: pick alpha, back up z,
        Newton-solve the mixed system, update diagnostics, shift
        psi_prev/u_prev.  Raises :class:`LVPPConvergenceError` if Newton keeps
        failing (see ``on_newton_failure``) or ``max_proximal_iterations`` is
        exhausted.  Returns ``self``; results are on ``self.z`` (mixed),
        ``self.u_out`` / ``self.psi_out`` (live subfunction views),
        ``self.u_tilde`` (bound-preserving reconstructions), plus
        ``self.history`` and the iteration counters.

        warm_start : bool
            If True, keep the current latent block and carry it into the
            proximal shift psi_prev instead of resetting both to zero.  Seed
            the latent block beforehand via ``lvpp.psi_out[j].assign(...)``
            (e.g. prolonged from a previous mesh level in an adaptive loop);
            the primal block is always reset from the user's ``u``.
        """
        n = len(self._u)
        m = len(self._constraint_unknowns)
        subs = self._subfunctions
        for i in range(n):
            subs[i].assign(self._u[i])
        if warm_start:
            # the caller seeded the latent block (psi_out views); carry it
            # into the proximal shift so the first subproblem continues the
            # previous level's solution
            for j in range(m):
                self._psi_prev[j].assign(subs[n + j])
        else:
            for j in range(m):
                subs[n + j].zero()
            for f in self._psi_prev:
                f.zero()
        for i in range(n):
            self._u_prev[i].assign(self._u[i])

        history = {"alpha": [], "newton_iterations": [], "primal_increment": [],
                   "latent_increment": [], "energy": [], "feasibility": [],
                   "complementarity": [], "dual_feasibility": []}
        self.history = history
        self.alpha_history = history["alpha"]
        self.newton_iterations = history["newton_iterations"]

        if self._verbose:
            PETSc.Sys.Print(f"lvpp[{self._name}] alpha rule: {self.alpha_rule_description}, "
                            f"tol={tol:g}, max_proximal_iterations={max_proximal_iterations}")

        alpha_prev = 0.0
        newton_its_prev = 0
        consecutive_failures = 0
        for k in range(1, max_proximal_iterations + 1):
            alpha = float(self._rule_fn(k, alpha_prev, newton_its_prev))
            self._alpha.assign(alpha)
            self._z_backup.assign(self._z)
            halvings = 0
            while True:
                try:
                    self._solver.solve()
                    break
                except ConvergenceError:
                    consecutive_failures += 1
                    if self._on_newton_failure != "reduce_alpha":
                        raise
                    if consecutive_failures > self._max_consecutive_failures:
                        raise LVPPConvergenceError(
                            f"lvpp[{self._name}]: {consecutive_failures} consecutive Newton "
                            f"failures (limit {self._max_consecutive_failures})") from None
                    halvings += 1
                    if halvings > _MAX_ALPHA_HALVINGS:
                        raise LVPPConvergenceError(
                            f"lvpp[{self._name}]: more than {_MAX_ALPHA_HALVINGS} alpha "
                            f"halvings needed at proximal iteration {k}") from None
                    self._z.assign(self._z_backup)
                    alpha *= 0.5
                    self._alpha.assign(alpha)
                    if self._verbose:
                        PETSc.Sys.Print(
                            f"lvpp[{self._name}] {k:3d} Newton failed; retrying with "
                            f"alpha={alpha:.6e}")
            newton_its = self._solver.snes.getIterationNumber()
            consecutive_failures = 0

            # shift the reconstructions and re-interpolate them at the new
            # iterate (raw constraints: user bound data; pieces are live)
            for j, i in enumerate(self._constraint_unknowns):
                self._recon_prev[j].assign(self._recon[j])
                self._recon[j].interpolate(
                    self._raw_constraints[j].observable(subs[n + j]))

            # diagnostics: remapped constraints + live pieces, so QVI bounds
            # and the increments see the current iterate
            pinc = math.sqrt(sum(assemble(f) for f in self._primal_inc_forms))
            linc = math.sqrt(sum(assemble(f) for f in self._latent_inc_forms))
            e = assemble(self._energy_form) if self._energy_form is not None else None
            feas = sum(assemble(f) for forms in self._diag_feas_forms for f in forms)
            comp = sum(assemble(f) for f in self._diag_comp_forms if f is not None)
            dual = sum(assemble(f) for f in self._diag_dual_forms if f is not None)
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

            converged = pinc < tol
            for j in range(m):
                # capture the dual (multiplier) BEFORE shifting: the drift
                # (psi_prev - psi)/alpha persists on the contact set at the
                # fixed point and converges to the exact multiplier
                self._drift[j].assign(
                    (self._psi_prev[j] - subs[n + j]) / alpha)
                self._psi_prev[j].assign(subs[n + j])
            for i in range(n):
                self._u_prev[i].assign(subs[i])
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
        self.u_tilde = []
        for j, i in enumerate(self._constraint_unknowns):
            c_raw = self._raw_constraints[j]
            out = Function(self._recon[j].function_space(), name=f"{self._name}:u_tilde_{j}")
            out.interpolate(c_raw.observable(subs[n + j]))
            self.u_tilde.append(out)
        return self

    # ------------------------------------------------- post-solve diagnostics
    # All on the RAW constraints (user bound data, valid after the solve) and
    # the live subfunctions of z.

    def _require_solved(self):
        if self.alpha is None:
            raise RuntimeError("call solve() before requesting diagnostics")

    def energy(self):
        """Energy of the current iterate (energy path only)."""
        self._require_solved()
        if self._energy_form is None:
            raise RuntimeError("energy diagnostics need energy=, not residual=")
        return assemble(self._energy_form)

    def feasibility(self):
        """Total integral primal constraint violation of the iterate."""
        self._require_solved()
        total = 0.0
        for j, i in enumerate(self._constraint_unknowns):
            c = self._raw_constraints[j]
            total += sum(assemble(f) for f in
                         c.feasibility_forms(self._spaces[i], self._subfunctions[i]))
        return total

    def complementarity(self):
        """Total complementarity measure (vanishes at the solution)."""
        self._require_solved()
        n = len(self._u)
        total = 0.0
        for j, i in enumerate(self._constraint_unknowns):
            c = self._raw_constraints[j]
            f = c.complementarity_form(self._spaces[i], self._subfunctions[i],
                                       self._subfunctions[n + j], self._psi_prev[j],
                                       self.alpha)
            if f is not None:
                total += assemble(f)
        return total

    def dual_feasibility(self):
        """Total dual-variable progress measure (vanishes at the solution)."""
        self._require_solved()
        n = len(self._u)
        total = 0.0
        for j, i in enumerate(self._constraint_unknowns):
            c = self._raw_constraints[j]
            f = c.dual_feasibility_form(self._spaces[i], self._subfunctions[n + j],
                                        self._psi_prev[j], self.alpha)
            if f is not None:
                total += assemble(f)
        return total
