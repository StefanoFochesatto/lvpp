"""Build the mixed LVPP system (eq. 2.7 of LVPP) from a user problem.

This module owns everything structural: it normalizes the user's input, forms
the mixed function space ``Z = U_1 x ... x U_n x W_1 x ... x W_m`` with one
primal block per user unknown and one latent block per constraint, and
accumulates the residual ``F``, its Jacobian ``J``, and the diagnostic forms.
It deliberately knows nothing about how the system is solved (see
:mod:`lvpp.solver`) or how its Newton systems are approximately factored (see
:mod:`lvpp.preconditioners`).

Every block-structured factorization relies on the dof ordering: all primal
dofs come first, in the order of the unknowns, followed by the latent blocks in
``constraint_unknowns`` order.  ``n_primal = sum(V.dim() for V in primal
spaces)`` is the offset of the first latent dof, which is what a preconditioner
uses to split the two families.
"""

from dataclasses import dataclass

import ufl
from ufl.algorithms import replace

from firedrake import (Constant, DirichletBC, Function, MixedFunctionSpace,
                       TestFunction, TrialFunction, derivative, grad, inner,
                       split)

from .constraints import BoxConstraint, Constraint

__all__ = ["ProblemSpec", "MixedSystem", "build_spec"]


# --------------------------------------------------------------- input parsing

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

    ``mixed_flag`` marks the single-form-with-mixed-test case: one form carries
    all the unknowns, and its mixed test argument is replaced by the solver's
    mixed test function so the user's rows land on the unknown blocks.
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
    # firedrake mixed spaces wrap to WithGeometry like any other space; the
    # UFL element type tells them apart, matched on the class name because
    # MixedElement has sat in different ufl modules across versions
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


def _normalize_latent_spaces(latent_spaces, n):
    if latent_spaces is None:
        return [None] * n
    if isinstance(latent_spaces, (list, tuple)):
        if len(latent_spaces) != n:
            raise ValueError(
                f"latent_spaces has {len(latent_spaces)} entries; expected one per "
                f"unknown ({n})")
        return list(latent_spaces)
    if n == 1:
        return [latent_spaces]
    raise TypeError("latent_spaces must be None or a list with one entry per unknown")


# ------------------------------------------------------------------ the spec

@dataclass(frozen=True)
class ProblemSpec:
    """Normalized problem data, validated once and then data-only.

    ``constraints`` is flattened across unknowns: one entry per latent block.
    ``constraint_unknowns[j]`` is the index of the unknown that block ``j``
    constrains.  ``u`` and ``spaces`` are per user unknown.
    """

    u: tuple
    spaces: tuple
    constraints: tuple
    constraint_unknowns: tuple
    latent_spaces: tuple
    bcs: tuple
    energy: object
    residual_forms: tuple
    residual_is_mixed: bool

    @property
    def n_unknowns(self):
        return len(self.u)

    @property
    def n_constraints(self):
        return len(self.constraints)


def build_spec(energy=None, residual=None, u=None, bounds=None, bcs=None,
               latent_spaces=None):
    """Validate and normalize user input into a :class:`ProblemSpec`.

    Exactly one of ``energy`` (a scalar 0-form) and ``residual`` (weak
    1-form(s)) must be given.  ``bounds`` follows the
    :class:`~lvpp.solver.LVPP` contract; ``latent_spaces`` optionally overrides
    the constraint-provided latent space per unknown.
    """
    if (energy is None) == (residual is None):
        raise ValueError("exactly one of energy= or residual= must be given")

    u_list = _as_function_list(u, "u")
    n = len(u_list)
    if len({id(ui) for ui in u_list}) != n:
        raise ValueError("the same Function was passed twice in u")
    spaces = [ui.function_space() for ui in u_list]

    energy_form = None if energy is None else _as_form(energy, "energy")
    if energy_form is not None and len(energy_form.arguments()) != 0:
        raise ValueError("energy must be a scalar 0-form (no test/trial functions)")

    residual_forms, residual_mixed = (
        _normalize_residual(residual, spaces) if residual is not None else (None, False))

    constraint_lists = _normalize_bounds(bounds, n)
    raw_constraints = []
    constraint_unknowns = []
    for i, clist in enumerate(constraint_lists):
        for c in clist:
            raw_constraints.append(c)
            constraint_unknowns.append(i)

    overrides = _normalize_latent_spaces(latent_spaces, n)
    latent = []
    for j, i in enumerate(constraint_unknowns):
        override = overrides[i]
        c = raw_constraints[j]
        latent.append(override if override is not None else c.latent_space(spaces[i]))

    bcs_list = [] if bcs is None else list(bcs if isinstance(bcs, (list, tuple)) else [bcs])

    return ProblemSpec(
        u=tuple(u_list), spaces=tuple(spaces), constraints=tuple(raw_constraints),
        constraint_unknowns=tuple(constraint_unknowns), latent_spaces=tuple(latent),
        bcs=tuple(bcs_list), energy=energy_form,
        residual_forms=None if residual_forms is None else tuple(residual_forms),
        residual_is_mixed=residual_mixed)


# ------------------------------------------------------------ the mixed system

class MixedSystem:
    """The static mixed weak form plus the live mixed unknown.

    The residual has two families of rows.  At proximal iteration ``k`` there
    is one primal row per user unknown, built from the user's energy or
    residual and shifted by the proximal term,

        alpha_k <J'(u^k), v> + (psi^k, B v) - (psi^{k-1}, B v) = 0     (2.7a)

    and one state row per constraint, which drives the latent variable to the
    point where its reconstruction matches the constrained unknown,

        (B u^k, w) - (grad R*(psi^k), w) = 0                           (2.7b)

    A constraint contributes to both families through
    :meth:`~lvpp.constraints.Constraint.coupling_form` (the ``B v`` term and
    its proximal shift, row (2.7a)) and
    :meth:`~lvpp.constraints.Constraint.state_form` (the ``grad R*`` term, row
    (2.7b)).  The primal blocks of ``Z`` come first, one per user unknown in
    order, and the latent blocks follow in ``constraint_unknowns`` order.

    Constructed once; :meth:`refresh_reconstructions`, the drift capture, and
    the diagnostic helpers are called per proximal iteration.  Nothing here
    allocates PETSc objects, so the whole class can be exercised without a
    solver.
    """

    def __init__(self, spec, increment_norm="L2", name="lvpp"):
        if increment_norm not in ("L2", "H1"):
            raise ValueError(f"increment_norm must be 'L2' or 'H1', got {increment_norm!r}")
        self.spec = spec
        self.name = name
        self.increment_norm = increment_norm
        n = spec.n_unknowns
        spaces = spec.spaces
        latent_spaces = spec.latent_spaces

        self.alpha = Constant(1.0, name=f"{name}:alpha")

        # --- mixed space and live pieces ------------------------------------
        self.Z = MixedFunctionSpace(list(spaces) + list(latent_spaces))
        self.z = Function(self.Z, name=f"{name}:z")
        self.subfunctions = self.z.subfunctions  # live views, so they track z
        self.mapping = {ui: split(self.z)[i] for i, ui in enumerate(spec.u)}
        # bounds whose data reference the user unknowns (QVI type) see the
        # current iterate through these remapped copies
        self.constraints = [c.remap(self.mapping) for c in spec.constraints]

        # --- scratch state ---------------------------------------------------
        self.u_prev = [Function(V) for V in spaces]
        self.psi_prev = [Function(W) for W in latent_spaces]
        self.z_backup = Function(self.Z)
        # per-iteration dual lambda_j = (psi_prev - psi)/alpha, the discrete
        # multiplier: at the fixed point it persists on the contact set because
        # psi itself drifts to -infinity there.  Created here so a
        # preconditioner's floor form can reference it; updated in place.
        self.drift = [Function(W, name=f"{name}:drift_{j}")
                      for j, W in enumerate(latent_spaces)]

        # --- boundary conditions ---------------------------------------------
        self.bcs = []
        for bc in spec.bcs:
            for i, V in enumerate(spaces):
                if bc.function_space() == V:
                    self.bcs.append(DirichletBC(self.Z.sub(i), bc.function_arg,
                                                bc.sub_domain))
                    break
            else:
                raise ValueError(f"{bc!r} does not live on any unknown's function space")

        # --- mixed weak form: rows (2.7a) and (2.7b) -------------------------
        zt = TestFunction(self.Z)
        ztr = TrialFunction(self.Z)
        if spec.energy is not None:
            R = derivative(replace(spec.energy, self.mapping), self.z, zt)
        elif spec.residual_is_mixed:
            F_user = spec.residual_forms[0]
            v_user = F_user.arguments()[0]
            R = replace(F_user, {v_user: zt, **self.mapping})
        else:
            R = 0
            for i, F_i in enumerate(spec.residual_forms):
                v_i = F_i.arguments()[0]
                R = R + replace(F_i, {v_i: split(zt)[i],
                                      spec.u[i]: split(self.z)[i]})
        F = self.alpha * R
        for j, i in enumerate(spec.constraint_unknowns):
            c = self.constraints[j]
            V = spaces[i]
            psi = split(self.z)[n + j]
            dw = split(zt)[n + j]
            F = (F
                 + c.coupling_form(V, split(zt)[i], psi, self.psi_prev[j])
                 + c.state_form(V, split(self.z)[i], dw, psi))
        self.F = F
        self.J = derivative(F, self.z, ztr)
        self.z_test = zt
        self.z_trial = ztr

        # --- increment and diagnostic forms: built once, evaluated per iterate
        def _increment_sq(expr, V):
            e = inner(expr, expr)
            if increment_norm == "H1":
                e = e + inner(grad(expr), grad(expr))
            return e * ufl.Measure("dx", domain=V.mesh())

        self.primal_increment_forms = [
            _increment_sq(self.subfunctions[i] - self.u_prev[i], spaces[i])
            for i in range(n)]

        self.recon = []
        self.recon_prev = []
        self.latent_increment_forms = []
        self.diag_feasibility_forms = []
        self.diag_complementarity_forms = []
        self.diag_dual_forms = []
        for j, i in enumerate(spec.constraint_unknowns):
            V = spaces[i]
            c_raw = spec.constraints[j]
            c = self.constraints[j]
            W_obs = c_raw.observable_space(V)
            r = Function(W_obs, name=f"{name}:recon_{j}")
            rp = Function(W_obs, name=f"{name}:recon_prev_{j}")
            self.recon.append(r)
            self.recon_prev.append(rp)
            self.latent_increment_forms.append(_increment_sq(r - rp, W_obs))
            u_piece = self.subfunctions[i]
            psi_piece = self.subfunctions[n + j]
            self.diag_feasibility_forms.append(c.feasibility_forms(V, u_piece))
            self.diag_complementarity_forms.append(
                c.complementarity_form(V, u_piece, psi_piece, self.psi_prev[j], self.alpha))
            self.diag_dual_forms.append(
                c.dual_feasibility_form(V, psi_piece, self.psi_prev[j], self.alpha))

        self.energy_form = (replace(spec.energy, self.mapping)
                            if spec.energy is not None else None)

    # ------------------------------------------------------------- jacobians

    def jacobian_with(self, corrections):
        """Return ``Jp = J + sum(corrections)`` (``None`` entries ignored)."""
        out = self.J
        for extra in corrections or ():
            if extra is not None:
                out = out + extra
        return out

    # ---------------------------------------------------- per-iterate updates

    def refresh_reconstructions(self):
        """Shift ``recon_prev <- recon`` and re-interpolate ``grad R*(psi)``.

        Raw constraints are used so QVI bound data is the user's; the pieces
        are live views of the current iterate.
        """
        n = self.spec.n_unknowns
        for j, i in enumerate(self.spec.constraint_unknowns):
            self.recon_prev[j].assign(self.recon[j])
            self.recon[j].interpolate(
                self.spec.constraints[j].observable(self.subfunctions[n + j]))

    def capture_drift(self):
        """Capture ``(psi_prev - psi)/alpha`` BEFORE shifting ``psi_prev``.

        At the fixed point this persists on the contact set and converges to
        the exact discrete multiplier.
        """
        n = self.spec.n_unknowns
        alpha = float(self.alpha)
        for j in range(self.spec.n_constraints):
            self.drift[j].assign(
                (self.psi_prev[j] - self.subfunctions[n + j]) / alpha)

    def shift_previous(self):
        """``psi_prev <- psi``, ``u_prev <- u`` (the proximal shift)."""
        n = self.spec.n_unknowns
        for j in range(self.spec.n_constraints):
            self.psi_prev[j].assign(self.subfunctions[n + j])
        for i in range(n):
            self.u_prev[i].assign(self.subfunctions[i])

    # -------------------------------------------------------- diagnostics

    def primal_increment(self):
        """Norm of ``u^k - u^{k-1}`` over all unknowns (L2 or H1 per
        ``increment_norm``); this is what :class:`~lvpp.schedules.PrimalIncrement`
        tests for convergence."""
        return float(sum(_assemble(f) for f in self.primal_increment_forms)) ** 0.5

    def latent_increment(self):
        """Norm of the change ``grad R*(psi^k) - grad R*(psi^{k-1})`` in the
        reconstructed (observable) latent variable."""
        return float(sum(_assemble(f) for f in self.latent_increment_forms)) ** 0.5

    def energy_value(self):
        """The user energy at the current iterate, or ``None`` for a problem
        given as a residual."""
        return None if self.energy_form is None else float(_assemble(self.energy_form))

    def feasibility_value(self):
        """Total integral primal constraint violation, summed over the
        constraints; it is zero exactly when the iterate is admissible."""
        n = self.spec.n_unknowns
        total = 0.0
        for j, i in enumerate(self.spec.constraint_unknowns):
            c = self.spec.constraints[j]
            total += sum(_assemble(f) for f in
                         c.feasibility_forms(self.spec.spaces[i], self.subfunctions[i]))
        return float(total)

    def complementarity_value(self):
        """Total complementarity measure, which vanishes at the solution of the
        variational inequality."""
        total = 0.0
        for f in self.diag_complementarity_forms:
            if f is not None:
                total += _assemble(f)
        return float(total)

    def dual_feasibility_value(self):
        """Total dual-feasibility measure, i.e. how far the latent variables
        still sit from the multiplier at the solution; it vanishes there."""
        total = 0.0
        for f in self.diag_dual_forms:
            if f is not None:
                total += _assemble(f)
        return float(total)

    def reconstruct(self, j):
        """A fresh copy of ``grad R*(psi_j)`` on the observable space, the same
        quantity the state row (2.7b) drives.  This is the bound-preserving
        ``u_tilde``: ``grad R*`` maps into the interior of the admissible set,
        so the reconstruction is pointwise feasible for any ``psi``."""
        n = self.spec.n_unknowns
        c_raw = self.spec.constraints[j]
        out = Function(self.recon[j].function_space(),
                       name=f"{self.name}:u_tilde_{j}")
        out.interpolate(c_raw.observable(self.subfunctions[n + j]))
        return out


def _assemble(f):
    from firedrake import assemble
    return assemble(f)
