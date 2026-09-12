"""Alpha schedules and stopping rules for the proximal point iteration.

Two orthogonal strategies drive :meth:`lvpp.solver.LVPP.solve`:

* :class:`AlphaSchedule` -- the proximal parameter ``alpha_k`` of the mixed
  system (2.7a)-(2.7b) of LVPP.  :func:`make_schedule` resolves the
  ``alpha_rule`` entry point (the names ``"constant"``, ``"geometric"``,
  ``"linear"``, ``"double_exponential"``, ``"newton_adaptive"``, or a user
  callable) into one of the immutable dataclasses below, and :func:`describe`
  reproduces the description used by the verbose line.
* :class:`StoppingRule` -- when the outer iteration has converged.
  :class:`PrimalIncrement` is the LVPP default; :class:`AlphaPlateau` is the
  termination rule used by the examples of hpG and is an opt-in choice.

An alpha schedule is the paper's freedom in choosing ``alpha_k``: it sees the
1-based iteration index, the parameter of the previous iteration, and the
Newton count of the previous proximal solve, and it returns the parameter for
the next one.  :meth:`DoubleExponential.__call__` is eq. (3.8) of LVPP, and it
is evaluated in log space: the naive ``C * r**(q**k)`` overflows to ``inf``
near ``k ~ 19`` for the defaults ``r = q = 1.5``, whereas the log form stays
finite until ``alpha_max`` saturates it.
"""

import math
import typing
from dataclasses import dataclass, fields

__all__ = [
    "AlphaSchedule",
    "Constant",
    "Geometric",
    "Linear",
    "DoubleExponential",
    "NewtonAdaptive",
    "ALPHA_RULE_DEFAULTS",
    "make_schedule",
    "describe",
    "StoppingRule",
    "PrimalIncrement",
    "AlphaPlateau",
]


class AlphaSchedule(typing.Protocol):
    """A rule ``alpha_k = f(k, alpha_prev, newton_its)``.

    ``k`` is the 1-based proximal iteration, ``alpha_prev`` the parameter of
    the previous iteration (0.0 before the first), and ``newton_its`` the
    Newton iterations the previous proximal solve took.
    """

    def __call__(self, k: int, alpha_prev: float, newton_its: int) -> float:
        """Return the proximal parameter for iteration ``k``."""
        ...


@dataclass(frozen=True)
class Constant:
    """``alpha_k = C``: a fixed proximal parameter."""

    C: float = 1.0

    def __call__(self, k: int, alpha_prev: float, newton_its: int) -> float:
        """Return ``C``, ignoring the iteration state."""
        return self.C


@dataclass(frozen=True)
class Geometric:
    """``alpha_k = C * r**k``, evaluated as ``exp(min(log C + k log r, 700))``.

    The ``min`` is an overflow guard: ``r**k`` itself overflows much earlier
    than the exponential form.
    """

    C: float = 1.0
    r: float = 1.5

    def __call__(self, k: int, alpha_prev: float, newton_its: int) -> float:
        """Return the overflow-guarded geometric ramp at iteration ``k``."""
        return math.exp(min(math.log(self.C) + k * math.log(self.r), 700.0))


@dataclass(frozen=True)
class Linear:
    """``alpha_1 = alpha0``, then ``alpha_k = min(c * alpha_{k-1}, C_max)``.

    A geometric ramp of ratio ``c`` that turns constant once it reaches
    ``C_max``; the values ``alpha0 = 2**-7``, ``c = sqrt(2)``, and
    ``C_max = 2**-3`` are the capped sequence used in §6.1-6.2 of hpG.
    """

    alpha0: float = 1.0
    c: float = 1.5
    C_max: float = 1e10

    def __call__(self, k: int, alpha_prev: float, newton_its: int) -> float:
        """Return ``alpha0`` at ``k == 1``, else the capped ramp step."""
        a = self.alpha0 if k == 1 else self.c * alpha_prev
        return min(a, self.C_max)


@dataclass(frozen=True)
class DoubleExponential:
    """``alpha_k = min(max(C*r**(q**k) - alpha_prev, C), alpha_max)``.

    This is eq. (3.8) of LVPP: the schedule has an exponent ``q**k`` nested
    inside the exponent ``r**(q**k)``, so alpha grows doubly exponentially and
    saturates at ``alpha_max``.  It is evaluated in log space because
    ``C * r**(q**k)`` overflows a Python float near ``k ~ 19`` for the defaults
    ``r = q = 1.5``; the ``y > 700`` and ``log_term < 700`` branches are the
    guards that keep the intermediate finite.
    """

    C: float = 1.0
    r: float = 1.5
    q: float = 1.5
    alpha_max: float = 10.0

    def __call__(self, k: int, alpha_prev: float, newton_its: int) -> float:
        """Return the double-exponential value at iteration ``k``.

        The nested exponent is what forces the log-space evaluation; the
        subtraction of ``alpha_prev`` is part of the published rule.
        """
        logC = math.log(self.C)
        logq = math.log(self.q)
        logr = math.log(self.r)
        y = k * logq
        if y > 700.0:
            qk_logr = float("inf") * math.copysign(1.0, logr)
        else:
            qk_logr = math.exp(y) * logr
        log_term = logC + qk_logr
        term = math.exp(log_term) if log_term < 700.0 else float("inf")
        return min(max(term - alpha_prev, self.C), self.alpha_max)


@dataclass(frozen=True)
class NewtonAdaptive:
    """Scale by the previous Newton iteration count (fracture heuristic).

    ``alpha_1 = alpha0``; then ``2 * alpha_prev`` when the previous proximal
    solve needed at most 4 Newton iterations, ``0.5 * alpha_prev`` when it
    needed 10 or more, and hold otherwise.  Capped at ``alpha_max``.
    """

    alpha0: float = 1.0
    alpha_max: float = 1e6

    def __call__(self, k: int, alpha_prev: float, newton_its: int) -> float:
        """Return the Newton-count-adapted value at iteration ``k``."""
        if k == 1:
            return self.alpha0
        if newton_its <= 4:
            a = 2.0 * alpha_prev
        elif newton_its >= 10:
            a = 0.5 * alpha_prev
        else:
            a = alpha_prev
        return min(a, self.alpha_max)


_RULES = {
    "constant": Constant,
    "geometric": Geometric,
    "linear": Linear,
    "double_exponential": DoubleExponential,
    "newton_adaptive": NewtonAdaptive,
}
"""Rule name -> the dataclass implementing it."""

_RULE_NAMES = {rule_cls: name for name, rule_cls in _RULES.items()}
"""Inverse of :data:`_RULES`, for :func:`describe`."""


ALPHA_RULE_DEFAULTS = {
    name: {field.name: field.default for field in fields(rule_cls)}
    for name, rule_cls in _RULES.items()
}
"""Per-rule default parameters, read off the fields of the rule dataclasses so
that the defaults and the constructors cannot drift apart."""


def make_schedule(rule, params=None):
    """Resolve ``rule`` into an :class:`AlphaSchedule`.

    ``rule`` is either a rule name -- a key of :data:`ALPHA_RULE_DEFAULTS` --
    in which case that rule's defaults are used and ``params`` overrides them
    (an unknown parameter name raises ``TypeError``, since the fields of the
    rule dataclass are fixed), or an :class:`AlphaSchedule` instance or any
    other callable ``f(k, alpha_prev, newton_its) -> alpha``, which is returned
    unchanged with ``params`` ignored.  An unrecognized name raises
    :class:`ValueError`.
    """
    if callable(rule):
        return rule
    if rule not in ALPHA_RULE_DEFAULTS:
        raise ValueError(
            f"alpha_rule {rule!r} not recognized; expected one of "
            f"{sorted(ALPHA_RULE_DEFAULTS)} or a callable f(k, alpha_prev, newton_its)")
    return _RULES[rule](**{**ALPHA_RULE_DEFAULTS[rule], **dict(params or {})})


def describe(schedule):
    """Return the verbose-line description of ``schedule``.

    A rule dataclass is rendered as ``name(key=value, ...)`` with the parameter
    keys sorted and the values formatted with ``%g``, e.g.
    ``"double_exponential(C=1, q=1.5, r=1.5, alpha_max=10)"``.  Any other
    callable is rendered as ``"custom(<__name__ or 'callable'>)"``.
    """
    name = _RULE_NAMES.get(type(schedule))
    if name is not None:
        params = {field.name: getattr(schedule, field.name)
                  for field in fields(_RULES[name])}
        text = ", ".join(f"{key}={params[key]:g}" for key in sorted(params))
        return f"{name}({text})"
    return f"custom({getattr(schedule, '__name__', 'callable')})"


class StoppingRule(typing.Protocol):
    """Decides when the outer proximal iteration has converged."""

    def converged(self, k: int, alpha: float, alpha_prev: float, history: dict) -> bool:
        """Return whether the iteration should stop after proximal step ``k``.

        ``alpha`` is the parameter used by step ``k`` and ``alpha_prev`` the
        one used by step ``k - 1`` (0.0 before the first); ``history`` holds
        the per-iteration records accumulated so far.
        """
        ...


@dataclass(frozen=True)
class PrimalIncrement:
    """Stop when the primal increment falls below ``tol``.

    The criterion reads ``history["primal_increment"][-1] < tol``, i.e. the
    norm of ``u^k - u^{k-1}`` from the most recent proximal step.  Missing or
    empty history means "not converged" -- the first step has nothing to
    compare against.
    """

    tol: float

    def converged(self, k: int, alpha: float, alpha_prev: float, history: dict) -> bool:
        """Return whether the latest primal increment is below ``tol``."""
        increments = (history or {}).get("primal_increment")
        if not increments:
            return False
        return increments[-1] < self.tol


@dataclass(frozen=True)
class AlphaPlateau:
    """Stop once the proximal parameter stops growing.

    The criterion is ``k >= 2 and alpha == alpha_prev``, optionally restricted
    to a ``target`` value (``alpha == target``); with ``target=None`` any
    plateau triggers, which is the literal ``alpha_k = alpha_{k-1}``.  The
    sequence is capped (see :class:`Linear`), so a plateau arrives after
    finitely many steps; this is the termination rule used by the examples of
    hpG.

    Caveat.  That plateau arrives within a few steps, yet §6.2 of hpG reports
    24 Newton iterations over the whole run, which the two do not obviously
    reconcile.  The rule is therefore an option for the paper-faithful preset
    rather than the recorded default; the measured records use
    :class:`PrimalIncrement`.
    """

    target: float | None = None

    def converged(self, k: int, alpha: float, alpha_prev: float, history: dict) -> bool:
        """Return whether ``alpha`` has plateaued at (or above) step 2."""
        if k < 2 or alpha != alpha_prev:
            return False
        return self.target is None or alpha == self.target
