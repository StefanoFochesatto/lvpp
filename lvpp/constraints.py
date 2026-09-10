"""Constraint objects for LVPP: the operator B, feasible images C(x), and
constraint-related diagnostics.

A :class:`Constraint` bundles everything the LVPP solver needs to know about
one inequality constraint acting on one unknown:

  * :meth:`Constraint.latent_space` -- the FunctionSpace W for the latent
    variable psi (a vector space over the same mesh),
  * :meth:`Constraint.coupling_form` -- the weak term (psi, B du) and its
    proximal shift (psi_prev, B du) added to the unknown's residual equation
    (LVPP equation (2.7a)),
  * :meth:`Constraint.state_form` -- the weak equation (Bu, dw)
    - (grad R*(psi), dw) which ties the unknown to the latent variable (LVPP
    equation (2.7b)),
  * :meth:`Constraint.observable_space` / :meth:`Constraint.observable` --
    the space and UFL expression for the bound-preserving reconstruction
    grad R*(psi),
  * optional diagnostics: feasibility, complementarity, and dual feasibility.

Box constraints (lb <= u <= ub, possibly with lb or ub absent) are implemented
by :class:`BoxConstraint` on top of the Shannon / Fermi-Dirac Legendre
functions.  Further constraint families (gradient constraints via Hellinger,
multiphase via GibbsSimplex, Signorini boundary constraints, eigenvalue
constraints via matrix tanh) follow the same interface; see Section 2.3 and
Table 1 of the LVPP reference paper.
"""

import typing

import ufl
from ufl.algorithms import replace

from .legendre import FermiDirac, ShannonLower, ShannonUpper

__all__ = [
    "Constraint",
    "BoxConstraint",
]


def _pointwise_max(expr, zero=0.0):
    """Return max(expr, 0) pointwise; componentwise for vector expressions."""
    if expr.ufl_shape == ():
        return ufl.conditional(expr > 0.0, expr, zero)
    pos = None
    for i in range(expr.ufl_shape[0]):
        c = ufl.conditional(expr[i] > 0.0, expr[i], zero)
        pos = c if pos is None else pos + c
    return pos


class Constraint:
    """Abstract base class for one LVPP inequality constraint on one unknown.

    Subclasses must implement :meth:`latent_space`, :meth:`coupling_form`,
    :meth:`state_form`, and (for the bound-preserving output)
    :meth:`observable_space` and :meth:`observable`.  Diagnostics methods have
    harmless defaults and may be overridden.

    Constraints may hold data (bounds, obstacles) that references user
    unknowns (quasi-variational inequalities); :meth:`remap` returns a copy
    with that data rewritten through a coefficient mapping, which is how the
    LVPP solver binds user functions into the mixed system.
    """

    def __init__(self, legendre):
        self.legendre = legendre

    def latent_space(self, V):
        """Return the latent space W for an unknown living in V."""
        raise NotImplementedError

    def coupling_form(self, V, du, psi, psi_prev):
        """Return the 1-form (psi, B du) - (psi_prev, B du), du the test function."""
        raise NotImplementedError

    def state_form(self, V, u, dw, psi):
        """Return the 1-form (Bu, dw) - (grad R*(psi), dw), dw the latent test function."""
        raise NotImplementedError

    def observable_space(self, V):
        """Return the space in which the reconstruction grad R*(psi) lives."""
        return V

    def observable(self, psi):
        """Return the UFL expression grad R*(psi) in observable_space."""
        return self.legendre.reconstruction(psi)

    # Optional diagnostics ---------------------------------------------------

    def feasibility_forms(self, V, u):
        """Return a list of scalar 0-forms measuring primal constraint violation.

        ``V`` is the unknown's FunctionSpace (for the measure); ``u`` may be a
        Function or a ``split()`` piece of the mixed solution.
        """
        return []

    def complementarity_form(self, V, u, psi, psi_prev, alpha):
        """Return a scalar 0-form measuring complementarity, or None."""
        return None

    def dual_feasibility_form(self, V, psi, psi_prev, alpha):
        """Return a scalar 0-form measuring dual-variable progress, or None."""
        return None

    # Rewriting user data ----------------------------------------------------

    def remap(self, mapping):
        """Return an equivalent constraint with data replaced via ``mapping``.

        ``mapping`` is a dict {Coefficient: expression} passed to
        ``ufl.algorithms.replace``.  Constants are left untouched.
        """
        return self

    @staticmethod
    def _remap_expr(expr, mapping):
        # scalars (float/int) and Constants carry no user unknowns: pass through
        if (expr is None or isinstance(expr, ufl.Constant)
                or not isinstance(expr, ufl.core.expr.Expr)):
            return expr
        return replace(expr, mapping)


class BoxConstraint(Constraint):
    """Pointwise box constraint lb(x) <= u(x) <= ub(x) for one unknown.

    Either bound may be omitted (``None``), giving unilateral constraints.
    Bounds may be Constants, Functions on the unknown's space, or arbitrary
    UFL scalar expressions -- including expressions involving other unknowns
    of the same LVPP problem, which yields obstacle-type quasi-variational
    inequalities (paper Section 3.5).

    The Legendre function is chosen automatically:

      * both bounds  -> :class:`FermiDirac` (stable tanh form),
      * lower only   -> :class:`ShannonLower`,
      * upper only   -> :class:`ShannonUpper`,

    unless the user supplies ``legendre`` explicitly.  For vector-valued
    unknowns the bounds act componentwise.
    """

    def __init__(self, lower=None, upper=None, legendre=None, psi_space=None):
        if lower is None and upper is None:
            raise ValueError("BoxConstraint needs at least one of lower, upper")
        if legendre is None:
            if lower is not None and upper is not None:
                legendre = FermiDirac(lower, upper)
            elif lower is not None:
                legendre = ShannonLower(lower)
            else:
                legendre = ShannonUpper(upper)
        super().__init__(legendre)
        self.lower = lower
        self.upper = upper
        self.psi_space_override = psi_space
        if lower is not None and upper is not None:
            self.kind = "bilateral"
        elif lower is not None:
            self.kind = "lower"
        else:
            self.kind = "upper"

    def latent_space(self, V):
        """Latent space: a copy of V by default (equal-order discretization,
        which satisfies the compatibility condition of Keith & Surowiec 2024,
        Section 4.7).  Override with ``psi_space`` at construction."""
        if self.psi_space_override is not None:
            return self.psi_space_override
        return V

    def coupling_form(self, V, du, psi, psi_prev):
        """(psi, du) - (psi_prev, du): B = id for box constraints."""
        dx = ufl.Measure("dx", domain=V.mesh())
        return ufl.inner(psi, du) * dx - ufl.inner(psi_prev, du) * dx

    def state_form(self, V, u, dw, psi):
        """(u, dw) - (grad R*(psi), dw): Bu = u for box constraints."""
        dx = ufl.Measure("dx", domain=V.mesh())
        return ufl.inner(u, dw) * dx - ufl.inner(self.observable(psi), dw) * dx

    # Diagnostics ------------------------------------------------------------

    def _violation(self, u):
        """Pointwise violation expressions (not yet integrated)."""
        viol = []
        for bound, sign in ((self.lower, -1.0), (self.upper, +1.0)):
            if bound is None:
                continue
            if u.ufl_shape == ():
                if sign < 0.0:
                    viol.append(ufl.conditional(u < bound, bound - u, 0.0))
                else:
                    viol.append(ufl.conditional(u > bound, u - bound, 0.0))
            else:
                for i in range(u.ufl_shape[0]):
                    b = bound if bound.ufl_shape == () else bound[i]
                    if sign < 0.0:
                        viol.append(ufl.conditional(u[i] < b, b - u[i], 0.0))
                    else:
                        viol.append(ufl.conditional(u[i] > b, u[i] - b, 0.0))
        return viol

    def feasibility_forms(self, V, u):
        """Integral of max(0, lb - u) plus integral of max(0, u - ub)."""
        dx = ufl.Measure("dx", domain=V.mesh())
        return [v * dx for v in self._violation(u)]

    def complementarity_form(self, V, u, psi, psi_prev, alpha):
        """(psi_prev - psi) / alpha paired with u; vanishes at the solution."""
        dx = ufl.Measure("dx", domain=V.mesh())
        return ufl.inner((psi_prev - psi) / alpha, u) * dx

    def dual_feasibility_form(self, V, psi, psi_prev, alpha):
        """Integral of the positive part of the dual increment / alpha.

        For a lower bound the dual variable psi only needs to climb (paper's
        dolfinx reference uses max(0, (psi - psi_prev) / alpha)); for an upper
        bound it only needs to descend; for bilateral bounds both directions
        count, giving |psi - psi_prev| / alpha.
        """
        dx = ufl.Measure("dx", domain=V.mesh())
        dpsi = (psi - psi_prev) / alpha
        if self.kind == "lower":
            pos = _pointwise_max(dpsi)
        elif self.kind == "upper":
            pos = _pointwise_max(-dpsi)
        else:
            pos = _pointwise_max(dpsi) + _pointwise_max(-dpsi)
        return pos * dx

    # Rewriting user data ----------------------------------------------------

    def remap(self, mapping):
        """Rewrite bounds through ``mapping`` (binds user unknowns into the
        mixed system; required for QVI-type bounds)."""
        return BoxConstraint(
            lower=self._remap_expr(self.lower, mapping),
            upper=self._remap_expr(self.upper, mapping),
            legendre=self.legendre,
            psi_space=self.psi_space_override,
        )

    def __repr__(self):
        return f"BoxConstraint({self.kind}: {self.legendre!r})"
