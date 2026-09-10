"""Legendre functions for the latent variable proximal point (LVPP) algorithm.

A *Legendre function* R : R^m -> R u {+inf} encodes the geometry of a closed
convex set C = dom R with nonempty interior (Rockafellar 1967).  Its conjugate
gradient map

    grad R* : R^m -> int C,   (grad R*)^{-1} = grad R

is an isomorphism between the dual image R^m and the interior of the feasible
image C.  In LVPP the latent variable psi lives in the dual image and the
pointwise constraint Bu(x) in C(x) is enforced *exactly* through the state
equation

    (Bu^k, w) - (grad R*(psi^k), w) = 0  for all w,

so the *reconstruction* grad R*(psi^k) is feasible by construction, even after
discretization, for any value of psi.

Each subclass of :class:`LegendreFunction` implements
:meth:`LegendreFunction.reconstruction`, the UFL expression for grad R*(psi).
Choosing a Legendre function is how a new *family* of constraints (bounds,
gradient constraints, eigenvalue constraints, ...) is added to LVPP; see the
constraint table in Section 2.3 and examples in Sections 3-4 of the reference

    J. S. Dokken, P. E. Farrell, B. Keith, I. P. A. Papadopoulos, and
    T. M. Surowiec, "The latent variable proximal point algorithm for
    variational problems with inequality constraints", arXiv:2503.05672 (2025).
"""

import typing

import ufl

__all__ = [
    "LegendreFunction",
    "ShannonLower",
    "ShannonUpper",
    "FermiDirac",
    "Hellinger",
    "GibbsSimplex",
]


class LegendreFunction(typing.Protocol):
    """Protocol for a Legendre function R with conjugate gradient map grad R*.

    A Legendre function encodes a closed convex feasible image C = dom R with
    nonempty interior.  Implementations hold whatever data defines C(x) (for
    example obstacle Functions or Constants) and expose the single map LVPP
    needs, the reconstruction grad R*(psi).

    The latent argument psi may be a Firedrake Function or a ``split()`` piece
    of a mixed Function; the returned UFL expression must have the shape of an
    element of int C.
    """

    def reconstruction(self, psi):
        """Return the UFL expression grad R*(psi), a point of int C."""
        ...


class ShannonLower:
    """Generalized Shannon entropy for unilateral lower bounds u >= phi.

    Feasible image C = [phi(x), oo) with

        R(a) = (a - phi) ln(a - phi) - (a - phi),

    whose conjugate gradient map is

        grad R*(a*) = phi + exp(a*)                          [paper (3.4)]

    As psi -> -exp(+inf) the reconstruction approaches the obstacle phi from
    above but never reaches it, so grad R*(psi) in int C always.
    """

    def __init__(self, obstacle):
        """``obstacle``: UFL expression / Constant / Function giving phi(x)."""
        self.obstacle = obstacle

    def reconstruction(self, psi):
        if psi.ufl_shape == ():
            return self.obstacle + ufl.exp(psi)
        # componentwise for vector-valued unknowns (bounds may be scalar)
        comps = []
        for i in range(psi.ufl_shape[0]):
            o = self.obstacle if self.obstacle.ufl_shape == () else self.obstacle[i]
            comps.append(o + ufl.exp(psi[i]))
        return ufl.as_vector(comps)

    def __repr__(self):
        return "ShannonLower(phi)"


class ShannonUpper:
    """Generalized Shannon entropy for unilateral upper bounds u <= phi.

    Feasible image C = (-oo, phi(x)] with

        R(a) = (phi - a) ln(phi - a) - (phi - a),

    whose conjugate gradient map is

        grad R*(a*) = phi - exp(-a*)

    (the sign-flipped Shannon map; cf. paper (3.22c) for the QVI example).
    """

    def __init__(self, obstacle):
        """``obstacle``: UFL expression / Constant / Function giving phi(x)."""
        self.obstacle = obstacle

    def reconstruction(self, psi):
        if psi.ufl_shape == ():
            return self.obstacle - ufl.exp(-psi)
        comps = []
        for i in range(psi.ufl_shape[0]):
            o = self.obstacle if self.obstacle.ufl_shape == () else self.obstacle[i]
            comps.append(o - ufl.exp(-psi[i]))
        return ufl.as_vector(comps)

    def __repr__(self):
        return "ShannonUpper(phi)"


class FermiDirac:
    """Generalized Fermi-Dirac entropy for bilateral bounds phi1 <= u <= phi2.

    Feasible image C = [phi1(x), phi2(x)] with

        R(a) = (a - phi1) ln(a - phi1) + (phi2 - a) ln(phi2 - a),

    whose conjugate gradient map is

        grad R*(a*) = (phi1 + phi2 exp(a*)) / (1 + exp(a*))  [paper (3.5)]

    We evaluate the algebraically identical, but numerically stable form

        grad R*(psi) = phi1 + (phi2 - phi1) * (1 + tanh(psi / 2)) / 2,

    which avoids overflow of exp(psi) for large |psi| (e.g. during Newton
    exploration steps with tight bounds).
    """

    def __init__(self, lower, upper):
        """``lower``, ``upper``: UFL expressions / Constants / Functions."""
        self.lower = lower
        self.upper = upper

    def reconstruction(self, psi):
        if psi.ufl_shape == ():
            return self.lower + (self.upper - self.lower) \
                * (1.0 + ufl.tanh(psi / 2.0)) / 2.0
        # componentwise for vector-valued unknowns (bounds may be scalar)
        comps = []
        for i in range(psi.ufl_shape[0]):
            lo = self.lower if self.lower.ufl_shape == () else self.lower[i]
            hi = self.upper if self.upper.ufl_shape == () else self.upper[i]
            comps.append(lo + (hi - lo) * (1.0 + ufl.tanh(psi[i] / 2.0)) / 2.0)
        return ufl.as_vector(comps)

    def __repr__(self):
        return "FermiDirac(phi1, phi2)"


class Hellinger:
    """Modified Hellinger entropy for gradient magnitude constraints |grad u| <= phi.

    Feasible image C = Euclidean ball of radius phi(x) > 0 in R^n with

        R(a) = -sqrt(phi^2 - |a|^2),

    whose conjugate gradient map is

        grad R*(a*) = phi a* / sqrt(1 + |a*|^2)              [paper (4.2)]

    The latent variable psi is vector-valued, with the geometric dimension of
    the domain.  (Used by gradient constraints, not by box constraints; see
    paper Section 4.1.  Provided here as part of the Legendre family.)
    """

    def __init__(self, obstacle):
        """``obstacle``: strictly positive radius phi(x), a scalar expression."""
        self.obstacle = obstacle

    def reconstruction(self, psi):
        if psi.ufl_shape == ():
            return self.obstacle * psi / ufl.sqrt(1.0 + psi**2)
        return self.obstacle * psi / ufl.sqrt(1.0 + ufl.dot(psi, psi))

    def __repr__(self):
        return "Hellinger(phi)"


class GibbsSimplex:
    """Boltzmann-Gibbs entropy for the Gibbs simplex sum_i u_i = 1, u_i >= 0.

    Feasible image C = {a in R^m : sum_i a_i = 1, a_i >= 0} with

        R(a) = sum_i a_i ln(a_i),

    whose conjugate gradient map is componentwise

        grad R*(a*)_i = exp(a*_i) / sum_j exp(a*_j)          [paper (3.17)]

    The latent variable psi is vector-valued with m components.  (Used by
    multiphase constraints; see paper Section 3.4.  Provided here as part of
    the Legendre family.)
    """

    def reconstruction(self, psi):
        m = psi.ufl_shape[0]
        e = ufl.exp(psi)
        total = e[0]
        for i in range(1, m):
            total = total + e[i]
        return e / total

    def __repr__(self):
        return "GibbsSimplex()"
