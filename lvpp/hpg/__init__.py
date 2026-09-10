"""hpG: the hierarchical Proximal-Galerkin subsystem.

This package owns everything specific to the paper's framework
(Papadopoulos, *Hierarchical proximal Galerkin*, arXiv:2412.13733v4):

* :mod:`lvpp.hpg.spaces` -- the tensor-product ``CG_p x DQ_{p-2}`` spectral
  discretization and its mesh builders;
* :mod:`lvpp.hpg.spectral` -- the per-cell spectral-Galerkin algebra
  (dimension-generic: ``d = 1, 2, 3``);
* :mod:`lvpp.hpg.twostage` -- the sequential ``P_F`` preconditioner
  (eq. 4.5) with the cached alpha-independent primal factorization;
* :mod:`lvpp.hpg.solver` -- :class:`HPG`, the thin preset that binds the two.

Nothing in the proximal algorithm lives here: :class:`HPG` overrides no
behaviour of :class:`lvpp.solver.LVPP`, it only chooses the spaces and the
preconditioner.
"""

from lvpp.hpg.solver import HPG
from lvpp.hpg.spaces import HPGDiscretization, graded_axis, latent_degree
from lvpp.hpg.spectral import (CellGeometry, SpectralGalerkin,
                               blocks_from_petsc, discover_vandermonde)
from lvpp.hpg.twostage import HPGTwoStage, SP_TWOSTAGE

__all__ = [
    "HPG",
    "HPGTwoStage",
    "SP_TWOSTAGE",
    "HPGDiscretization",
    "graded_axis",
    "latent_degree",
    "SpectralGalerkin",
    "CellGeometry",
    "blocks_from_petsc",
    "discover_vandermonde",
]
