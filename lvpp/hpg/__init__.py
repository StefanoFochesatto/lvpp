"""hpG: the hierarchical Proximal-Galerkin (hpG) subsystem.

This subpackage holds what is specific to the framework of Papadopoulos,
*Hierarchical proximal Galerkin* (arXiv:2412.13733), which is the same
proximal point algorithm as LVPP on other spaces:

* :mod:`lvpp.hpg.spaces` -- the ``CG_p x DQ_{p-2}`` spectral discretization
  on tensor-product meshes, and the builders for uniform and graded ones;
* :mod:`lvpp.hpg.spectral` -- the per-cell spectral-Galerkin algebra, in any
  dimension ``d = 1, 2, 3``;
* :mod:`lvpp.hpg.twostage` -- the sequential ``P_F`` preconditioner of hpG
  section 4.4, with the alpha-free primal factorization cached for the whole
  nonlinear solve;
* :mod:`lvpp.hpg.solver` -- :class:`HPG`, the preset that binds the two.

The proximal loop of Papadopoulos (arXiv:2412.13733) is lvpp's
:class:`lvpp.solver.LVPP`, so nothing of it lives here: :class:`HPG` overrides
no method of :class:`~lvpp.solver.LVPP`, it chooses only the spaces and the
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
