"""Meshes and finite-element spaces for the hpG discretization.

Tensor-product cells are a hard requirement
-------------------------------------------
The hpG latent operator is a *cellwise* spectral-Galerkin matrix: ``Â`` and
``Ŝ`` are tensor products of one-dimensional bases over each cell (paper
§3 and §7).  An hpG mesh must therefore consist of **tensor-product cells** --
intervals, quadrilaterals or hexahedra.  For ``d >= 2`` that means quad or hex
cells; simplex meshes are not supported by :mod:`lvpp.hpg.spectral`.
:class:`HPGDiscretization` refuses anything else on construction.

The spaces, and what the hierarchy is
-------------------------------------
The inf-sup stable pair of the paper (§4.1-§4.2) is

    primal ``u``   in ``CG_p``      -- the obstacle unknown,
    latent ``psi`` in ``DQ_{p-2}``  -- the multiplier, cellwise discontinuous,

with the latent space in Firedrake's ``variant="spectral"`` flavour (the
Legendre-modal basis of §3, whose mass matrix is diagonal, [60, Lem. B.3]).

The paper presents ``u`` in a *hierarchical* basis (P1 "hats" plus Jacobi
bubbles).  **``CG_p`` is the polynomial space of that basis**: P1 hats and
Jacobi bubbles up to degree ``p`` span exactly ``P_p``, so standard nodal
``CG_p`` is the same discretization -- the hierarchy is a *basis choice*.
What the hierarchical basis buys is the cellwise preconditioner of §4.5
(:mod:`lvpp.hpg.twostage`), never a different answer.

``p`` is fixed and uniform over the mesh here: this is high-order hpG with
mesh grading standing in for the AMR of the archived P1 experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from firedrake import BoxMesh, FunctionSpace, IntervalMesh, Mesh, RectangleMesh

__all__ = ["HPGDiscretization", "graded_axis", "latent_degree"]

_TENSOR_PRODUCT_CELLS = ("interval", "quadrilateral", "hexahedron")


def latent_degree(p: int, family: str = "obstacle") -> int:
    """Latent degree of the inf-sup stable hpG pair at primal degree ``p``.

    Obstacle-type pairs (paper §4.1, [60, Lem. B.3]) pair ``CG_p`` with
    ``DQ_{p-2}``; gradient-type pairs (paper §4.2, [35, Ex. 6]) pair it with
    ``DQ_{p-1}`` and change ``E_beta`` and ``Ŝ``.

    ``family`` is ``"obstacle"`` (default) or ``"gradient"``.  A degree that
    would come out negative is an error rather than a silent clamp: ``p = 1``
    cannot carry an obstacle pair.

    >>> [latent_degree(p) for p in (2, 3, 4)], latent_degree(3, "gradient")
    ([0, 1, 2], 2)
    """
    if family == "obstacle":
        if p < 2:
            raise ValueError(
                f"obstacle-type hpG pairs need p >= 2, got p={p} "
                "(the latent degree is p - 2)")
        return p - 2
    if family == "gradient":
        if p < 1:
            raise ValueError(
                f"gradient-type hpG pairs need p >= 1, got p={p} "
                "(the latent degree is p - 1)")
        return p - 1
    raise ValueError(
        f"unknown hpG constraint family {family!r}, "
        "expected 'obstacle' or 'gradient'")


def graded_axis(n_total: int, x_lo: float, x_hi: float, band_center: float,
                band_halfwidth: float, ratio: float) -> tuple:
    """One graded coordinate axis: ``n_total`` cells over ``[x_lo, x_hi]``.

    A uniform fine *band* of ``max(2, round(0.12*n_total))`` cells centred on
    ``band_center``, with geometrically coarsening chains away from it on both
    sides.  Every chain fills its side exactly, so the axis has exactly
    ``n_total`` cells whatever ``ratio`` is asked for.

    This is the archived ``hpg_common.graded_axis`` -- same band rule, same
    exact-fit geometric chains, same cell count -- with one correction: the
    shared per-cell growth factor is calibrated (monotone bisection) so that
    the measured grading really is the requested one.  The archived growth
    ``ratio**(1/n)`` spans only ``ratio**((n-1)/n)`` across a chain, and the
    two sides have different lengths, so the archived helper measured ~1.2x
    the ratio it was given (26.41 for ``n=64, ratio=21.8``).

    Returns ``(coords, hmin, hmax)``: the ``n_total + 1`` vertex coordinates
    and the extreme cell sizes.
    """
    n_band = max(2, round(0.12 * n_total))
    n_side = n_total - n_band
    if n_side < 2:
        raise ValueError(
            f"a graded axis needs at least one cell outside the band on each "
            f"side, got n_total={n_total} (band of {n_band}, {n_side} left)")
    n_left = n_side // 2
    n_right = n_side - n_left

    # Band width: the archived rule.  It is the finest cell size a chain of the
    # nominal band width (band_center -+ band_halfwidth) would have at the
    # archived growth ratio**(1/n), taken over the smaller of the two sides, so
    # the band is the finest part of the axis.  Fixing it up front keeps the
    # refined region at the requested place while the growth is calibrated.
    def nominal_finest(n_cells, length):
        return _chain_sizes(n_cells, length, ratio ** (1.0 / n_cells))[0]

    band = min(nominal_finest(n_left, (band_center - band_halfwidth) - x_lo),
               nominal_finest(n_right, x_hi - (band_center + band_halfwidth)))
    width = band * n_band

    lo = band_center - width / 2.0
    hi = band_center + width / 2.0
    lengths = (lo - x_lo, x_hi - hi)

    growth = _calibrate_growth(n_band, n_left, n_right, lengths, band, ratio)
    sizes = _axis_sizes(n_band, n_left, n_right, lengths, band, growth)
    coords = np.concatenate(([x_lo], x_lo + np.cumsum(sizes)))
    return coords, float(sizes.min()), float(sizes.max())


def _chain_sizes(n_cells: int, length: float, growth: float) -> np.ndarray:
    """Ascending geometric cell sizes ``h, h*g, ..., h*g**(n-1)`` that sum to
    ``length`` exactly (``h`` finest, ``h*g**(n-1)`` coarsest)."""
    if n_cells == 0:
        return np.zeros(0)
    sizes = growth ** np.arange(n_cells)
    return length * sizes / float(sizes.sum())


def _axis_sizes(n_band: int, n_left: int, n_right: int, lengths: tuple,
                band: float, growth: float) -> np.ndarray:
    """All ``n_band + n_left + n_right`` cell sizes, ordered left to right:
    the left chain coarsening away from the band, the band, the right chain."""
    return np.concatenate((
        _chain_sizes(n_left, lengths[0], growth)[::-1],
        np.full(n_band, band),
        _chain_sizes(n_right, lengths[1], growth)))


def _calibrate_growth(n_band: int, n_left: int, n_right: int, lengths: tuple,
                      band: float, ratio: float) -> float:
    """Shared chain growth factor whose axis measures ``hmax/hmin == ratio``.

    ``hmax/hmin`` increases monotonically in the growth factor -- the chains
    coarsen while the band stays put -- so a bisection is exact and robust.
    Growth 1 (uniform chains) already grades the axis, because the band is
    finer than either chain; a smaller request than that floor is unreachable
    with this band geometry and is rejected.
    """
    if ratio <= 1.0:
        raise ValueError(f"grading ratio must exceed 1, got {ratio:g}")

    def grade(growth: float) -> float:
        sizes = _axis_sizes(n_band, n_left, n_right, lengths, band, growth)
        return float(sizes.max() / sizes.min())

    floor = grade(1.0)
    if floor > ratio:
        raise ValueError(
            f"cannot grade this axis by {ratio:g}: a band of {n_band} cells "
            f"centred here admits no grading below {floor:.4g}; ask for a "
            "larger ratio, or use a uniform mesh")
    lo, hi = 1.0, ratio
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if grade(mid) < ratio:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _tensor_product_box(n: int, length: float, dim: int):
    """``n`` cells per axis of ``[0, length]^dim``, tensor-product cells."""
    if dim == 1:
        return IntervalMesh(n, length)
    if dim == 2:
        return RectangleMesh(n, n, length, length, quadrilateral=True)
    if dim == 3:
        return BoxMesh(n, n, n, length, length, length, hexahedral=True)
    raise ValueError(f"hpG supports dim in (1, 2, 3), got dim={dim}")


def _tensor_product_spaces(mesh, p: int) -> tuple:
    """``(CG_p, DQ_{p-2} spectral)`` on ``mesh``.

    On an interval the latent element is requested as ``"DG"``: Firedrake's
    ``"DQ"`` family is registered for interval cells but not implemented there
    (``ValueError: DQ is supported, but handled incorrectly``).  A one-factor
    tensor-product element *is* ``DG``, so this is the same space, and the
    ``spectral`` variant keeps the diagonal latent mass matrix.
    """
    family = "DG" if mesh.topological_dimension == 1 else "DQ"
    primal = FunctionSpace(mesh, "CG", p)
    latent = FunctionSpace(mesh, family, latent_degree(p), variant="spectral")
    return primal, latent


def _regrade(mesh, n: int, length: float, band_center: float,
             band_halfwidth: float, ratio: float) -> None:
    """Regrade every axis of a uniform tensor-product mesh, independently but
    with the same parameters.

    Vertices of a tensor-product mesh carry exactly ``n + 1`` distinct
    coordinate values per axis; each axis is relabelled by :func:`graded_axis`.
    """
    graded, _, _ = graded_axis(n, 0.0, length, band_center, band_halfwidth,
                               ratio)
    coords = mesh.coordinates.dat.data
    flat = coords.ndim == 1
    axes = range(1) if flat else range(coords.shape[1])
    for axis in axes:
        column = coords if flat else coords[:, axis]
        uniform = np.unique(column)
        if uniform.size != graded.size:
            raise ValueError(
                f"axis {axis} of this mesh has {uniform.size} distinct "
                f"coordinates, expected the {graded.size} (n + 1) of a "
                "structured tensor-product mesh")
        column[:] = graded[np.searchsorted(uniform, column)]


@dataclass(frozen=True)
class HPGDiscretization:
    """An hpG discretization: mesh + primal and latent spaces.

    ``primal`` is ``CG_p``, the obstacle unknown; ``latent`` is ``DQ_{p-2}``
    with the spectral (Legendre-modal) variant -- the inf-sup stable pair of
    paper §4.1.  ``CG_p`` *is* the polynomial space of the paper's
    hierarchical basis; the hierarchy is a basis choice, not another space
    (see the module docstring).

    Build with :meth:`uniform` or :meth:`graded`.  The mesh must have
    tensor-product cells (interval, quadrilateral or hexahedron), which the
    constructor enforces; see :meth:`grading_ratio` for the mesh spacing.
    """

    mesh: Mesh
    p: int
    primal: FunctionSpace
    latent: FunctionSpace

    def __post_init__(self):
        if self.p < 1:
            raise ValueError(f"primal degree must be >= 1, got p={self.p}")
        cell = self.mesh.ufl_cell().cellname
        if cell not in _TENSOR_PRODUCT_CELLS:
            raise ValueError(
                f"hpG needs tensor-product cells (paper sections 3 and 7), "
                f"got {cell!r}; expected one of {_TENSOR_PRODUCT_CELLS}")

    @classmethod
    def uniform(cls, n: int, p: int, length: float = 2.0,
                dim: int = 2) -> HPGDiscretization:
        """``n`` cells per axis, uniformly spaced, on ``[0, length]^dim``.

        ``dim=1`` builds an interval mesh, ``dim=2`` a quadrilateral mesh
        (the archived ``uniform_quad``: ``n x n`` over ``2.0 x 2.0``) and
        ``dim=3`` a hexahedral box.
        """
        return cls._on(_tensor_product_box(n, length, dim), p)

    @classmethod
    def graded(cls, n: int, p: int, ratio: float, band_center: float = 0.9,
               band_halfwidth: float = 0.10, dim: int = 2,
               length: float = 2.0) -> HPGDiscretization:
        """``n`` cells per axis, geometrically graded toward a fine band.

        Each axis is graded independently by :func:`graded_axis`, all with the
        same parameters, so the band runs through every coordinate direction;
        the mesh measures ``hmax/hmin == ratio`` (:meth:`grading_ratio`).

        This replaces the archive's ``graded_quad``, which was 2D-only and
        measured ~1.2x the ratio it was given.  As there, it is *not* viamr's
        SBR grading: SBR produces triangles, and hpG requires tensor-product
        cells.  ``length`` defaults to the archived benchmark domain, the
        ``2.0 x 2.0`` square of ``uniform``.

        Weak grading is limited by the band geometry (a band finer than its
        chains cannot grade the axis down to 1): a ``ratio`` below what ``n``
        admits is rejected rather than silently returning a mesh graded
        differently from the request.
        """
        mesh = _tensor_product_box(n, length, dim)
        _regrade(mesh, n, length, band_center, band_halfwidth, ratio)
        return cls._on(mesh, p)

    @classmethod
    def _on(cls, mesh, p: int) -> HPGDiscretization:
        primal, latent = _tensor_product_spaces(mesh, p)
        return cls(mesh=mesh, p=p, primal=primal, latent=latent)

    def grading_ratio(self) -> float:
        """Measured ``hmax/hmin`` of the mesh.

        Taken from the mesh's vertices -- the axis-aligned spacings between
        the distinct coordinate values of each axis -- which is exact for the
        structured tensor-product meshes built here: ``1.0`` for
        :meth:`uniform`, the requested ``ratio`` for :meth:`graded`.
        """
        coords = self.mesh.coordinates.dat.data
        columns = ([coords] if coords.ndim == 1
                   else [coords[:, axis] for axis in range(coords.shape[1])])
        extents = np.concatenate([np.diff(np.unique(column))
                                  for column in columns])
        return float(extents.max() / extents.min())
