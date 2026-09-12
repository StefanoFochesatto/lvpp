"""Meshes and finite-element spaces for the hpG discretization.

Tensor-product cells are a hard requirement
-------------------------------------------
The hpG latent operator is a cellwise spectral-Galerkin matrix: the modal mass
matrix and the cellwise factorizations of Shat are tensor products of 1D bases
over each cell (sections 3 and 7 of hpG).  An hpG mesh must therefore consist
of **tensor-product cells** -- intervals, quadrilaterals or hexahedra, so that
each cell's basis is the product of one 1D basis per axis.  For ``d >= 2`` that
means quad or hex cells; simplex meshes cannot be carried by the cellwise
algebra of :mod:`lvpp.hpg.spectral`, and :class:`HPGDiscretization` refuses
them on construction.

The spaces, and what the hierarchy is
-------------------------------------
The inf-sup stable pair of hpG section 4.1 is

    primal ``u``   in ``CG_p``      -- the obstacle unknown,
    latent ``psi`` in ``DQ_{p-2}``  -- the multiplier, cellwise discontinuous,

with the latent space in Firedrake's ``variant="spectral"`` flavour (the
Legendre-modal basis of section 3, whose mass matrix is diagonal -- Lemma 3.2
of hpG).  The degree pairing is the inf-sup stable one of Lemma B.3 in Keith &
Surowiec (2024), used for the obstacle-type constraints of LVPP here.

hpG presents ``u`` in a *hierarchical* basis (P1 "hats" plus Jacobi bubbles).
``CG_p`` **is** the polynomial space of that basis: the P1 hats together with
the Jacobi bubbles up to degree ``p`` span exactly ``P_p``, so standard nodal
``CG_p`` is the same discretization, and the hierarchy is a choice of basis.
What the hierarchical basis buys is the cellwise preconditioner of hpG section
4.4 (:mod:`lvpp.hpg.twostage`), never a different answer.

``p`` is fixed and uniform over the mesh here: this is high-order hpG with mesh
grading standing in for the AMR of the older scripts in ``experiments/hpg/``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from firedrake import BoxMesh, FunctionSpace, IntervalMesh, Mesh, RectangleMesh

__all__ = ["HPGDiscretization", "graded_axis", "latent_degree"]

_TENSOR_PRODUCT_CELLS = ("interval", "quadrilateral", "hexahedron")


def latent_degree(p: int, family: str = "obstacle") -> int:
    """The latent degree that pairs with primal degree ``p``, inf-sup stably.

    The obstacle-type pair of hpG section 4.1 -- pair ``CG_p`` with
    ``DQ_{p-2}`` -- is the default, and is the degree pairing that the inf-sup
    stability result (Lemma B.3 of Keith & Surowiec (2024)) admits.  The
    gradient-type pair of hpG section 4.2 pairs ``CG_p`` with ``DQ_{p-1}``
    instead, which changes ``E_beta`` and the cellwise Shat as well.

    ``family`` is ``"obstacle"`` or ``"gradient"``.  A degree that would come
    out negative is an error rather than a silent clamp: ``p = 1`` cannot
    carry an obstacle pair.

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

    The axis is a uniform fine *band* of ``max(2, round(0.12*n_total))`` cells
    centred on ``band_center``, with geometrically coarsening chains away from
    it on both sides.  Each chain fills its side exactly, so the axis has
    exactly ``n_total`` cells whatever grading is asked for: grading here is a
    redistribution of a fixed cell count, not a refinement of it.

    The band rule, the exact-fit geometric chains and the cell count are those
    of the older ``graded_axis`` in the hpG experiment scripts, with one
    correction: the shared per-cell growth factor is calibrated (monotone
    bisection) so that the measured grading really is the one requested.  A
    growth of ``ratio**(1/n)`` spans only ``ratio**((n-1)/n)`` across a chain,
    and the two sides have different lengths, so that older helper measured
    about 1.2 times the ratio it was given (26.41 for ``n=64, ratio=21.8``).

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

    # band width by the older rule: the finest cell size a chain of the nominal
    # band width (band_center +- band_halfwidth) would have at a growth of
    # ratio**(1/n), taken over the smaller of the two sides.  Fixing it here
    # holds the refined region at the requested place while the growth factor
    # is calibrated against the requested ratio.
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
    ``length`` exactly, so the finest is ``h`` and the coarsest ``h*g**(n-1)``.
    Rescaling by the sum is what makes a chain fill its side exactly."""
    if n_cells == 0:
        return np.zeros(0)
    sizes = growth ** np.arange(n_cells)
    return length * sizes / float(sizes.sum())


def _axis_sizes(n_band: int, n_left: int, n_right: int, lengths: tuple,
                band: float, growth: float) -> np.ndarray:
    """All ``n_band + n_left + n_right`` cell sizes, ordered left to right:
    the left chain coarsening away from the band, the uniform band, then the
    right chain.  The left chain is reversed because cells grow away from the
    band on both sides."""
    return np.concatenate((
        _chain_sizes(n_left, lengths[0], growth)[::-1],
        np.full(n_band, band),
        _chain_sizes(n_right, lengths[1], growth)))


def _calibrate_growth(n_band: int, n_left: int, n_right: int, lengths: tuple,
                      band: float, ratio: float) -> float:
    """Shared chain growth factor whose axis measures ``hmax/hmin == ratio``.

    ``hmax/hmin`` increases monotonically in the growth factor -- the chains
    coarsen while the band stays put -- which is what makes the bisection
    below monotone and its result deterministic.

    Growth 1 (uniform chains) already grades the axis, because the band is
    finer than either chain; the floor of ``grade`` is therefore the weakest
    grading this band geometry admits, and a request below it is unreachable
    and rejected rather than quietly rounded up.
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
    """``n`` cells per axis of ``[0, length]^dim``, with tensor-product cells:
    interval, quadrilateral or hexahedral, as hpG requires (sections 3 and
    7)."""
    if dim == 1:
        return IntervalMesh(n, length)
    if dim == 2:
        return RectangleMesh(n, n, length, length, quadrilateral=True)
    if dim == 3:
        return BoxMesh(n, n, n, length, length, length, hexahedral=True)
    raise ValueError(f"hpG supports dim in (1, 2, 3), got dim={dim}")


def _tensor_product_spaces(mesh, p: int) -> tuple:
    """The inf-sup stable pair ``(CG_p, DQ_{p-2} spectral)`` on ``mesh``, i.e.
    the tensor product of a continuous primal space with a cellwise
    discontinuous latent one (hpG section 4.1).

    On an interval the latent element is requested as ``"DG"``: Firedrake's
    ``"DQ"`` family is registered for interval cells but not implemented there
    (``ValueError: DQ is supported, but handled incorrectly``).  A one-factor
    tensor-product element *is* ``DG``, so this is the same space, and the
    ``spectral`` variant still makes the latent mass matrix diagonal.
    """
    family = "DG" if mesh.topological_dimension == 1 else "DQ"
    primal = FunctionSpace(mesh, "CG", p)
    latent = FunctionSpace(mesh, family, latent_degree(p), variant="spectral")
    return primal, latent


def _regrade(mesh, n: int, length: float, band_center: float,
             band_halfwidth: float, ratio: float) -> None:
    """Regrade every axis of a uniform tensor-product mesh, each independently
    but with the same parameters, so that the fine band runs through every
    coordinate direction.

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
    """An HPGDiscretization holds the mesh, the primal space and the latent
    space of one hpG discretization.

    ``primal`` is ``CG_p``, which carries the obstacle unknown, and ``latent``
    is ``DQ_{p-2}`` in the spectral (Legendre-modal) variant, which carries the
    multiplier: this is the inf-sup stable pair of hpG section 4.1.  ``CG_p``
    is the same polynomial space as the paper's hierarchical basis, so the
    hierarchy is a basis choice rather than another space (see the module
    docstring).

    The public API of an HPGDiscretization is:

      uniform():  ``n`` cells per axis over ``[0, length]^dim``, uniformly
            spaced -- an interval mesh in 1D, a quadrilateral mesh in 2D, a
            hexahedral box in 3D

      graded():  the same, with every axis geometrically graded toward a fine
            band, so that the mesh measures ``hmax/hmin == ratio``

      grading_ratio():  the measured ``hmax/hmin`` of the mesh

    The mesh's cells must be tensor-product cells (interval, quadrilateral or
    hexahedron), which the constructor enforces, because only such cells carry
    the cellwise spectral algebra of :mod:`lvpp.hpg.spectral`.  A typical call
    is:

    .. code-block:: python3

      disc = HPGDiscretization.graded(n=64, p=3, ratio=21.8)   # CG_3 x DQ_1 spectral
      u = Function(disc.primal)
      psi = Function(disc.latent)
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

        ``dim=1`` builds an interval mesh, ``dim=2`` a quadrilateral mesh --
        ``n x n`` over ``2.0 x 2.0`` with the default ``length``, the benchmark
        domain -- and ``dim=3`` a hexahedral box.
        """
        return cls._on(_tensor_product_box(n, length, dim), p)

    @classmethod
    def graded(cls, n: int, p: int, ratio: float, band_center: float = 0.9,
               band_halfwidth: float = 0.10, dim: int = 2,
               length: float = 2.0) -> HPGDiscretization:
        """``n`` cells per axis, geometrically graded toward a fine band.

        Each axis is graded independently by :func:`graded_axis`, all with the
        same parameters, so the band runs through every coordinate direction
        and the mesh measures ``hmax/hmin == ratio`` (:meth:`grading_ratio`).
        The cell count is untouched: grading redistributes cells, it does not
        add them.

        ``length`` defaults to the benchmark domain, the ``2.0 x 2.0`` square
        of :meth:`uniform`.  This is not the skeleton-based refinement of
        viamr, which produces triangles; hpG requires tensor-product cells.

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
        """Measured ``hmax/hmin`` of the mesh, i.e. how graded it actually is.

        The extents are read from the mesh's vertices -- the axis-aligned
        spacings between the distinct coordinate values of each axis -- which
        is exact for the structured tensor-product meshes built here: ``1.0``
        for :meth:`uniform`, the requested ``ratio`` for :meth:`graded`.
        """
        coords = self.mesh.coordinates.dat.data
        columns = ([coords] if coords.ndim == 1
                   else [coords[:, axis] for axis in range(coords.shape[1])])
        extents = np.concatenate([np.diff(np.unique(column))
                                  for column in columns])
        return float(extents.max() / extents.min())
