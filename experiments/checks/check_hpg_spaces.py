"""Self-check for :mod:`lvpp.hpg.spaces`: meshes, hpG spaces and refusals.

:class:`HPGDiscretization` pairs the primal ``CG_p`` space with a latent
discontinuous spectral space of degree ``p - 2``, the pair used by the hpG
solver, and can grade the mesh along one axis.  The expected dimensions below
are those of that pair; the 21.8 grading is the request of the first graded
level (64 cells per axis, ``g4``) of the hpG results in
``experiments/hpg/RESULTS.md``.

Checks
  (a) dims of ``CG_p x DQ_{p-2}`` on the 16x16 quad mesh, p = 2 and 4;
  (b) the graded mesh: measured hmax/hmin within 5% of the requested 21.8,
      every cell a quadrilateral, band exactly the uniform fine block;
  (c) the 1D (interval) and 3D (hexahedral) builders;
  (d) the documented refusals: simplex mesh, p < 2, unreachable grading.

Run from the repository root with the env guard:

  PETSC_DIR=/home/stefano/firedrake/petsc PETSC_ARCH=arch-firedrake-default \
  OMP_NUM_THREADS=1 /home/stefano/firedrake/venv-firedrake/bin/python \
  experiments/checks/check_hpg_spaces.py
"""

import numpy as np
from firedrake import assemble, dx

from lvpp.hpg import HPGDiscretization, latent_degree


def check_dimensions():
    """(a) space dimensions on the archived 16x16 quadrilateral benchmark."""
    print("(a) uniform meshes: primal CG_p and latent DQ_{p-2} spectral")
    expected = {2: (1089, 256), 4: (4225, 2304)}
    for p, (primal_dim, latent_dim) in expected.items():
        disc = HPGDiscretization.uniform(16, p)
        got = (disc.primal.dim(), disc.latent.dim())
        print(f"    uniform(16, p={p}): primal.dim()={got[0]} "
              f"latent.dim()={got[1]}  (DQ degree {latent_degree(p)})")
        assert disc.p == p
        assert got == (primal_dim, latent_dim), (p, got, (primal_dim, latent_dim))
        assert disc.grading_ratio() == 1.0, disc.grading_ratio()
    print(f"    latent_degree: {[latent_degree(p) for p in (2, 3, 4)]}, "
          f"gradient(3)={latent_degree(3, 'gradient')}")


def check_graded():
    """(b) measured grading, cell type and band structure of graded(64, 2)."""
    ratio = 21.8
    n = 64
    disc = HPGDiscretization.graded(n, p=2, ratio=ratio)
    measured = disc.grading_ratio()
    mesh = disc.mesh
    cells = mesh.num_cells()
    cell = mesh.ufl_cell().cellname
    x = np.unique(mesh.coordinates.dat.data[:, 0])
    y = np.unique(mesh.coordinates.dat.data[:, 1])
    sizes = np.diff(x)
    band_cells = max(2, round(0.12 * n))
    band = np.isclose(sizes, sizes.min())
    print(f"(b) graded(n={n}, p=2, ratio={ratio}): "
          f"measured hmax/hmin = {measured:.6f}")
    print(f"    hmin={sizes.min():.6g} hmax={sizes.max():.6g} "
          f"cells={cells} cell={cell!r} vertices/axis={x.size}")
    print(f"    uniform band: {int(band.sum())} cells of the finest size "
          f"(requested {band_cells})")
    assert abs(measured - ratio) / ratio < 0.05, (measured, ratio)
    assert cell == "quadrilateral", cell
    assert cells == n * n, (cells, n * n)
    assert x.size == y.size == n + 1, (x.size, y.size)
    assert int(band.sum()) == band_cells, int(band.sum())
    # the regraded vertices must be the ones the assembly sees
    area = assemble(1 * dx(domain=mesh))
    print(f"    domain measure = {area:.12g} (2.0 x 2.0 = 4.0)")
    assert abs(area - 4.0) < 1e-12, area


def check_other_dimensions():
    """(c) the 1D and 3D builders, plus graded meshes in 1D/3D."""
    print("(c) dimension generality")
    one_d = HPGDiscretization.uniform(4, 2, dim=1)
    print(f"    uniform(4, p=2, dim=1): cell={one_d.mesh.ufl_cell().cellname!r} "
          f"primal.dim()={one_d.primal.dim()} latent.dim()={one_d.latent.dim()}")
    assert one_d.mesh.ufl_cell().cellname == "interval"
    assert one_d.primal.dim() == 9 and one_d.latent.dim() == 4

    three_d = HPGDiscretization.uniform(2, 2, dim=3)
    print(f"    uniform(2, p=2, dim=3): cell={three_d.mesh.ufl_cell().cellname!r} "
          f"primal.dim()={three_d.primal.dim()} latent.dim()={three_d.latent.dim()}")
    assert three_d.mesh.ufl_cell().cellname == "hexahedron"
    assert three_d.primal.dim() == 125 and three_d.latent.dim() == 8

    for dim, n in ((1, 64), (3, 8)):
        disc = HPGDiscretization.graded(n, p=2, ratio=21.8, dim=dim)
        print(f"    graded(n={n}, p=2, ratio=21.8, dim={dim}): "
              f"measured {disc.grading_ratio():.6f} "
              f"cell={disc.mesh.ufl_cell().cellname!r} "
              f"cells={disc.mesh.num_cells()}")
        assert abs(disc.grading_ratio() - 21.8) / 21.8 < 0.05
        assert disc.mesh.num_cells() == n ** dim


def check_rejections():
    """The documented refusals: simplex mesh, p < 2, unreachable grading."""
    print("(d) rejections")
    from firedrake import UnitSquareMesh
    for what, fn in (
            ("simplex mesh", lambda: HPGDiscretization(
                mesh=UnitSquareMesh(2, 2), p=2,
                primal=None, latent=None)),
            ("p = 1 obstacle pair", lambda: latent_degree(1)),
            ("unknown family", lambda: latent_degree(2, "cubic")),
            ("unreachable grading", lambda: HPGDiscretization.graded(64, 2, 1.2)),
            ("no room for chains", lambda: HPGDiscretization.graded(3, 2, 30.0))):
        try:
            fn()
        except ValueError as exc:
            print(f"    {what}: ValueError({exc})")
        else:
            raise AssertionError(f"{what} was not rejected")


if __name__ == "__main__":
    check_dimensions()
    check_graded()
    check_other_dimensions()
    check_rejections()
    print("\ncheck_hpg_spaces: all checks passed")
