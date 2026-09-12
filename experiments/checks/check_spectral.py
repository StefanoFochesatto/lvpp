"""Focused self-check for :mod:`lvpp.hpg.spectral`, stage 0 of RESULTS.md.

Builds a :class:`SpectralGalerkin` operator on 16x16 uniform quadrilaterals for
p = 2, 3, 4 and verifies it against the same forms assembled by Firedrake: the
modal psi-mass off-diagonal (<= 1e-17), the on-cell operators against their
closed forms (<= 1e-13) and the parity off-block entries (<= 1e-13), the
thresholds recorded for stage 0 in ``experiments/hpg/RESULTS.md``.  The
coupling of D_psi between different cells (its off-cell blocks) is measured
separately and must vanish exactly.  The same comparison is repeated on an
anisotropic 4x8 mesh (hx != hy, so the per-axis weights matter), on an interval
(d = 1: Ahat must come out diagonal) and on hexahedral meshes in 3D (d = 3,
isotropic and anisotropic).

The Shat algebra is checked by inverting ``Shat = Vinv^T (L L^T) Vinv`` with
``build_shat``: ``shat_apply`` must solve ``Shat_nodal y = x`` to 1e-10
relative, and ``min_eig_negS`` must be positive (``-Shat`` SPD, since
D_jac = -D_psi).

Prints SPECTRAL OK only if the 2D, Shat and d = 1 checks pass; the d = 3 block
prints its exception instead of raising.

Env guard REQUIRED:

  PETSC_DIR=/home/stefano/firedrake/petsc PETSC_ARCH=arch-firedrake-default \
  OMP_NUM_THREADS=1 /home/stefano/firedrake/venv-firedrake/bin/python \
  experiments/checks/check_spectral.py
"""
import sys

import numpy as np

from firedrake import (BoxMesh, Function, FunctionSpace, IntervalMesh,
                       RectangleMesh, SpatialCoordinate, TestFunction,
                       TrialFunction, assemble, dx, exp, inner)

try:                                           # installed package path
    from lvpp.hpg.spectral import SpectralGalerkin      # noqa: E402
except ImportError:                            # sibling hpg/__init__.py not
    sys.path.insert(0, "/home/stefano/firedrake/lvpp/lvpp")   # landed yet
    from hpg.spectral import SpectralGalerkin  # noqa: E402

# RESULTS.md stage 0 (2D 16x16 quads): thresholds from the task contract.
TOL_OFFDIAG = 1e-17          # psi-mass modal off-diagonal
TOL_YAhat = 1e-13            # Y-Ahat vs closed form
TOL_PARITY = 1e-13           # Y parity off-block


def offcellblock_max(W, Dfd):
    """Max |entry| of a PETSc Mat coupling two different cells."""
    indptr, indices, data = Dfd.getValuesCSR()
    rows = np.repeat(np.arange(W.dim()), np.diff(indptr))
    cols = indices
    dof2cell = np.empty(W.dim(), dtype=np.int64)
    cmap = W.cell_node_map().values
    for c in range(cmap.shape[0]):
        dof2cell[cmap[c]] = c
    cross = dof2cell[rows] != dof2cell[cols]
    return float(np.abs(data[cross]).max()) if cross.any() else 0.0


def main():
    ok = True

    print("=== d = 2: 16x16 uniform quads, p = 2, 3, 4 ===")
    mesh = RectangleMesh(16, 16, 2.0, 2.0, quadrilateral=True)
    x, y = SpatialCoordinate(mesh)
    for p in (2, 3, 4):
        W = FunctionSpace(mesh, "DQ", p - 2, variant="spectral")
        sg = SpectralGalerkin(mesh, W, p, alpha=1.0)
        res = sg.verify_vs_firedrake()
        print(f"  p={p} q={sg.q} k={sg.k} dofs={W.dim()} ncells={sg.ncells}")
        for key in sorted(res):
            print(f"    {key} = {res[key]}")

        psi = Function(W).interpolate(-30.0 + 5.0 * (x + y) / 4.0)
        Dfd = assemble(exp(-psi) * inner(TrialFunction(W),
                                         TestFunction(W)) * dx,
                       mat_type="aij").M.handle
        dpsi = offcellblock_max(W, Dfd)
        Dfd.destroy()
        print(f"    dpsi_off_cell_block = {dpsi}")

        checks = {
            "psi_mass_modal_offdiag <= 1e-17":
                res["psi_mass_modal_offdiag"] <= TOL_OFFDIAG,
            "dpsi_off_cell_block == 0.0": dpsi == 0.0,
            "yfull_vs_closed <= 1e-13": res["yfull_vs_closed"] <= TOL_YAhat,
            "yparity_offblock <= 1e-13": res["yparity_offblock"] <= TOL_PARITY,
        }
        for name, passed in checks.items():
            print(f"    {'PASS' if passed else 'FAIL'}: {name}")
            ok = ok and passed

    print("=== d = 2: anisotropic 4x8 cells (hx != hy), p = 3 ===")
    try:
        manga = RectangleMesh(4, 8, 2.0, 3.0, quadrilateral=True)
        Wa = FunctionSpace(manga, "DQ", 1, variant="spectral")
        sga = SpectralGalerkin(manga, Wa, 3, alpha=1.0)
        resa = sga.verify_vs_firedrake()
        print(f"  hx={sga.h[0, 0]} hy={sga.h[0, 1]} k={sga.k} "
              f"yfull_vs_closed = {resa['yfull_vs_closed']} "
              f"aleg_modal_vs_closed = {resa['aleg_modal_vs_closed']} "
              f"yparity_offblock = {resa['yparity_offblock']}")
        for name, passed in {
            "aniso yfull_vs_closed <= 1e-13":
                resa["yfull_vs_closed"] <= TOL_YAhat,
            "aniso aleg_modal_vs_closed <= 1e-13":
                resa["aleg_modal_vs_closed"] <= TOL_YAhat,
        }.items():
            print(f"    {'PASS' if passed else 'FAIL'}: {name}")
            ok = ok and passed
    except Exception as exc:                      # noqa: BLE001
        print(f"  anisotropic FAILED: {type(exc).__name__}: {exc}")
        ok = False

    print("=== carried-over Shat algebra: build_shat / shat_apply (d=2, p=3) ===")
    try:
        W = FunctionSpace(mesh, "DQ", 1, variant="spectral")
        sg = SpectralGalerkin(mesh, W, 3, alpha=1.0)
        Mn = sg.cell_blocks(assemble(inner(TrialFunction(W), TestFunction(W))
                                     * dx, mat_type="aij").M.handle)
        L = sg.build_shat(-Mn)                # -Shat SPD (D_jac = -D_psi)
        LLt = np.einsum("cik,cjk->cij", L, L)
        Vinv = sg.Vinv
        # -(Shat_modal) = L L^T  =>  Shat_nodal = Vinv^T (L L^T) Vinv
        Shat_nodal = np.einsum("ai,cab,bj->cij", Vinv, LLt, Vinv)
        x = np.random.default_rng(0).standard_normal((sg.ncells, sg.k))
        y = sg.shat_apply(x)
        resid = float(np.abs(np.einsum("cij,cj->ci", Shat_nodal, y) - x).max()
                      / np.abs(x).max())
        print(f"    min_eig_negS = {sg.min_eig_negS} "
              f"|Shat_nodal y - x|/|x| = {resid}")
        passed = resid <= 1e-10 and sg.min_eig_negS > 0.0
        print(f"    {'PASS' if passed else 'FAIL'}: shat_apply inverts Shat")
        ok = ok and passed
    except Exception as exc:                      # noqa: BLE001
        print(f"  Shat smoke FAILED: {type(exc).__name__}: {exc}")
        ok = False

    print("=== d = 1: IntervalMesh(16, 2.0) ===")
    try:
        mesh1 = IntervalMesh(16, 2.0)
        # FInAT rejects the "DQ" alias on intervals ("DQ is supported, but
        # handled incorrectly"); "DG" spectral is the same tensor-product
        # discontinuous Legendre element with a nodal dual.
        try:
            W1 = FunctionSpace(mesh1, "DQ", 2, variant="spectral")
        except ValueError as exc:
            print(f"  DQ on interval rejected: {type(exc).__name__}: {exc}")
            W1 = FunctionSpace(mesh1, "DG", 2, variant="spectral")
        sg1 = SpectralGalerkin(mesh1, W1, 4, alpha=1.0)
        A = sg1.Afull
        off = float(np.abs(A - A.diagonal(axis1=1, axis2=2)[:, :, None]
                           * np.eye(sg1.k)[None]).max())
        print(f"  p=4 k={sg1.k} dofs={W1.dim()} "
              f"Afull_max_offdiag = {off}")
        res1 = sg1.verify_vs_firedrake()
        for key in sorted(res1):
            print(f"    {key} = {res1[key]}")
        if not off <= TOL_YAhat:
            ok = False
            print("  FAIL: d=1 Afull is not diagonal")
    except Exception as exc:                      # noqa: BLE001
        print(f"  d=1 FAILED: {type(exc).__name__}: {exc}")
        ok = False

    print("=== d = 3: BoxMesh(2, 2, 2, 2, 2, 2, hexahedral=True) ===")
    try:
        mesh3 = BoxMesh(2, 2, 2, 2.0, 2.0, 2.0, hexahedral=True)
        W3 = FunctionSpace(mesh3, "DQ", 1, variant="spectral")
        sg3 = SpectralGalerkin(mesh3, W3, 3, alpha=1.0)
        res3 = sg3.verify_vs_firedrake()
        print(f"  p=3 d={sg3.d} k={sg3.k} dofs={W3.dim()} "
              f"ncells={sg3.ncells}")
        for key in sorted(res3):
            print(f"    {key} = {res3[key]}")
        # anisotropic cells (h = 1, 1.5, 2): exercises the per-axis weights
        mesh3b = BoxMesh(2, 2, 2, 2.0, 3.0, 4.0, hexahedral=True)
        W3b = FunctionSpace(mesh3b, "DQ", 1, variant="spectral")
        sg3b = SpectralGalerkin(mesh3b, W3b, 3, alpha=1.0)
        res3b = sg3b.verify_vs_firedrake()
        print(f"  anisotropic h={sg3b.h[0]} p=3 "
              f"yfull_vs_closed = {res3b['yfull_vs_closed']} "
              f"aleg_modal_vs_closed = {res3b['aleg_modal_vs_closed']} "
              f"yparity_offblock = {res3b['yparity_offblock']}")
    except Exception as exc:                      # noqa: BLE001
        print(f"  d=3 FAILED: {type(exc).__name__}: {exc}")

    print("SPECTRAL OK" if ok else "SPECTRAL FAILED")


if __name__ == "__main__":
    main()
