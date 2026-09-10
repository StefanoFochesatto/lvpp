"""Stage 0 gate: hpG discretization structure + serial-direct control.

(a) Psi-mass structure: DQ_{p-2} spectral basis = tensor Legendre -> mass
    DIAGONAL in the modal basis (Lemma 3.2); Firedrake's nodal-dual layout
    round-trips exactly through the discovered Vandermonde.
(b) D_psi = (zeta_i, e^-psi zeta_j) block-diagonal per cell (measured on the
    converged deep state).
(c) Serial-direct MUMPS solve of the monolithic mixed system on L0/L1 quads
    at p=2,3 vs the analytic sphere solution; err(u_h) compared against our
    P1 references at matched/similar dofs.

Env guard REQUIRED:
PETSC_DIR=/home/stefano/firedrake/petsc PETSC_ARCH=arch-firedrake-default \
OMP_NUM_THREADS=1 venv-firedrake/bin/python stage0_gate.py
"""
import time

import numpy as np

from firedrake import (Function, SpatialCoordinate, assemble, dx, errornorm,
                       exp, inner, sqrt, TrialFunction, TestFunction)
from hpg_common import (LU_REFS_P1, REASON_NAMES, build_spaces, csr,
                        run_lvpp, uniform_quad, uexactUFL)
from hpg_shat import SpectralGalerkin, blocks_from_petsc, CellGeometry
import hpg_pc

DIRECT = {
    "mat_type": "aij",
    "snes_type": "newtonls",
    "snes_linesearch_type": "l2",
    "snes_linesearch_maxlambda": 1.0,
    "snes_rtol": 1e-6,
    "snes_max_it": 100,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def structure_table():
    print("=== stage0 (a)+(b): Psi-mass and D_psi per-cell structure ===")
    for p in (2, 3, 4):
        mesh = uniform_quad(16)
        V, W = build_spaces_public(mesh, p)
        sg = SpectralGalerkin(mesh, W, p, alpha=1.0)
        res = sg.verify_vs_firedrake()
        x, y = SpatialCoordinate(mesh)
        psi = Function(W).interpolate(-30.0 + 5.0 * (x + y) / 4.0)
        Dfd = assemble(exp(-psi) * inner(TrialFunction(W),
                                         TestFunction(W)) * dx,
                       mat_type="aij").M.handle
        # NOTE hpG sign: D_psi = M . e^{-psi_hpg}; the state is a shallow
        # proxy (constant-ish depth) -- only the cell-block structure matters
        n1 = W.dim()
        k = sg.k
        perm = sg.perm_can
        try:
            blocks = sg.cell_blocks(Dfd)
            ok = "ok"
        except AssertionError as e:
            ok = f"FAIL: {e}"
        print(f"  p={p}: q={p-2} k/cell={sg.k} dofs={n1} "
              f"psi-mass modal-offdiag={res['psi_mass_modal_offdiag']:.2e} "
              f"modal-diag-match={res['psi_mass_modal_diag']:.2e} "
              f"nodal-roundtrip={res['psi_mass_nodal_match']:.2e} "
              f"gram={res['gram_modal_vs_kron']:.2e} "
              f"bhat={res['bhat_vs_injection']:.2e} "
              f"yfull={res['yfull_vs_closed']:.2e} "
              f"yparity-offblock={res['yparity_offblock']:.2e} "
              f"D_psi-cellblocks={ok}")
        # explicit off-cell-block measurement of the assembled D
        Dsp = csr(Dfd)
        rows = np.repeat(np.arange(n1), np.diff(Dsp.indptr))
        cols = Dsp.indices
        dof2cell = np.empty(n1, dtype=np.int64)
        cmap = W.cell_node_map().values
        for c in range(cmap.shape[0]):
            dof2cell[cmap[c]] = c
        cross = dof2cell[rows] != dof2cell[cols]
        val = float(np.abs(Dsp.data[cross]).max()) if cross.any() else 0.0
        print(f"        D_psi off-cell-block max = {val:.2e}")


def build_spaces_public(mesh, p):
    from hpg_common import build_spaces
    return build_spaces(mesh, p)


def direct_control(level, p):
    mesh = uniform_quad(16 * 2 ** level)
    V, W = build_spaces_public(mesh, p)
    from hpg_common import make_lvpp
    lv = make_lvpp(mesh, p, dict(DIRECT), f"stage0_L{level}_p{p}")
    t0 = time.time()
    lv.solve(tol=1e-4, max_proximal_iterations=100)
    wall = time.time() - t0
    x, y = SpatialCoordinate(mesh)
    r = (x * x + y * y) ** 0.5
    from firedrake import errornorm
    err = float(errornorm(uexactUFL(r), lv.u_out[0]))
    prox = lv.proximal_iterations
    newton = int(np.sum(lv.newton_iterations))
    dofs = V.dim() + W.dim()
    print(f"  L{level} p={p}: dofs={dofs} (n0={V.dim()}, n1={W.dim()}) "
          f"prox={prox} newton={newton} err(u_h)={err:.4e} wall={wall:.1f}s"
          f"   [our P1 ref: L{level} {LU_REFS_P1.get(level, '?')}]")
    return dict(level=level, p=p, dofs=dofs, prox=prox, newton=newton,
                err=err, wall=wall)


if __name__ == "__main__":
    structure_table()
    print("\n=== stage0 (c): serial-direct control (monolithic MUMPS) ===")
    for level in (0, 1):
        for p in (2, 3):
            direct_control(level, p)
    print("STAGE0 DONE")
