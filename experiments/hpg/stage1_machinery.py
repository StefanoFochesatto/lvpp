"""Stage 1 gates: (a) alpha-independence of the primal block, (b) Shat SPD
at deep states, (c) cold inner-GMRES on the true Schur with the Shat PC.

Env guard REQUIRED:
PETSC_DIR=/home/stefano/firedrake/petsc PETSC_ARCH=arch-firedrake-default \
OMP_NUM_THREADS=1 venv-firedrake/bin/python stage1_machinery.py
"""
import time

import numpy as np

from firedrake import assemble
from hpg_common import (SolveMonitor, build_spaces, csr, field_ises_petsc,
                        make_lvpp, split_blocks_pet, uniform_quad)
from hpg_shat import SpectralGalerkin
from hpg_pc import HPGCtx, SShellCtx, ShatBlockPC

DIRECT = {
    "mat_type": "aij",
    "snes_type": "newtonls",
    "snes_linesearch_type": "l2",
    "snes_rtol": 1e-6,
    "snes_max_it": 100,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def alpha_gate():
    print("=== stage1 (a): alpha-independence of the primal block ===")
    mesh = uniform_quad(16)
    V, W = build_spaces(mesh, 2)
    lv = make_lvpp(mesh, 2, dict(DIRECT), "stage1_alpha")
    out = {}
    for a in (1.0, 5.0):
        lv._alpha.assign(a)
        Jfd = assemble(lv._J, mat_type="aij", bcs=lv._bcs)
        Jm = Jfd.M.handle
        Km = split_blocks_pet(Jm, V.dim(), W.dim())[0]
        out[a] = csr(Km)
        Km.destroy()
        Jfd.M.handle.destroy()
    K1, K5 = out[1.0], out[5.0]
    bc = np.asarray(lv._bcs[0].nodes, dtype=np.int64)
    interior = np.ones(K1.shape[0], dtype=bool)
    interior[bc] = False
    d_int = float(np.abs(K5[interior][:, interior]
                         - 5.0 * K1[interior][:, interior]).max())
    bc_rows_ok = (float(np.abs(np.asarray(K5[bc][:, bc].diagonal()).ravel()
                               - 1.0).max()) if bc.size else 0.0)
    print(f"  max |K(5) - 5 K(1)| on interior = {d_int:.2e}")
    print(f"  max |diag K(5) on bc rows - 1|  = {bc_rows_ok:.2e}")
    ok = d_int < 1e-10 and bc_rows_ok < 1e-10
    print("  ALPHA-INDEPENDENCE:", "PASS" if ok else "FAIL")
    return ok


def deep_and_shallow_states(p=2, level=0):
    """Direct control; returns (lv, Jm, Km, Bm, Dm, rhs_deep, rhs_shallow,
    alpha, V, W).  Deep = final linearization; shallow = first Newton rhs."""
    mesh = uniform_quad(16 * 2 ** level)
    V, W = build_spaces(mesh, p)
    lv = make_lvpp(mesh, p, dict(DIRECT), f"stage1_state_L{level}")
    mon = SolveMonitor()
    lv._solver.snes.ksp.setMonitor(mon)
    lv.solve(tol=1e-4, max_proximal_iterations=100)
    Jfd = assemble(lv._J, mat_type="aij", bcs=lv._bcs)
    Jm = Jfd.M.handle
    is0, is1 = field_ises_petsc(Jm, V.dim(), W.dim())
    Km = Jm.createSubMatrix(is0, is0)
    Bm = Jm.createSubMatrix(is0, is1)
    Dm = Jm.createSubMatrix(is1, is1)
    is0.destroy()
    is1.destroy()
    rhs_deep = mon.rhss[-1].copy()
    rhs_shallow = mon.rhss[0].copy()
    return lv, Jm, Km, Bm, Dm, rhs_deep, rhs_shallow, float(lv._alpha), V, W


def main():
    alpha_gate()
    print("\n=== stage1 (b)+(c): Shat SPD + cold S-solves, p=2 L0 ===")
    lv, Jm, Km, Bm, Dm, rhs_deep, rhs_shallow, alpha, V, W = \
        deep_and_shallow_states(p=2, level=0)
    sg = SpectralGalerkin(V.mesh(), W, 2, alpha=alpha)
    Dn = sg.cell_blocks(Dm)
    for beta in (0.0, 1e-5, 1e-4, 1e-3):
        sg.build_shat(Dn, beta=beta, check=True)
        print(f"  beta_shat={beta:g}: min eig(-Shat_c) = "
              f"{sg.min_eig_negS:.3e}  (SPD: {sg.min_eig_negS > 0})")
    ctx = HPGCtx(V.dim(), W.dim(), np.asarray(lv._bcs[0].nodes), sg,
                 alpha_getter=lambda: alpha)
    ctx.B = csr(Bm)
    ctx.BT = ctx.B.T.tocsr()
    ctx.D = csr(Dm)
    ctx._build_kspA_lu(Km, alpha)
    ctx._refresh_scale(alpha)
    from firedrake.petsc import PETSc
    from hpg_pc import SShellCtx, ShatBlockPC
    for tag, rhs in (("shallow(1st newton)", rhs_shallow),
                     ("deep(converged)", rhs_deep)):
        xr = rhs.getArray(readonly=True).copy()
        b1, b2 = xr[:ctx.n0], xr[ctx.n0:]
        y2 = b2 - ctx.BT.dot(ctx.Ainv(b1))
        sg.build_shat(sg.cell_blocks(Dm), beta=0.0)
        matS = PETSc.Mat().createPython((ctx.n1, ctx.n1), comm=Jm.comm)
        matS.setPythonContext(SShellCtx(ctx))
        matS.setUp()
        ksp = PETSc.KSP().create(comm=Jm.comm)
        ksp.setType("gmres")
        ksp.setTolerances(rtol=1e-6, max_it=500)
        ksp.setGMRESRestart(500)
        ksp.setOperators(matS)
        pin = ksp.getPC()
        pin.setType("python")
        pin.setPythonContext(ShatBlockPC(ctx))
        ksp.setUp()
        b = matS.createVecLeft()
        x = matS.createVecRight()
        b.setArray(np.ascontiguousarray(y2))
        t0 = time.time()
        ksp.solve(b, x)
        wall = time.time() - t0
        r = matS.createVecLeft()
        matS.mult(x, r)
        r.axpy(-1.0, b)
        rel = float(r.norm() / max(b.norm(), 1e-300))
        print(f"  cold S-solve [{tag}]: inner-its="
              f"{ksp.getIterationNumber()} rel={rel:.2e} wall={wall:.2f}s "
              f"||y2||={float(b.norm()):.2e}")
        ksp.destroy()
        matS.destroy()
    print("STAGE1 DONE")


if __name__ == "__main__":
    main()
