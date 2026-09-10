"""Stage 2/3 promotion driver for the hpG port (HPG_PORT_SPEC.md stages 2-3).

Attaches the gate-verified HPGTwoStagePC (paper arXiv:2412.13733v4 eq 4.5,
sequential P_F) to end-to-end LVPP solves on the hpG quad discretization,
following the archive promob.py promotion pattern (python PC on snes.ksp,
KSP monitor for outer-its curves, failure_reason diagnostics).  No existing
file is modified; this file is the only new artifact.

One machinery-level incompatibility is handled HERE (driver-side, machinery
untouched): HPGTwoStagePC.setUp calls ctx.sg.cell_blocks(ctx.D) after ctx.D
has been converted to a scipy csr_matrix, while the stage-1-verified path
passed the raw PETSc latent block.  SGDriver dispatches on input type
(PETSc Mat -> original path; scipy csr -> equivalent per-cell extraction;
D_psi is exactly cell-block-diagonal per stage 0, so dropping off-cell
entries is a no-op).

Stages (argv[1]):
  uniform   L0-L2 uniform quads x p in {2,4}: per (level,p) serial-direct
            MUMPS control + two-stage run (published arm; p=4 capped by
            wall caps after the diag-established stall) + our capped-inexact
            arm (inner rtol 1e-4, cap 40) at p=4.
  uniform2  stage 2 restricted to p=2 (budgeted).
  uniform4  stage 2 restricted to p=4 (budgeted).
  beta      beta_Shat sweep {0, 1e-5, 1e-4, 1e-3} at L0 p=2.
  graded    graded_quad band chain (g4/g5/g6/g7 ~ ratios 22/45/94/187x) at
            p=2, published vs capped-inexact arms; optional tag list:
            `graded g4 g5` etc.  Divergence check at the finest level run.
  diag      bounded-cost diagnostic of the L0 p=4 published-arm stall.

Env guard REQUIRED (all invocations):
  PETSC_DIR=/home/stefano/firedrake/petsc PETSC_ARCH=arch-firedrake-default \
  OMP_NUM_THREADS=1 /home/stefano/firedrake/venv-firedrake/bin/python \
    experiments/hpg/promote_hpg.py <stage> [args]   (cwd = lvpp root)

Serial only.  petsc4py idiom: x.getArray(readonly=True).copy()/y.setArray.
"""
import argparse
import gc
import os
import statistics
import sys
import time

import numpy as np

if (os.environ.get("OMP_NUM_THREADS") != "1"
        or os.environ.get("PETSC_DIR") != "/home/stefano/firedrake/petsc"
        or os.environ.get("PETSC_ARCH") != "arch-firedrake-default"):
    sys.exit("env guard: PETSC_DIR/PETSC_ARCH/OMP_NUM_THREADS required "
             "(see module docstring)")

from firedrake import SpatialCoordinate, errornorm, sqrt
from firedrake.petsc import PETSc

import hpg_pc
from hpg_common import (LU_REFS_P1, REASON_NAMES, SolveMonitor, graded_quad,
                        make_lvpp, uexactUFL, uniform_quad)
from hpg_pc import HPGCtx, HPGTwoStagePC
from hpg_shat import SpectralGalerkin

print = PETSc.Sys.Print

T_START = time.time()


def head(s):
    print(f"\n=== [{time.time() - T_START:7.1f}s] {s} "
          + "=" * max(4, 60 - len(s)), flush=True)


# Two-stage config: paper eq 4.5 -- outer FGMRES (restart 250) on the
# monolithic mixed Jacobian, PC = sequential P_F python PC.  E_beta = 0 in
# the operator (obstacle case; beta enters only Shat via ctx.beta_shat).
SP_TWOSTAGE = {
    "mat_type": "matfree",
    "pmat_type": "aij",
    "snes_type": "newtonls",
    "snes_linesearch_type": "l2",
    "snes_rtol": 1e-6,
    "snes_max_it": 100,
    "ksp_type": "fgmres",
    "ksp_rtol": 1e-6,
    "ksp_max_it": 1000,
    "ksp_gmres_restart": 250,
    "pc_type": "python",
}

# Serial-direct control (stage0/stage1 monolithic MUMPS pattern).
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


def failure_reason(exc, lv=None):
    reason = ""
    if lv is not None:
        try:
            snes = lv._solver.snes
            reason = (f" [snes={snes.getConvergedReason()} "
                      f"ksp={snes.ksp.getConvergedReason()}]")
        except Exception:
            pass
    msg = str(exc).strip().replace("\n", " ")
    return f"{type(exc).__name__}: {msg[:220]}{reason}"


def group_seqs(seqs, counts):
    """Split a flat list of per-solve records into per-prox groups."""
    out, i = [], 0
    for n in counts:
        out.append(seqs[i:i + n])
        i += n
    return out, seqs[i:]


def flatl(chunks):
    """Flatten one level: list of segments -> list of ints."""
    return [i for ch in chunks for i in ch]


def flat2(chunks):
    """Flatten two levels: list of prox groups (lists of segments) -> ints."""
    return [i for grp in chunks for seg in grp for i in seg]


class SGDriver(SpectralGalerkin):
    """Attaches HPGTwoStagePC end-to-end without touching the gate-verified
    machinery.  HPGTwoStagePC.setUp calls ctx.sg.cell_blocks(ctx.D) where
    ctx.D has ALREADY been converted to a scipy csr_matrix (setUp keeps the
    csr copy for D_mult); the stage-1 path passed the raw PETSc latent block
    instead, so the csr route was never exercised.  Dispatch on input type:
    PETSc Mat -> original machinery path; scipy csr -> equivalent per-cell
    block extraction (D_psi is exactly cell-block-diagonal, stage-0
    bit-exact, so dropping off-cell entries is a no-op)."""

    def cell_blocks(self, mat):
        if hasattr(mat, "getValuesCSR"):
            return super().cell_blocks(mat)
        indptr, indices, data = mat.indptr, mat.indices, mat.data
        k = self.k
        g2cl = {}
        for c in range(self.ncells):
            for l, g in enumerate(self.perm_can[c * k:(c + 1) * k]):
                g2cl[int(g)] = (c, l)
        rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
        rc = np.array([g2cl.get(int(g), (-1, -1)) for g in rows])
        cc = np.array([g2cl.get(int(g), (-1, -1)) for g in indices])
        mask = (rc[:, 0] == cc[:, 0]) & (rc[:, 0] >= 0)
        out = np.zeros((self.ncells, k, k))
        out[rc[mask, 0], rc[mask, 1], cc[mask, 1]] = data[mask]
        return out


def metrics(lv, mesh):
    x, y = SpatialCoordinate(mesh)
    r = sqrt(x * x + y * y)
    err, prox, newton = float("nan"), -1, -1
    try:
        err = float(errornorm(uexactUFL(r), lv.u_out[0]))
        prox = int(lv.proximal_iterations)
        newton = int(sum(lv.newton_iterations or []))
    except Exception:
        pass
    return err, prox, newton


def run_direct(mesh, p, name, max_prox=100):
    """Serial-direct control at the same (level, p): monolithic MUMPS LU."""
    lv = make_lvpp(mesh, p, dict(DIRECT), name)
    t0 = time.time()
    failed = ""
    try:
        lv.solve(tol=1e-4, max_proximal_iterations=max_prox)
    except Exception as e:
        failed = failure_reason(e, lv)
    wall = time.time() - t0
    err, prox, newton = metrics(lv, mesh)
    del lv
    gc.collect()
    return dict(p=p, prox=prox, newton=newton, err=err, wall=wall,
                failed=failed)


def run_two_stage(mesh, p, name, inner_rtol=1e-6, inner_maxit=500,
                  beta_shat=0.0, max_prox=100, progress=0, wall_cap=0.0,
                  run_cap=0.0):
    """End-to-end LVPP solve with HPGTwoStagePC (paper eq 4.5) on snes.ksp.

    progress: print a heartbeat every N PC applies (0 = off).
    wall_cap: abort one linear solve after this many seconds (0 = off).
    run_cap:  abort the whole run after this many seconds (0 = off).
    """
    lv = make_lvpp(mesh, p, dict(SP_TWOSTAGE), name)
    V0 = lv._z.subfunctions[0].function_space()
    W = lv._z.subfunctions[1].function_space()
    n0, n1 = V0.dim(), W.dim()
    sg = SGDriver(mesh, W, p, alpha=float(lv._alpha))
    ctx = HPGCtx(n0, n1, np.asarray(lv._bcs[0].nodes), sg,
                 alpha_getter=(lambda: 1.0),
                 inner_rtol=inner_rtol, inner_maxit=inner_maxit,
                 beta_shat=beta_shat)

    # alpha_getter reads the LIVE lvpp Constant and keeps the LU arm's
    # D_alpha scaling current (the K0 factor itself is alpha-free, built
    # once per mesh; this is the paper's sec-4.4 cache trick).
    def alpha_getter():
        a = float(lv._alpha)
        ctx._refresh_scale(a)
        return a
    ctx.alpha_getter = alpha_getter

    # Instrumentation: one segment per linear solve (each PC setUp), holding
    # the inner-GMRES its of every PC apply in that linear solve.
    segs = []
    pcctx = HPGTwoStagePC()
    orig_setup = pcctx.setUp
    t_solve = [time.time()]
    t0_run = [time.time()]

    def pc_setUp(pc, orig=orig_setup):
        segs.append([])
        t_solve[0] = time.time()
        if progress:
            print(f"      [prog] solve#{len(segs)} alpha="
                  f"{float(lv._alpha):.3g}", flush=True)
        orig(pc)
    pcctx.setUp = pc_setUp

    orig_inner = ctx.inner_solve

    def inner_w(y2, orig=orig_inner):
        r = orig(y2)
        its = int(ctx.inner_its_per_apply[-1])
        segs[-1].append(its)
        if progress and ctx.n_applies % progress == 0:
            print(f"      [prog] applies={ctx.n_applies} lastinner={its} "
                  f"innerksp={ctx.inner_ksp.getConvergedReason()} "
                  f"solve_wall={time.time() - t_solve[0]:6.0f}s", flush=True)
        if wall_cap and time.time() - t_solve[0] > wall_cap:
            raise RuntimeError(
                f"diagnostic wall cap {wall_cap:.0f}s hit on one linear "
                f"solve (applies={ctx.n_applies}, lastinner={its}, "
                f"innerksp={ctx.inner_ksp.getConvergedReason()})")
        if run_cap and time.time() - t0_run[0] > run_cap:
            raise RuntimeError(
                f"diagnostic run cap {run_cap:.0f}s hit "
                f"(applies={ctx.n_applies}, lastinner={its})")
        return r
    ctx.inner_solve = inner_w

    hpg_pc.CTX = ctx
    pc = lv._solver.snes.ksp.getPC()
    pc.setType(PETSc.PC.Type.PYTHON)
    pc.setPythonContext(pcctx)

    mon = SolveMonitor()
    lv._solver.snes.ksp.setMonitor(mon)

    t0 = time.time()
    t0_run[0] = t0
    failed = ""
    try:
        lv.solve(tol=1e-4, max_proximal_iterations=max_prox)
    except Exception as e:
        failed = failure_reason(e, lv)
    wall = time.time() - t0
    if ctx.bad:
        failed += " | HPGTwoStagePC produced non-finite PC output"

    err, prox, newton = metrics(lv, mesh)
    curve, maxits, medits = [], 0, float("nan")
    inner_avg = float("nan")
    inner_tot = 0
    i_first = (float("nan"), 0)
    i_deep = (float("nan"), 0)
    try:
        nits = list(lv.newton_iterations or [])
        outer_groups, leftover = mon.per_prox(nits)
        all_groups = outer_groups + ([leftover] if leftover else [])
        flatits = flatl(all_groups)
        curve = [max(ch) if ch else 0 for ch in outer_groups]
        maxits = max(flatits) if flatits else 0
        medits = statistics.median(flatits) if flatits else float("nan")
        iproxs, ileft = group_seqs(segs, nits)
        inner_all = flat2(iproxs + ([ileft] if ileft else []))
        if inner_all:
            inner_avg = statistics.mean(inner_all)
            inner_tot = int(sum(inner_all))
        if iproxs:
            f = flatl(iproxs[0])
            if f:
                i_first = (statistics.mean(f), max(f))
            d = flatl(iproxs[-1])
            if d:
                i_deep = (statistics.mean(d), max(d))
    except Exception as e:
        failed += " | metric extraction: " + failure_reason(e)

    res = dict(tag=name, p=p, n0=n0, n1=n1, prox=prox, newton=newton,
               maxits=maxits, medits=medits, curve=curve, err=err, wall=wall,
               inner_avg=inner_avg, inner_tot=inner_tot,
               inner_first=i_first, inner_deep=i_deep,
               n_applies=ctx.n_applies, n_setup=ctx.n_setup,
               setup_wall=ctx.setup_wall, apply_wall=ctx.apply_wall,
               min_eig=sg.min_eig_negS, failed=failed)
    del lv
    gc.collect()
    return res


def print_two_stage_row(tag, res):
    f0, m0 = res["inner_first"]
    print(f"  {tag:>16} {res['n0']:>7} {res['n1']:>6} {res['prox']:>4} "
          f"{res['newton']:>6} {res['maxits']:>6} {res['medits']:>6.1f} "
          f"{res['err']:>9.3e} {res['wall']:>6.1f}s "
          f"{res['inner_avg']:>8.1f}/{res['inner_tot']:>7} "
          f"{res['n_setup']:>3}x{res['setup_wall']:>5.1f}s "
          f"{'ok' if not res['failed'] else 'FAIL':>9}"
          + (f"  FAILED: {res['failed']}" if res["failed"] else ""), flush=True)


ROW_U = (f"  {'tag':>16} {'u-dofs':>7} {'psi':>6} {'prox':>4} {'newton':>6} "
         f"{'maxits':>6} {'medits':>6} {'err(u_h)':>9} {'wall':>7} "
         f"{'innAvg':>8} {'innTot':>7} {'PCsetup':>9} {'status':>9}")


def stage_uniform_levels(levels, ps):
    head(f"stage 2: uniform quads levels={list(levels)} p={list(ps)}")
    print(f"  LU_REFS_P1 (our P1 floorless, cross-discr.): {LU_REFS_P1}")
    print(ROW_U)
    for level in levels:
        n = 16 * 2 ** level
        mesh = uniform_quad(n)
        for p in ps:
            d = run_direct(mesh, p, f"dir_L{level}_p{p}")
            print(f"  {'direct ctrl':>16} {'-':>7} {'-':>6} {d['prox']:>4} "
                  f"{d['newton']:>6} {'-':>6} {'-':>6} {d['err']:>9.3e} "
                  f"{d['wall']:>6.1f}s {'-':>8} {'-':>7} {'-':>9} "
                  f"{'-':>9}"
                  + (f"  FAILED: {d['failed']}" if d["failed"] else ""),
                  flush=True)
            res = run_two_stage(mesh, p, f"2stage_L{level}_p{p}",
                                progress=(50 if p == 4 else 0),
                                wall_cap=(240.0 if p == 4 else 0.0),
                                run_cap=(420.0 if p == 4 else 0.0))
            print_two_stage_row(res["tag"], res)
            if p == 4:
                resc = run_two_stage(mesh, p, f"2stagecap_L{level}_p{p}",
                                     inner_rtol=1e-4, inner_maxit=40,
                                     progress=50, run_cap=700.0)
                print_two_stage_row(resc["tag"], resc)
        del mesh
        gc.collect()


def stage_uniform():
    stage_uniform_levels((0, 1, 2), (2, 4))


def stage_uniform2():
    stage_uniform_levels((0, 1, 2), (2,))


def stage_uniform4():
    stage_uniform_levels((0, 1, 2), (4,))


def stage_beta():
    head("stage 3.1: beta_Shat sweep, L0 p=2 (worst-case first prox + deep)")
    print(f"  {'beta':>7} {'minEig(-Shat)@1st':>18} {'prox':>4} {'newton':>6} "
          f"{'outer max':>9} {'inner 1st-prox avg/max':>23} "
          f"{'inner deep avg/max':>19} {'wall':>8}  notes")
    for beta in (0.0, 1e-5, 1e-4, 1e-3):
        res = run_two_stage(uniform_quad(16), 2, f"beta{beta:g}_L0p2",
                            beta_shat=beta)
        f0, m0 = res["inner_first"]
        d0, dm = res["inner_deep"]
        notes = (f"FAILED: {res['failed']}" if res["failed"] else
                 "clean solve; deep-state eig ref stage1: "
                 "8.5e-7/1.0e-6/2.4e-6/1.6e-5 ladder")
        me = res["min_eig"]
        me = float("nan") if me is None else me
        print(f"  {beta:>7g} {me:>18.3e} {res['prox']:>4} "
              f"{res['newton']:>6} {res['maxits']:>9} "
              f"{f0:>10.1f}/{m0:>5.0f} {d0:>8.1f}/{dm:>4.0f} "
              f"{res['wall']:>7.1f}s  {notes}", flush=True)


def stage_graded(which=None):
    head("stage 3.2: graded_quad chain p=2 (band analogue) + divergence check")
    print(f"  {'tag':>16} {'cells':>5} {'ratio':>7} {'arm':>13} "
          f"{'prox':>4} {'newton':>6} {'maxits':>6} {'medits':>6} "
          f"{'err(u_h)':>9} {'wall':>7} {'inner 1st|deep avg/max':>23} "
          f" status")
    all_levels = {"g4": (64, 21.8), "g5": (80, 45.3), "g6": (96, 93.5),
                  "g7": (112, 187.0)}
    tags = which or ["g4", "g5", "g6", "g7"]
    for tag in tags:
        n, ratio = all_levels[tag]
        mesh, meas = graded_quad(n, ratio=ratio)
        for arm, rtol, cap in (("published", 1e-6, 500),
                               ("capped-inx", 1e-4, 40)):
            res = run_two_stage(mesh, 2, f"{tag}_{arm}",
                                inner_rtol=rtol, inner_maxit=cap,
                                progress=(25 if n >= 96 else 0),
                                run_cap=420.0)
            f0, m0 = res["inner_first"]
            d0, dm = res["inner_deep"]
            print(f"  {res['tag']:>16} {n * n:>5} {meas:>7.1f}x "
                  f"r{rtol:g}/c{cap:>3} {res['prox']:>4} {res['newton']:>6} "
                  f"{res['maxits']:>6} {res['medits']:>6.1f} "
                  f"{res['err']:>9.3e} {res['wall']:>6.1f}s "
                  f"{f0:>9.1f}/{m0:>3.0f}|{d0:>4.1f}/{dm:>3.0f} "
                  f"{'ok' if not res['failed'] else 'FAIL':>7}"
                  + (f"  FAILED: {res['failed']}" if res["failed"] else ""),
                  flush=True)
        del mesh
        gc.collect()
    head(f"stage 3.2 graded chain done ({','.join(tags)})")


def stage_diag():
    """Bounded-cost diagnostic of the L0 p=4 two-stage stall."""
    head("diag: L0 p=4 two-stage, per-apply progress, caps")
    res = run_two_stage(uniform_quad(16), 4, "diag_L0p4", progress=10,
                        wall_cap=240.0, run_cap=420.0)
    print_two_stage_row(res["tag"], res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["uniform", "uniform2", "uniform4",
                                      "beta", "graded", "diag"])
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    if a.stage == "uniform":
        stage_uniform()
    elif a.stage == "uniform2":
        stage_uniform2()
    elif a.stage == "uniform4":
        stage_uniform4()
    elif a.stage == "beta":
        stage_beta()
    elif a.stage == "graded":
        stage_graded(a.args or None)
    else:
        stage_diag()
    print(f"[{time.time() - T_START:7.1f}s] {a.stage} DONE")


if __name__ == "__main__":
    main()