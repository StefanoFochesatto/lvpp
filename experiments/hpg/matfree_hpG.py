"""TRUE-matfree hpG solver driver (one new file; no existing file edited).

Goal (session deliverable): remove every GLOBAL factorization from the hpG
solve path and benchmark the p-scaling / graded arms against the
assembled+cached-Cholesky arm measured at the SAME states.

What "truly matfree" means here (the deliverable definition):
  Allowed:  assembly of small/local matrices for PRECONDITIONER
            construction (hpG's own claim is sparsity-promoting, not
            kernel-only).  Forbidden: any solve-path dependence on global
            factorizations (no cached MUMPS/Cholesky of K, no global LU
            anywhere in the apply path).

The five components:
1. K-block (A-action):  CG + GAMG on the alpha-FREE K0 = D_alpha @ A(alpha)
   = stiffness + bc-identity rows (stage-1-verified machine-precision
   alpha-independence, max |K(5)-5K(1)| = 3.6e-15).  ONE GAMG hierarchy is
   built from the assembled K0 (preconditioner construction from an
   assembled operator -- allowed, no factorization; PETSc's GAMG coarse
   solvers live on the tiny coarse hierarchy and are part of AMG setup,
   standard practice, documented here deliberately).  Every apply is a
   FRESH capped CG solve:
       A(alpha)^-1 b = K0^-1 (D_alpha b),  D_alpha from ctx.scale_vec.
   The A-solve is therefore a VARIABLE-accuracy inner iteration (its vary
   with rhs/alpha/state); the outer FGMRES tolerates that by design (PC
   variability is what FGMRES is for).  Tolerance discipline (documented,
   measured):
       A-CG rtol 1e-8, cap 60 -- tight because the A-solve feeds BOTH the
       true-S matvec (S v = D v - B^T K^-1 (B v): an inexact K^-1 inside a
       matvec perturbs the operator seen by the inner GMRES; 1e-8 is far
       below the inner rtol 1e-4 so the perturbation is negligible) and the
       P_F backward solve.  A-CG its per call and per outer apply are
       recorded; non-converged (reason<0) calls are counted.
2. True-S matvec:  the existing matfree SShellCtx (S v = D v - B^T A^-1 (B v))
   is reused UNCHANGED -- now with the CG+GAMG Ainv above.  B/D/B^T applies
   are scipy-csr matvecs off the per-setUp assembled blocks (matvec-only in
   the apply path; the block extraction is the allowed preconditioner
   assembly).
3. Shat PC:  batched CELLWISE Cholesky of -Shat_c (hpg_shat.shat_apply) --
   cell-local dense k x k (k <= 9 at p=4, k=25 at p=6) factorizations; NO
   global factorization; allowed by definition.
4. E_beta decoupling:  operator G has E_beta = 0 ALWAYS (lvpp lower-bound
   residual, our Finding-1 rule; obstacle case beta=0 in the paper too).
   beta enters ONLY the Shat preconditioner (ctx.beta_shat, default 0).
   Safety-margin datum (measured, stage-1/RESULTS.md): min eig(-Shat_c) at
   the deep converged state = 8.5e-7 / 1.0e-6 / 2.4e-6 / 1.6e-5 for
   beta_Shat = 0 / 1e-5 / 1e-4 / 1e-3 -- beta=0 stays SPD, and the beta
   sweep (promote_hpg stage beta) showed beta=0 viable end-to-end.
5. Inner Krylov:  GMRES (NOT CG -- our band_l5 proximal-stall datum) on the
   true Schur, rtol 1e-4 cap 40 (the capped-inexact discipline that rescued
   the published p=4 wedge), PC = ShatBlockPC.  Outer FGMRES rtol 1e-6,
   restart 250, on the monolithic matfree G (firedrake action) with the
   sequential P_F python PC (paper eq 4.5).

Wall attribution:  PC-setup wall (block split + Shat chol + inner ksp setUp
+ ONE GAMG hierarchy build), outer-PC-apply wall, inner-S wall (true-S
GMRES incl. the in-shell A-CG time), A-CG wall split shell/direct, A-CG
its totals.  The residual (solve wall - setup - apply) is the outer FGMRES
operator action + SNES residual/linesearch cost, not attributable to the
PC.

Arms compared AT THE SAME STATES (the fair high-order comparison; both arms
share the identical outer FGMRES + capped-inner-GMRES discipline and differ
ONLY in the A-action: matfree CG+GAMG vs cached MUMPS-LU of K0):
  - matfree : MatfreeTwoStagePC (this file)
  - assembled-cache : promote_hpg.run_two_stage (HPGTwoStagePC, a_action lu)

Env guard REQUIRED (every invocation):
  PETSC_DIR=/home/stefano/firedrake/petsc PETSC_ARCH=arch-firedrake-default \
  OMP_NUM_THREADS=1 /home/stefano/firedrake/venv-firedrake/bin/python \
    experiments/hpg/matfree_hpG.py <stage>   (cwd = lvpp root)

Stages:  smoke (L0 p2 sanity) | correct (L0/L1 p2 + direct controls)
       | scale (L0/L1 x p2/p4, both arms + directs) | graded48 | graded80
       | p6 (L0 p6 stretch) | all
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

from firedrake.petsc import PETSc

import hpg_pc
from hpg_common import (SolveMonitor, graded_quad, make_lvpp, uniform_quad)
from hpg_pc import HPGCtx, HPGTwoStagePC, split_blocks_pet
from promote_hpg import (SP_TWOSTAGE, SGDriver, failure_reason, flatl,
                         flat2, group_seqs, metrics, run_direct,
                         run_two_stage)

print = PETSc.Sys.Print

T_START = time.time()


def head(s):
    print(f"\n=== [{time.time() - T_START:7.1f}s] {s} "
          + "=" * max(4, 60 - len(s)), flush=True)


# ============================================================ matfree machinery

class MatfreeCtx(HPGCtx):
    """HPGCtx with the cached-K0 MUMPS-LU A-inverse replaced by capped
    CG+GAMG on the alpha-free K0.  Subclassing only: the gate-verified
    machinery files are untouched."""

    def __init__(self, n0, n1, bc_nodes, sg, alpha_getter,
                 inner_rtol=1e-4, inner_maxit=40, beta_shat=0.0,
                 a_rtol=1e-8, a_maxit=60, a_pc="gamg"):
        super().__init__(n0, n1, bc_nodes, sg, alpha_getter,
                         inner_rtol=inner_rtol, inner_maxit=inner_maxit,
                         beta_shat=beta_shat, a_action="matfree",
                         gamg_rtol=a_rtol)
        self.a_rtol = float(a_rtol)
        self.a_maxit = int(a_maxit)
        self.a_pc = str(a_pc)
        # attribution instrumentation (superset of HPGCtx's)
        self.a_wall = 0.0            # total A-CG wall (shell + direct)
        self.a_wall_shell = 0.0      # A-CG time inside the true-S matvec
        self.a_wall_direct = 0.0     # A-CG time in the P_F body
        self.inner_wall = 0.0        # inner GMRES wall incl. in-shell A
        self.a_calls = 0
        self.a_its_total = 0
        self.a_its_per_call = []
        self.a_maxits_seen = 0
        self.a_nonconv = 0
        self.in_shell = False

    def _build_kspA_matfree(self, Km, alpha):
        """K0 = diag(1/alpha interior, 1 on bc) @ Km = stiffness + bc-rows
        (alpha-free, stage-1 verified to 3.6e-15).  Build ONE CG+GAMG
        solver on K0.  Reuses whatever alpha is live at first setUp -- the
        scaling cancels by construction."""
        interior = np.ones(self.n0, dtype=bool)
        interior[self.bc_nodes] = False
        scale = np.ones(self.n0)
        scale[interior] = 1.0 / alpha
        K0 = Km.copy()
        v = K0.createVecLeft()
        v.setArray(scale)
        K0.diagonalScale(L=v)
        ksp = PETSc.KSP().create(comm=K0.comm)
        ksp.setOperators(K0)
        ksp.setType("cg")
        ksp.setTolerances(rtol=self.a_rtol, max_it=self.a_maxit)
        ksp.getPC().setType(self.a_pc)
        if self.a_pc == "hypre":
            ksp.getPC().setHYPREType("boomeramg")
        ksp.setUp()
        self.kspA, self.K0 = ksp, K0
        self.interior = interior
        self._av = K0.createVecRight()
        self._ax = K0.createVecRight()

    def Ainv(self, b):
        """A(alpha)^-1 b = K0^-1 (D_alpha b), D_alpha = scale_vec kept
        current by alpha_getter at every PC setUp.  Fresh CG per call:
        its vary with rhs/state (variable inner iteration, documented)."""
        t0 = time.time()
        self._av.setArray(b * self.scale_vec)
        self.kspA.solve(self._av, self._ax)
        its = int(self.kspA.getIterationNumber())
        if int(self.kspA.getConvergedReason()) < 0:
            self.a_nonconv += 1
        self.a_calls += 1
        self.a_its_total += its
        self.a_its_per_call.append(its)
        self.a_maxits_seen = max(self.a_maxits_seen, its)
        dt = time.time() - t0
        self.a_wall += dt
        if self.in_shell:
            self.a_wall_shell += dt
        else:
            self.a_wall_direct += dt
        out = self._ax.getArray(readonly=True).copy()
        if not np.all(np.isfinite(out)):
            self.bad = True
        return out


class MatfreeTwoStagePC(HPGTwoStagePC):
    """Sequential P_F (paper eq 4.5) with the matfree A-action.  setUp
    builds the GAMG hierarchy on K0 once (first linear solve); every later
    setUp only refreshes B/D (per-iterate), the Shat cellwise Cholesky and
    the inner-GMRES -- identical to HPGTwoStagePC.setUp for everything
    except the A arm."""

    def setUp(self, pc):
        ctx = hpg_pc.CTX
        alpha = ctx.alpha_getter()          # refreshes D_alpha scale_vec
        if ctx.kspA is None:
            A, P = pc.getOperators()
            Km, Bm, Dm = split_blocks_pet(P, ctx.n0, ctx.n1)
            ctx._build_kspA_matfree(Km, alpha)
            Km.destroy()
        super().setUp(pc)

    def apply(self, pc, x, y):
        ctx = hpg_pc.CTX
        t0 = time.time()
        xr = x.getArray(readonly=True).copy()
        b1, b2 = xr[:ctx.n0], xr[ctx.n0:]
        t = ctx.Ainv(b1)                          # A_alpha^-1 b_u
        y2 = b2 - ctx.BT_mult(t)
        t_i = time.time()
        ctx.in_shell = True                       # A-CG calls inside the
        dpsi = ctx.inner_solve(y2)                #   true-S matvec count
        ctx.in_shell = False                      #   as shell time
        ctx.inner_wall += time.time() - t_i
        y1 = ctx.Ainv(b1 - ctx.B_mult(dpsi))
        out = np.concatenate([y1, dpsi])
        if not np.all(np.isfinite(out)):
            ctx.bad = True
        y.setArray(out)
        ctx.n_applies += 1
        ctx.apply_wall += time.time() - t0


def run_matfree(mesh, p, name, inner_rtol=1e-4, inner_maxit=40,
                beta_shat=0.0, max_prox=100, progress=0, run_cap=0.0,
                a_rtol=1e-8, a_maxit=60, a_pc="gamg"):
    """End-to-end LVPP solve with MatfreeTwoStagePC on snes.ksp.  Mirrors
    promote_hpg.run_two_stage (same promotion pattern) with per-linear-
    solve A/inner attribution."""
    lv = make_lvpp(mesh, p, dict(SP_TWOSTAGE), name)
    V0 = lv._z.subfunctions[0].function_space()
    W = lv._z.subfunctions[1].function_space()
    n0, n1 = V0.dim(), W.dim()
    sg = SGDriver(mesh, W, p, alpha=float(lv._alpha))
    ctx = MatfreeCtx(n0, n1, np.asarray(lv._bcs[0].nodes), sg,
                     alpha_getter=(lambda: 1.0),
                     inner_rtol=inner_rtol, inner_maxit=inner_maxit,
                     beta_shat=beta_shat, a_rtol=a_rtol, a_maxit=a_maxit,
                     a_pc=a_pc)

    def alpha_getter():
        a = float(lv._alpha)
        ctx._refresh_scale(a)
        return a
    ctx.alpha_getter = alpha_getter

    segs = []
    pcctx = MatfreeTwoStagePC()
    orig_setup = pcctx.setUp
    t_solve = [time.time()]
    t0_run = [time.time()]

    def pc_setUp(pc, orig=orig_setup):
        segs.append({"inner": [], "a0": ctx.a_calls,
                     "ait0": ctx.a_its_total})
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
        segs[-1]["inner"].append(its)
        if progress and ctx.n_applies % progress == 0:
            la = ctx.a_its_per_call[-1] if ctx.a_calls else -1
            print(f"      [prog] applies={ctx.n_applies} lastinner={its} "
                  f"lastaCG={la} "
                  f"solve_wall={time.time() - t_solve[0]:6.0f}s", flush=True)
        if run_cap and time.time() - t0_run[0] > run_cap:
            raise RuntimeError(
                f"run cap {run_cap:.0f}s hit (applies={ctx.n_applies}, "
                f"lastinner={its})")
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
        failed += " | non-finite PC output (A-CG or inner)"

    err, prox, newton = metrics(lv, mesh)
    curve, maxits, medits = [], 0, float("nan")
    inner_avg, inner_tot = float("nan"), 0
    a_avg = float("nan")
    a_calls_tot, a_its_tot = 0, 0
    inner_first = (float("nan"), 0)
    inner_deep = (float("nan"), 0)
    setup_wall = apply_wall = 0.0
    try:
        nits = list(lv.newton_iterations or [])
        outer_groups, leftover = mon.per_prox(nits)
        all_groups = outer_groups + ([leftover] if leftover else [])
        flatits = flatl(all_groups)
        curve = [max(ch) if ch else 0 for ch in outer_groups]
        maxits = max(flatits) if flatits else 0
        medits = statistics.median(flatits) if flatits else float("nan")
        iproxs, ileft = group_seqs(segs, nits)
        all_segs = iproxs + ([ileft] if ileft else [])
        inner_all = flat2([[s["inner"] for s in grp] for grp in all_segs])
        if inner_all:
            inner_avg = statistics.mean(inner_all)
            inner_tot = int(sum(inner_all))
        if iproxs:
            f = flatl([s["inner"] for s in iproxs[0]])
            if f:
                inner_first = (statistics.mean(f), max(f))
            d = flatl([s["inner"] for s in iproxs[-1]])
            if d:
                inner_deep = (statistics.mean(d), max(d))
        a_calls_tot = int(ctx.a_calls)
        a_its_tot = int(ctx.a_its_total)
        a_avg = (ctx.a_its_total / ctx.a_calls) if ctx.a_calls \
            else float("nan")
        setup_wall = ctx.setup_wall
        apply_wall = ctx.apply_wall
    except Exception as e:
        failed += " | metric extraction: " + failure_reason(e)

    res = dict(tag=name, p=p, n0=n0, n1=n1, prox=prox, newton=newton,
               maxits=maxits, medits=medits, curve=curve, err=err,
               wall=wall, inner_avg=inner_avg, inner_tot=inner_tot,
               inner_first=inner_first, inner_deep=inner_deep,
               a_avg=a_avg, a_calls=a_calls_tot, a_its=a_its_tot,
               a_maxits_seen=ctx.a_maxits_seen, a_nonconv=ctx.a_nonconv,
               a_wall=ctx.a_wall, a_wall_shell=ctx.a_wall_shell,
               a_wall_direct=ctx.a_wall_direct, inner_wall=ctx.inner_wall,
               setup_wall=setup_wall, apply_wall=apply_wall,
               n_applies=ctx.n_applies, n_setup=ctx.n_setup,
               min_eig=sg.min_eig_negS, failed=failed)
    del lv
    gc.collect()
    return res


# ================================================================== printing

ROW_MF = (f"  {'tag':>14} {'u-dofs':>7} {'psi':>6} {'prox':>4} {'newton':>6} "
          f"{'maxits':>6} {'medits':>6} {'err(u_h)':>9} {'wall':>7} "
          f"{'innAvg':>7} {'innTot':>6} {'aCGav':>6} {'aCGits':>7} "
          f"{'setupW':>6} {'applyW':>6} {'innerW':>6} {'aCG_W':>6} "
          f"{'status':>7}")


def _g(res, key, fmt, dash="-"):
    v = res.get(key)
    if v is None or (isinstance(v, float) and not np.isfinite(v)
                     and fmt.endswith("f")):
        return dash.rjust(len(f"{0:{fmt}}"))
    return f"{v:{fmt}}"


def print_row(tag, res):
    f0, m0 = res.get("inner_first", (float("nan"), 0))
    print(f"  {tag:>14} {_g(res, 'n0', 'd')} {_g(res, 'n1', 'd')} "
          f"{_g(res, 'prox', 'd')} {_g(res, 'newton', 'd')} "
          f"{_g(res, 'maxits', 'd')} {_g(res, 'medits', '.1f')} "
          f"{_g(res, 'err', '.3e')} {_g(res, 'wall', '.1f')}s "
          f"{_g(res, 'inner_avg', '.1f')}/{_g(res, 'inner_tot', 'd')} "
          f"{_g(res, 'a_avg', '.1f')}/{_g(res, 'a_its', 'd')} "
          f"{_g(res, 'setup_wall', '.1f')} {_g(res, 'apply_wall', '.1f')} "
          f"{_g(res, 'inner_wall', '.1f')} {_g(res, 'a_wall', '.1f')} "
          f"{'ok' if not res.get('failed') else 'FAIL':>7}"
          + (f"  FAILED: {res['failed']}" if res.get("failed") else ""),
          flush=True)


def print_attr(res):
    """Wall attribution line; tolerant of assembled-arm res dicts."""
    setup = res.get("setup_wall")
    apply_w = res.get("apply_wall")
    parts = []
    if setup is not None:
        parts.append(f"setup={setup:.1f}s (x{res.get('n_setup', '?')})")
    if apply_w is not None:
        parts.append(f"apply={apply_w:.1f}s (x{res.get('n_applies', '?')})")
    if res.get("inner_wall") is not None:
        parts.append(f"innerS={res['inner_wall']:.1f}s")
    if res.get("a_wall") is not None:
        parts.append(f"A-CG={res['a_wall']:.1f}s (shell "
                     f"{res.get('a_wall_shell', 0.0):.1f} + direct "
                     f"{res.get('a_wall_direct', 0.0):.1f}), "
                     f"calls={res.get('a_calls', '?')} "
                     f"avg={res.get('a_avg', float('nan')):.1f} "
                     f"max={res.get('a_maxits_seen', '?')} "
                     f"nonconv={res.get('a_nonconv', '?')}")
    if setup is not None and apply_w is not None:
        parts.append(f"snes+outer_action="
                     f"{max(res['wall'] - setup - apply_w, 0.0):.1f}s")
    if res.get("min_eig") is not None:
        parts.append(f"min_eig(-Shat)={res['min_eig']:.2e}")
    print(f"    [attr] {res.get('tag', '?')}: " + "  ".join(parts), flush=True)


def print_direct_row(d):
    print(f"  {'direct':>14} {'-':>7} {'-':>6} {_g(d, 'prox', 'd')} "
          f"{_g(d, 'newton', 'd')} {'-':>6} {'-':>6} {_g(d, 'err', '.3e')} "
          f"{_g(d, 'wall', '.1f')}s"
          + (f"  FAILED: {d['failed']}" if d.get("failed") else ""),
          flush=True)


# ==================================================================== stages

def stage_smoke():
    head("smoke: L0 p2 matfree (GAMG-on-K0 sanity, correctness vs direct)")
    mesh = uniform_quad(16)
    d = run_direct(mesh, 2, "dir_L0_p2")
    print_direct_row(d)
    res = run_matfree(mesh, 2, "mf_L0_p2")
    print_row(res["tag"], res)
    print_attr(res)
    ok = (not res["failed"] and abs(res["err"] - d["err"]) < 5e-4
          and res["prox"] == d["prox"])
    print(f"  SMOKE {'PASS' if ok else 'CHECK'}: err mf {res['err']:.3e} "
          f"vs direct {d['err']:.3e}, prox {res['prox']} vs {d['prox']}",
          flush=True)
    return ok


def stage_correct():
    head("correctness: L0/L1 p2, matfree vs serial-direct controls "
         "(hpg serial-direct refs: L0 8/22 1.19e-3; L1 7/20 1.04e-4)")
    print(ROW_MF)
    for level in (0, 1):
        mesh = uniform_quad(16 * 2 ** level)
        d = run_direct(mesh, 2, f"dir_L{level}_p2")
        print_direct_row(d)
        res = run_matfree(mesh, 2, f"mf_L{level}_p2")
        print_row(res["tag"], res)
        print_attr(res)
        del mesh
        gc.collect()


def stage_scale(levels=(0, 1), ps=(2, 4), arms=("direct", "assem", "matfr"),
                cap_L1p4=900.0, cap_p4=600.0):
    head(f"p-scaling: levels={list(levels)} p={list(ps)} arms={list(arms)}"
         f" -- matfree vs assembled-cache (same capped-inexact config "
         f"r1e-4/c40, beta_Shat=0)")
    print(ROW_MF)
    for level in levels:
        n = 16 * 2 ** level
        mesh = uniform_quad(n)
        for p in ps:
            cap = 0.0
            if p == 4:
                cap = cap_L1p4 if level == 1 else cap_p4
            if "direct" in arms:
                d = run_direct(mesh, p, f"dir_L{level}_p{p}")
                print_direct_row(d)
            if "assem" in arms:
                ra = run_two_stage(mesh, p, f"assem_L{level}_p{p}",
                                   inner_rtol=1e-4, inner_maxit=40,
                                   run_cap=cap)
                print_row(ra["tag"], ra)
                print_attr(ra)
            if "matfr" in arms:
                rm = run_matfree(mesh, p, f"matfr_L{level}_p{p}",
                                 run_cap=cap)
                print_row(rm["tag"], rm)
                print_attr(rm)
        del mesh
        gc.collect()


def stage_graded(n=48, p=4, cap=900.0, arms=("direct", "assem", "matfr"),
                 a_pc="gamg"):
    head(f"graded robustness: graded_quad({n}, ratio~45x) p={p} -- matfree "
         f"(A-PC={a_pc}) + E_beta-decoupled (beta_Shat=0) + capped GMRES "
         f"inner vs assembled-cache")
    mesh, meas = graded_quad(n, ratio=45.3)
    print(f"  measured grading ratio hmax/hmin = {meas:.1f}x", flush=True)
    if "direct" in arms:
        d = run_direct(mesh, p, f"dir_g{n}_p{p}")
        print_direct_row(d)
    print(ROW_MF)
    if "assem" in arms:
        ra = run_two_stage(mesh, p, f"assem_g{n}_p{p}",
                           inner_rtol=1e-4, inner_maxit=40, run_cap=cap)
        print_row(ra["tag"], ra)
        print_attr(ra)
    if "matfr" in arms:
        rm = run_matfree(mesh, p, f"matfr_g{n}_p{p}", run_cap=cap,
                         a_pc=a_pc)
        print_row(rm["tag"], rm)
        print_attr(rm)


def stage_p6(cap=600.0):
    head("stretch: L0 p=6 (k=25/cell latent) matfree vs assembled-cache")
    mesh = uniform_quad(16)
    d = run_direct(mesh, 6, "dir_L0_p6")
    print_direct_row(d)
    print(ROW_MF)
    ra = run_two_stage(mesh, 6, "assem_L0_p6",
                       inner_rtol=1e-4, inner_maxit=40, run_cap=cap)
    print_row(ra["tag"], ra)
    print_attr(ra)
    rm = run_matfree(mesh, 6, "matfr_L0_p6", run_cap=cap)
    print_row(rm["tag"], rm)
    print_attr(rm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["smoke", "correct", "scale",
                                      "graded48", "graded80", "p6", "all"])
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    if a.stage == "smoke":
        stage_smoke()
    elif a.stage == "correct":
        stage_correct()
    elif a.stage == "scale":
        lv_ = [int(x) for x in a.args if x.isdigit()] or [0, 1]
        ps_ = [int(x[1:]) for x in a.args if x.startswith("p")
               and x[1:].isdigit()] or (2, 4)
        arms_ = tuple(x for x in a.args
                      if x in ("direct", "assem", "matfr")) \
            or ("direct", "assem", "matfr")
        stage_scale(levels=lv_, ps=ps_, arms=arms_)
    elif a.stage == "graded48":
        arms_ = [x for x in a.args if x in ("direct", "assem", "matfr")] \
            or ("direct", "assem", "matfr")
        a_pc = "hypre" if "hypre" in a.args else "gamg"
        stage_graded(48, 4, arms=arms_, a_pc=a_pc)
    elif a.stage == "graded80":
        stage_graded(80, 4, cap=1500.0)
    elif a.stage == "p6":
        stage_p6()
    else:
        stage_smoke()
        stage_graded(48, 4)
    print(f"[{time.time() - T_START:7.1f}s] {a.stage} DONE", flush=True)


if __name__ == "__main__":
    main()
