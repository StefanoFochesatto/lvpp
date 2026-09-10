"""Scaling experiment on the FLOORLESS Schur configuration (Finding 6 /
RESOLVED block): can it be pushed past the 8,321-dof level-2 baseline?

Three independent questions on the production config (matfree Amat +
assembled UNfloored Pmat, psi_floor=0, schur/upper/selfp, MUMPS-LU on
both fieldsplit blocks, outer GMRES rtol 1e-6 restart 250, snes_rtol
1e-6, l2 linesearch maxlambda 1.0, alpha_max 10, quadrature_degree 6,
tol=1e-4, increment_norm H1):

  A. Levels 3-4 (MeshHierarchy(base, 4): L3 = 33k dofs/field, L4 =
     132k) with the production config as-is.  Solvable?  prox/newton/
     err vs the LU refs (8/21/1.553e-02, 11/23/3.599e-03, 8/19/8.375e-
     04)?  max-its + per-prox outer-GMRES curve, wall, MUMPS
     setup/factor wall (PETSc log events MatLUFactorSym/Num, cumulative
     parse per level), peak RSS (resource.getrusage ru_maxrss).  An L4
     memory failure is a recorded scaling datum, not a defeat.

  B. CG+GAMG-on-K variant: fieldsplit_0 ksp cg + pc gamg (agg) instead
     of preonly+lu(MUMPS), fieldsplit_1 stays preonly+lu(MUMPS) on the
     selfp Sp.  Levels 0-3 (L4 if memory allows).  Question: does the
     iterative K-solve preserve LU-identical Newton counts, and what is
     the wall tradeoff at each level?  Inner CG iteration counts are
     monitored lazily on the fieldsplit_0 sub-KSP (attached at the
     first outer apply, so the first Newton solve is uncounted).

  C. Saturation-corner regression (Finding 5) at L2: the floorless
     config with tol=1e-6 (tighter than the production 1e-4) -- does
     the latent psi drift deeper (Finding 5: the OLD additive config
     hit MUMPS PC_FAILED at |psi| = 1351) and does the solve survive?
     tol=1e-4 is re-run in the same stage for the depth baseline.

Usage (env guard REQUIRED):
  PETSC_DIR=/home/stefano/firedrake/petsc PETSC_ARCH=arch-firedrake-default \
  OMP_NUM_THREADS=1 venv-firedrake/bin/python experiments/scale_up.py \
      --scaleup --levels 3,4
  ... --gamg --levels 0,1,2,3 [--gamg-rtol 1e-8]
  ... --corner --levels 2 --tols 1e-4,1e-6

Serial only.  Read-only access to lvpp internals; no lvpp.py edits.
Problem setup, KSP-monitor grouping, failure handling and the winner
config dict are reused from eps_schedule.py (single source of truth).
"""
import argparse
import gc
import os
import re
import resource
import statistics
import time

import numpy as np
from firedrake import (DirichletBC, Function, FunctionSpace,
                       MeshHierarchy, RectangleMesh, SpatialCoordinate,
                       assemble, dx, errornorm, grad, inner, sqrt)
from firedrake.petsc import PETSc
from lvpp import LVPP

from eps_schedule import (FCP, LU_REFS, REASON_NAMES, SP_WINNER,
                          failure_reason, per_prox_its, psiUFL, uexactUFL)

PETSc.Log.begin()

# log summary parsed from a temp viewer; cumulative MatLUFactor totals
_LOGFILE = f"/tmp/scale_up_petsc_log_{os.getpid()}.txt"
_cum_factor = {"sym": 0.0, "num": 0.0, "count": 0}


def _rss_mib():
    """Peak RSS of this process in MiB (ru_maxrss is KiB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _factor_totals():
    """Cumulative MUMPS setup/factor wall from the PETSc log summary.

    Event line layout: Name Count TimeRatio TimeSec ... (serial run:
    the Max column IS the total).  Returns (sym_s, num_s, count)."""
    v = PETSc.Viewer().createASCII(_LOGFILE)
    PETSc.Log.view(v)
    v.destroy()
    sym = num = 0.0
    count = 0
    with open(_LOGFILE) as fh:
        for line in fh:
            m = re.match(r"\s*(MatLUFactorSym|MatLUFactorNum)\s+(\d+)\s+\S+\s+(\S+)",
                         line)
            if m:
                t = float(m.group(3))
                if m.group(1) == "MatLUFactorSym":
                    sym += t
                else:
                    num += t
                count += int(m.group(2))
    return sym, num, count


def factor_report():
    """Per-run MUMPS factor wall as the delta of cumulative totals."""
    sym, num, count = _factor_totals()
    dsym, dnum, dcount = (sym - _cum_factor["sym"], num - _cum_factor["num"],
                          count - _cum_factor["count"])
    _cum_factor.update(sym=sym, num=num, count=count)
    return dsym, dnum, dcount


def field_ises(Jm, n0, n1):
    """Field ISes for the 2-field mixed system (u block first)."""
    is0 = PETSc.IS().createGeneral(list(range(n0)), comm=Jm.getComm())
    is1 = PETSc.IS().createGeneral(list(range(n0, n0 + n1)),
                                   comm=Jm.getComm())
    return is0, is1


# CG+GAMG variant: only fieldsplit_0 differs from the production config
# (psi_floor is NOT passed to LVPP -> 0.0, the floorless config).
SP_GAMG_BASE = dict(SP_WINNER)
SP_GAMG_BASE.update({
    "fieldsplit_0_ksp_type": "cg",
    "fieldsplit_0_ksp_rtol": None,      # set per-run (--gamg-rtol)
    "fieldsplit_0_ksp_converged_reason": None,
    "fieldsplit_0_pc_type": "gamg",
    "fieldsplit_0_pc_gamg_type": "agg",
    "fieldsplit_0_pc_gamg_agg_nsmooths": "1",
    "fieldsplit_0_mg_levels_ksp_type": "chebyshev",
    "fieldsplit_0_mg_levels_pc_type": "jacobi",
})


def run_solve(mesh, tag, sp_dict, tol=1e-4, max_prox=500):
    """One full floorless LVPP solve with the given solver config."""
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name="u")
    x, y = SpatialCoordinate(mesh)
    r = sqrt(x * x + y * y)
    energy = 0.5 * inner(grad(u), grad(u)) * dx  # f = 0
    psi = Function(V, name="psi").interpolate(psiUFL(r))
    bc = DirichletBC(V, uexactUFL(r), "on_boundary")
    lvpp = LVPP(energy=energy, u=u, bounds=(psi, None), bcs=bc,
                alpha_rule="double_exponential",
                alpha_parameters={"alpha_max": 10.0},
                increment_norm="H1", verbose=False,
                form_compiler_parameters=FCP,
                solver_parameters=dict(sp_dict), name=tag)

    state = {"groups": [], "inner": [], "inner_attach": False}

    def inner_mon(ksp, its, rnorm):
        if its == 1:
            state["inner"].append(1)
        elif its > 1:
            if state["inner"]:
                state["inner"][-1] = its
            else:
                state["inner"].append(its)

    def ksp_mon(ksp, its, rnorm):
        if its == 1:
            state["groups"].append(1)
            if not state["inner_attach"]:
                state["inner_attach"] = True
                try:
                    subs = lvpp._solver.snes.ksp.pc.getFieldSplitSubKSP()
                    subs[0].setMonitor(inner_mon)
                except Exception:
                    state["inner_attach"] = False
        elif its > 1:
            if state["groups"]:
                state["groups"][-1] = its
            else:
                state["groups"].append(its)

    lvpp._solver.snes.ksp.setMonitor(ksp_mon)
    t0 = time.time()
    lvpp.solve(tol=tol, max_proximal_iterations=max_prox)
    wall = time.time() - t0

    per_prox, _ = per_prox_its(state["groups"],
                               list(lvpp.newton_iterations))
    curve = [max(ch) if ch else 0 for ch in per_prox]

    err = float(errornorm(uexactUFL(r), lvpp.u_out[0]))

    # psi-depth + dead-row anatomy of the final Jacobian (read-only)
    psi_min = psi_absmax = None
    dead8 = dead12 = None
    try:
        lat = lvpp._z.subfunctions[1].dat.data
        psi_min, psi_absmax = float(lat.min()), float(np.abs(lat).max())
    except Exception:
        pass
    try:
        W = lvpp._z.subfunctions[1].function_space()
        n0 = V.dim()
        Jfd = assemble(lvpp._J, mat_type="aij", bcs=lvpp._bcs)
        is0, is1 = field_ises(Jfd.M.handle, n0, W.dim())
        Dm = Jfd.M.handle.createSubMatrix(is1, is1)
        dd = Dm.getDiagonal().getArray(readonly=True).copy()
        dead8 = int(np.sum(np.abs(dd) < 1e-8))
        dead12 = int(np.sum(np.abs(dd) < 1e-12))
        Dm.destroy()
        is0.destroy()
        is1.destroy()
    except Exception:
        pass

    fsym, fnum, fcount = factor_report()
    snes = lvpp._solver.snes
    return dict(tag=tag, dofs=V.dim(), prox=lvpp.proximal_iterations,
                newton=sum(lvpp.newton_iterations), err=err, wall=wall,
                curve=curve, its=state["groups"],
                inner_total=sum(state["inner"]),
                inner_attached=state["inner_attach"],
                snes=int(snes.getConvergedReason()),
                ksp=int(snes.ksp.getConvergedReason()),
                psi_min=psi_min, psi_absmax=psi_absmax,
                dead8=dead8, dead12=dead12,
                fsym=fsym, fnum=fnum, fcount=fcount,
                rss_mib=_rss_mib())


def print_row(lev, v, refs=True):
    if v is None:
        print(f"  L{lev}: FAILED")
        return
    p, n, e = LU_REFS.get(lev, (0, 0, 0.0))
    med = statistics.median(v["its"]) if v["its"] else float("nan")
    print(f"  L{lev}: dofs={v['dofs']} prox={v['prox']} newton={v['newton']} "
          f"err={v['err']:.3e}"
          + (f" (LU ref {p}/{n} {e:.3e})" if refs and p else "")
          + f" snes={v['snes']}({REASON_NAMES.get(v['snes'], '?')}) "
          f"ksp={v['ksp']}({REASON_NAMES.get(v['ksp'], '?')})")
    print(f"       outer-its max={max(v['its']) if v['its'] else 0} "
          f"med={med:.1f} wall={v['wall']:.1f}s  "
          f"per-prox curve: {v['curve']}")
    if v["inner_attached"]:
        print(f"       inner cg total its (after lazy attach): "
              f"{v['inner_total']}")
    ffac = v["fsym"] + v["fnum"]
    print(f"       MUMPS factor: sym {v['fsym']:.2f}s + num {v['fnum']:.2f}s "
          f"({v['fcount']} factors) = {ffac:.2f}s "
          f"({ffac / max(v['wall'], 1e-9) * 100:.0f}% of wall); "
          f"peak RSS {v['rss_mib']:.0f} MiB")
    if v["psi_min"] is not None:
        print(f"       psi(latent) min={v['psi_min']:.4g} absmax="
              f"{v['psi_absmax']:.4g}  dead rows D: |d|<1e-8: {v['dead8']}  "
              f"<1e-12: {v['dead12']}")


def run_stage(title, levels, sp_dict, hier, tagfmt, tols=(1e-4,), refs=True,
              max_prox=500):
    print("=" * 78)
    print(title)
    if refs:
        print("LU refs: " + "  ".join(
            f"L{k}: {v[0]}/{v[1]}/{v[2]:.3e}" for k, v in LU_REFS.items()))
    print("=" * 78)
    for lev, mesh in enumerate(hier):
        if lev not in levels:
            continue
        for tol in tols:
            suffix = "" if len(tols) == 1 else f" tol={tol:g}"
            try:
                v = run_solve(mesh, tagfmt.format(lev=lev, tol=f"{tol:g}"),
                              sp_dict, tol=tol, max_prox=max_prox)
            except Exception as exc:
                print(f"  L{lev}{suffix}: FAILED: {failure_reason(exc)}")
                print(f"  peak RSS at failure: {_rss_mib():.0f} MiB")
                gc.collect()
                continue
            print(f"  L{lev}{suffix}:")
            print_row(lev, v, refs=refs and tol == 1e-4)
            del v
            gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaleup", action="store_true",
                    help="production config (both blocks MUMPS-LU)")
    ap.add_argument("--gamg", action="store_true",
                    help="CG+GAMG on K (fieldsplit_0), MUMPS on Sp")
    ap.add_argument("--corner", action="store_true",
                    help="saturation-corner regression: tol sweep")
    ap.add_argument("--levels", default=None,
                    help="comma list; defaults per stage")
    ap.add_argument("--gamg-rtol", type=float, default=1e-8)
    ap.add_argument("--tols", default="1e-4,1e-6")
    ap.add_argument("--max-prox", type=int, default=500)
    args = ap.parse_args()
    if not (args.scaleup or args.gamg or args.corner):
        ap.error("choose --scaleup and/or --gamg and/or --corner")

    base = RectangleMesh(16, 16, 2.0, 2.0, originX=-2.0, originY=-2.0,
                         diagonal="crossed")
    hier = MeshHierarchy(base, 4)

    if args.scaleup:
        levels = [int(s) for s in (args.levels or "3,4").split(",")]
        run_stage(
            "STAGE A: production floorless Schur config, both blocks "
            "MUMPS-LU, tol=1e-4",
            levels, SP_WINNER, hier, "scale_l{lev}", max_prox=args.max_prox)

    if args.gamg:
        levels = [int(s) for s in (args.levels or "0,1,2,3").split(",")]
        sp_gamg = dict(SP_GAMG_BASE)
        sp_gamg["fieldsplit_0_ksp_rtol"] = float(args.gamg_rtol)
        run_stage(
            f"STAGE B: CG+GAMG on K (fieldsplit_0 cg rtol "
            f"{args.gamg_rtol:g}, agg + chebyshev/jacobi smoothers), "
            "fieldsplit_1 preonly+lu(MUMPS) on the selfp Sp",
            levels, sp_gamg, hier, "gamg_l{lev}", max_prox=args.max_prox)

    if args.corner:
        levels = [int(s) for s in (args.levels or "2").split(",")]
        tols = [float(s) for s in args.tols.split(",")]
        run_stage(
            "STAGE C: saturation-corner regression, tol sweep "
            f"{tols} (Finding 5: MUMPS PC_FAILED at |psi|=1351 on the "
            "OLD additive config)",
            levels, SP_WINNER, hier, "corner_l{lev}_tol{tol}",
            tols=tuple(tols), refs=False, max_prox=args.max_prox)


if __name__ == "__main__":
    main()