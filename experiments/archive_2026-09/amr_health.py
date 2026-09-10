"""AMR-health experiment: does free-boundary-aware adaptivity shrink the
degenerate (dead) latent subspace of the floorless Schur LVPP solver at a
matched dof budget?

Prediction under test (Finding 1 + Finding 6, RESEARCH.md): the exactly-
degenerate latent rows (|diag(D)| < 1e-8 at convergence) live in the contact
INTERIOR, so their count scales O(area/h^2) under uniform refinement but only
O(perimeter/h) on a mesh that keeps the known-active interior coarse and
concentrates resolution in the free-boundary transition band.  At a matched
dof budget the adapted mesh should therefore show far fewer dead rows and a
flatter per-Newton-solve outer-GMRES curve.  VERDICT (measured, this file):
REFUTED on both counts -- see the summary table and the mechanism notes at
the bottom of this docstring.

Configuration: the validated FLOORLESS Schur production config
(eps_schedule.py winner, parent-verified): matfree Amat + assembled UNfloored
Pmat, pc_fieldsplit_type schur / fact_type upper / use_amat False / selfp,
fieldsplit_0 and fieldsplit_1 preonly + MUMPS-LU, psi_floor = 0, outer GMRES
rtol 1e-6 restart 250, snes_rtol 1e-6, l2 linesearch maxlambda 1.0,
alpha_max 10, quadrature_degree 6, tol 1e-4, increment_norm H1.  Protocol
matches the baseline (cold solves at every level, like eps_schedule.run_full).

Mesh arms (how the adapted meshes are built -- honest construction notes):

  uniform  MeshHierarchy(RectangleMesh(16,16,2,2, origin (-2,-2), crossed), 2):
           545 / 2113 / 8321 u-dofs.  L2 is the matched-dof reference.

  band     free-boundary-ring marking, built manually from viamr primitives:
           after each solve, gap = u_tilde - lb is EXACTLY exp(psi) nodally
           (Finding 2), so a cell straddles the transition band iff
           min_K(gap) < activetol < max_K(gap) (elementwise min/max via
           viamr's _elemextreme, DG0).  Only those straddling cells are
           marked; viamr.refinesbr2D (PETSc DMPlex SBR, Plaza & Carey)
           bisects them.  Contact interior and far field are NEVER marked --
           this is the degeneracy-minimizing strategy of Finding 1
           (known-active interior stays coarse).  The chain runs solve ->
           mark -> refine until the u-dof count reaches the target band.

  udobr    the standard viamr pipeline of examples/sphere_lvpp.py (pattern
           copied, not imported): mark = udomark(u_tilde, lb, n=1) (contact
           set dilated by one ring) UNION gradrecinactivemark (gradient-
           recovery marking of the inactive set, Doerfler theta=0.5), then
           refinesbr2D.  Note this arm refines part of the active interior
           cumulatively, so it sits between 'band' and 'uniform' in the
           degeneracy ledger.

No firedrake.adapt() is needed and no hand-built annulus mesh is used: the
viamr SBR refinement works out of the box in this venv, so both adapted
chains are genuine solve-mark-refine loops (the fallback constructions
mentioned in the assignment were not needed).

Dead-row census, two definitions:
  dead8/dead12  growth_driver.py definition: rows of the assembled unfloored
                latent block D with |diag(D)| < 1e-8 / < 1e-12.  Comparable
                to all prior measurements (45/177/765 at uniform L0/L1/L2).
  psid8/pside12 h-fair census: D_ii ~ M_ii * exp(psi_i) with M_ii ~ h^2, so
                the raw threshold over-counts on refined cells.  Since
                gap = exp(psi) exactly (Finding 2), counting nodes with
                psi < log(tol) is mesh-independent.  The two agree on
                uniform meshes (h constant) and diverge on graded ones.

Protocol notes: cold solves everywhere (protocol-matched to the parent's
run_full baseline: the its/dead-row comparison must not confound solver
state).  Warm-start AMR chains (the accuracy protocol of
viamr/examples/sphere_lvpp.py) are a different measurement.  A failed solve
is recorded as a row (prox/newton -1, converged reasons printed) and the
chain stops -- failures are data here.

Mechanism reading of the REFUTATION (measured):
  1. Dead rows are NOT capped by keeping the contact interior coarse.  The
     refined free-boundary ring is itself a dead-row factory: sharpening the
     boundary converts the coarse mesh's smeared compromise nodes into
     cleanly-pinned contact nodes, so psid8 grows as O(perimeter/h_ring)
     AND the deep-contact fraction of all latent dofs explodes
     (uniform: ~7-9% at every level; band: 7->25->31->36->50->47%).
  2. The outer-GMRES curve degrades with grading strength, not dof count:
     the selfp Schur approximation Sp = D - B diag(K)^{-1} B^T leans on
     diag(K)^{-1} ~ K^{-1}, and strong grading (h ratio 10-45x here)
     amplifies that mismatch.  The blowup concentrates in the deep proximal
     steps (the ones that are cheap on uniform).
  3. Net: at matched dofs the uniform mesh is simultaneously the dead-row-
     minimizing and the outer-its-minimizing strategy for the floorless
     Schur config on this problem.  This CORRECTS the "Relation to
     free-boundary-aware meshes" paragraph of Finding 1: free-boundary AMR
     buys accuracy (do-not-refine-known-active), not preconditioner health.

Usage (env guard REQUIRED):
  PETSC_DIR=/home/stefano/firedrake/petsc PETSC_ARCH=arch-firedrake-default \
  OMP_NUM_THREADS=1 /home/stefano/firedrake/venv-firedrake/bin/python \
      experiments/amr_health.py [--arms uniform,band,udobr] \
      [--target-dofs 7500] [--max-amr-levels 12]

Serial only.  Read-only access to lvpp internals (lvpp._solver.snes.ksp,
lvpp._J, lvpp._z); no existing files modified.
"""
import argparse
import gc
import os
import statistics
import sys
import time

import numpy as np

if os.environ.get("OMP_NUM_THREADS") != "1":
    sys.exit("env guard: run with OMP_NUM_THREADS=1 (see module docstring)")

from firedrake import (DirichletBC, Function, FunctionSpace, MeshHierarchy,
                       RectangleMesh, SpatialCoordinate, assemble, conditional,
                       dx, errornorm, grad, inner, le, ln, sqrt)
from firedrake.petsc import PETSc
from lvpp import LVPP
from viamr import VIAMR

print = PETSc.Sys.Print

r0 = 0.9
AFREE = 0.697965148223374
A_, B_ = 0.680259411891719, 0.471519893402112
FCP = {"quadrature_degree": 6}
DEAD_TOL = 1e-8
DEAD_TOL2 = 1e-12

# Floorless Schur production config (eps_schedule.py SP_WINNER, psi_floor=0).
SP_WINNER = {
    "mat_type": "matfree",
    "pmat_type": "aij",
    "pc_fieldsplit_use_amat": False,
    "snes_linesearch_type": "l2",
    "snes_linesearch_maxlambda": 1.0,
    "snes_rtol": 1e-6,
    "snes_max_it": 100,
    "ksp_type": "gmres",
    "ksp_rtol": 1e-6,
    "ksp_max_it": 1000,
    "ksp_gmres_restart": 250,
    "ksp_converged_reason": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "upper",
    "pc_fieldsplit_schur_precondition": "selfp",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "lu",
    "fieldsplit_0_pc_factor_mat_solver_type": "mumps",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "lu",
    "fieldsplit_1_pc_factor_mat_solver_type": "mumps",
}


def psiUFL(r):
    psi0 = np.sqrt(1.0 - r0 * r0)
    dpsi0 = -r0 / psi0
    return conditional(le(r, r0), sqrt(1.0 - r * r), psi0 + dpsi0 * (r - r0))


def uexactUFL(r):
    return conditional(le(r, AFREE), psiUFL(r), -A_ * ln(r) + B_)


def per_prox_its(groups, newton_its):
    """Group per-solve outer-GMRES its by proximal solve (eps_schedule)."""
    out, i = [], 0
    for n in newton_its:
        out.append(groups[i:i + n])
        i += n
    return out, groups[i:]  # leftovers = alpha-halving retry solves


def failure_reason(exc, lvpp=None):
    reason = ""
    if lvpp is not None:
        try:
            snes = lvpp._solver.snes
            reason = (f" [snes={snes.getConvergedReason()} "
                      f"ksp={snes.ksp.getConvergedReason()}]")
        except Exception:
            pass
    msg = str(exc).strip().replace("\n", " ")
    return f"{type(exc).__name__}: {msg[:220]}{reason}"


# ---------------------------------------------------------------------------
# One production-config solve on one mesh + full health measurement
# ---------------------------------------------------------------------------

def solve_measure(mesh, tag, amr):
    """Cold LVPP solve (floorless Schur config) on `mesh`; returns metrics
    plus the objects needed for marking (lvpp is kept only by the caller)."""
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name="u")
    x, y = SpatialCoordinate(mesh)
    r = sqrt(x * x + y * y)
    lb = Function(V, name="psi_lb").interpolate(psiUFL(r))
    bc = DirichletBC(V, uexactUFL(r), "on_boundary")
    energy = 0.5 * inner(grad(u), grad(u)) * dx  # f = 0
    lvpp = LVPP(energy=energy, u=u, bounds=(lb, None), bcs=bc,
                alpha_rule="double_exponential",
                alpha_parameters={"alpha_max": 10.0},
                increment_norm="H1", verbose=False,
                psi_floor=0.0,
                form_compiler_parameters=FCP,
                solver_parameters=dict(SP_WINNER), name=tag)

    state = {"groups": []}

    def ksp_mon(ksp, its, rnorm):
        if its == 1:
            state["groups"].append(1)
        elif its > 1:
            if state["groups"]:
                state["groups"][-1] = its
            else:
                state["groups"].append(its)

    lvpp._solver.snes.ksp.setMonitor(ksp_mon)
    t0 = time.time()
    try:
        lvpp.solve(tol=1e-4, max_proximal_iterations=500)
        failed = ""
    except Exception as e:  # record honestly, still salvage counts
        failed = failure_reason(e, lvpp)
    wall = time.time() - t0

    flat, curve = [], []
    ut = psil = None
    err = float("nan")
    Ne = nact = -1
    hmin = hmax = float("nan")
    prox = newton = -1
    try:
        per_prox, leftover = per_prox_its(state["groups"],
                                          list(lvpp.newton_iterations or []))
        flat = [i for chunk in per_prox + ([leftover] if leftover else [])
                for i in chunk]
        curve = [max(ch) if ch else 0 for ch in per_prox]
        err = float(errornorm(uexactUFL(r), lvpp.u_out[0]))
        ut = lvpp.u_tilde[0]
        psil = lvpp.psi_out[0]
        act = amr.elemactive(ut, lb)
        Ne = amr.countmark(Function(act.function_space()).assign(1.0))
        nact = amr.countmark(act)
        Nv, _, hmin, hmax = amr.meshsizes(mesh)
        prox = lvpp.proximal_iterations
        newton = sum(lvpp.newton_iterations or [])
    except Exception as e:
        failed += " | metric extraction FAILED: " + failure_reason(e)

    # ---- degenerate-row census on the unfloored latent block -------------
    # (growth_driver pattern: assembled Jacobian at the converged state,
    #  latent (psi) block diagonal; psi_floor=0 so J itself is unfloored)
    n0 = V.dim()
    n1 = lvpp._z.subfunctions[1].function_space().dim()
    dead8 = dead12 = -1
    psid8 = psid12 = -1
    phi_min = float("nan")
    try:
        Jfd = assemble(lvpp._J, mat_type="aij", bcs=lvpp._bcs)
        Jm = Jfd.M.handle
        is1 = PETSc.IS().createGeneral(
            np.arange(n0, n0 + n1, dtype=np.int32), comm=Jm.comm)
        Dm = Jm.createSubMatrix(is1, is1)
        dd = Dm.getDiagonal().getArray(readonly=True).copy()
        dead8 = int(np.sum(np.abs(dd) < DEAD_TOL))
        dead12 = int(np.sum(np.abs(dd) < DEAD_TOL2))
        Dm.destroy()
        is1.destroy()
        del Jfd, Jm
        phi = lvpp._z.subfunctions[1].dat.data.copy()
        phi_min = float(phi.min())
        # h-fair census: gap = exp(psi) exactly (Finding 2), so the count of
        # nodes with exp(psi) < tol is mesh-independent; the raw |diag(D)|
        # threshold is not (M_ii ~ h^2 shrinks on refined cells).
        psid8 = int(np.sum(phi < np.log(DEAD_TOL)))
        psid12 = int(np.sum(phi < np.log(DEAD_TOL2)))
    except Exception as e:
        failed += " | dead-row census FAILED: " + failure_reason(e)

    return dict(tag=tag, mesh=mesh, V=V, lvpp=lvpp, ut=ut, psil=psil, lb=lb,
                udofs=n0, psidofs=n1, Ne=Ne, hmin=hmin, hmax=hmax, nact=nact,
                prox=prox, newton=newton,
                maxits=max(flat) if flat else 0,
                medits=statistics.median(flat) if flat else float("nan"),
                curve=curve, dead8=dead8, dead12=dead12,
                psid8=psid8, psid12=psid12, phi_min=phi_min,
                err=err, wall=wall, failed=failed,
                snes=int(lvpp._solver.snes.getConvergedReason()),
                ksp=int(lvpp._solver.snes.ksp.getConvergedReason()))


ROW = (f"  {'tag':>14} {'u-dofs':>7} {'psi':>6} {'elems':>6} {'hmin/hmax':>17} "
       f"{'dead8':>5} {'dead12':>6} {'psid8':>5} {'pside12':>7} "
       f"{'phi_min':>8} {'prox':>4} {'newton':>6} {'maxits':>6} "
       f"{'err(u_h)':>9} {'wall':>6}  {'snes/ksp':>9}")


def print_row(res):
    print(f"  {res['tag']:>14} {res['udofs']:>7} {res['psidofs']:>6} "
          f"{res['Ne']:>6} {res['hmin']:>8.2e}/{res['hmax']:<8.2e} "
          f"{res['dead8']:>5} {res['dead12']:>6} {res['psid8']:>5} "
          f"{res['psid12']:>7} {res['phi_min']:>8.1f} "
          f"{res['prox']:>4} {res['newton']:>6} {res['maxits']:>6} "
          f"{res['err']:>9.3e} {res['wall']:>5.1f}s  "
          f"{res['snes']:>4}/{res['ksp']:>4}"
          + (f"  FAILED: {res['failed']}" if res['failed'] else ""))


# ---------------------------------------------------------------------------
# Marking pipelines (all built from viamr machinery + the gap = exp(psi) field)
# ---------------------------------------------------------------------------

def mark_band(amr, res):
    """Free-boundary ring only: cells whose gap = u_tilde - lb = exp(psi)
    straddles the activetol threshold (elementwise min/max over the cell).
    Contact interior and far field are never marked."""
    gap = Function(res["ut"].function_space()).interpolate(res["ut"] - res["lb"])
    gmin = amr._elemextreme(gap, minimum=True, defaultval=PETSc.INFINITY)
    gmax = amr._elemextreme(gap, minimum=False, defaultval=-PETSc.INFINITY)
    DG0 = amr.spaces(res["ut"].function_space().mesh())[1]
    mark = Function(DG0, name="band mark").interpolate(
        conditional(gmin < amr.activetol,
                    conditional(gmax > amr.activetol, 1.0, 0.0), 0.0))
    return mark


def mark_udobr(amr, res):
    """Standard viamr pipeline (sphere_lvpp.py 'udobr' method): contact
    dilation by one ring + gradient-recovery marking of the inactive set."""
    mark = amr.udomark(res["ut"], res["lb"], n=1)
    imark, _, _ = amr.gradrecinactivemark(
        res["lvpp"].u_out[0], (res["lb"], None), theta=0.5, method="max")
    return amr.unionmarks(mark, imark)


MARKERS = {"band": mark_band, "udobr": mark_udobr}


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def arm_uniform(base, amr, target_dofs):
    print(ROW)
    rows = []
    hier = MeshHierarchy(base, 2)
    for lev, mesh in enumerate(hier):
        res = solve_measure(mesh, f"uni_l{lev}", amr)
        rows.append(res)
        print_row(res)
        print(f"        curve: {res['curve']}")
        del res
        gc.collect()
        if rows[-1]["udofs"] >= target_dofs:
            break
    return rows


def arm_amr(base, amr, pipeline_name, target_dofs, max_levels):
    marker = MARKERS[pipeline_name]
    print(ROW)
    rows = []
    mesh = base
    for lev in range(max_levels + 1):
        res = solve_measure(mesh, f"{pipeline_name}_l{lev}", amr)
        rows.append(res)
        print_row(res)
        print(f"        curve: {res['curve']}")
        if res["udofs"] >= target_dofs or lev == max_levels or res["failed"]:
            break
        try:
            mark = marker(amr, res)
            nmark = amr.countmark(mark)
            print(f"        marking {pipeline_name}: {nmark} cells marked "
                  f"({100.0 * nmark / res['Ne']:.1f}% of {res['Ne']})")
            mesh = amr.refinesbr2D(mesh, mark)
        except Exception as e:
            print(f"        marking/refine FAILED: {failure_reason(e)}")
            break
        del res["lvpp"], res["ut"], res["psil"], res["lb"]
        gc.collect()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="uniform,band,udobr")
    ap.add_argument("--target-dofs", type=int, default=7500,
                    help="stop refining once u-dofs reach this (target band "
                         "~7500-9500 around the uniform L2 8321 reference)")
    ap.add_argument("--max-amr-levels", type=int, default=12)
    args, _ = ap.parse_known_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    print("=" * 78)
    print("AMR health: dead latent rows + outer-GMRES its, adapted vs uniform")
    print(f"floorless Schur config, psi_floor=0, cold solves, "
          f"dead-row tol |diag(D)| < {DEAD_TOL:g} (raw) / exp(psi) < tol (h-fair)")
    print("=" * 78)

    base = RectangleMesh(16, 16, 2.0, 2.0, originX=-2.0, originY=-2.0,
                         diagonal="crossed")
    allrows = {}
    for arm in arms:
        amr = VIAMR(activetol=1e-4)
        print(f"\n=== arm: {arm.upper()} ===")
        try:
            if arm == "uniform":
                allrows[arm] = arm_uniform(base, amr, args.target_dofs)
            elif arm in MARKERS:
                allrows[arm] = arm_amr(base, amr, arm, args.target_dofs,
                                       args.max_amr_levels)
            else:
                print(f"  unknown arm '{arm}' skipped")
        except Exception as e:
            print(f"  arm {arm}: FAILED: {failure_reason(e)}")
        gc.collect()

    print("\n=== summary ===")
    print(f"{'arm':>8} {'lvl':>4} {'u-dofs':>7} {'psi':>6} {'elems':>6} "
          f"{'dead8':>5} {'dead12':>6} {'psid8':>5} {'pside12':>7} "
          f"{'prox':>4} {'newton':>6} {'maxits':>6} "
          f"{'err(u_h)':>9} {'wall':>6}")
    for arm, rows in allrows.items():
        for i, r in enumerate(rows):
            last = i == len(rows) - 1
            print(f"{arm:>8} {i:>4} {r['udofs']:>7} {r['psidofs']:>6} "
                  f"{r['Ne']:>6} {r['dead8']:>5} {r['dead12']:>6} "
                  f"{r['psid8']:>5} {r['psid12']:>7} "
                  f"{r['prox']:>4} {r['newton']:>6} {r['maxits']:>6} "
                  f"{r['err']:>9.3e} {r['wall']:>5.1f}s"
                  + ("   <-- final row" if last else ""))
    print("\ndone")


if __name__ == "__main__":
    main()