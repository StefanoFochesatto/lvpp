"""Epsilon-schedule experiment: is psi_floor a tunable knob for the
outer-GMRES growth of the Schur-fieldsplit LVPP solver?

Builds on growth_driver.py (Finding 6 + growth-driver decomposition): the
growth driver is the eps=1e-2 psi_floor -- an exact block-UL PC with the
EXACT UNfloored S^-1 is iteration-flat at every depth, while the same PC
with the FLOORED S reproduces the whole growth curve (145/895/diverged).

Stage 1 (cold-solve epsilon ladder).  For each extracted deep state (J and
RHS are eps-independent; only the PC changes), build the production-semantics
diagonal Schur PC exactly as selfp does,

    Sp = (D + eps*M_latent) - B diag(K)^{-1} B^T   (sparse),

factor Sp with sparse LU (mirrors fieldsplit_1 preonly+lu), keep exact K^-1
(mirrors fieldsplit_0 preonly+mumps-lu), and measure cold outer-GMRES its
(rtol 1e-6, restart 250) for eps in {1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8}.
Also one drift-concentrated entry: Sp built from
base*M + c*assemble(alpha*|lambda| inner(phi_i, phi_j) dx) (the exact lvpp
psi_floor_drift form with alpha frozen at the converged value).

Stage 2 (full-solve promotion).  End-to-end LVPP solves with the winner
config (schur/upper/selfp/LUS, matfree Amat + assembled floored Pmat) with
psi_floor=<eps>, levels 0-2, per-Newton-solve outer-its monitoring.

Stage 3 (drift-concentrated floor, full solve): psi_floor=1e-6 +
psi_floor_drift=2.5e-5  (contact-scale floor alpha*|lambda|*c ~ 10*4*2.5e-5
= 1e-3).

Differences from growth_driver.py:
  * gmres_its computes the honest rel-resid ||Jx - b|| / ||b|| (the parent
    harness reported ||Jx||/||b||; its and reason were always valid).
  * only the sparse diagonal-approximation Schur is used (no dense true-S
    variants): this experiment varies eps, not the coupling treatment.
  * extract_state additionally stores the drift-weighted latent mass matrix
    Mdrift = assemble(alpha*|lambda| inner(trial,test) dx) and the final
    multiplier, so the drift floor can be tested cold without re-solving.

Usage (env guard REQUIRED):
  PETSC_DIR=/home/stefano/firedrake/petsc PETSC_ARCH=arch-firedrake-default \
  OMP_NUM_THREADS=1 venv-firedrake/bin/python experiments/eps_schedule.py \
      --stage1 [--levels 0,1,2]
  ... --stage2 --eps-list 1e-3,1e-4 [--levels 0,1,2]
  ... --stage3 [--levels 0,1,2]

Serial only.  Read-only access to lvpp internals; no lvpp.py edits.
"""
import argparse
import gc
import statistics
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from firedrake import (Constant, DirichletBC, Function, FunctionSpace,
                       MeshHierarchy, RectangleMesh, SpatialCoordinate,
                       TestFunction, TrialFunction, assemble, conditional, dx,
                       errornorm, exp, grad, inner, le, ln, sqrt)
from firedrake.petsc import PETSc
from lvpp import LVPP

r0 = 0.9
AFREE = 0.697965148223374
A_, B_ = 0.680259411891719, 0.471519893402112

BASELINE_PSI_FLOOR = 1e-2
FCP = {"quadrature_degree": 6}

EPS_LADDER = [0.0, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]

# drift-concentrated floor: contact-scale alpha*|lambda|*c ~ 1e-3 with
# lambda ~ 4 at the cap and alpha_max = 10  ->  c ~ 2.5e-5
DRIFT_BASE = 1e-6
DRIFT_C = 2.5e-5

LU_REFS = {0: (8, 21, 1.553e-2), 1: (11, 23, 3.599e-3), 2: (8, 19, 8.375e-4)}
BASELINE_CURVE = "45 (L0) -> 130-560 (L1) -> 400-926 (L2)"

# Winner config "schur upper selfp LUS" (schur_probe.py), matfree Amat +
# assembled floored Pmat + selfp Schur + LU blocks.
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

REASON_NAMES = {
    2: "RTOL", 3: "ATOL", 4: "ITS",
    -3: "DIVERGED_ITS", -4: "DIVERGED_DTOL", -5: "DIVERGED_BREAKDOWN",
    -6: "DIVERGED_BICG", -7: "DIVERGED_NONSYMM", -8: "DIVERGED_INDEF_PC",
    -9: "DIVERGED_INDEF_MAT", -11: "DIVERGED_PCSETUP",
}


def psiUFL(r):
    psi0 = np.sqrt(1.0 - r0 * r0)
    dpsi0 = -r0 / psi0
    return conditional(le(r, r0), sqrt(1.0 - r * r), psi0 + dpsi0 * (r - r0))


def uexactUFL(r):
    return conditional(le(r, AFREE), psiUFL(r), -A_ * ln(r) + B_)


def csr(m):
    r, c, v = m.getValuesCSR()
    return sp.csr_matrix((v, c, r), shape=m.getSize())


def field_ises(Jm, n0, n1):
    """Field ISes for the 2-field mixed system (u block first, latent last)."""
    is0 = PETSc.IS().createGeneral(np.arange(n0, dtype=np.int32), comm=Jm.comm)
    is1 = PETSc.IS().createGeneral(np.arange(n0, n0 + n1, dtype=np.int32),
                                   comm=Jm.comm)
    return is0, is1


def per_prox_its(groups, newton_its):
    """Group per-solve outer-GMRES its by proximal solve (schur_percell)."""
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
# State extraction at a converged deep state (extended growth_driver copy)
# ---------------------------------------------------------------------------

def extract_state(mesh, tag, max_prox=500):
    """Run the winner config (psi_floor 1e-2) to convergence; return the
    blocks J/K/B/D/M, a real Newton RHS, and the drift-weighted latent mass
    matrix.  J and the RHS do not depend on eps, so ONE extraction per level
    serves the whole epsilon ladder."""
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name="u")
    x, y = SpatialCoordinate(mesh)
    r = sqrt(x * x + y * y)
    energy = 0.5 * inner(grad(u), grad(u)) * dx  # f = 0
    psi = Function(V, name="psi").interpolate(psiUFL(r))
    bc = DirichletBC(V, uexactUFL(r), "on_boundary")
    sp_dict = dict(SP_WINNER)
    lvpp = LVPP(energy=energy, u=u, bounds=(psi, None), bcs=bc,
                alpha_rule="double_exponential",
                alpha_parameters={"alpha_max": 10.0},
                increment_norm="H1", verbose=False,
                psi_floor=BASELINE_PSI_FLOOR,
                form_compiler_parameters=FCP,
                solver_parameters=sp_dict, name=tag)

    state = {"groups": [], "rhss": []}

    def ksp_mon(ksp, its, rnorm):
        if its == 1:
            state["groups"].append(1)
            state["rhss"].append(ksp.getRhs().copy())
        elif its > 1:
            if state["groups"]:
                state["groups"][-1] = its
            else:
                state["groups"].append(its)

    lvpp._solver.snes.ksp.setMonitor(ksp_mon)
    t0 = time.time()
    lvpp.solve(tol=1e-4, max_proximal_iterations=max_prox)
    wall = time.time() - t0

    per_prox, leftover = per_prox_its(state["groups"],
                                      list(lvpp.newton_iterations))
    curve = [max(ch) if ch else 0 for ch in per_prox]

    # the RHS of the linear solve that needed the most outer its
    idx = int(np.argmax(state["groups"]))
    bvec = state["rhss"][idx]

    # exact (unfloored) monolithic Jacobian at the converged state
    W = lvpp._z.subfunctions[1].function_space()
    n0 = V.dim()
    n1 = W.dim()
    Jfd = assemble(lvpp._J, mat_type="aij", bcs=lvpp._bcs)
    Jm = Jfd.M.handle
    is0, is1 = field_ises(Jm, n0, n1)
    Km = Jm.createSubMatrix(is0, is0)
    Bm = Jm.createSubMatrix(is0, is1)
    Dm = Jm.createSubMatrix(is1, is1)
    K = csr(Km)
    B = csr(Bm)
    D = csr(Dm)
    dk = Km.getDiagonal().getArray(readonly=True).copy()
    Km.destroy()
    Bm.destroy()
    Dm.destroy()
    is0.destroy()
    is1.destroy()

    # latent mass matrix M (the floor is a FULL mass matrix, Finding 6.4)
    Mfd = assemble(inner(TrialFunction(W), TestFunction(W)) * dx,
                   mat_type="aij")
    M = csr(Mfd.M.handle)

    # drift-weighted latent mass matrix: assemble(alpha*|lambda| inner) with
    # alpha and the multiplier frozen at the converged state -- the exact
    # lvpp psi_floor_drift form.  base + c*Mdrift == the variable-coefficient
    # floor for any (base, c) pair, so this ONE assembly serves the ladder.
    drift = lvpp._drift[0]
    alpha = float(lvpp._alpha)
    ztr1, zt1 = TrialFunction(W), TestFunction(W)
    Mdriftfd = assemble(alpha * abs(drift) * inner(ztr1, zt1) * dx,
                        mat_type="aij", form_compiler_parameters=FCP)
    Mdrift = csr(Mdriftfd.M.handle)

    lam = drift.dat.data.copy()
    phi = lvpp._z.subfunctions[1].dat.data.copy()
    dd = D.diagonal().copy()
    dead8 = int(np.sum(np.abs(dd) < 1e-8))
    dead12 = int(np.sum(np.abs(dd) < 1e-12))

    err = errornorm(uexactUFL(r), lvpp.u_out[0])
    return dict(level_tag=tag, n0=n0, n1=n1, dofs=n0 + n1, alpha=alpha,
                Jm=Jm, Jfd=Jfd, K=K, B=B, BT=B.T.tocsr(), D=D, M=M,
                Mdrift=Mdrift, dk=dk, bvec=bvec, bnorm=float(bvec.norm()),
                phi=phi, lam=lam, lam_max=float(np.abs(lam).max()),
                phi_min=float(phi.min()), dead8=dead8, dead12=dead12,
                prox=lvpp.proximal_iterations,
                newton=sum(lvpp.newton_iterations), err=float(err), wall=wall,
                curve=curve, max_its_captured=int(state["groups"][idx]))


# ---------------------------------------------------------------------------
# Stage 1: cold-solve epsilon ladder through a production-semantics PC
# ---------------------------------------------------------------------------

class BlockLUPC:
    """Python PC: exact block-UL factorization P = [[K, B], [0, S_pc]].

    apply(x, y): w2 = S^-1(b2 - B^T (K^-1 b1)); w1 = K^-1(b1 - B w2).
    K^-1 is a sparse LU (exact, mirrors fieldsplit_0 mumps-lu); S^-1 is a
    sparse-LU callable on the floored diagonal-approximation Sp (mirrors
    fieldsplit_1 mumps-lu on the selfp-built Sp).
    """

    def __init__(self, n0, klu, B, BT, ssolver):
        self.n0 = n0
        self.klu = klu
        self.B = B
        self.BT = BT
        self.ssolver = ssolver

    def setUp(self, pc):
        pass

    def apply(self, pc, x, y):
        # petsc4py idiom: the context-manager form raises PETSc error 101 here
        xr = x.getArray(readonly=True).copy()
        b1 = xr[:self.n0]
        b2 = xr[self.n0:]
        t = self.klu.solve(b1)
        w2 = self.ssolver(b2 - self.BT.dot(t))
        w1 = self.klu.solve(b1 - self.B.dot(w2))
        y.setArray(np.concatenate([w1, w2]))


def gmres_its(Jm, bvec, ctx):
    """Cold outer GMRES on fixed J with a python PC.

    rel-resid is the honest ||Jx - b|| / ||b|| (NOT ||Jx||/||b||)."""
    ksp = PETSc.KSP().create(comm=Jm.comm)
    ksp.setTolerances(rtol=1e-6, max_it=1000)
    ksp.setGMRESRestart(250)
    ksp.setOperators(Jm)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.PYTHON)
    pc.setPythonContext(ctx)
    x = Jm.createVecRight()
    ksp.setUp()
    t0 = time.time()
    ksp.solve(bvec, x)
    solve_wall = time.time() - t0
    its = ksp.getIterationNumber()
    reason = int(ksp.getConvergedReason())
    r = Jm.createVecLeft()
    Jm.mult(x, r)
    r.axpy(-1.0, bvec)  # r = Jx - b
    rel = float(r.norm() / bvec.norm())
    x.destroy()
    r.destroy()
    ksp.destroy()
    return its, reason, rel, solve_wall


def diag_schur_pc(st, Dfloor):
    """Build the production-semantics diagonal-Schur PC:
    Sp = Dfloor - B diag(K)^{-1} B^T (sparse), exact K^-1 via sparse LU.
    Returns (ctx, klu) or raises on singular Sp."""
    n0 = st["n0"]
    klu = splu(st["K"].tocsc())
    Bsc = st["B"] @ sp.diags(1.0 / st["dk"])
    Sp = (Dfloor.tocsr() - (Bsc @ st["B"].T).tocsr()).tocsc()
    sslu = splu(Sp)
    # sanity: the factorization must actually solve (catch NaN/Inf pivots)
    probe = np.ones(Sp.shape[0])
    out = sslu.solve(probe)
    if not np.all(np.isfinite(out)):
        raise RuntimeError("Sp sparse LU produced non-finite solution")
    rel_probe = float(np.linalg.norm(Sp @ out - probe)
                      / max(np.linalg.norm(probe), 1e-300))
    ctx = BlockLUPC(n0, klu, st["B"], st["BT"], sslu.solve)
    return ctx, rel_probe


def run_ladder_entry(st, Dfloor, label):
    """One cold-solve measurement; returns dict or an error marker."""
    try:
        ctx, rel_probe = diag_schur_pc(st, Dfloor)
    except Exception as e:
        msg = str(e).strip().replace("\n", " ")[:120]
        print(f"   {label:<16} PC BUILD FAILED: {msg}")
        return dict(label=label, status="pcfail", detail=msg)
    try:
        its, reason, rel, swall = gmres_its(st["Jm"], st["bvec"], ctx)
    except Exception as e:
        msg = str(e).strip().replace("\n", " ")[:120]
        print(f"   {label:<16} SOLVE RAISED: {msg}")
        return dict(label=label, status="raise", detail=msg)
    rname = REASON_NAMES.get(reason, str(reason))
    ok = reason == 2 and np.isfinite(rel)
    print(f"   {label:<16} its={its:>5}  reason={rname:<16} "
          f"rel-resid={rel:.2e}  Sp-solve={rel_probe:.1e}  "
          f"solve={swall:6.1f}s")
    return dict(label=label, status="ok" if ok else "bad", its=its,
                reason=rname, rel=rel, wall=swall, rel_probe=rel_probe)


def stage1_level(st, ladder=None):
    """Epsilon ladder on one extracted state."""
    ladder = ladder if ladder is not None else EPS_LADDER
    print(f"\n== LEVEL state '{st['level_tag']}' | dofs {st['dofs']} "
          f"(n0={st['n0']}, n1={st['n1']}) | alpha={st['alpha']:.4f} | "
          f"phi_min={st['phi_min']:.1f} | lam_max={st['lam_max']:.2f} | "
          f"dead rows |d|<1e-8: {st['dead8']} "
          f"(|d|<1e-12: {st['dead12']}) ==")
    print(f"   baseline run: prox={st['prox']} newton={st['newton']} "
          f"err={st['err']:.3e} wall={st['wall']:.1f}s   "
          f"(LU ref {LU_REFS.get(int(st['level_tag'][-1]), ('?', '?', '?'))})")
    print(f"   per-prox outer-its curve (max per solve): {st['curve']}   "
          f"[baseline {BASELINE_CURVE}]")
    print(f"   captured rhs: from the solve with its={st['max_its_captured']} "
          f"||b||={st['bnorm']:.3e}")

    results = {}
    for eps in ladder:
        label = f"eps={eps:g}"
        Dfloor = (st["D"] + eps * st["M"]).tocsr()
        results[label] = run_ladder_entry(st, Dfloor, label)

    # drift-concentrated floor, cold: eps(x) = base + c*alpha*|lambda|,
    # contact-scale floor ~ c * alpha * lam_max ~ 1e-3 at (c=2.5e-5)
    label = f"drift b={DRIFT_BASE:g} c={DRIFT_C:g}"
    Dfloor = (DRIFT_BASE * st["M"] + DRIFT_C * st["Mdrift"]).tocsr()
    contact_scale = DRIFT_C * st["alpha"] * st["lam_max"]
    print(f"   (drift contact-scale floor alpha*lam_max*c = "
          f"{contact_scale:.2e})")
    results[label] = run_ladder_entry(st, Dfloor, label)
    return results


# ---------------------------------------------------------------------------
# Stage 2: end-to-end promotion of surviving epsilons
# ---------------------------------------------------------------------------

def instrument(lvpp):
    """Outer-GMRES its per Newton solve (schur_percell pattern)."""
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
    return state


def run_full(mesh, tag, psi_floor, psi_floor_drift=0.0, max_prox=500):
    """Full LVPP solve with the winner config at the given floor setting.
    selfp builds Sp from the assembled floored Pmat, so no PC surgery."""
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
                psi_floor=psi_floor, psi_floor_drift=psi_floor_drift,
                form_compiler_parameters=FCP,
                solver_parameters=dict(SP_WINNER), name=tag)
    state = instrument(lvpp)
    t0 = time.time()
    lvpp.solve(tol=1e-4, max_proximal_iterations=500)
    wall = time.time() - t0
    err = errornorm(uexactUFL(r), lvpp.u_out[0])
    per_prox, leftover = per_prox_its(state["groups"],
                                      list(lvpp.newton_iterations))
    flat = [i for chunk in per_prox + ([leftover] if leftover else [])
            for i in chunk]
    curve = [max(ch) if ch else 0 for ch in per_prox]
    snes = lvpp._solver.snes
    return dict(tag=tag, dofs=V.dim(), prox=lvpp.proximal_iterations,
                newton=sum(lvpp.newton_iterations), err=float(err), wall=wall,
                its=flat, curve=curve,
                snes=int(snes.getConvergedReason()),
                ksp=int(snes.ksp.getConvergedReason()))


def print_full_row(lev, v):
    if v is None:
        print(f"  L{lev}: FAILED")
        return
    med = statistics.median(v["its"]) if v["its"] else float("nan")
    p, n, e = LU_REFS.get(lev, (0, 0, 0.0))
    print(f"  L{lev}: prox={v['prox']} newton={v['newton']} "
          f"err={v['err']:.3e} (LU ref {p}/{n} {e:.3e}) "
          f"snes={v['snes']} ksp={v['ksp']} "
          f"outer-its max={max(v['its']) if v['its'] else 0} "
          f"med={med:.1f} wall={v['wall']:.1f}s\n"
          f"       per-prox curve: {v['curve']}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--stage3", action="store_true")
    ap.add_argument("--eps-list", default=",".join(f"{e:g}" for e in EPS_LADDER))
    ap.add_argument("--levels", default="0,1,2")
    ap.add_argument("--max-prox", type=int, default=500)
    args = ap.parse_args()
    if not (args.stage1 or args.stage2 or args.stage3):
        ap.error("choose --stage1 and/or --stage2 and/or --stage3")

    ladder = [float(s) for s in args.eps_list.split(",")]
    base = RectangleMesh(16, 16, 2.0, 2.0, originX=-2.0, originY=-2.0,
                         diagonal="crossed")
    hier = MeshHierarchy(base, 2)
    levels = [int(s) for s in args.levels.split(",")]

    if args.stage1:
        print("=" * 78)
        print("STAGE 1: cold-solve epsilon ladder (fixed J + real RHS; only "
              "the PC changes)")
        print(f"baseline outer-GMRES per-solve growth: {BASELINE_CURVE}")
        print("=" * 78)
        for lev, mesh in enumerate(hier):
            if lev not in levels:
                continue
            try:
                st = extract_state(mesh, f"eps1_l{lev}",
                                   max_prox=args.max_prox)
            except Exception as e:
                print(f"level {lev}: extraction FAILED: {failure_reason(e)}")
                continue
            try:
                stage1_level(st, ladder=ladder)
            except Exception as e:
                print(f"level {lev}: ladder FAILED: {failure_reason(e)}")
            del st
            gc.collect()

    if args.stage2:
        print("\n" + "=" * 78)
        print("STAGE 2: end-to-end promotion (winner config, psi_floor=eps)")
        print(f"baseline outer-GMRES per-solve growth: {BASELINE_CURVE}")
        print("=" * 78)
        for eps in ladder:
            print(f"\n-- psi_floor = {eps:g} --")
            for lev, mesh in enumerate(hier):
                if lev not in levels:
                    continue
                try:
                    v = run_full(mesh, f"eps2_l{lev}_eps{eps:g}", psi_floor=eps)
                except Exception as e:
                    print(f"  L{lev}: FAILED: {failure_reason(e)}")
                    continue
                print_full_row(lev, v)
                del v
                gc.collect()

    if args.stage3:
        print("\n" + "=" * 78)
        print(f"STAGE 3: drift-concentrated floor (psi_floor={DRIFT_BASE:g}, "
              f"psi_floor_drift={DRIFT_C:g}, contact-scale "
              f"~{DRIFT_C * 10 * 4:.1e})")
        print(f"baseline outer-GMRES per-solve growth: {BASELINE_CURVE}")
        print("=" * 78)
        for lev, mesh in enumerate(hier):
            if lev not in levels:
                continue
            try:
                v = run_full(mesh, f"eps3_drift_l{lev}",
                             psi_floor=DRIFT_BASE,
                             psi_floor_drift=DRIFT_C)
            except Exception as e:
                print(f"  L{lev}: FAILED: {failure_reason(e)}")
                continue
            print_full_row(lev, v)
            del v
            gc.collect()


if __name__ == "__main__":
    main()
