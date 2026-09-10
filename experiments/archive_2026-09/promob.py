"""Variant-B promotion: END-TO-END LVPP solves with the matfree true-S
inner-CG Schur treatment (grade_pc.py winner) on the graded band-mesh chain.

Question (RESEARCH.md, "Grading-robust S-PC experiment"): the floorless
production Schur config (schur/upper/selfp, both blocks preonly+MUMPS-LU,
psi_floor 0, outer GMRES rtol 1e-6 restart 250) has ~N^0.8 outer-its growth
on uniform meshes (14/29/48/105/329 at L0-L4) but BLOWS UP on graded
(band-adapted) meshes (max-its 297 at band_l4, 323 at band_l5, and the band
chain DIVERGES ksp=-5 at 14245 dofs).  In the COLD harness (grade_pc.py,
exact block-UL PC on the assembled J) variant B -- replacing the Schur-block
solve by a capped inner CG on -S (S = D - B^T K^{-1} B applied matrix-free,
one exact K solve per matvec, Jacobi-scaled, cap 40, inner rtol 1e-4) --
gives outer its 6/10/15 on uni_l2/band_l4/band_l5 vs base 22/39/51.  The
cold harness UNDER-reproduces graded production max-its (39/51 vs 297/323),
so this file measures the promotion END-TO-END through full LVPP solves.

Attachment route (documented, the assignment's fallback -- chosen for a
measured reason): petsc4py instantiates a `pc_python_type` context with NO
constructor arguments (libpetsc4py.pyx `createcontext`: `return cls()`), so
a sub-PC PYTHON context on fieldsplit_1 cannot receive the lvpp handle and
cannot rebuild its matrices per proximal solve.  Route taken: a TOP-LEVEL
pc_type python context (petsc4py PCType PYTHON, attached manually via
pc.setPythonContext, rankk_correction.py style) whose apply replicates the
production upper-factorization semantics with the Schur block replaced:

    w2 = S_solve(b2)              inner CG (or gmres) on -S, matfree S,
                                  Jacobi-scaled, rtol 1e-4, cap 40
    w1 = K^{-1}(b1 - B w2)        exact sparse-LU solve on K

Deviations from the production config, both structural no-ops for the
mathematics and documented honestly:
  * K^{-1} is a scipy splu (serial SuperLU) instead of MUMPS-LU -- same
    exact factorization, this harness is serial-only; PETSc fieldsplit
    plumbing is bypassed wholesale by the monolithic PC, so the MUMPS
    sub-KSPs are simply not instantiated.
  * the Pmat is the SAME assembled unfloored Jacobian the production config
    uses (mat_type matfree, pmat_type aij, psi_floor 0), so K/B/D extracted
    in setUp are the exact production blocks.
  * outer KSP is FGMRES (production: GMRES) -- the inner CG makes the Schur
    apply variable-iteration, so outer flexibility is required; same
    rtol 1e-6, restart 250, max_it 1000.

Per-prox refresh: lvpp._solver.solve is wrapped to set ctx.pending (rankk
pattern); setUp rebuilds K/B/D + splu on every pending flag, so each Newton
linear solve uses the current Jacobian blocks.

CRITICAL petsc4py idiom: in PC callbacks use x.getArray(readonly=True).copy()
and y.setArray(arr) -- NEVER the context-manager form (PETSc error 101).

Inner-solver symmetry note: the S action is applied via plain matvecs
(D v - B^T (K^{-1} (B v))) through a scipy LinearOperator -- NO symmetric
assumption anywhere in the operator.  Only the CG *solver* assumes -S is
SPD (true for this obstacle/energy problem: S strictly negative definite,
h^2-scaled).  For nonsymmetric/indefinite LVPP systems the same structure
runs unchanged with --inner-method gmres (scipy gmres on the same
operator/preconditioner pair), measured below as a second arm.

MEASURED VERDICT (2026-09-10, this file, serial, budget-capped):

  Uniform control (inner cg rtol 1e-4 cap 40) -- LU-identical solves:
    uni_l0  545 dofs   8/21  err 1.553e-02  maxits 62  wall 3.0s
    uni_l1  2113       11/23 err 3.599e-03  maxits 65  wall 8.7s
    uni_l2  8321        8/19 err 8.365e-04  maxits 70  wall 28.9s
  (LU refs 8/21/1.553e-02, 11/23/3.599e-3, 8/19/8.375e-4; selfp maxits
  14/29/48.)  prox/newton/err match the LU refs exactly at every level
  (uni_l2 err differs in the 4th digit).  The maxits picture INVERTS the
  cold-harness prediction: end-to-end the FIRST (coldest) Newton solve
  costs 62-70 outer its at every uniform level (vs selfp <=48 anywhere),
  while the DEEP proximal steps collapse to 3-10 -- where the cold harness
  measured (converged deep state) variant B gave 6.  Wall: the inner CG
  costs one exact K solve per matvec (avg 13-17 inner its/solve, 4-7k
  matvecs/level; s_wall 22.0s of 28.9s at uni_l2) -- a ~4.5x wall
  regression vs production (~6.5s) on uniform.

  Band chain (inner cg): l0-l4 all converge, band_l4 (3613 u-dofs,
  22.8x grading) 39 prox/49 newton, err 5.020e-3, maxits 161 vs selfp 297;
  the deep proximal solves (where selfp's graded blowup lives) collapse
  from 297 to 4-5 outer its.  band_l5 STALLS: every linear solve
  converges in 4-5 its but the proximal increment plateaus at 4.243e-4
  (tol 1e-4) for hundreds of proximal steps on the variant-B l5 chain
  (6318 dofs; the chain's own marking diverged from production's 6861-dof
  l5 through roundoff-driven mark differences).  Mechanism (corroborated
  by the gmres arm below): at l5-depth the inner CG hits the cap-40
  before reaching rtol 1e-4, the Schur apply is too inexact for the
  nonlinear fixed point, and the proximal map stalls inside the tol band.

  Band chain (inner gmres arm, Main request): same chain, scipy gmres
  inner.  CAVEAT: scipy gmres maxiter counts RESTART CYCLES (40 cycles x
  restart 40 = up to 1600 matvecs), so this arm ran a LOOSER effective
  cap than the cg arm (avg inner its/solve 15.0/21.8/24.4/26.2/89.6/54.8
  at l0-l5 vs cg's 13.3/16.2/16.6/17.6/25.6).  Results:
    band_l4  9/19 err 5.027e-3 maxits 280 wall 160.0s
    band_l5 (6661 u-dofs, 45.6x) SOLVES: 8/17 err 4.995e-3 maxits 188
            wall 116.6s   (selfp production: maxits 323)
    band_l6: 14 linear solves recorded, ALL CONVERGED_RTOL (2-192 its)
            before the wall budget cut the run mid-proximal-solve -- the
            selfp ksp=-5 DIVERGENCE at l6 does not occur under variant B.
  Symmetry note for Main: the S action is applied via plain matvecs
  (D v - B^T K^{-1} B v) with no symmetric assumption anywhere; ONLY the
  cg solver kernel assumes -S is SPD.  The gmres arm ran unchanged with
  the same operator/preconditioner code -- the structure survives the
  inner-GMRES swap for nonsymmetric/indefinite LVPP systems, at ~2-3x
  the inner-iteration cost (89.6 vs 25.6 avg at band_l4).

  Production recommendation: variant B (matfree true-S inner Krylov,
  Jacobi, rtol 1e-4) FIXES the graded-mesh outer-its blowup end-to-end
  (297->161 at band_l4, 323->188 at band_l5, no l6 divergence) and
  preserves LU-identical Newton/error quality, at two costs: (a) a
  62-70-its coldest-solve penalty on uniform meshes (deep steps improve),
  and (b) a ~2-5x wall penalty from the exact-K-per-matvec inner solve.
  The cap-40 stall at l5 depth says the cap must scale with problem
  difficulty (or the rtol tightened) -- cap 40/rtol 1e-4 is NOT safe
  end-to-end at 45.6x grading with cg; the gmres arm (effectively
  uncapped) is safe but its cap semantics must be converted to
  total-matvec accounting in any production rollout.

References (RESEARCH.md, not re-measured here):
  uniform LU refs:        8/21/1.553e-02, 11/23/3.599e-3, 8/19/8.375e-4
  uniform selfp max-its:  14/29/48 (L0/L1/L2; 105/329 at L3/L4)
  band selfp max-its:     297 (band_l4, 3613 u-dofs, 22.8x grading),
                          323 (band_l5, 6861 u-dofs, 45.6x grading);
                          band chain DIVERGES (ksp=-5) at l6, 14245 u-dofs
"""

import argparse
import gc
import os
import statistics
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg, gmres, splu

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

# Production floorless Schur config (amr_health.py SP_WINNER, psi_floor=0),
# with the outer KSP switched to FGMRES and the PC switched to the monolithic
# python PC (TrueSPC below).  The fieldsplit sub-KSP options are kept in the
# dict for provenance but are inert here: TrueSPC replaces the whole PC.
SP_VARIANT_B = {
    "mat_type": "matfree",
    "pmat_type": "aij",
    "pc_fieldsplit_use_amat": False,
    "snes_linesearch_type": "l2",
    "snes_linesearch_maxlambda": 1.0,
    "snes_rtol": 1e-6,
    "snes_max_it": 100,
    "ksp_type": "fgmres",
    "ksp_rtol": 1e-6,
    "ksp_max_it": 1000,
    "ksp_gmres_restart": 250,
    "ksp_converged_reason": None,
    "pc_type": "python",
    # production block config (inert under TrueSPC; kept for documentation)
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

LU_REFS = {0: (8, 21, 1.553e-2), 1: (11, 23, 3.599e-3), 2: (8, 19, 8.375e-4)}
SELFP_MAXITS = {("uniform", 0): 14, ("uniform", 1): 29, ("uniform", 2): 48,
                ("uniform", 3): 105, ("uniform", 4): 329,
                ("band", 4): 297, ("band", 5): 323}

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
# Variant-B PC: production block-UL with the Schur block replaced by
# matfree true-S inner CG (grade_pc.py TrueSCG machinery, validated)
# ---------------------------------------------------------------------------

class TrueSPC:
    """Top-level python PC.  apply = block-UL (upper fact) semantics:

        w2 = S_solve(b2)          capped inner CG on -S (matfree S),
        w1 = K^{-1}(b1 - B w2)    exact sparse-LU on K.

    S action (matvec only, no symmetry assumed):
        S v = D v - B^T (K^{-1} (B v)).
    K/B/D are rebuilt from the assembled unfloored Pmat whenever the LVPP
    solver re-solves (per proximal step), via the rankk wrap_solve pattern.
    """

    def __init__(self, lvpp, inner_rtol=1e-4, inner_cap=40,
                 inner_method="cg"):
        self.lvpp = lvpp
        self.rtol = float(inner_rtol)
        self.cap = int(inner_cap)
        self.method = inner_method
        self.pending = True
        self.pc = lvpp._solver.snes.ksp.getPC()
        self.klu = None
        self.n0 = self.lvpp.u_out[0].function_space().dim()
        self.n1 = (self.lvpp._z.subfunctions[1].function_space().dim())
        # stats
        self.s_calls = 0        # Schur-solve (PC apply) count
        self.inner_total = 0    # total inner Krylov iterations
        self.inner_its = []     # per-S-solve inner iteration counts
        self.s_wall = 0.0       # wall inside solve_S (includes matvecs)
        self.matvecs = 0        # S matvec count (== exact K solves)
        self.pc_wall = 0.0      # wall inside apply
        self.finite_fail = 0

    # -- petsc4py python-PC interface ---------------------------------------
    def setUp(self, pc):
        if self.pending or self.klu is None:
            self._rebuild()
            self.pending = False

    def apply(self, pc, x, y):
        t0 = time.time()
        # CRITICAL idiom: plain getArray(readonly=True).copy() / setArray
        xr = x.getArray(readonly=True).copy()
        b1 = xr[:self.n0]
        b2 = xr[self.n0:]
        w2 = self.solve_S(b2)
        w1 = self.klu.solve(b1 - self.B.dot(w2))
        out = np.concatenate([w1, w2])
        if not np.all(np.isfinite(out)):
            self.pc_wall += time.time() - t0
            raise RuntimeError("TrueSPC apply produced non-finite output")
        y.setArray(out)
        self.pc_wall += time.time() - t0

    # -- setup / refresh -----------------------------------------------------
    def _rebuild(self):
        A, P = self.pc.getOperators()
        n0, n1 = self.n0, self.n1
        n = P.getSize()[0]
        assert n == n0 + n1, f"P size {n} != n0+n1 {n0}+{n1}"
        is0 = PETSc.IS().createGeneral(np.arange(n0, dtype=np.int32),
                                       comm=P.comm)
        is1 = PETSc.IS().createGeneral(np.arange(n0, n, dtype=np.int32),
                                       comm=P.comm)
        Km = P.createSubMatrix(is0, is0)
        Bm = P.createSubMatrix(is0, is1)
        Dm = P.createSubMatrix(is1, is1)
        K = csr(Km)
        self.B = csr(Bm)
        self.D = csr(Dm)
        Km.destroy()
        Bm.destroy()
        Dm.destroy()
        is0.destroy()
        is1.destroy()
        self.BT = self.B.T.tocsr()
        self.klu = splu(K.tocsc())
        # Jacobi of -S: diag(-S) = -diag(D) + diag(B^T diag(K)^{-1} B)
        dk = K.diagonal().copy()
        coup = self.B.power(2).T.dot(1.0 / np.maximum(dk, 1e-300))
        d = -self.D.diagonal() + coup
        self.minv = 1.0 / np.maximum(d, 1e-300)
        self.dead8 = int(np.sum(np.abs(self.D.diagonal()) < DEAD_TOL))

    # -- Schur solve: capped inner Krylov on -S ------------------------------
    def _s_action(self, v):
        """-S v (SPD for cg).  Pure matvecs: no symmetric assumption."""
        self.matvecs += 1
        return -(self.D.dot(v) - self.BT.dot(self.klu.solve(self.B.dot(v))))

    def solve_S(self, rhs):
        t0 = time.time()
        n1 = self.n1
        Aop = LinearOperator((n1, n1), matvec=self._s_action,
                             dtype=np.float64)
        minv = self.minv
        Mop = LinearOperator((n1, n1), matvec=lambda v: minv * v,
                             dtype=np.float64)
        count = [0]

        def cb(x):
            count[0] += 1

        if self.method == "cg":
            x, info = cg(Aop, rhs, rtol=self.rtol, atol=0.0,
                         maxiter=self.cap, M=Mop, callback=cb)
        elif self.method == "gmres":
            x, info = gmres(Aop, rhs, rtol=self.rtol, atol=0.0,
                            maxiter=self.cap, restart=self.cap, M=Mop,
                            callback=cb, callback_type="pr_norm")
        else:
            raise ValueError(f"unknown inner method {self.method}")
        self.s_wall += time.time() - t0
        self.s_calls += 1
        self.inner_total += count[0]
        self.inner_its.append(count[0])
        if not np.all(np.isfinite(x)):
            raise RuntimeError(
                f"inner {self.method} produced non-finite solution "
                f"(info={info})")
        return -x  # S^{-1} rhs = -(-S)^{-1} rhs  (S negative definite)

    # -- per-prox refresh hook ------------------------------------------------
    def wrap_solve(self):
        orig = self.lvpp._solver.solve

        def wrapped(*a, **k):
            self.pending = True
            return orig(*a, **k)
        self.lvpp._solver.solve = wrapped


# ---------------------------------------------------------------------------
# One end-to-end solve + metrics (eps_schedule.run_full + amr_health census)
# ---------------------------------------------------------------------------

def run_level(mesh, tag, inner_rtol, inner_cap, inner_method, max_prox=500):
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
                solver_parameters=dict(SP_VARIANT_B), name=tag)

    ctx = TrueSPC(lvpp, inner_rtol=inner_rtol, inner_cap=inner_cap,
                  inner_method=inner_method)
    pc = lvpp._solver.snes.ksp.getPC()
    pc.setType(PETSc.PC.Type.PYTHON)
    pc.setPythonContext(ctx)
    ctx.wrap_solve()

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
    failed = ""
    try:
        lvpp.solve(tol=1e-4, max_proximal_iterations=max_prox)
    except Exception as e:
        failed = failure_reason(e, lvpp)
    wall = time.time() - t0

    flat, curve = [], []
    ut = psil = None
    err = float("nan")
    hmin = hmax = float("nan")
    psid8 = -1
    phi_min = float("nan")
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
        _, _, hmin, hmax = VIAMR(activetol=1e-4).meshsizes(mesh)
        prox = lvpp.proximal_iterations
        newton = sum(lvpp.newton_iterations or [])
        phi = lvpp._z.subfunctions[1].dat.data.copy()
        phi_min = float(phi.min())
        psid8 = int(np.sum(phi < np.log(DEAD_TOL)))
    except Exception as e:
        failed += " | metric extraction FAILED: " + failure_reason(e)

    istats = dict(calls=ctx.s_calls, inner_total=ctx.inner_total,
                  matvecs=ctx.matvecs, s_wall=ctx.s_wall,
                  pc_wall=ctx.pc_wall,
                  inner_avg=(ctx.inner_total / max(ctx.s_calls, 1)))
    return dict(tag=tag, mesh=mesh, V=V, lvpp=lvpp, ut=ut, psil=psil, lb=lb,
                udofs=V.dim(), hmin=hmin, hmax=hmax, prox=prox, newton=newton,
                maxits=max(flat) if flat else 0,
                medits=statistics.median(flat) if flat else float("nan"),
                curve=curve, psid8=psid8, phi_min=phi_min, err=err, wall=wall,
                failed=failed, inner=istats,
                snes=int(lvpp._solver.snes.getConvergedReason()),
                ksp=int(lvpp._solver.snes.ksp.getConvergedReason()))


ROW = (f"  {'tag':>9} {'u-dofs':>7} {'h ratio':>8} {'psid8':>6} "
       f"{'prox':>4} {'newton':>6} {'maxits':>6} {'err(u_h)':>9} "
       f"{'wall':>7} {'snes/ksp':>9}  {'innerCG(avg its/matvec tot)':>26}")


def print_row(res):
    print(f"  {res['tag']:>9} {res['udofs']:>7} "
          f"{res['hmax'] / max(res['hmin'], 1e-300):>7.1f}x "
          f"{res['psid8']:>6} {res['prox']:>4} {res['newton']:>6} "
          f"{res['maxits']:>6} {res['err']:>9.3e} {res['wall']:>6.1f}s  "
          f"{res['snes']:>4}/{res['ksp']:>4}   "
          f"inner {res['inner']['inner_avg']:>5.1f}/solve "
          f"tot {res['inner']['inner_total']:>6}"
          + (f"  FAILED: {res['failed']}" if res["failed"] else ""))


def udofs_of(mesh):
    return FunctionSpace(mesh, "CG", 1).dim()


# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------

def arm_uniform(base, amr, levels, cfg):
    print(ROW)
    rows = []
    hier = MeshHierarchy(base, max(levels))
    for lev, mesh in enumerate(hier):
        if lev not in levels:
            continue
        res = run_level(mesh, f"uni_l{lev}", **cfg)
        res["udofs"] = udofs_of(mesh)
        rows.append(res)
        print_row(res)
        print(f"          curve: {res['curve']}")
        ref = LU_REFS.get(lev)
        if ref:
            print(f"          LU ref {ref[0]}/{ref[1]}/{ref[2]:.3e};  "
                  f"selfp production max-its "
                  f"{SELFP_MAXITS[('uniform', lev)]}")
        del res
        gc.collect()
    return rows


def mark_band(amr, res):
    """Free-boundary ring marking (amr_health.mark_band copy)."""
    gap = Function(res["ut"].function_space()).interpolate(
        res["ut"] - res["lb"])
    gmin = amr._elemextreme(gap, minimum=True, defaultval=PETSc.INFINITY)
    gmax = amr._elemextreme(gap, minimum=False, defaultval=-PETSc.INFINITY)
    DG0 = amr.spaces(res["ut"].function_space().mesh())[1]
    mark = Function(DG0, name="band mark").interpolate(
        conditional(gmin < amr.activetol,
                    conditional(gmax > amr.activetol, 1.0, 0.0), 0.0))
    return mark


def arm_band(base, amr, max_level, cfg, deadline, t_start):
    print(ROW)
    rows = []
    mesh = base
    for lev in range(max_level + 1):
        if time.time() - t_start > deadline:
            print(f"  wall budget reached; stopping band chain before l{lev}")
            break
        tag = f"band_l{lev}"
        res = run_level(mesh, tag, **cfg)
        res["udofs"] = udofs_of(mesh)
        rows.append(res)
        print_row(res)
        print(f"          curve: {res['curve']}")
        if res["failed"]:
            print(f"  band chain STOPS at l{lev}: {res['failed']}")
            break
        if lev == max_level:
            break
        try:
            mark = mark_band(amr, res)
            nmark = amr.countmark(mark)
            print(f"          marking band: {nmark} cells "
                  f"({100.0 * nmark / mesh.num_cells():.1f}% of "
                  f"{mesh.num_cells()})")
            mesh = amr.refinesbr2D(mesh, mark)
        except Exception as e:
            print(f"          marking/refine FAILED: {failure_reason(e)}")
            break
        del res["lvpp"], res["ut"], res["psil"], res["lb"]
        gc.collect()
    return rows


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summary(allrows, inner_method, inner_rtol, inner_cap):
    print("\n" + "=" * 78)
    print(f"SUMMARY: variant-B end-to-end (inner {inner_method}, rtol "
          f"{inner_rtol:g}, cap {inner_cap}) vs LU refs + selfp production")
    print("=" * 78)
    hdr = (f"{'mesh':>9} {'u-dofs':>7} {'h-ratio':>8} {'prox':>4} "
           f"{'newton':>6} {'err':>9} {'maxits':>6} {'wall':>7} "
           f"{'inner tot':>9} {'avg':>5} {'matvec':>7} {'s-wall':>7}")
    print(hdr)
    for arm, rows in allrows.items():
        for r in rows:
            print(f"{r['tag']:>9} {r['udofs']:>7} "
                  f"{r['hmax'] / max(r['hmin'], 1e-300):>7.1f}x "
                  f"{r['prox']:>4} {r['newton']:>6} {r['err']:>9.3e} "
                  f"{r['maxits']:>6} {r['wall']:>6.1f}s "
                  f"{r['inner']['inner_total']:>9} "
                  f"{r['inner']['inner_avg']:>5.1f} "
                  f"{r['inner']['matvecs']:>7} {r['inner']['s_wall']:>6.1f}s"
                  + ("  FAILED: " + r["failed"] if r["failed"] else ""))


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner-method", default="cg", choices=("cg", "gmres"))
    ap.add_argument("--inner-rtol", type=float, default=1e-4)
    ap.add_argument("--inner-cap", type=int, default=40)
    ap.add_argument("--max-band-level", type=int, default=6)
    ap.add_argument("--uniform-levels", default="0,1,2")
    ap.add_argument("--skip-uniform", action="store_true")
    ap.add_argument("--skip-band", action="store_true")
    ap.add_argument("--budget-min", type=float, default=45.0,
                    help="soft wall deadline for the band chain (minutes)")
    args, _ = ap.parse_known_args()
    levels = {int(s) for s in args.uniform_levels.split(",") if s.strip()}

    print("=" * 78)
    print(f"PROMOB: variant-B promotion (matfree true-S inner-{args.inner_method}"
          f" Schur, rtol {args.inner_rtol:g}, cap {args.inner_cap}) "
          f"END-TO-END on uniform L0-L2 + graded band chain")
    print("=" * 78)

    cfg = dict(inner_rtol=args.inner_rtol, inner_cap=args.inner_cap,
               inner_method=args.inner_method)
    t_start = time.time()
    deadline = args.budget_min * 60.0

    base = RectangleMesh(16, 16, 2.0, 2.0, originX=-2.0, originY=-2.0,
                         diagonal="crossed")
    allrows = {}

    if not args.skip_uniform:
        amr = VIAMR(activetol=1e-4)
        print("\n=== UNIFORM control L0-L2 (LU refs 8/21/1.553e-02, "
              "11/23/3.599e-3, 8/19/8.375e-4; selfp max-its 14/29/48) ===")
        try:
            allrows["uniform"] = arm_uniform(base, amr, levels, cfg)
        except Exception as e:
            print(f"  uniform arm FAILED: {failure_reason(e)}")
        gc.collect()

    if not args.skip_band:
        amr = VIAMR(activetol=1e-4)
        print("\n=== BAND chain l0-"
              f"{args.max_band_level} (selfp production max-its: "
              "l4 297, l5 323, l6 DIVERGED ksp=-5 at 14245 dofs) ===")
        try:
            allrows["band"] = arm_band(base, amr, args.max_band_level, cfg,
                                       deadline, t_start)
        except Exception as e:
            print(f"  band arm FAILED: {failure_reason(e)}")
        gc.collect()

    summary(allrows, args.inner_method, args.inner_rtol, args.inner_cap)
    print(f"\ntotal wall {time.time() - t_start:.0f}s")
    print("done")


if __name__ == "__main__":
    main()
