"""Rank-k dead-subspace correction for the Schur-fieldsplit LVPP solver.

Implements and validates the identified scalable path after the two refuted
hypotheses (RESEARCH.md Finding 6, "Growth-driver decomposition"):

  variant (a) exact block-LU PC (exact K^-1 + exact UNfloored S^-1):
      outer-GMRES its = 1 FLAT at every level  -> growth is 100% PC-fixable.
  variant (b) floored exact-S PC reproduces the whole growth curve
      (145 / 895 / 1000-diverged at L0/L1/L2): the eps=1e-2 psi_floor mass
      on the latent block is the driver (O(eps) PC-vs-operator mismatch on
      the dead subspace).
  variant (c) naive identity rows on dead dofs DIVERGES: dead rows must be
      corrected consistently with the coupling, not patched.

THE CORRECTION (derived here; the PC apply is a right preconditioner
M ~ J^-1; GMRES sees G = J M; V = latent unit vectors on the dead rows,
|diag(D)| < tol, k = 37..765):

  Goal: the corrected PC reproduces the EXACT PC's (variant a) action on
  span(V) while keeping the floored (safe) action elsewhere:

      P_c = P_b + (P_a - P_b) V V^T
          = P_b + Z V^T,   Z = (P_a - P_b) V,

  implemented as one PC apply + a k-dimensional correction
  w += Z (V^T x) driven ONLY by the input's dead-dof components.
  Exactness on span(V) is by construction: P_c v = P_a v for v in span(V).
  NOTE on the alternative "P_c V = J^-1 V" goal: J^-1 [0; e_j] ~ 1e28
  here (the dead latent block of J is ~1e-28, while the exact Schur
  S_a = D - B^T K^-1 B is healthy) -- that Z is numerically explosive and
  is NOT used; matching the perfect PC on span(V) is the operative goal.

  Cost: Z needs S_a^-1 e_j and S_f^-1 e_j for the k dead unit vectors
  (k dense triangular solves each, after the n1 x n1 dense LU) plus
  K^-1 (sparse LU) -- NO k x k inverse of V^T J V is formed.  For the
  record: V^T J V = D_dd, the dead-dead block of the row-scaled mass D
  (a full mass matrix scaled by exp(psi) ~ 1e-28 on the dead rows), is
  TINY but its condition number is mass-matrix-like (measured/reported);
  the k x k Schur-Gram V^T S_a V = D_dd - (BV)^T K^-1 (BV) is also
  measured/reported.

Stage 1 (fixed extracted J + real captured RHS, outer GMRES rtol 1e-6
restart 250, same harness as experiments/growth_driver.py) measures on the
SAME state: baseline (b) its, rank-k-corrected its, variant-(a) reference
its, plus k, the phi < -50 count, conditioning of V^T J V and V^T S_a V,
and the exactness check ||P_c V - P_a V||.

Stage 2 (path B, only if stage 1 validates): the correction is attached
around the production fieldsplit PC via a Python PC whose apply =
fieldsplit PC apply + Z (V^T x); V, Z refreshed per proximal solve
(schur_percell.py pattern).  In this prototype Z = (P_a - P_b) V is built
from a MUMPS-LU of the fresh K block + dense LU of the fresh exact Schur
-- a prototype cost, documented.

CRITICAL petsc4py idiom (burned a previous agent): in PC callbacks use
x.getArray(readonly=True).copy() and y.setArray(arr) -- NEVER the
context-manager form (raises TypeError -> PETSc error 101).

Run (serial only, no petsc4py.init()):
  PETSC_DIR=/home/stefano/firedrake/petsc PETSC_ARCH=arch-firedrake-default \\
  OMP_NUM_THREADS=1 /home/stefano/firedrake/venv-firedrake/bin/python \\
  lvpp/experiments/rankk_correction.py --stage1 [--levels 0,1,2]
MEASURED VERDICT (stage 1, 2026-09-09; outer GMRES rtol 1e-6 restart 250
on the same extracted J + captured max-its RHS, levels 0/1/2):

  | variant                      | L0  | L1                 | L2                |
  |------------------------------|-----|--------------------|-------------------|
  | b_floored  (control)         | 145 | 895 RTOL           | 1000 DIVERGED     |
  | rankk      (exact on span V) |  11 | 1000 DIVERGED      | 1000 DIVERGED     |
  | rankk_eig  (dominant band)   |   9 | 1000 DIVERGED      | 1000 DIVERGED     |
  | a_exact    (reference)       |   1 | 1                  | 1                 |

  The correction is verified active and exact (||P_c V - P_a V||/||P_a V||
  = 4.7e-17 / 8.5e-17 / 1.1e-17) and the k x k Schur-Gram V^T S_a V is
  benign (cond 1.7e3 / 5.5e3 / 2.0e4) -- so the failure is NOT an
  implementation artifact.  The raw V^T J V is effectively singular
  (cond 9.2e16 / 4.1e59 / 8.1e58 at scale 3.8e-11 / 2.0e-20 / 1.3e-10)
  and is never inverted (Z = (P_a - P_b) V needs only the two Schur
  solves).

  WHY IT FAILS AT DEPTH (decisive diagnostic, in-file): the floored PC's
  deficiency E_b - I with E_b = D S_f^{-1} (latent block of the
  preconditioned operator) has slowly-decaying singular values:
  sigma_max = 1.0e2 / 2.1e2 / 3.4e1 and the band above sigma 0.1
  has rank >= 70 / 241 / 829 -- LARGER than the dead-row count
  k = 45 / 177 / 765, with sigma_k = 0.998 / 0.974 / 0.85 still ~O(1).
  The eps-floor therefore damages a whole TRANSITION BAND of the latent
  block, not just the strictly-dead rows; even a correction over the full
  visible band (rankk_eig, k+64 columns) stagnates (GMRES residual floors
  at 3.9e-4 / 1.6e-1 vs baseline 4.6e-6 / DIVERGED), i.e. the band's tail
  is long.  Consistent with refuted variant (c): partially patching the
  latent block is worse than the consistent floor.

  CONSEQUENCE: no rank-k correction on the dead (or even band) subspace
  collapses the growth toward its=1; the scalable treatment must reproduce
  the exact unfloored Schur action across the band (variant (a) structure)
  or decay eps with alpha.  PATH B (full-solve promotion) NOT RUN: its
  gate -- stage-1 validation -- FAILED (the corrected PC diverges where
  the baseline converges at L1/L2); promoting it to the full solver would
  be pointless.
"""

import argparse
import gc
import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import splu

from firedrake import (DirichletBC, Function, FunctionSpace, MeshHierarchy,
                       RectangleMesh, SpatialCoordinate, TestFunction,
                       TrialFunction, assemble, conditional, dx, errornorm,
                       grad, inner, le, ln, sqrt)
from firedrake.petsc import PETSc
from lvpp import LVPP

r0 = 0.9
AFREE = 0.697965148223374
A_, B_ = 0.680259411891719, 0.471519893402112

PSI_FLOOR = 1e-2
FCP = {"quadrature_degree": 6}
DEAD_TOL = 1e-8      # degenerate-row threshold on |diag(D)|
PHI_TOL = -50.0      # analytic dead-dof test from Finding 1

LU_REFS = {0: (8, 21, 1.553e-2), 1: (11, 23, 3.599e-3), 2: (8, 19, 8.375e-4)}
BASELINE_CURVE = "45 (L0) -> 130-560 (L1) -> 400-926 (L2)"
VARIANT_B_REFS = {0: 145, 1: 895, 2: 1000}
VARIANT_A_REFS = {0: 1, 1: 1, 2: 1}

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

RK_PREFIX = "rk_"     # options prefix for the inner fieldsplit PC (path B)

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
# Stage 1: state extraction at a converged deep state (growth_driver pattern)
# ---------------------------------------------------------------------------

def extract_state(mesh, tag, max_prox=500):
    """Run the winner config to convergence; return J blocks + a real RHS.

    The RHS is the outer-GMRES rhs captured by a KSP monitor at the start of
    the linear solve that needed the MOST iterations (a genuine Newton
    linearization, not a random vector).  J is assembled at the converged
    state (the final linearization point differs from it only within
    snes_rtol; documented).
    """
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
                psi_floor=PSI_FLOOR,
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

    idx = int(np.argmax(state["groups"]))
    bvec = state["rhss"][idx]

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

    ztr1, zt1 = TrialFunction(W), TestFunction(W)
    Mfd = assemble(inner(ztr1, zt1) * dx, mat_type="aij")
    M = csr(Mfd.M.handle)

    alpha = float(lvpp._alpha)
    phi = lvpp._z.subfunctions[1].dat.data.copy()
    dd = D.diagonal().copy()
    dead8 = int(np.sum(np.abs(dd) < 1e-8))
    dead12 = int(np.sum(np.abs(dd) < 1e-12))
    phi_dead = int(np.sum(phi < PHI_TOL))

    err = errornorm(uexactUFL(r), lvpp.u_out[0])
    return dict(level_tag=tag, n0=n0, n1=n1, dofs=n0 + n1, alpha=alpha,
                Jm=Jm, K=K, B=B, BT=B.T.tocsr(), D=D, M=M, dk=dk,
                bvec=bvec, bnorm=float(bvec.norm()), phi=phi,
                phi_min=float(phi.min()), dead8=dead8, dead12=dead12,
                phi_dead=phi_dead,
                prox=lvpp.proximal_iterations,
                newton=sum(lvpp.newton_iterations), err=float(err), wall=wall,
                curve=curve, max_its_captured=int(state["groups"][idx]))


# ---------------------------------------------------------------------------
# Stage 1: floored block-UL PC + rank-k dead-subspace correction
# ---------------------------------------------------------------------------

class RankKBlockPC:
    """Python PC: block-UL P = [[K, B], [0, S_pc]] + optional rank-k fix.

    apply(x, y): w2 = S^-1(b2 - B^T K^-1 b1); w1 = K^-1(b1 - B w2);
    then, if enabled, w += Z (V^T x) -- makes the corrected PC reproduce
    the exact PC's action on span(V) while keeping the floored action
    elsewhere (P_c V = P_a V by construction of Z = (P_a - P_b) V).
    """

    def __init__(self, n0, klu, B, BT, ssolver, VT=None, Z=None):
        self.n0 = n0
        self.klu = klu
        self.B = B
        self.BT = BT
        self.ssolver = ssolver
        self.VT = VT
        self.Z = Z
        self.bad = False

    def setUp(self, pc):
        pass

    def apply(self, pc, x, y):
        # CRITICAL petsc4py idiom: plain getArray(readonly=True).copy() /
        # setArray() -- the context-manager form raises PETSc error 101.
        xr = x.getArray(readonly=True).copy()
        b1 = xr[:self.n0]
        b2 = xr[self.n0:]
        t = self.klu.solve(b1)
        w2 = self.ssolver(b2 - self.BT.dot(t))
        w1 = self.klu.solve(b1 - self.B.dot(w2))
        w = np.concatenate([w1, w2])
        if self.Z is not None:
            w = w + self.Z.dot(self.VT.dot(xr))
        if not np.all(np.isfinite(w)):
            self.bad = True
        y.setArray(w)


def gmres_its(Jm, bvec, ctx):
    """Outer GMRES on fixed J with a python PC; its + properly scaled resid."""
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
    r.axpy(-1.0, bvec)   # r = J x - b (growth_driver's rel column was
    rel = float(r.norm() / bvec.norm())  # mis-scaled; fixed here)
    x.destroy()
    r.destroy()
    ksp.destroy()
    return its, reason, rel, solve_wall


def stage1_level(st, dead_tol=DEAD_TOL):
    """Measure baseline (b), rank-k-corrected, and (a) on one state."""
    n0, n1 = st["n0"], st["n1"]
    Jm, K, B, BT, D, M = st["Jm"], st["K"], st["B"], st["BT"], st["D"], st["M"]
    lev = int(st["level_tag"][-1])
    print(f"\n== LEVEL state '{st['level_tag']}' | dofs {st['dofs']} "
          f"(n0={n0}, n1={n1}) | alpha={st['alpha']:.4f} | "
          f"phi_min={st['phi_min']:.1f} | "
          f"dead rows |d|<{dead_tol:g}: {st['dead8']} "
          f"(|d|<1e-12: {st['dead12']}, phi<{PHI_TOL}: {st['phi_dead']}) ==")
    print(f"   winner run: prox={st['prox']} newton={st['newton']} "
          f"err={st['err']:.3e} wall={st['wall']:.1f}s   "
          f"(LU ref {LU_REFS.get(lev, ('?', '?', '?'))});  "
          f"per-prox curve: {st['curve']}  [baseline {BASELINE_CURVE}]")
    print(f"   captured rhs: its={st['max_its_captured']} "
          f"||b||={st['bnorm']:.3e}")

    t0 = time.time()
    klu = splu(K.tocsc())
    X = klu.solve(B.toarray())          # K^-1 B, dense (n0 x n1)
    BtKB = np.asarray(BT.dot(X))
    del X
    print(f"   factored K + dense true coupling B^T K^-1 B in "
          f"{time.time() - t0:.1f}s")

    Df_dense = (D + PSI_FLOOR * M).toarray()
    S_f = Df_dense - BtKB              # variant (b): floored exact Schur
    S_a = D.toarray() - BtKB           # variant (a): exact unfloored Schur
    Sf_fact = lu_factor(S_f, check_finite=False)
    Sa_fact = lu_factor(S_a, check_finite=False)
    sf_solver = lambda v: lu_solve(Sf_fact, v, check_finite=False)
    sa_solver = lambda v: lu_solve(Sa_fact, v, check_finite=False)

    dd = D.diagonal()
    dead = np.abs(dd) < dead_tol
    idx_dead = np.where(dead)[0]
    k = int(dead.sum())

    def block_apply(ssolver, X):
        """Block-UL PC action on the (n x m) dense matrix X."""
        T = klu.solve(X[:n0])
        W2 = ssolver(X[n0:] - BT.dot(T))
        W1 = klu.solve(X[:n0] - B.dot(W2))
        return np.vstack([W1, W2])

    # V: latent unit vectors on dead rows (u components zero)
    n = n0 + n1
    V = np.zeros((n, k))
    V[n0 + idx_dead, np.arange(k)] = 1.0
    VT = V.T

    # Z = (P_a - P_b) V: difference of exact and floored PC on span(V)
    t0 = time.time()
    Z = block_apply(sa_solver, V) - block_apply(sf_solver, V)
    t_z = time.time() - t0

    # exactness check: P_c V == P_a V, and the k x k conditioning report
    D_dd = D[np.ix_(idx_dead, idx_dead)].toarray()       # = V^T J V
    BV = B.dot(V[n0:])
    KBV = klu.solve(BV)
    cond_vtjv = float(np.linalg.cond(D_dd)) if k else float("nan")
    Gd = D_dd - BV.T.dot(KBV)                            # = V^T S_a V
    scale_vtjv = float(np.max(np.abs(D_dd))) if k else float("nan")
    cond_gd = float(np.linalg.cond(Gd)) if k else float("nan")
    PcV = block_apply(sf_solver, V) + Z
    PaV = block_apply(sa_solver, V)
    exact_err = float(np.max(np.abs(PcV - PaV))) if k else 0.0
    exact_rel = exact_err / max(float(np.max(np.abs(PaV))), 1e-300)
    zscale = float(np.max(np.abs(Z))) if k else 0.0
    print(f"   k(|d|<{dead_tol:g})={k}  k(phi<{PHI_TOL:g})={st['phi_dead']}  "
          f"Z built in {t_z:.1f}s (|Z|max={zscale:.2e})")
    print(f"   cond(V^T J V)={cond_vtjv:.2e} at scale {scale_vtjv:.2e}  "
          f"cond(V^T S_a V)={cond_gd:.2e}")
    print(f"   exactness ||P_c V - P_a V||_max={exact_err:.2e} "
          f"(rel {exact_rel:.2e})")

    # ---- decisive diagnostic: HOW BIG is the floored PC's deficiency? ----
    # Latent block of the preconditioned operator G = [[I,0],[0, D S^-1]]:
    # GMRES convergence is governed by E_b = D S_f^{-1} (variant b) vs
    # E_a = D S_a^{-1} (variant a; clustered {~1} u {~0} -> its=1).
    # The rank of the deviation E_b - I = (D - S_f) S_f^{-1} bounds what
    # ANY rank-k correction on span(V) can fix.
    Dm = D.toarray()
    Eb = Dm @ lu_solve(Sf_fact, np.eye(n1), check_finite=False)
    Da_ = Eb - np.eye(n1)
    rng = np.random.default_rng(20260909)
    p = 64
    Yq = Da_ @ rng.standard_normal((n1, min(k + p, n1)))
    Qq, _ = np.linalg.qr(Yq)
    Umat, sig, _ = np.linalg.svd(Qq.T @ Da_, full_matrices=False)
    Uq = Qq @ Umat         # approximate top-(k+p) LEFT singular vectors
    m_bad = int(np.sum(sig > 0.1))
    sig_flat = np.ravel(sig)
    sig_k = float(sig_flat[min(k, sig_flat.size - 1)])
    print(f"   deficiency D S_f^-1 - I: sigma_max={sig[0]:.2e}  "
          f"sigma_k={sig_k:.2e}  "
          f"#(sigma>0.1)>={m_bad}  (n1={n1}; rank needed vs k={k})")
    # rank-k variant on the DOMINANT deficiency subspace (not span(V)):
    Veig = np.zeros((n, Uq.shape[1]))
    Veig[n0:, :] = Uq
    Zeig = block_apply(sa_solver, Veig) - block_apply(sf_solver, Veig)


    def finish(name, ssolver, VT=None, Z=None):
        ctx = RankKBlockPC(n0, klu, B, BT, ssolver, VT=VT, Z=Z)
        its, reason, rel, solve_wall = gmres_its(Jm, st["bvec"], ctx)
        rname = REASON_NAMES.get(reason, str(reason))
        results[name] = (its, rname, rel, solve_wall)
        print(f"   {name:<14} its={its:>5}  reason={rname:<16} "
              f"rel-resid={rel:.2e}  solve={solve_wall:6.1f}s"
              + ("  [NaN/Inf in PC apply!]" if ctx.bad else ""))
        gc.collect()

    results = {}
    results.update(k=k, cond_vtjv=cond_vtjv, scale_vtjv=scale_vtjv,
                   cond_gd=cond_gd, exact_rel=exact_rel)
    finish("b_floored", sf_solver)                       # control (no fix)
    finish("rankk", sf_solver, VT, Z)
    finish("rankk_eig", sf_solver, Veig.T, Zeig)
    finish("a_exact", sa_solver)
    gc.collect()
    return results


# ---------------------------------------------------------------------------
# Stage 2 (path B): correction attached to the production fieldsplit PC
# ---------------------------------------------------------------------------

class RankKUpdator:
    """Python PC wrapping the production fieldsplit PC + rank-k correction.

    apply(x, y) = inner fieldsplit PC apply(x, y) + Z (V^T x), with
    Z = (P_a - P_b) V.  The inner fieldsplit PC is built programmatically
    (own options prefix) from the SAME options as the production config.
    V and Z are refreshed per proximal solve (lvpp._solver.solve wrap,
    schur_percell pattern).  Z is built from the freshly assembled exact
    blocks: P_a - P_b differs from the floored PC only through
    S_a^{-1} - S_f^{-1} on the latent block, so
        Z = [K^-1 B W ; W],  W = (S_a^{-1} - S_f^{-1}) E_dead,
    with K exact (MUMPS LU), S_a = D - B^T K^-1 B (dense LU on the k
    coupling columns only), S_f = D + eps M (dense LU).  Prototype cost:
    a dense n1 x n1 exact-Schur assembly per refresh; documented.
    """

    def __init__(self, lvpp, dead_tol=DEAD_TOL, correction=True):
        self.lvpp = lvpp
        self.pc = lvpp._solver.snes.ksp.getPC()
        self.W = lvpp._z.subfunctions[1].function_space()
        self.dead_tol = dead_tol
        self.correction = correction
        self.inner = None
        self.Z = None
        self.VT = None
        self.pending = True
        self.n_refresh = 0
        self.refresh_wall = 0.0
        self._sig = None

    # -- python PC interface ------------------------------------------------
    def setUp(self, pc):
        if self.inner is None:
            self._build_inner(pc)
        if self.pending:
            self.refresh()
            self.pending = False

    def apply(self, pc, x, y):
        # CRITICAL petsc4py idiom (no context-manager getArray!).
        xr = x.getArray(readonly=True).copy()
        self.inner.apply(x, y)
        Z, VT = self.Z, self.VT
        if Z is not None:
            yr = y.getArray(readonly=True).copy()
            yr = yr + Z.dot(VT.dot(xr))
            y.setArray(yr)

    # -- setup ---------------------------------------------------------------
    def _build_inner(self, pc):
        A, P = pc.getOperators()
        opts = PETSc.Options()
        for key, val in [
                ("pc_type", "fieldsplit"),
                ("pc_fieldsplit_type", "schur"),
                ("pc_fieldsplit_schur_fact_type", "upper"),
                ("pc_fieldsplit_schur_precondition", "selfp"),
                ("pc_fieldsplit_use_amat", False),
                ("fieldsplit_0_ksp_type", "preonly"),
                ("fieldsplit_0_pc_type", "lu"),
                ("fieldsplit_0_pc_factor_mat_solver_type", "mumps"),
                ("fieldsplit_1_ksp_type", "preonly"),
                ("fieldsplit_1_pc_type", "lu"),
                ("fieldsplit_1_pc_factor_mat_solver_type", "mumps")]:
            opts[f"{RK_PREFIX}{key}"] = val
        inner = PETSc.PC().create(comm=pc.comm)
        inner.setOptionsPrefix(RK_PREFIX)
        inner.setOperators(A, P)
        n1 = self.W.dim()
        n0 = A.getSize()[0] - n1
        is0, is1 = field_ises(A, n0, n1)
        inner.setFieldSplitIS("0", is0)
        inner.setFieldSplitIS("1", is1)
        inner.setFromOptions()
        inner.setUp()
        is0.destroy()
        is1.destroy()
        self.inner = inner

    # -- refresh (per proximal solve) ----------------------------------------
    def refresh(self):
        t0 = time.time()
        A = self.inner.getOperators()[0]
        n1 = self.W.dim()
        n = A.getSize()[0]
        n0 = n - n1
        z1 = self.lvpp._z.subfunctions[1]
        # latent block D = -exp(psi) * M (row-scaled mass), exact form
        Dfd = assemble(inner(-exp(z1) * TrialFunction(self.W),
                             TestFunction(self.W)) * dx, mat_type="aij",
                       form_compiler_parameters=FCP)
        Ds = csr(Dfd.M.handle)
        Mfd = assemble(inner(TrialFunction(self.W), TestFunction(self.W))
                       * dx, mat_type="aij", form_compiler_parameters=FCP)
        Ms = csr(Mfd.M.handle)
        dd = Ds.diagonal()
        dead = np.abs(dd) < self.dead_tol
        idx_dead = np.where(dead)[0]
        k = int(dead.sum())
        if k == 0 or not self.correction:
            self.Z, self.VT = None, None
            self.n_refresh += 1
            self.refresh_wall = time.time() - t0
            return
        # fresh exact blocks K, B from the assembled J at the current state
        Jfd = assemble(self.lvpp._J, mat_type="aij", bcs=self.lvpp._bcs)
        Jm = Jfd.M.handle
        is0, is1 = field_ises(Jm, n0, n1)
        Km = Jm.createSubMatrix(is0, is0)
        Bm = Jm.createSubMatrix(is0, is1)
        K = csr(Km)
        B = csr(Bm)
        BT = B.T.tocsr()
        Km.destroy()
        Bm.destroy()
        is0.destroy()
        is1.destroy()
        Jm.destroy()
        klu = splu(K.tocsc())
        Ed = np.zeros((n1, k))
        Ed[idx_dead, np.arange(k)] = 1.0
        # W = (S_a^{-1} - S_f^{-1}) E_dead with the EXACT Schur
        # S_a = D - B^T K^{-1} B and the floored S_f = S_a + eps*M:
        # dense n1 x n1 LU per distinct (alpha, dead set) -- prototype
        # cost, documented; Z then matches stage-1's exact Z.
        sig = (float(self.lvpp._alpha), tuple(int(i) for i in idx_dead))
        if sig == self._sig:
            self.n_refresh += 1
            self.refresh_wall = time.time() - t0
            return
        self._sig = sig
        X = klu.solve(B.toarray())           # K^{-1} B, dense (n0 x n1)
        S_a = Ds.toarray() - BT.dot(X)
        del X
        S_f = Ds.toarray() + PSI_FLOOR * Ms.toarray()
        Wa = lu_solve(lu_factor(S_a, check_finite=False), Ed,
                      check_finite=False)
        Wf = lu_solve(lu_factor(S_f, check_finite=False), Ed,
                      check_finite=False)
        W = Wa - Wf
        self.Z = np.vstack([klu.solve(B.dot(W)), W])
        if not np.all(np.isfinite(self.Z)):
            print("   [rankk] Z has NaN/Inf -- correction DISABLED")
            self.Z = None
        V = np.zeros((n, k))
        V[n0 + idx_dead, np.arange(k)] = 1.0
        self.VT = V.T
        self.n_refresh += 1
        self.refresh_wall = time.time() - t0

    def wrap_solve(self):
        orig = self.lvpp._solver.solve

        def wrapped(*a, **k):
            self.pending = True
            return orig(*a, **k)
        self.lvpp._solver.solve = wrapped


def instrument(lvpp):
    """Outer-GMRES its per Newton solve (growth_driver pattern)."""
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


def run_stage2_level(mesh, tag, dead_tol=DEAD_TOL, correction=True,
                     max_prox=500):
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name="u")
    x, y = SpatialCoordinate(mesh)
    r = sqrt(x * x + y * y)
    energy = 0.5 * inner(grad(u), grad(u)) * dx  # f = 0
    psi = Function(V, name="psi").interpolate(psiUFL(r))
    bc = DirichletBC(V, uexactUFL(r), "on_boundary")
    sp_dict = dict(SP_WINNER)
    sp_dict["pc_type"] = "python"
    sp_dict["pc_python_type"] = "rankk_correction.RankKUpdator"
    lvpp = LVPP(energy=energy, u=u, bounds=(psi, None), bcs=bc,
                alpha_rule="double_exponential",
                alpha_parameters={"alpha_max": 10.0},
                increment_norm="H1", verbose=False,
                psi_floor=PSI_FLOOR,
                form_compiler_parameters=FCP,
                solver_parameters=sp_dict, name=tag)
    upd = RankKUpdator(lvpp, dead_tol=dead_tol, correction=correction)
    upd.wrap_solve()
    state = instrument(lvpp)
    t0 = time.time()
    lvpp.solve(tol=1e-4, max_proximal_iterations=max_prox)
    wall = time.time() - t0
    err = errornorm(uexactUFL(r), lvpp.u_out[0])
    per_prox, leftover = per_prox_its(state["groups"],
                                      list(lvpp.newton_iterations))
    flat = [i for chunk in per_prox + ([leftover] if leftover else [])
            for i in chunk]
    curve = [max(ch) if ch else 0 for ch in per_prox]
    return dict(tag=tag, dofs=V.dim(), prox=lvpp.proximal_iterations,
                newton=sum(lvpp.newton_iterations), err=float(err), wall=wall,
                its=flat, curve=curve, n_refresh=upd.n_refresh,
                refresh_wall=upd.refresh_wall, lvpp=lvpp)


def print_stage2_row(lev, v):
    if v is None:
        print(f"  L{lev}: FAILED")
        return
    med = (float(np.median(v["its"])) if v["its"] else float("nan"))
    p, n, e = LU_REFS.get(lev, (0, 0, 0.0))
    print(f"  L{lev}: prox={v['prox']} newton={v['newton']} "
          f"err={v['err']:.3e} (LU ref {p}/{n} {e:.3e}) "
          f"outer-its max={max(v['its']) if v['its'] else 0} "
          f"med={med:.1f} wall={v['wall']:.1f}s "
          f"refreshes={v['n_refresh']} (last {v['refresh_wall']:.1f}s)\n"
          f"       per-prox curve: {v['curve']}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--levels", default="0,1,2")
    ap.add_argument("--dead-tol", type=float, default=DEAD_TOL)
    ap.add_argument("--max-prox", type=int, default=500)
    args = ap.parse_args()
    if not (args.stage1 or args.stage2):
        ap.error("choose --stage1 and/or --stage2")

    base = RectangleMesh(16, 16, 2.0, 2.0, originX=-2.0, originY=-2.0,
                         diagonal="crossed")
    hier = MeshHierarchy(base, 2)
    levels = [int(s) for s in args.levels.split(",")]

    if args.stage1:
        print("=" * 78)
        print("STAGE 1: rank-k dead-subspace correction (fixed J + real RHS)")
        print(f"refs: variant (b) floored {list(VARIANT_B_REFS.values())}, "
              f"variant (a) exact {list(VARIANT_A_REFS.values())}")
        print("=" * 78)
        for lev, mesh in enumerate(hier):
            if lev not in levels:
                continue
            try:
                st = extract_state(mesh, f"rk1_l{lev}", max_prox=args.max_prox)
            except Exception as e:
                print(f"level {lev}: extraction FAILED: {failure_reason(e)}")
                continue
            try:
                stage1_level(st, dead_tol=args.dead_tol)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"level {lev}: measurement FAILED: {failure_reason(e)}")
            del st
            gc.collect()

    if args.stage2:
        print("\n" + "=" * 78)
        print(f"STAGE 2 (path B): production fieldsplit PC + rank-k "
              f"correction (dead_tol={args.dead_tol:g})")
        print(f"baseline outer-GMRES per-solve growth: {BASELINE_CURVE}")
        print("=" * 78)
        for corr in (False, True):
            label = "rankk-ON " if corr else "rankk-OFF"
            for lev, mesh in enumerate(hier):
                if lev not in levels:
                    continue
                try:
                    v = run_stage2_level(mesh, f"rk2_{int(corr)}_l{lev}",
                                         dead_tol=args.dead_tol,
                                         correction=corr,
                                         max_prox=args.max_prox)
                except Exception as e:
                    print(f"  {label} L{lev}: FAILED: {failure_reason(e)}")
                    continue
                print(f"  {label}", end="")
                print_stage2_row(lev, v)


if __name__ == "__main__":
    main()
