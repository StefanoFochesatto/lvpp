"""Grading-robust Schur-PC experiment: diagnose and fix the outer-GMRES
blowup of the FLOORLESS Schur config on GRADED (band-adapted) meshes.

Background (RESEARCH.md, Scaling workstream results): the floorless
production config (matfree Amat + assembled unfloored Pmat, schur/upper/
use_amat False/selfp, both blocks preonly+MUMPS-LU, psi_floor 0, outer
GMRES rtol 1e-6 restart 250) has ~N^0.8 outer-its growth on UNIFORM meshes
(14/29/48/105/329 at L0-L4) but BLOWS UP on graded meshes, correlating with
grading strength h_max/h_min (1x: 48 its; 10x: 78-275; 45x: 323) and the
band-adapted chain diverges (ksp=-5) at 14245 dofs.  Mechanism hypothesis:
the selfp Schur approximation Sp = D - B^T diag(K)^{-1} B degrades under
grading because diag(K)^{-1} mismatches K^{-1} by ~(h_max/h_min)^2.

Stages (all on the SAME extracted states + captured max-its RHS, cold
outer GMRES rtol 1e-6 restart 250 max_it 1000 -- the growth_driver /
eps_schedule / rankk_correction protocol):

  0. State extraction: production-config cold solves on
       uni_l2   uniform MeshHierarchy level 2 (8321 u-dofs, h ratio 1,
                control; production max-its 48)
       band_l4  band-adapted chain level 4 (amr_health construction:
                free-boundary ring marking via viamr._elemextreme +
                refinesbr2D; ~6.9k u-dofs, h ratio ~45x)
       band_l5  chain level 5 (~10k u-dofs)
     At each converged state: exact monolithic J (unfloored), blocks
     K/B/D, the RHS of the max-its linear solve, per-cell broken-basis
     quantities built from the ACTUAL K block (variant A diagonals and
     variant C coupling), dead-row census (raw + h-fair).

  1. DIAGNOSIS per state (band vs uniform):
       kappa(diag(K)^{-1/2} K diag(K)^{-1/2})   -- is diag(K) the problem?
       ||Sp - S||_2 / ||S||_2 for the selfp Sp and each fix's Sp
         (true S = D - B^T K^{-1} B applied matrix-free via sparse LU of K).

  2. COLD-SOLVE its per state:
       base    production selfp semantics: Sp0 = D - B^T diag(K)^{-1} B,
               sparse LU (splu) -- must reproduce the production curve.
       gold    variant-(a) reference: exact block-LU with the EXACT
               unfloored S (dense assembly + LU; growth_driver measured
               its=1 on uniform; here it validates the harness on graded
               states).
       A-pinv  graded-aware diagonal Sp: diag(K)^{-1} replaced by the
               per-cell broken-basis diagonal of K^{-1} (pinv variant,
               per-cell constant mode dropped).
       A-reg   same with the mass-regularized per-cell inverse
               (tau = 1e-2 tr(k3)/tr(m3), schur_percell "reg").
       B1/B2   true-S matfree inner CG: the fieldsplit_1 block solve is
               replaced by capped CG on -S (S negative definite; -S SPD,
               h^2-scaled) applied matrix-free (one sparse-LU K solve per
               matvec), Jacobi-scaled, rtol 1e-2 / 1e-4, maxit 40.  The
               PC action stays linear/fixed, so plain GMRES is valid.
       C       band/per-cell Sp upgrade: SpC = D - C0 with
               C0 = sum_cells m_c^T (K3_c)^+ m_c (full per-cell broken
               Schur coupling from the actual K; schur_percell machinery,
               no alpha assumption).

CRITICAL petsc4py idiom: in PC callbacks use x.getArray(readonly=True).copy()
and y.setArray(arr) -- NEVER the context-manager form (PETSc error 101).

Run (env guard REQUIRED, serial only, no petsc4py.init()):
  PETSC_DIR=/home/stefano/firedrake/petsc PETSC_ARCH=arch-firedrake-default \
  OMP_NUM_THREADS=1 /home/stefano/firedrake/venv-firedrake/bin/python \
      lvpp/experiments/grade_pc.py [--max-band-level 5] [--from-cache] \
      [--skip-gold]

States are cached to /tmp/grade_pc_cache_<pid-independent>/ so a crash in a
later stage does not force re-solving; --from-cache skips stage 0.

MEASURED VERDICT (2026-09-09, this file; states uni_l2 8321 / band_l4 3613
(h-ratio 22.8x, dead8 2137) / band_l5 6861 (45.6x, 5169); captured max-its
RHS per state; cold GMRES rtol 1e-6 restart 250 max 1000):

  | variant              | uni_l2 | band_l4            | band_l5            |
  |----------------------|--------|--------------------|--------------------|
  | base (selfp Sp, LU)  | 22     | 39                 | 51                 |
  | gold(a) exact S      | 1      | 1                  | 1                  |
  | A-pinv (per-cell dg) | 89     | 85                 | 90                 |
  | A-reg                | 92     | 85                 | 90                 |
  | B inner-CG rtol 1e-2 | 8      | 30 DIVERGED_BREAK. | 30 DIVERGED_BREAK. |
  | B inner-CG rtol 1e-3 | 5      | 12                 | 21                 |
  | B inner-CG rtol 1e-4 | 6      | 10                 | 15                 |
  | C per-cell Schur     | 29     | 53                 | 83                 |

  Diagnosis: the diag(K)^{-1}-mismatch hypothesis is REFUTED at the norm
  level.  kappa(diag(K)^{-1}K) grows only 3.3e3 -> 4.4e3 -> 8.9e3 across
  h-ratios 1x -> 22.8x -> 45.6x (vs the predicted (h_max/h_min)^2 ~ 2000x),
  and ||Sp(selfp) - S||_2/||S||_2 stays ~1.6e-3 on graded meshes
  (1.6e-4 uniform) -- Sp remains an excellent Schur approximation in norm;
  grading degrades it only ~10x.  Variant A (built on that hypothesis) makes
  BOTH the norm (5.7e-3) and the its worse on every mesh; variant C (per-cell
  coupling) is no better (norm ~1.55e-3, its worse than base everywhere).
  Variant B is the only fix that helps: inner CG on -S (matrix-free, capped
  40, Jacobi) gives 6/10/15 its (rtol 1e-4) vs base 22/39/51 -- the only
  affordable variant that both beats the baseline at every state and stays
  nearly flat in absolute its under 45.6x grading.  rtol 1e-2 is a graded-
  mesh knife-edge (inner CG too inexact -> outer DIVERGED_BREAKDOWN);
  rtol <= 1e-4 is safe.  gold(a) = 1/1/1 confirms the growth is 100%
  PC-fixable on graded meshes.  CAVEAT: the cold harness (exact block-LU
  PC on the assembled J) reproduces production counts on uniform
  (extraction run: prox/newton/err/maxits exactly 8/19/48/8.375e-04 at
  uni_l2, 8/21/14/1.553e-02 at band_l0) but UNDER-reproduces the graded
  production max-its (harness base 39/51 vs production 297/323 at
  band_l4/l5); all variant comparisons above are same-state, same-RHS and
  internally consistent, but production-magnitude promotion numbers need
  the end-to-end run (parent).
"""

import argparse
import gc
import os
import statistics
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator, cg, eigsh, splu

if os.environ.get("OMP_NUM_THREADS") != "1":
    sys.exit("env guard: run with OMP_NUM_THREADS=1 (see module docstring)")

from firedrake import (DirichletBC, Function, FunctionSpace, MeshHierarchy,
                       RectangleMesh, SpatialCoordinate, assemble, conditional,
                       dx, errornorm, grad, inner, le, ln, sqrt)
from firedrake.petsc import PETSc
from lvpp import LVPP
from viamr import VIAMR

print = PETSc.Sys.Print

CACHE_DIR = "/tmp/grade_pc_cache"

r0 = 0.9
AFREE = 0.697965148223374
A_, B_ = 0.680259411891719, 0.471519893402112
FCP = {"quadrature_degree": 6}
DEAD_TOL = 1e-8

# Floorless Schur production config (amr_health.py SP_WINNER, psi_floor=0).
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

UNIFORM_REFS = {0: (8, 21, 1.553e-2), 1: (11, 23, 3.599e-3),
                2: (8, 19, 8.375e-4)}


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
# Per-cell broken-basis quantities from the ACTUAL Jacobian blocks
# ---------------------------------------------------------------------------

def percell_quantities(mesh, W, K, B):
    """Per-cell quantities on the (u=latent, shared CG1) dof set:

    kd_pinv[i] = diag( sum_c Lm_c^T (K3_c)^+ Lm_c )   (pinv variant)
    kd_reg[i]  = same with the mass-regularized per-cell inverse
    C0         = sum_c m_c^T (K3_c)^+ m_c             (full coupling, pinv)

    where K3_c = K[cell dofs, cell dofs] is taken from the ACTUAL assembled
    K block (no alpha / pure-stiffness assumption) and m_c = B[cell, cell]
    (B is the mass-type coupling; symmetric here).  K3_c^+ is the 3x3
    Moore-Penrose pseudoinverse dropping the near-null constant mode
    (schur_percell "pinv"), or the regularized inverse k3 + tau*m3 with
    tau = 1e-2 tr(k3)/tr(m3) (schur_percell "reg").
    """
    nc = mesh.num_cells()
    vn = np.asarray(W.cell_node_map().values, dtype=np.int32)
    assert vn.shape == (nc, 3), f"expected triangles, got {vn.shape}"
    n1 = W.dim()
    Kc = K.tocsr()
    Bc = B.tocsr()
    kd_pinv = np.zeros(n1)
    kd_reg = np.zeros(n1)
    c0_rows, c0_cols, c0_vals = [], [], []
    for c in range(nc):
        idx = vn[c]
        k3 = np.asarray(Kc[np.ix_(idx, idx)].todense())
        m3 = np.asarray(Bc[np.ix_(idx, idx)].todense())
        tr_k = np.trace(k3)
        tr_m = np.trace(m3)
        # pinv: drop the per-cell (near-)constant mode
        w, Q = np.linalg.eigh(k3)
        keep = w > 1e-10 * np.abs(w).max()
        inv_w = np.where(keep, 1.0 / np.where(keep, w, 1.0), 0.0)
        ksp_p = (Q * inv_w) @ Q.T
        # reg: all modes invertible, tau = 1e-2 tr(k)/tr(m) (schur_percell)
        if tr_m > 0:
            tau = 1e-2 * tr_k / tr_m
            wr, Qr = np.linalg.eigh(k3 + tau * m3)
            ksp_r = (Qr * (1.0 / wr)) @ Qr.T
        else:
            ksp_r = ksp_p
        kd_pinv[idx] += np.diag(ksp_p)
        kd_reg[idx] += np.diag(ksp_r)
        c0c = m3.T @ ksp_p @ m3
        ii, jj = np.meshgrid(idx, idx, indexing="ij")
        c0_rows.append(ii.ravel())
        c0_cols.append(jj.ravel())
        c0_vals.append(c0c.ravel())
    C0 = sp.coo_matrix((np.concatenate(c0_vals),
                        (np.concatenate(c0_rows), np.concatenate(c0_cols))),
                       shape=(n1, n1)).tocsr()
    C0 = ((C0 + C0.T) * 0.5).tocsr()
    return kd_pinv, kd_reg, C0


# ---------------------------------------------------------------------------
# Stage 0: state extraction at a converged deep state (eps_schedule pattern)
# ---------------------------------------------------------------------------

def extract(mesh, tag, amr):
    """Cold production-config solve (psi_floor=0) to convergence; extract
    the exact unfloored J blocks K/B/D, the RHS of the max-its linear
    solve, health census, and the per-cell quantities."""
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
    failed = ""
    try:
        lvpp.solve(tol=1e-4, max_proximal_iterations=500)
    except Exception as e:
        failed = failure_reason(e, lvpp)
    wall = time.time() - t0

    per_prox, leftover = per_prox_its(state["groups"],
                                      list(lvpp.newton_iterations or []))
    flat = [i for chunk in per_prox + ([leftover] if leftover else [])
            for i in chunk]
    curve = [max(ch) if ch else 0 for ch in per_prox]
    if not state["groups"]:
        return dict(tag=tag, failed=failed or "no linear solves recorded",
                    mesh=mesh, lvpp=lvpp)
    idx = int(np.argmax(state["groups"]))
    bvec = state["rhss"][idx]
    captured_max_its = int(state["groups"][idx])

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
    Km.destroy()
    Bm.destroy()
    Dm.destroy()
    is0.destroy()
    is1.destroy()

    ut = lvpp.u_tilde[0]
    dk = K.diagonal().copy()
    phi = lvpp._z.subfunctions[1].dat.data.copy()
    dd = D.diagonal().copy()
    dead8 = int(np.sum(np.abs(dd) < DEAD_TOL))
    dead12 = int(np.sum(np.abs(dd) < 1e-12))
    psid8 = int(np.sum(phi < np.log(DEAD_TOL)))  # h-fair census (Finding 2)

    err = float(errornorm(uexactUFL(r), lvpp.u_out[0]))
    Nv, _, hmin, hmax = amr.meshsizes(mesh)

    kd_pinv, kd_reg, C0 = percell_quantities(mesh, W, K, B)

    out = dict(tag=tag, mesh=mesh, W=W, V=V, n0=n0, n1=n1, dofs=n0 + n1,
               alpha=float(lvpp._alpha), K=K, B=B, D=D, dk=dk,
               bvec=bvec, bnorm=float(bvec.norm()), phi=phi,
               phi_min=float(phi.min()), dead8=dead8, dead12=dead12,
               psid8=psid8, prox=lvpp.proximal_iterations,
               newton=sum(lvpp.newton_iterations or []), err=err, wall=wall,
               curve=curve, captured_max_its=captured_max_its,
               hmin=hmin, hmax=hmax, Ne=Nv, failed=failed,
               kd_pinv=kd_pinv, kd_reg=kd_reg, C0=C0,
               snes=int(lvpp._solver.snes.getConvergedReason()),
               ksp=int(lvpp._solver.snes.ksp.getConvergedReason()),
               Jm=Jm, Jfd=Jfd, ut=ut, lb=lb)
    return out


def to_petsc(M):
    """scipy CSR -> serial PETSc AIJ (schur_percell pattern)."""
    M = M.tocsr()
    indptr = np.asarray(M.indptr, dtype=np.int32)
    indices = np.asarray(M.indices, dtype=np.int32)
    out = PETSc.Mat().createAIJ(size=M.shape, csr=(indptr, indices, M.data),
                                comm=PETSc.COMM_SELF)
    return out


def save_state(st):
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag = st["tag"]
    sp.save_npz(f"{CACHE_DIR}/{tag}_K.npz", st["K"])
    sp.save_npz(f"{CACHE_DIR}/{tag}_B.npz", st["B"])
    sp.save_npz(f"{CACHE_DIR}/{tag}_D.npz", st["D"])
    sp.save_npz(f"{CACHE_DIR}/{tag}_C0.npz", st["C0"])
    r, c, v = st["Jm"].getValuesCSR()
    sp.save_npz(f"{CACHE_DIR}/{tag}_J.npz",
                sp.csr_matrix((v, c, r), shape=st["Jm"].getSize()))
    np.savez(f"{CACHE_DIR}/{tag}_meta.npz", dk=st["dk"], bvec=st["bvec"],
             phi=st["phi"], kd_pinv=st["kd_pinv"], kd_reg=st["kd_reg"],
             n0=st["n0"], n1=st["n1"], alpha=st["alpha"],
             dead8=st["dead8"], dead12=st["dead12"], psid8=st["psid8"],
             prox=st["prox"], newton=st["newton"], err=st["err"],
             captured_max_its=st["captured_max_its"], hmin=st["hmin"],
             hmax=st["hmax"], failed=st["failed"])




def load_state(tag):
    K = sp.load_npz(f"{CACHE_DIR}/{tag}_K.npz").tocsr()
    B = sp.load_npz(f"{CACHE_DIR}/{tag}_B.npz").tocsr()
    D = sp.load_npz(f"{CACHE_DIR}/{tag}_D.npz").tocsr()
    C0 = sp.load_npz(f"{CACHE_DIR}/{tag}_C0.npz").tocsr()
    J = sp.load_npz(f"{CACHE_DIR}/{tag}_J.npz").tocsr()
    meta = np.load(f"{CACHE_DIR}/{tag}_meta.npz", allow_pickle=True)
    n0, n1 = int(meta["n0"]), int(meta["n1"])
    return dict(tag=tag, K=K, B=B, BT=B.T.tocsr(), D=D, C0=C0, dk=meta["dk"],
                bvec=np.asarray(meta["bvec"]), phi=np.asarray(meta["phi"]),
                kd_pinv=meta["kd_pinv"], kd_reg=meta["kd_reg"],
                n0=n0, n1=n1, dofs=n0 + n1, alpha=float(meta["alpha"]),
                dead8=int(meta["dead8"]), dead12=int(meta["dead12"]),
                psid8=int(meta["psid8"]), err=float(meta["err"]),
                captured_max_its=int(meta["captured_max_its"]),
                prox=int(meta["prox"]), newton=int(meta["newton"]),
                hmin=float(meta["hmin"]), hmax=float(meta["hmax"]),
                failed=str(meta["failed"]), Jcsr=J, newton_total=-1,
                wall=float("nan"), curve=[], phi_min=float("nan"),
                bnorm=float(np.linalg.norm(meta["bvec"])))



def build_states(args):
    """Extract uni_l2 + band chain l0..max_band_level; keep band_l4/l5."""
    states = {}
    tags = ("uni_l2", "band_l4", "band_l5")
    if args.from_cache and all(
            os.path.exists(f"{CACHE_DIR}/{t}_meta.npz") for t in tags):
        for t in tags:
            states[t] = load_state(t)
            print(f"  loaded cached state {t} (dofs {states[t]['dofs']})")
        return states
    if args.from_cache:
        print("  cache incomplete; running full extraction")

    base = RectangleMesh(16, 16, 2.0, 2.0, originX=-2.0, originY=-2.0,
                         diagonal="crossed")
    amr = VIAMR(activetol=1e-4)

    # uniform L2 control (8321 u-dofs; production max-its 48)
    if "uni_l2" not in states:
        hier = MeshHierarchy(base, 2)
        st = extract(list(hier)[2], "uni_l2", amr)
        print_extract_row(st, ref=UNIFORM_REFS[2])
        if not st["failed"]:
            save_state(st)
            states["uni_l2"] = st
        else:
            print(f"  uni_l2 FAILED: {st['failed']}")
        st.pop("lvpp", None)
        gc.collect()

    # band-adapted chain (amr_health construction: free-boundary ring only)
    mesh = base

    for lev in range(args.max_band_level + 1):
        tag = f"band_l{lev}"
        st = extract(mesh, tag, amr)
        print_extract_row(st)
        keep = lev >= 4 or st["failed"]
        if keep and not st["failed"]:
            save_state(st)
            states[tag] = st
        if st["failed"]:
            print(f"  band chain stops at l{lev}: {st['failed']}")
            break
        if lev == args.max_band_level:
            break
        mesh = amr.refinesbr2D(mesh, mark_band(amr, st))
        st.pop("lvpp", None)
        gc.collect()
    if "band_l4" not in states or "band_l5" not in states:
        print("  WARNING: band_l4/l5 not both available; "
              f"have {sorted(states)}")
    return states


def mark_band(amr, st):
    """Free-boundary ring marking (amr_health.mark_band copy): cells whose
    gap = exp(psi) straddles activetol."""
    from firedrake import Function as _F
    gap = _F(st["V"]).interpolate(st["ut"] - st["lb"])
    gmin = amr._elemextreme(gap, minimum=True, defaultval=PETSc.INFINITY)
    gmax = amr._elemextreme(gap, minimum=False, defaultval=-PETSc.INFINITY)
    DG0 = amr.spaces(st["V"].mesh())[1]
    from firedrake import conditional
    mark = _F(DG0, name="band mark").interpolate(
        conditional(gmin < amr.activetol,
                    conditional(gmax > amr.activetol, 1.0, 0.0), 0.0))
    return mark


def print_extract_row(st, ref=None):
    if st.get("failed") and "n0" not in st:
        print(f"  {st['tag']}: EXTRACT FAILED: {st['failed']}")
        return
    refstr = f"  (LU ref {ref[0]}/{ref[1]}/{ref[2]:.3e})" if ref else ""
    print(f"  {st['tag']:>8}: u-dofs {st['n0']:>6}  hmin/hmax "
          f"{st['hmin']:.2e}/{st['hmax']:.2e} (ratio "
          f"{st['hmax'] / max(st['hmin'], 1e-300):.1f}x)  "
          f"dead8 {st['dead8']:>5} (h-fair {st['psid8']:>5})  "
          f"prox {st['prox']} newton {st['newton']}  "
          f"maxits {st['captured_max_its']}  err {st['err']:.3e}  "
          f"wall {st['wall']:.1f}s  ||b|| {st['bnorm']:.2e}"
          + (f"  FAILED: {st['failed']}" if st["failed"] else "")
          + refstr)
    print(f"           per-prox outer-its curve: {st['curve']}")


# ---------------------------------------------------------------------------
# Stage 1: diagnosis
# ---------------------------------------------------------------------------

def make_klu(st):
    return splu(st["K"].tocsc())


def s_matvec_factory(st, klu):
    D, B, BT = st["D"], st["B"], st["BT"]

    def s_mat(v):
        return D.dot(v) - BT.dot(klu.solve(B.dot(v)))

    return s_mat


def spectral_norm(op_matvec, n, ends=("LA", "SA"), maxiter=600):
    """Spectral-norm proxy of a symmetric operator: max |eig| from the
    algebraic extremes (for a symmetric matrix max(|lam_max|, |lam_min|)
    IS the spectral norm).  ARPACK which='LM' is unreliable on these
    clustered matrix-free spectra; S is negative definite, so its callers
    probe 'SA' only.  Degrades gracefully on nonconvergence (partial
    extreme = lower bound; reported)."""
    op = LinearOperator((n, n), matvec=op_matvec, dtype=np.float64)
    vals = []
    for which in ends:
        try:
            w = eigsh(op, k=2, which=which, maxiter=maxiter, ncv=40,
                      tol=1e-6, return_eigenvectors=False)
            vals.extend(np.abs(w).tolist())
        except Exception as e:
            print(f"   (eigsh {which}: no convergence: "
                  f"{type(e).__name__}; result is a lower bound)")
    return float(max(vals)) if vals else float("nan")




def build_sp(st, mid_diag):
    """Sp = D - B^T diag(mid_diag)^{-1} B  (mid_diag over the u dofs)."""
    Bsc = st["B"].T @ sp.diags(1.0 / mid_diag)
    return (st["D"] - (Bsc @ st["B"])).tocsr()


def diag_state(st):
    """Diagnosis: is diag(K) the problem on graded meshes, and how far is
    each Sp from the true Schur S?"""
    n0, n1 = st["n0"], st["n1"]
    klu = make_klu(st)
    s_mat = s_matvec_factory(st, klu)
    out = {}

    # kappa(diag(K)^{-1} K) via the symmetrized congruence
    dks = 1.0 / np.sqrt(np.maximum(st["dk"], 1e-300))
    Ks = sp.diags(dks) @ st["K"] @ sp.diags(dks)
    Ks = ((Ks + Ks.T) * 0.5).tocsr()
    wmax = eigsh(Ks, k=3, which="LM", return_eigenvectors=False)
    wmin = eigsh(Ks, k=3, sigma=0.0, which="LM", return_eigenvectors=False)
    lam_min = float(np.min(np.abs(wmin)))
    kappa_k = float(np.max(wmax) / lam_min)

    # true Schur norm (negative definite -> most negative extreme)
    norm_S = spectral_norm(s_mat, n1, ends=("SA",))

    # Sp quality: baseline selfp + the two variant-A diagonals + per-cell C0
    sps = {
        "selfp": build_sp(st, st["dk"]),
        "A-pinv": build_sp(st, st["kd_pinv"]),
        "A-reg": build_sp(st, st["kd_reg"]),
    }
    norms = {}
    for name, Sp in sps.items():
        Spc = ((Sp + Sp.T) * 0.5).tocsr()
        norms[name] = spectral_norm(
            lambda v, Sp=Spc: Spc.dot(v) - s_mat(v), n1)
    C0 = st["C0"]
    SpC = (st["D"] - C0).tocsr()
    norms["C-percell"] = spectral_norm(
        lambda v: SpC0_matvec(v, C0, st["D"]) - s_mat(v), n1)

    out.update(kappa_diagK=kappa_k, norm_S=norm_S, sp_norms=norms)
    return out


def SpC0_matvec(v, C0, D):
    return D.dot(v) - C0.dot(v)


def print_diag(tag, d):
    print(f"  {tag:>8}: kappa(diag(K)^-1 K) = {d['kappa_diagK']:.3e}   "
          f"||S||_2 = {d['norm_S']:.3e}")
    for name, val in d["sp_norms"].items():
        print(f"           ||Sp({name}) - S||_2 / ||S||_2 = {val:.3e}")


# ---------------------------------------------------------------------------
# Stage 2: cold-solve variants (eps_schedule BlockLUPC + gmres_its protocol)
# ---------------------------------------------------------------------------

class BlockLUPC:
    """Python PC: exact block-UL factorization P = [[K, B], [0, S_pc]].

    apply(x, y): w2 = S^-1(b2 - B^T (K^-1 b1)); w1 = K^-1(b1 - B w2).
    CRITICAL petsc4py idiom: plain getArray(readonly=True).copy() /
    setArray() -- the context-manager form raises PETSc error 101.
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
        xr = x.getArray(readonly=True).copy()
        b1 = xr[:self.n0]
        b2 = xr[self.n0:]
        t = self.klu.solve(b1)
        w2 = self.ssolver(b2 - self.BT.dot(t))
        w1 = self.klu.solve(b1 - self.B.dot(w2))
        y.setArray(np.concatenate([w1, w2]))


class TrueSCG:
    """Capped CG solve of S x = rhs via -S (SPD) applied matrix-free.

    Each matvec costs one sparse-LU solve on K.  S is negative definite
    (Finding 6.3), so -S is SPD; CG on -S returns x with S x = -(-S x) ...
    x = S^{-1} rhs requires x = -(-S)^{-1} rhs.
    """

    def __init__(self, st, klu, rtol, maxit):
        self.n1 = st["n1"]
        self.D, self.B, self.BT = st["D"], st["B"], st["BT"]
        self.klu = klu
        self.rtol = rtol
        self.maxit = maxit
        # Jacobi of -S: diag(-S) = -diag(D) + diag(B^T diag(K)^{-1} B)
        coup = self.B.power(2).T.dot(1.0 / st["dk"])
        d = -st["D"].diagonal() + coup
        self.minv = 1.0 / np.maximum(d, 1e-300)
        self.calls = 0
        self.last_inner = 0
        self.total_inner = 0

    def matvec(self, v):
        return -(self.D.dot(v) - self.BT.dot(self.klu.solve(self.B.dot(v))))

    def __call__(self, rhs):
        Aop = LinearOperator((self.n1, self.n1), matvec=self.matvec,
                             dtype=np.float64)
        Mop = LinearOperator((self.n1, self.n1),
                             matvec=lambda v: self.minv * v,
                             dtype=np.float64)
        self.last_inner = 0

        def cb(x):
            self.last_inner += 1

        x, info = cg(Aop, rhs, rtol=self.rtol, atol=0.0,
                     maxiter=self.maxit, M=Mop, callback=cb)
        self.calls += 1
        self.total_inner += self.last_inner
        if not np.all(np.isfinite(x)):
            raise RuntimeError("inner CG produced non-finite solution "
                               f"(info={info})")
        return -x  # S^{-1} rhs = -(-S)^{-1} rhs


def exact_s_dense(st, klu):
    """Dense exact unfloored Schur S = D - B^T K^{-1} B (variant-(a) gold;
    rankk_correction pattern, chunked to bound peak memory)."""
    n0, n1 = st["n0"], st["n1"]
    Bcsc = st["B"].tocsc()
    D = st["D"].tocsr()
    BT = st["BT"]
    S = np.empty((n1, n1))
    blk = 512
    for s in range(0, n1, blk):
        sl = slice(s, min(s + blk, n1))
        X = klu.solve(Bcsc[:, sl].toarray())
        S[:, sl] = D[:, sl].toarray() - BT.dot(X)
    return S


def gmres_its(Jm, bvec, ctx):
    """Cold outer GMRES on fixed J with a python PC (eps_schedule copy:
    honest ||Jx - b||/||b||)."""
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


def run_variant(st, klu, label, mid_diag=None, dense_S=None, scg=None):
    """One cold-solve measurement; returns dict or an error marker."""
    n0 = st["n0"]
    try:
        if dense_S is not None:
            fact = lu_factor(dense_S, check_finite=False)
            ssolver = lambda v: lu_solve(fact, v, check_finite=False)
        elif scg is not None:
            ssolver = scg
        elif mid_diag is not None:
            Sp = build_sp(st, mid_diag).tocsc()
            sslu = splu(Sp)
            probe = sslu.solve(np.ones(Sp.shape[0]))
            if not np.all(np.isfinite(probe)):
                raise RuntimeError("Sp LU produced non-finite solution")
            ssolver = sslu.solve
        else:
            Sp = (st["D"] - st["C0"]).tocsc()  # variant C: per-cell Schur
            sslu = splu(Sp)
            probe = sslu.solve(np.ones(Sp.shape[0]))
            if not np.all(np.isfinite(probe)):
                raise RuntimeError("SpC LU produced non-finite solution")
            ssolver = sslu.solve
        ctx = BlockLUPC(n0, klu, st["B"], st["BT"], ssolver)
        its, reason, rel, swall = gmres_its(st["Jm"], st["bvec"], ctx)
    except Exception as e:
        msg = str(e).strip().replace("\n", " ")[:140]
        print(f"   {label:<10} RAISED: {msg}")
        return dict(label=label, status="raise", detail=msg)
    rname = REASON_NAMES.get(reason, str(reason))
    inner = ""
    if scg is not None:
        inner = (f"  inner-CG total {scg.total_inner} "
                 f"(avg {scg.total_inner / max(scg.calls, 1):.1f}/apply)")
    ok = reason == 2 and np.isfinite(rel)
    print(f"   {label:<10} its={its:>5}  reason={rname:<18} "
          f"rel-resid={rel:.2e}  solve={swall:6.1f}s{inner}")
    return dict(label=label, status="ok" if ok else "bad", its=its,
                reason=rname, rel=rel, wall=swall)


def stage_table(states, diagnoses, skip_gold):
    print("\n" + "=" * 78)
    print("COLD-SOLVE TABLE: outer-GMRES its per state (rtol 1e-6, "
          "restart 250, max 1000) on the captured max-its RHS")
    print("=" * 78)
    summary = {}
    for tag, st in states.items():
        print(f"\n== {tag}: dofs {st['dofs']} (n0={st['n0']}, "
              f"n1={st['n1']}) h-ratio "
              f"{st['hmax'] / max(st['hmin'], 1e-300):.1f}x  "
              f"dead8 {st['dead8']}  captured production max-its "
              f"{st['captured_max_its']} ==")
        klu = make_klu(st)
        res = {}
        res["base"] = run_variant(st, klu, "base", mid_diag=st["dk"])
        if not skip_gold and st["n1"] <= 13000:
            t0 = time.time()
            S = exact_s_dense(st, klu)
            print(f"   (exact dense S assembled in {time.time() - t0:.1f}s)")
            res["gold"] = run_variant(st, klu, "gold(a)", dense_S=S)
            del S
            gc.collect()
        res["A-pinv"] = run_variant(st, klu, "A-pinv",
                                    mid_diag=st["kd_pinv"])
        res["A-reg"] = run_variant(st, klu, "A-reg", mid_diag=st["kd_reg"])
        for rtol, lbl in ((1e-2, "B rtol1e-2"), (1e-3, "B rtol1e-3"),
                          (1e-4, "B rtol1e-4")):
            scg = TrueSCG(st, klu, rtol=rtol, maxit=40)
            res[lbl] = run_variant(st, klu, lbl, scg=scg)
        res["C-percell"] = run_variant(st, klu, "C-percell")
        summary[tag] = res
        klu = None
        gc.collect()

    print("\n=== SUMMARY (outer-GMRES its; RTOL unless noted) ===")
    variants = ["base", "gold", "A-pinv", "A-reg",
                "B rtol1e-2", "B rtol1e-3", "B rtol1e-4", "C-percell"]
    hdr = f"{'variant':<12}" + "".join(f"{t:>24}" for t in summary)
    print(hdr)
    for v in variants:
        cells = []
        for t in summary:
            r = summary[t].get(v)
            if r is None:
                c = "--"
            elif r.get("status") == "ok":
                c = str(r["its"])
            elif r.get("status") == "raise":
                c = "RAISED"
            else:
                c = r.get("reason", "?")
            cells.append(c)
        print(f"{v:<12}" + "".join(f"{c:>24}" for c in cells))
    return summary


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-band-level", type=int, default=5)
    ap.add_argument("--from-cache", action="store_true",
                    help="load extracted states from " + CACHE_DIR)
    ap.add_argument("--skip-gold", action="store_true",
                    help="skip the dense exact-S variant-(a) reference")
    args, _ = ap.parse_known_args()

    print("=" * 78)
    print("GRADE_PC: grading-robust Schur-PC for the floorless LVPP config")
    print("=" * 78)

    t_all = time.time()
    states = build_states(args)
    if not states:
        sys.exit("no states extracted; aborting")

    # cached states carry only the scipy J; build PETSc handles from it
    for st in states.values():
        st.setdefault("BT", st["B"].T.tocsr())
        if "Jm" not in st:
            st["Jm"] = to_petsc(st.pop("Jcsr"))
        for key in ("bvec",):
            if isinstance(st[key], np.ndarray):
                st[key] = PETSc.Vec().createWithArray(
                    np.ascontiguousarray(st[key], dtype=np.float64),
                    comm=PETSc.COMM_SELF)

    print("\n=== STAGE 1: DIAGNOSIS ===")
    diagnoses = {}
    for tag, st in states.items():
        d = diag_state(st)
        diagnoses[tag] = d
        print_diag(tag, d)
        gc.collect()

    summary = stage_table(states, diagnoses, args.skip_gold)

    print("\n=== VERDICT ===")
    print(f"total wall {time.time() - t_all:.0f}s")
    print("done")


if __name__ == "__main__":
    main()
