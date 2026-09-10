"""Per-cell block Schur approximation for the LVPP saddle solver.

Finding 6 follow-up: replace the diagonal Schur approximation of the winner
config (`schur upper selfp LUS`, Sp_diag = D_floor - B diag(K)^{-1} B^T,
outer-GMRES growth 45 -> 130-560 -> 400-926 across levels) by the per-cell
broken-basis recipe of the hierarchical-Proximal-Galerkin paper
(arXiv:2412.13733):

    Sp = D_floor - (1/alpha) * C0,
    C0 = sum_cells Lm_c^T (ks_c)^+ Lm_c

where, on the per-cell broken basis (Vb = DG1 on the same mesh),

  * ks_c: 3x3 per-cell block of the alpha*stiffness form assembled on
    Vb x Vb (per-cell block-diagonal by construction; singular with the
    per-cell constant in the kernel because the broken stiffness has no BC
    rows) -- (alpha*ks_c)^+ = ks_c^+/alpha;
  * Lm_c: 3x3 per-cell local mass coupling Vb(test) x latent(trial)
    (the latent space is an equal-order CG1 copy of the u space);
  * (ks_c)^+ : Moore-Penrose pseudoinverse per cell (variant "pinv": the
    per-cell constant mode gets zero coupling) or the inverse of the
    regularized block ks_c + tau*lm_c with tau = 1e-2 tr(ks_c)/tr(lm_c)
    (variant "reg": constant mode gets a mass-scaled coupling);
  * D_floor: the latent block of the Pmat, D = -(exp(psi)-weighted mass)
    plus the eps = 1e-2 psi_floor mass, assembled live from the current
    latent iterate: D_floor_ij = int (eps - exp(psi)) phi_i phi_j dx.

Mechanism (path A, PETSc-native user Schur preconditioner): after LVPP
construction,

    pc.setFieldSplitSchurPreType(PETSc.PC.FieldSplitSchurPreType.USER, Sp)

attaches Sp as the matrix that fieldsplit_1 (the Schur KSP, preonly + MUMPS
LU) factors, replacing PETSc's selfp-computed diagonal-approximation Schur
Pmat.  Verified against petsc src/ksp/pc/impls/fieldsplit/fieldsplit.c: the
USER slot keeps the user matrix across every PCSetUp (only the FULL path
recomputes it) and KSPSetOperators(kspschur, schur, FieldSplitSchurPre(jac))
hands it to the Schur KSP.  Sp is refreshed per SNES solve by wrapping
lvpp._solver.solve (read-only instrumentation, no lvpp.py edits): D_floor
tracks the current latent iterate and the coupling scales with the current
proximal alpha.

Usage:
  PETSC_DIR=/home/stefano/firedrake/petsc PETSC_ARCH=arch-firedrake-default \
  OMP_NUM_THREADS=1 venv-firedrake/bin/python experiments/schur_percell.py \
      [--probe] [--variant pinv|reg] [--levels 0,1,2]

  --probe   L0 mechanism check: PCView must report "user provided matrix"
            and a coupling-killed control run (C0 scaled by 1e-6) compares
            full-L0 per-prox outer-its curves, proving the attached Sp is
            actually used.

MEASURED OUTCOME (2026-09-08): mechanism verified (PCView reports "user
provided matrix"; Sp refreshed per SNES solve), Newton/prox/err LU-identical
at every level -- but outer-GMRES growth is NOT reduced (61 -> 580 -> 890 vs
baseline 45 -> 130-560 -> 400-926).  Coupling-kill control: scaling C0 by
1e-6 leaves the per-solve outer-its curves BIT-IDENTICAL, and the attached
matrices measure |C0| ~ 7.3e-4 x |D_floor|: the per-cell (and any) Sp
coupling term B^T Khat^{-1} B is O(h^4) here while D is O(h^2) (B = mass,
K = alpha*stiffness), i.e. structurally subdominant for this LVPP system --
unlike the divergence-coupled saddle systems the recipe targets.  Finding 6's
attribution of the growth to "diag(K)^{-1} vs K^{-1} coupling quality" is
refuted by measurement; the growth driver is the floored-D/degenerate-row
mechanics of the coupled system, not the Sp coupling approximation.

LU references (schur_probe.py): 8/21 1.553e-2 | 11/23 3.599e-3 | 8/19 8.375e-4.
"""
import argparse
import statistics
import time

import numpy as np
import scipy.sparse as sp

from firedrake import (DirichletBC, Function, FunctionSpace, MeshHierarchy,
                       RectangleMesh, SpatialCoordinate, TestFunction,
                       TrialFunction, assemble, conditional, dx, errornorm,
                       exp, grad, inner, le, ln, sqrt)
from firedrake.petsc import PETSc
from scipy.sparse import coo_matrix

from lvpp import LVPP

_LAST = None  # most recent LVPP object (reachable when run() raises)

r0 = 0.9
AFREE = 0.697965148223374
A, B = 0.680259411891719, 0.471519893402112

PSI_FLOOR = 1e-2
KILL = 1e-6      # coupling scale for the --probe control run
FCP = {"quadrature_degree": 6}

SP = {
    "mat_type": "matfree",
    "pmat_type": "aij",
    # fieldsplit must extract its blocks from the assembled Pmat
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
    # the user matrix is attached via petsc4py below; this also sets the type
    "pc_fieldsplit_schur_precondition": "user",
    # exact K: block solve AND the K^{-1} inside the Schur matvec
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "lu",
    "fieldsplit_0_pc_factor_mat_solver_type": "mumps",
    # LU of OUR per-cell Sp (replaces the selfp diag-approximation Pmat)
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "lu",
    "fieldsplit_1_pc_factor_mat_solver_type": "mumps",
}

LU_REFS = {0: (8, 21, 1.553e-2), 1: (11, 23, 3.599e-3), 2: (8, 19, 8.375e-4)}
BASELINE_CURVE = "45 (L0) -> 130-560 (L1) -> 400-926 (L2)"


def psiUFL(r):
    psi0 = np.sqrt(1.0 - r0 * r0)
    dpsi0 = -r0 / psi0
    return conditional(le(r, r0), sqrt(1.0 - r * r), psi0 + dpsi0 * (r - r0))


def uexactUFL(r):
    return conditional(le(r, AFREE), psiUFL(r), -A * ln(r) + B)


def to_petsc(M):
    """scipy CSR -> serial PETSc AIJ."""
    M = M.tocsr()
    indptr = np.asarray(M.indptr, dtype=np.int32)
    indices = np.asarray(M.indices, dtype=np.int32)
    out = PETSc.Mat().createAIJ(size=M.shape, csr=(indptr, indices, M.data),
                                comm=PETSc.COMM_SELF)
    return out


def csr_of(mat):
    r, c, v = mat.getValuesCSR()
    return sp.csr_matrix((v, c, r), shape=mat.getSize())


def build_c0(mesh, W, variant):
    """C0 = sum_cells Lm_c^T (ks_c)^+ Lm_c as a scipy CSR on W's dofs.

    ks_c from the DG1-broken stiffness form, Lm_c from the DG1 x W mass.
    """
    Vb = FunctionSpace(mesh, "DG", 1)
    ztrb, ztb = TrialFunction(Vb), TestFunction(Vb)
    ztr1 = TrialFunction(W)
    Kb = assemble(inner(grad(ztrb), grad(ztb)) * dx, mat_type="aij").M.handle
    Mb = assemble(inner(ztr1, ztb) * dx, mat_type="aij").M.handle
    n1 = W.dim()
    nc = mesh.num_cells()
    # PETSc getValues wants int32 indices here (serial 32-bit indexing)
    vn_b = np.asarray(Vb.cell_node_map().values, dtype=np.int32)
    vn_w = np.asarray(W.cell_node_map().values, dtype=np.int32)
    assert vn_b.shape == (nc, 3) and vn_w.shape == (nc, 3)
    k3 = np.empty((nc, 3, 3))
    m3 = np.empty((nc, 3, 3))
    for c in range(nc):
        rb, rw = vn_b[c], vn_w[c]
        k3[c] = Kb.getValues(rb, rb)
        m3[c] = Mb.getValues(rb, rw)
    tr_k = np.trace(k3, axis1=1, axis2=2)
    tr_m = np.trace(m3, axis1=1, axis2=2)
    if variant == "reg":
        # regularized per-cell inverse: ks_c + tau*lm_c, all modes invertible
        tau = 1e-2 * tr_k / tr_m
        areg = k3 + tau[:, None, None] * m3
        w, Q = np.linalg.eigh(areg)
        inv_w = 1.0 / w
    else:  # pinv (default): drop the per-cell constant mode
        w, Q = np.linalg.eigh(k3)
        keep = w > 1e-10 * np.abs(w).max(axis=1, keepdims=True)
        inv_w = np.where(keep, 1.0 / np.where(keep, w, 1.0), 0.0)
    ksp3 = np.einsum("nij,nj,nkj->nik", Q, inv_w, Q)
    c0 = np.einsum("nij,njk,nlk->nil", ksp3, m3, m3)   # = m^T ksp m, symmetric
    rows = np.broadcast_to(vn_w[:, :, None], (nc, 3, 3))
    cols = np.broadcast_to(vn_w[:, None, :], (nc, 3, 3))
    C0 = coo_matrix((c0.ravel(), (rows.ravel(), cols.ravel())),
                    shape=(n1, n1)).tocsr()
    asym = float(abs(C0 - C0.T).max()) if C0.nnz else 0.0
    Kb.destroy()
    Mb.destroy()
    return C0, asym


def d_floor_csr(lvpp):
    """D_floor = int (eps - exp(psi)) phi_i phi_j dx from the current iterate."""
    z1 = lvpp._z.subfunctions[1]
    W = z1.function_space()
    ztr1, zt1 = TrialFunction(W), TestFunction(W)
    form = inner((PSI_FLOOR - exp(z1)) * ztr1, zt1) * dx
    handle = assemble(form, mat_type="aij",
                      form_compiler_parameters=FCP).M.handle
    out = csr_of(handle)
    handle.destroy()
    return out


class SpUpdator:
    """Refreshes and attaches the user Schur matrix before every SNES solve."""

    def __init__(self, lvpp, C0, coup_scale=1.0):
        self.lvpp = lvpp
        self.C0 = C0 * coup_scale
        self.pc = lvpp._solver.snes.ksp.getPC()
        self.prev = None
        self.n_refresh = 0

    def attach(self):
        alpha = float(self.lvpp._alpha)
        Sp = (d_floor_csr(self.lvpp) - self.C0 / alpha).tocsr()
        mat = to_petsc(Sp)
        self.pc.setFieldSplitSchurPreType(
            PETSc.PC.FieldSplitSchurPreType.USER, mat)
        if self.prev is not None:
            self.prev.destroy()
        self.prev = mat
        self.n_refresh += 1

    def wrap_solve(self):
        orig = self.lvpp._solver.solve

        def wrapped(*a, **k):
            self.attach()
            return orig(*a, **k)
        self.lvpp._solver.solve = wrapped


def instrument(lvpp):
    """Outer-GMRES its per Newton solve (same grouping as schur_sp_upgrade)."""
    state = {"groups": [], "solves": 0}

    def ksp_mon(ksp, its, rnorm):
        if its == 1:
            state["groups"].append(1)
        elif its > 1:
            if state["groups"]:
                state["groups"][-1] = its
            else:
                state["groups"].append(its)

    def snes_mon(snes, its, norm):
        if its == 1:
            state["solves"] += 1

    lvpp._solver.snes.ksp.setMonitor(ksp_mon)
    lvpp._solver.snes.setMonitor(snes_mon)
    return state

def per_prox_its(state, newton_its):
    groups = state["groups"]
    out, i = [], 0
    for n in newton_its:
        out.append(groups[i:i + n])
        i += n
    return out, groups[i:]  # leftovers = alpha-halving retry solves


def run(mesh, tag, variant, coup_scale=1.0, max_prox=500, snes_max_it=None):
    global _LAST
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name="u")
    x, y = SpatialCoordinate(mesh)
    r = sqrt(x * x + y * y)
    energy = 0.5 * inner(grad(u), grad(u)) * dx  # f = 0
    psi = Function(V, name="psi").interpolate(psiUFL(r))
    bc = DirichletBC(V, uexactUFL(r), "on_boundary")
    sp = dict(SP)
    if snes_max_it is not None:
        sp["snes_max_it"] = snes_max_it
    lvpp = LVPP(energy=energy, u=u, bounds=(psi, None), bcs=bc,
                alpha_rule="double_exponential",
                alpha_parameters={"alpha_max": 10.0},
                increment_norm="H1", verbose=False,
                psi_floor=PSI_FLOOR,
                form_compiler_parameters=FCP,
                solver_parameters=sp, name=tag)
    _LAST = lvpp
    C0, asym = build_c0(mesh, lvpp._z.subfunctions[1].function_space(), variant)
    upd = SpUpdator(lvpp, C0, coup_scale=coup_scale)
    upd.attach()
    upd.wrap_solve()
    state = instrument(lvpp)
    t0 = time.time()
    lvpp.solve(tol=1e-4, max_proximal_iterations=max_prox)
    wall = time.time() - t0
    err = errornorm(uexactUFL(r), lvpp.u_out[0])
    per_prox, leftover = per_prox_its(state, list(lvpp.newton_iterations))
    flat = [i for chunk in per_prox + ([leftover] if leftover else [])
            for i in chunk]
    return dict(tag=tag, dofs=V.dim(), prox=lvpp.proximal_iterations,
                newton=sum(lvpp.newton_iterations), err=err, wall=wall,
                its=flat, curve=[max(ch) if ch else 0 for ch in per_prox],
                asym=asym, n_refresh=upd.n_refresh, lvpp=lvpp)


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

def print_header():
    print(f"{'level':>5} {'dofs':>7} {'variant':>8} {'prox':>5} {'newton':>7} "
          f"{'err(u_h)':>10} {'wall(s)':>8} {'oits_max':>9} {'oits_med':>9} "
          "outer-its curve (max per prox solve)")


def print_row(lev, v, variant="pinv"):
    if v is None:
        print(f"{lev:>5} {'FAILED':>7}")
        return
    med = statistics.median(v["its"]) if v["its"] else float("nan")
    print(f"{lev:>5} {v['dofs']:>7} {variant:>8} {v['prox']:>5} "
          f"{v['newton']:>7} {v['err']:>10.3e} {v['wall']:>8.2f} "
          f"{max(v['its']) if v['its'] else 0:>9} {med:>9.1f}  {v['curve']}")


def probe(hier):
    """Mechanism check at L0: PCView + FULL-L0 coupling-killed control."""
    print("--- probe: FULL L0 run with user per-cell Sp ---")
    v, k = None, None
    try:
        v = run(hier[0], "probe_percell", "pinv", max_prox=500)
    except Exception as e:
        print(f"probe run raised: {failure_reason(e, _LAST)}")
    lv = v["lvpp"] if v is not None else _LAST
    try:
        print("--- PCView (fieldsplit schur, user Sp) ---")
        lv._solver.snes.ksp.getPC().view()
    except Exception as e:
        print(f"PC view failed: {failure_reason(e, lv)}")
    if v is not None:
        print(f"probe run: prox={v['prox']} newton={v['newton']} "
              f"oits={v['its']} refreshes={v['n_refresh']} "
              f"C0 asym={v['asym']:.2e} curve={v['curve']}")
    print("--- probe control: FULL L0 run with C0 scaled by 1e-6 "
          "(kills the coupling; per-prox outer-its curves must differ) ---")
    try:
        k = run(hier[0], "probe_kill", "pinv", coup_scale=KILL,
                max_prox=500)
    except Exception as e:
        print(f"kill run raised: {failure_reason(e, _LAST)}")
    if v is not None and k is not None and v["its"] and k["its"]:
        if abs(statistics.median(v["its"])
               - statistics.median(k["its"])) < 1:
            print("WARNING: identical outer-GMRES counts -- the user Schur "
                  "matrix may not be reaching the PC!")
        else:
            print(f"OK: counts differ -- the user Sp is being used.\n"
                  f"  Sp run    curve: {v['curve']}\n"
                  f"  kill run  curve: {k['curve']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--variant", default="pinv", choices=["pinv", "reg"])
    ap.add_argument("--levels", default="0,1,2")
    args = ap.parse_args()

    base = RectangleMesh(16, 16, 2.0, 2.0, originX=-2.0, originY=-2.0,
                         diagonal="crossed")
    hier = MeshHierarchy(base, 2)

    if args.probe:
        probe(hier)
        return

    levels = [int(s) for s in args.levels.split(",")]
    print(f"variant={args.variant}  baseline outer-GMRES per-solve growth: "
          f"{BASELINE_CURVE}")
    print_header()
    rows = {}
    for lev, mesh in enumerate(hier):
        if lev not in levels:
            continue
        try:
            v = run(mesh, f"percell{lev}_{args.variant}", args.variant)
        except Exception as e:
            print(f"level {lev}: {failure_reason(e)}")
            rows[lev] = None
            continue
        rows[lev] = v
        print_row(lev, v, args.variant)

    print(f"\nSummary (variant={args.variant}; LU refs "
          f"{' | '.join(f'{k}:{v[0]}/{v[1]} {v[2]:.3e}' for k, v in sorted(LU_REFS.items()))})")
    print(f"baseline outer-GMRES per-solve growth: {BASELINE_CURVE}")
    for lev, v in sorted(rows.items()):
        if v is None:
            print(f"  L{lev}: FAILED")
            continue
        med = statistics.median(v["its"]) if v["its"] else float("nan")
        p, n, e = LU_REFS[lev]
        print(f"  L{lev}: prox={v['prox']} newton={v['newton']} "
              f"err={v['err']:.3e} (LU {p}/{n} {e:.3e}) "
              f"outer-its max={max(v['its']) if v['its'] else 0} "
              f"med={med:.1f} wall={v['wall']:.1f}s")


if __name__ == "__main__":
    main()
