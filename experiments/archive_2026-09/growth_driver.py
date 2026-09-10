"""Growth-driver decomposition for the Schur-fieldsplit LVPP solver.

Finding 6 follow-up (see RESEARCH.md): the validated schur/upper/selfp/LUS
configuration converges LU-identically at every level, but its outer-GMRES
iteration count grows with mesh depth (45 -> 130-560 -> 400-926, max per
proximal solve).  The "Schur coupling approximation quality" attribution was
REFUTED by measurement (schur_percell.py: the exact per-cell broken-basis
Schur leaves every solve bit-identical), leaving two candidate mechanisms:

  (1) the eps = 1e-2 psi_floor dominating the latent block at depth: the PC
      inverts a system differing from the true J by O(eps) on the degenerate
      subspace (the floor form is a FULL mass matrix eps*M, not a diagonal);
  (2) the dead latent rows themselves (exp(psi) collapses to 1e-28..1e-80;
      37/177/749 rows with |diag(D)| < 1e-12 at convergence across levels
      0/1/2) -- a near-null manifold of the MONOLITHIC operator that no PC
      of J can remove.

Stage 1 (linear-solve-level decomposition; no full solves beyond the one
needed to reach the deep state):

At a converged deep state on each level, assemble the exact monolithic
Jacobian J = [[K, B], [B^T, D]] (psi_floor is a Jp-only term, so lvpp._J is
the unfloored true Jacobian) and capture the actual RHS of a real Newton
linearization (the outer-GMRES rhs of the solve that needed the most
iterations).  Then measure outer-GMRES its (rtol 1e-6, restart 250,
max_it 1000) for four preconditioners, all built as the exact block-UL
factorization

    P = [[K, B], [0, S_pc]],
    apply: w2 = S_pc^{-1}(b2 - B^T K^{-1} b1);  w1 = K^{-1}(b1 - B w2)

with EXACT K^{-1} (sparse LU) and only S_pc varying:

  a exact      S_pc = D - B^T K^{-1} B             (true unfloored Schur,
                                                    assembled dense, dense LU)
  b floored    S_pc = D + eps*M - B^T K^{-1} B     (floor = FULL mass matrix,
                                                    as the real Pmat has)
  c activeset  S_pc = D_c - B^T K^{-1} B, with D_c = D + eps*M and the dead
               latent rows/cols (|diag(D)| < 1e-8) replaced by identity rows
               BEFORE forming the Schur (active-set row treatment, PC-only)
  d diagref    S_pc = D + eps*M - B^T diag(K)^{-1} B  (what PETSc selfp +
               floor effectively factors; the baseline reference)

Variant a is the best possible PC of J: if its counts stay ~flat while
b/d reproduce the baseline growth, the driver is PC-internal, and the
b-vs-c comparison isolates the active-set cure.  If even a grows, the
driver is the operator itself and the cure must be operator-level
(deflation of the dead subspace).

Stage 2 (end-to-end promotion): if the driver is PC-internal, the winning
treatment is implemented PETSc-natively via setFieldSplitSchurPreType(USER,
Sp) with Sp refreshed per SNES solve (the schur_percell.py precedent),
keeping the exact K (fieldsplit_0 preonly+MUMPS LU) and the winner's outer
KSP settings, and run levels 0-2 with the per-solve outer-GMRES monitor.

Usage (from /home/stefano/firedrake/lvpp):
  PETSC_DIR=/home/stefano/firedrake/petsc PETSC_ARCH=arch-firedrake-default \
  OMP_NUM_THREADS=1 venv-firedrake/bin/python experiments/growth_driver.py \
      --stage1 [--levels 0,1,2]
  PETSC_DIR=/home/stefano/firedrake/petsc PETSC_ARCH=arch-firedrake-default \
  OMP_NUM_THREADS=1 venv-firedrake/bin/python experiments/growth_driver.py \
      --stage2 [--stage2-mode activeset|floor] [--levels 0,1,2]

Serial only.  Read-only access to lvpp internals; PETSc-level surgery on
PCs/matrices only (no lvpp.py edits).
"""
import argparse
import gc
import statistics
import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import coo_matrix
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

PSI_FLOOR = 1e-2
FCP = {"quadrature_degree": 6}
DEAD_TOL = 1e-8      # degenerate-row threshold on |diag(D)| (spectrum TINY)

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


def to_petsc(M):
    """scipy CSR -> serial PETSc AIJ (schur_percell pattern)."""
    M = M.tocsr()
    indptr = np.asarray(M.indptr, dtype=np.int32)
    indices = np.asarray(M.indices, dtype=np.int32)
    out = PETSc.Mat().createAIJ(size=M.shape, csr=(indptr, indices, M.data),
                                comm=PETSc.COMM_SELF)
    return out


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
# Stage 1: state extraction at a converged deep state
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
    flat = [i for chunk in per_prox + ([leftover] if leftover else [])
            for i in chunk]
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

    # floor form: the lvpp psi_floor term eps*inner(trial,test) on the latent
    # block is a FULL mass matrix (Finding 6.4) -- assemble it exactly.
    ztr1, zt1 = TrialFunction(W), TestFunction(W)
    Mfd = assemble(inner(ztr1, zt1) * dx, mat_type="aij")
    M = csr(Mfd.M.handle)
    # NOTE: assemble() returns a firedrake Matrix; let GC handle it.
    # cross-check B against the analytic coupling M/alpha (informational)
    Mulf = assemble(inner(TrialFunction(V), TestFunction(W)) * dx,
                    mat_type="aij")
    Mul = csr(Mulf.M.handle)
    # let GC handle it (firedrake Matrix has no destroy)
    alpha = float(lvpp._alpha)
    bc_mask = np.ones(n0, dtype=bool)
    bc_mask[np.asarray(sorted(set(int(i) for i in bc.nodes)),
                       dtype=np.int64)] = False
    dev = float(np.max(np.abs((B - Mul / alpha).data))) if B.nnz else 0.0
    scale = float(np.max(np.abs(Mul.data)))
    b_matches = dev / max(scale, 1e-300) < 1e-10

    phi = lvpp._z.subfunctions[1].dat.data.copy()
    dd = D.diagonal().copy()
    dead8 = int(np.sum(np.abs(dd) < 1e-8))
    dead12 = int(np.sum(np.abs(dd) < 1e-12))

    err = errornorm(uexactUFL(r), lvpp.u_out[0])
    bn = bvec.norm()
    return dict(level_tag=tag, n0=n0, n1=n1, dofs=n0 + n1, alpha=alpha,
                Jm=Jm, Jfd=Jfd, K=K, B=B, BT=B.T.tocsr(), D=D, M=M, dk=dk,
                bvec=bvec, bnorm=float(bn), phi=phi,
                phi_min=float(phi.min()), dead8=dead8, dead12=dead12,
                prox=lvpp.proximal_iterations,
                newton=sum(lvpp.newton_iterations), err=float(err), wall=wall,
                curve=curve, max_its_captured=int(state["groups"][idx]),
                b_matches=(bool(b_matches), dev / max(scale, 1e-300)))


# ---------------------------------------------------------------------------
# Stage 1: the four block-UL preconditioners
# ---------------------------------------------------------------------------

class BlockLUPC:
    """Python PC: exact block-UL factorization P = [[K, B], [0, S_pc]].

    apply(x, y): w2 = S^{-1}(b2 - B^T (K^{-1} b1)); w1 = K^{-1}(b1 - B w2).
    K^{-1} is a sparse LU (exact); S^{-1} is supplied as a callable.
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


def gmres_its(Jm, bvec, ctx, n0):
    """Run outer GMRES on the fixed J with the given python PC; return its."""
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
    rel = float(r.norm() / bvec.norm())
    x.destroy()
    r.destroy()
    ksp.destroy()
    return its, reason, rel, solve_wall


def stage1_level(st):
    """Measure the four PC variants on one extracted state; print the table."""
    n0, n1 = st["n0"], st["n1"]
    Jm, K, B, BT, D, M, dk = (st["Jm"], st["K"], st["B"], st["BT"],
                              st["D"], st["M"], st["dk"])
    print(f"\n== LEVEL state '{st['level_tag']}' | dofs {st['dofs']} "
          f"(n0={n0}, n1={n1}) | alpha={st['alpha']:.4f} | "
          f"phi_min={st['phi_min']:.1f} | "
          f"dead rows |d|<1e-8: {st['dead8']} (|d|<1e-12: {st['dead12']}) ==")
    print(f"   winner baseline run: prox={st['prox']} newton={st['newton']} "
          f"err={st['err']:.3e} wall={st['wall']:.1f}s   "
          f"(LU ref {LU_REFS.get(int(st['level_tag'][-1]), ('?', '?', '?'))})")
    print(f"   per-prox outer-its curve (max per solve): {st['curve']}   "
          f"[baseline {BASELINE_CURVE}]")
    print(f"   captured rhs: from the solve with its={st['max_its_captured']} "
          f"||b||={st['bnorm']:.3e}   "
          f"B==M/alpha check: {st['b_matches'][0]} "
          f"(rel dev {st['b_matches'][1]:.2e})")

    t0 = time.time()
    klu = splu(K.tocsc())
    t_k = time.time() - t0

    # dense true coupling B^T K^{-1} B (shared by variants a, b, c)
    t0 = time.time()
    X = klu.solve(B.toarray())          # K^{-1} B, dense (n0 x n1)
    BtKB = np.asarray(BT.dot(X))        # dense (n1 x n1)
    del X
    gc.collect()
    t_coupling = time.time() - t0
    print(f"   factored K (sparse LU) in {t_k:.1f}s; assembled true coupling "
          f"B^T K^-1 B in {t_coupling:.1f}s")

    Df_dense = (D + PSI_FLOOR * M).toarray()
    dead = np.abs(D.diagonal()) < DEAD_TOL
    idx_dead = np.where(dead)[0]

    results = {}

    def finish(name, Dmat, sparse_S=None):
        t0 = time.time()
        if sparse_S is not None:
            sslu = splu(sparse_S.tocsc())
            ssolver = sslu.solve
        else:
            S = Dmat
            Sfact = lu_factor(S, check_finite=False)
            ssolver = lambda v: lu_solve(Sfact, v, check_finite=False)
        t_build = time.time() - t0
        ctx = BlockLUPC(n0, klu, B, BT, ssolver)
        its, reason, rel, solve_wall = gmres_its(Jm, st["bvec"], ctx, n0)
        rname = REASON_NAMES.get(reason, str(reason))
        results[name] = (its, rname, rel, t_build, solve_wall)
        print(f"   {name:<12} its={its:>5}  reason={rname:<16} "
              f"rel-resid={rel:.2e}  build={t_build:6.1f}s  "
              f"solve={solve_wall:6.1f}s")
        gc.collect()

    # a: exact unfloored true Schur
    finish("a_exact", D.toarray() - BtKB)
    # b: floored Schur (D + eps*M, full mass floor), exact coupling
    finish("b_floored", Df_dense - BtKB)
    # c: active-set treatment: dead rows/cols of the floored latent block
    #    become identity rows BEFORE forming the Schur (keeps the off-block
    #    coupling entries in those rows).
    Dc = Df_dense.copy()
    Dc[idx_dead, :] = 0.0
    Dc[:, idx_dead] = 0.0
    Dc[idx_dead, idx_dead] = 1.0
    finish("c_activeset", Dc - BtKB)
    # d: baseline reference: what selfp + floor factors
    Df_sp = (D + PSI_FLOOR * M).tocsr()
    Bsc = BT @ sp.diags(1.0 / dk)
    S_d = (Df_sp - (Bsc @ B)).tocsr()
    finish("d_diagref", None, sparse_S=S_d)

    del Df_dense, BtKB
    gc.collect()
    return results


# ---------------------------------------------------------------------------
# Stage 2: end-to-end promotion of the winning treatment
# ---------------------------------------------------------------------------

class SpUpdator:
    """Refreshes and attaches the user Schur matrix before every SNES solve.

    Sp = D_treated - B^T diag(K)^{-1} B  (diag coupling, exactly what PETSc
    selfp would factor), with the chosen latent-block treatment:
      mode "activeset": dead latent rows/cols of the FLOORED block become
                        identity rows before the Schur is formed;
      mode "floor":     the plain floored block (baseline-equivalent control).
    K and B blocks are re-extracted only when the proximal alpha changes.
    """

    def __init__(self, lvpp, mode="activeset"):
        self.lvpp = lvpp
        self.pc = lvpp._solver.snes.ksp.getPC()
        self.mode = mode
        self.prev = None
        self.n_refresh = 0
        self.last_alpha = None
        self.Bsp = None
        self.dk = None
        self.W = lvpp._z.subfunctions[1].function_space()
        self.Msp = None

    def refresh_blocks(self, alpha):
        n1 = self.W.dim()
        Jfd = assemble(self.lvpp._J, mat_type="aij", bcs=self.lvpp._bcs)
        Jm = Jfd.M.handle
        n0 = Jm.getSize()[0] - n1
        is0, is1 = field_ises(Jm, n0, n1)
        Bm = Jm.createSubMatrix(is0, is1)
        Km = Jm.createSubMatrix(is0, is0)
        self.Bsp = csr(Bm)
        self.dk = Km.getDiagonal().getArray(readonly=True).copy()
        Bm.destroy()
        Km.destroy()
        is0.destroy()
        is1.destroy()
        Jm.destroy()
        self.last_alpha = alpha

    def attach(self):
        alpha = float(self.lvpp._alpha)
        if alpha != self.last_alpha or self.Bsp is None:
            self.refresh_blocks(alpha)
        z1 = self.lvpp._z.subfunctions[1]
        Dfd = assemble(inner(-exp(z1) * TrialFunction(self.W),
                             TestFunction(self.W)) * dx, mat_type="aij",
                       form_compiler_parameters=FCP)
        Ds = csr(Dfd.M.handle)
        # firedrake Matrix: let GC handle it
        if self.Msp is None:
            Mfd = assemble(inner(TrialFunction(self.W),
                                 TestFunction(self.W)) * dx, mat_type="aij")
            self.Msp = csr(Mfd.M.handle)
            # firedrake Matrix: let GC handle it
        Df = (Ds + PSI_FLOOR * self.Msp).tocsr()
        dead = np.abs(Ds.diagonal()) < DEAD_TOL
        if self.mode == "activeset":
            keep = sp.diags((~dead).astype(float))
            Dc = (keep @ Df @ keep
                  + sp.diags(dead.astype(float))).tocsr()
        else:  # "floor": plain floored block (baseline control)
            Dc = Df
        Bsc = self.Bsp @ sp.diags(1.0 / self.dk)
        coupling = (Bsc @ self.Bsp.T).tocsr()
        Sp = (Dc - coupling).tocsr()
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


def run_stage2_level(mesh, tag, mode, max_prox=500):
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name="u")
    x, y = SpatialCoordinate(mesh)
    r = sqrt(x * x + y * y)
    energy = 0.5 * inner(grad(u), grad(u)) * dx  # f = 0
    psi = Function(V, name="psi").interpolate(psiUFL(r))
    bc = DirichletBC(V, uexactUFL(r), "on_boundary")
    sp_dict = dict(SP_WINNER)
    sp_dict["pc_fieldsplit_schur_precondition"] = "user"
    lvpp = LVPP(energy=energy, u=u, bounds=(psi, None), bcs=bc,
                alpha_rule="double_exponential",
                alpha_parameters={"alpha_max": 10.0},
                increment_norm="H1", verbose=False,
                psi_floor=PSI_FLOOR,
                form_compiler_parameters=FCP,
                solver_parameters=sp_dict, name=tag)
    upd = SpUpdator(lvpp, mode=mode)
    upd.attach()
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
                its=flat, curve=curve, n_refresh=upd.n_refresh, lvpp=lvpp)


def print_stage2_row(lev, v, mode):
    if v is None:
        print(f"  L{lev}: FAILED")
        return
    med = statistics.median(v["its"]) if v["its"] else float("nan")
    p, n, e = LU_REFS.get(lev, (0, 0, 0.0))
    print(f"  L{lev}: prox={v['prox']} newton={v['newton']} "
          f"err={v['err']:.3e} (LU ref {p}/{n} {e:.3e}) "
          f"outer-its max={max(v['its']) if v['its'] else 0} "
          f"med={med:.1f} wall={v['wall']:.1f}s refreshes={v['n_refresh']}\n"
          f"       per-prox curve: {v['curve']}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--stage2-mode", default="activeset",
                    choices=["activeset", "floor"])
    ap.add_argument("--levels", default="0,1,2")
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
        print("STAGE 1: linear-solve-level decomposition (fixed J + real RHS)")
        print(f"baseline outer-GMRES per-solve growth: {BASELINE_CURVE}")
        print("=" * 78)
        for lev, mesh in enumerate(hier):
            if lev not in levels:
                continue
            try:
                st = extract_state(mesh, f"gd1_l{lev}",
                                   max_prox=args.max_prox)
            except Exception as e:
                print(f"level {lev}: extraction FAILED: {failure_reason(e)}")
                continue
            try:
                stage1_level(st)
            except Exception as e:
                print(f"level {lev}: measurement FAILED: {failure_reason(e)}")
            # free the big state (firedrake Matrix has no destroy; let GC handle it)
            del st
            gc.collect()

    if args.stage2:
        print("\n" + "=" * 78)
        print(f"STAGE 2: end-to-end LVPP solves, treated user Schur "
              f"(mode={args.stage2_mode})")
        print(f"baseline outer-GMRES per-solve growth: {BASELINE_CURVE}")
        print("=" * 78)
        for lev, mesh in enumerate(hier):
            if lev not in levels:
                continue
            try:
                v = run_stage2_level(mesh, f"gd2_{args.stage2_mode}_l{lev}",
                                     args.stage2_mode,
                                     max_prox=args.max_prox)
            except Exception as e:
                print(f"  L{lev}: FAILED: {failure_reason(e)}")
                continue
            print_stage2_row(lev, v, args.stage2_mode)


if __name__ == "__main__":
    main()
