"""hpG two-stage solver: outer FGMRES on the monolithic mixed Jacobian with
the sequential P_F preconditioner of Papadopoulos (arXiv:2412.13733v4,
eq 4.5), with our upgrades attachable:

    y = b_psi - B^T A_alpha^-1 b_u        (cached alpha-independent Cholesky)
    dpsi = S^-1 y   (inner GMRES, PC = cellwise Cholesky of Shat)
    du  = A_alpha^-1 (b_u - B dpsi)

The mixed Jacobian here (lvpp two-row residual, box lower bound, E_beta = 0
in the operator -- our Finding-1 rule) is EXACTLY the hpG Newton matrix
G = [[A_alpha, B], [B^T, -D_psi]] with D_psi = M . e^psi (their psi is the
negative of ours; sign conventions documented in the stage-0 report).

Alpha-independence (the paper's cache trick, sec 4.4): with Dirichlet rows
replaced by identity rows E, the assembled primal block is
    A(alpha) = alpha*K0 + (1-alpha)*E,   K0 = stiffness + E  (alpha-free),
so for any rhs:  A(alpha)^-1 b = K0^-1 (D_alpha b),
D_alpha = diag(1/alpha on interior, 1 on bc rows).  K0 is factored ONCE per
mesh (MUMPS LU) and reused across all proximal iterations and all alpha;
the current alpha is read from the live lvpp Constant by the driver.

petsc4py idiom (harness rule): x.getArray(readonly=True).copy() and
y.setArray(arr) -- NEVER the context-manager form (PETSc error 101).
"""
import time

import numpy as np
import scipy.sparse as sp
from firedrake.petsc import PETSc

CTX = None          # set by the driver script before lvpp.solve()


def csr(m):
    indptr, indices, data = m.getValuesCSR()
    return sp.csr_matrix((data, indices, indptr), shape=m.getSize())


def split_blocks_pet(P, n0, n1):
    """(K, B, D) PETSc submatrices of the monolithic Jacobian (primal dofs
    first, contiguous; lvpp mixed-space ordering)."""
    is0 = PETSc.IS().createGeneral(np.arange(n0, dtype=np.int32), comm=P.comm)
    is1 = PETSc.IS().createGeneral(np.arange(n0, n0 + n1, dtype=np.int32),
                                   comm=P.comm)
    Km = P.createSubMatrix(is0, is0)
    Bm = P.createSubMatrix(is0, is1)
    Dm = P.createSubMatrix(is1, is1)
    is0.destroy()
    is1.destroy()
    return Km, Bm, Dm


class HPGCtx:
    """Everything the Python PC needs, owned by the driver script."""

    def __init__(self, n0, n1, bc_nodes, sg, alpha_getter,
                 inner_rtol=1e-6, inner_maxit=500, beta_shat=0.0,
                 a_action="lu", gamg_rtol=1e-9):
        self.n0, self.n1 = n0, n1
        self.bc_nodes = np.asarray(bc_nodes, dtype=np.int64)
        self.sg = sg
        self.alpha_getter = alpha_getter
        self.inner_rtol = float(inner_rtol)
        self.inner_maxit = int(inner_maxit)
        self.beta_shat = float(beta_shat)
        self.a_action = a_action          # "lu" (cached K0) | "gamg" stretch
        self.gamg_rtol = float(gamg_rtol)
        self.interior = None
        self.alpha = 1.0
        self.last_alpha_seen = None
        self.kspA = None
        self.K0 = None
        self.K0diag_ref = None
        self.interior_ref = None
        self.scale_vec = None
        # per-setUp operators
        self.B = None
        self.BT = None
        self.D = None
        self.inner_ksp = None
        self.matS = None
        self._wb = None
        self._wx = None
        # instrumentation
        self.inner_its_per_apply = []
        self.n_applies = 0
        self.setup_wall = 0.0
        self.apply_wall = 0.0
        self.n_setup = 0
        self.bad = False
        self.gamg_alpha = None

    # ------------------------------------------------------- primal inverse
    def _build_kspA_lu(self, Km, alpha):
        """K0 = diag(1/alpha interior, 1 on bc) @ Km = stiffness + bc rows;
        factor once with MUMPS-LU (cached for the whole run)."""
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
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        ksp.setUp()
        self.kspA, self.K0 = ksp, K0
        self.interior_ref = int(np.flatnonzero(interior)[0])
        self.K0diag_ref = float(K0.getValue(self.interior_ref,
                                            self.interior_ref))

    def _rebuild_gamg(self, Km, alpha):
        """CG+GAMG on A(alpha) = alpha*K0 + (1-alpha)*E (stretch arm)."""
        scale = np.full(self.n0, alpha)
        scale[self.bc_nodes] = 1.0        # bc rows: alpha + (1-alpha) = 1
        A = Km.copy()
        v = A.createVecLeft()
        v.setArray(scale)
        A.diagonalScale(L=v)
        if self.kspA is not None:
            self.kspA.destroy()
        ksp = PETSc.KSP().create(comm=A.comm)
        ksp.setOperators(A)
        ksp.setType("cg")
        ksp.setTolerances(rtol=self.gamg_rtol, max_it=500)
        ksp.getPC().setType("gamg")
        ksp.setUp()
        self.kspA = ksp
        self.gamg_alpha = alpha
        self.interior = np.ones(self.n0, dtype=bool)
        self.interior[self.bc_nodes] = False

    def _refresh_scale(self, alpha):
        self.scale_vec = np.ones(self.n0)
        self.scale_vec[self.bc_nodes] = 1.0
        mask = np.ones(self.n0, dtype=bool)
        mask[self.bc_nodes] = False
        self.scale_vec[mask] = 1.0 / alpha

    def Ainv(self, b):
        """A(alpha)^-1 b (numpy in/out)."""
        if self.a_action == "lu":
            bs = b * self.scale_vec
            v = self.K0.createVecRight()
            x = self.K0.createVecRight()
            v.setArray(bs)
            self.kspA.solve(v, x)
            return x.getArray(readonly=True).copy()
        # gamg arm: plain solve on A(alpha)
        v = self.kspA.getOperators()[0].createVecRight()
        x = self.kspA.getOperators()[0].createVecRight()
        v.setArray(np.ascontiguousarray(b))
        self.kspA.solve(v, x)
        return x.getArray(readonly=True).copy()

    # -------------------------------------------------------- block applies
    def B_mult(self, v):
        return self.B.dot(v)

    def BT_mult(self, v):
        return self.BT.dot(v)

    def D_mult(self, v):
        return self.D.dot(v)

    def inner_solve(self, y2):
        """Inner GMRES: S dpsi = y2 with the cellwise-Shat PC."""
        self._wb.setArray(np.ascontiguousarray(y2))
        self.inner_ksp.solve(self._wb, self._wx)
        self.inner_its_per_apply.append(
            int(self.inner_ksp.getIterationNumber()))
        return self._wx.getArray(readonly=True).copy()


class SShellCtx:
    """MatShell context for the true Schur action with the SIGNED Jacobian
    latent block: S = J_psi_psi - B^T A^-1 B (lvpp lower bound:
    J_psi_psi = -D_psi, so S = -(D_psi + B^T A^-1 B), negative definite)."""

    def __init__(self, ctx):
        self.ctx = ctx

    def mult(self, mat, v, w):
        ctx = self.ctx
        vv = v.getArray(readonly=True).copy()
        w.setArray(ctx.D_mult(vv) - ctx.BT_mult(ctx.Ainv(ctx.B_mult(vv))))

class ShatBlockPC:
    """Inner PC: batched cellwise Cholesky solves of -Shat_c."""

    def __init__(self, ctx):
        self.ctx = ctx

    def apply(self, pc, x, y):
        ctx = self.ctx
        sg = ctx.sg
        xr = x.getArray(readonly=True).copy()
        X = xr[sg.perm_can].reshape(sg.ncells, sg.k)
        Y = sg.shat_apply(X)
        out = np.empty(ctx.n1)
        out[sg.perm_can] = Y.reshape(-1)
        y.setArray(out)


class HPGTwoStagePC:
    """Python PC implementing the paper's sequential P_F (eq 4.5)."""

    def setUp(self, pc):
        ctx = CTX
        t0 = time.time()
        A, P = pc.getOperators()
        Km, Bm, Dm = split_blocks_pet(P, ctx.n0, ctx.n1)
        alpha = float(ctx.alpha_getter())
        ctx.alpha = alpha
        ctx.last_alpha_seen = alpha
        ctx.B = csr(Bm)
        ctx.BT = ctx.B.T.tocsr()
        ctx.D = csr(Dm)
        Bm.destroy()
        Dm.destroy()
        # cached alpha-independent factorization (once per mesh, "lu" arm)
        if ctx.a_action == "lu" and ctx.kspA is None:
            ctx._build_kspA_lu(Km, alpha)
        elif ctx.a_action == "gamg" and ctx.gamg_alpha != alpha:
            ctx._rebuild_gamg(Km, alpha)
        Km.destroy()
        # per-iterate Shat from the fresh latent block
        Dn = ctx.sg.cell_blocks(ctx.D)
        ctx.sg.build_shat(Dn, beta=ctx.beta_shat, check=False)
        # inner GMRES on the true Schur S (matfree) with the Shat PC
        if ctx.matS is not None:
            ctx.matS.destroy()
        if ctx.inner_ksp is not None:
            ctx.inner_ksp.destroy()
        matS = PETSc.Mat().createPython((ctx.n1, ctx.n1), comm=P.comm)
        matS.setPythonContext(SShellCtx(ctx))
        matS.setUp()
        ksp = PETSc.KSP().create(comm=P.comm)
        ksp.setType("gmres")
        ksp.setTolerances(rtol=ctx.inner_rtol, max_it=ctx.inner_maxit)
        ksp.setGMRESRestart(ctx.inner_maxit)
        ksp.setOperators(matS)
        p_in = ksp.getPC()
        p_in.setType("python")
        p_in.setPythonContext(ShatBlockPC(ctx))
        ksp.setUp()
        ctx.inner_ksp = ksp
        ctx.matS = matS
        ctx._wb = matS.createVecLeft()
        ctx._wx = matS.createVecRight()
        ctx.inner_its_per_apply = []
        ctx.n_setup += 1
        ctx.setup_wall += time.time() - t0

    def apply(self, pc, x, y):
        ctx = CTX
        t0 = time.time()
        xr = x.getArray(readonly=True).copy()
        b1, b2 = xr[:ctx.n0], xr[ctx.n0:]
        t = ctx.Ainv(b1)                          # A_alpha^-1 b_u
        y2 = b2 - ctx.BT_mult(t)                  # y = b_psi - B^T A^-1 b_u
        dpsi = ctx.inner_solve(y2)                # S^-1 y (inner GMRES)
        y1 = ctx.Ainv(b1 - ctx.B_mult(dpsi))      # A^-1 (b_u - B dpsi)
        out = np.concatenate([y1, dpsi])
        if not np.all(np.isfinite(out)):
            ctx.bad = True
        y.setArray(out)
        ctx.n_applies += 1
        ctx.apply_wall += time.time() - t0
