"""The hpG two-stage preconditioner: the sequential ``P_F`` of section 4.4.

``HPGTwoStage`` is the Python ``PC`` attached to the outer FGMRES of the hpG
preset (:class:`lvpp.hpg.solver.HPG`).  It applies eq. (4.5) of Papadopoulos
(arXiv:2412.13733) to the mixed Newton matrix

    G = [[A_alpha, B], [B^T, J_psi_psi]]

one outer Krylov iteration at a time.  The three steps are sequential --
``S^-1`` is applied to the outcome of the first ``A_alpha^-1`` solve, and the
second one consumes ``dpsi`` -- so the object is a *preconditioner apply*
rather than a factorization:

    y    = b_psi - B^T A_alpha^-1 b_u
    dpsi = S^-1 y         # inner GMRES on the true Schur, PC = cellwise Shat
    du   = A_alpha^-1 (b_u - B dpsi)

The outer solver is FGMRES because the preconditioner is variable: ``S^-1`` is
itself an inner Krylov solve, so the outer iteration has to keep the vectors
the preconditioner applied.

Sign conventions
----------------
lvpp's lower-bound constraint block is ``J_psi_psi = -D_psi``, so

    A_alpha     alpha * K + E                     SPD primal block
    B           coupling ``(psi, B v)``           n0 x n1
    J_psi_psi   -D_psi                            negative definite
    S           J_psi_psi - B^T A_alpha^-1 B      negative definite
    -Shat_c     -D_jac + coupling + beta * M_c    SPD (cellwise)

``D_jac`` is the latent block *of the Jacobian* (``= -D_psi``), which is what
:meth:`~lvpp.hpg.spectral.SpectralGalerkin.build_shat` expects.  The true
Schur complement is ``S = -(D_psi + E_beta + B^T A_alpha^-1 B)``, so ``S`` is
negative definite and ``-Shat_c`` is SPD for ``D_psi`` positive semidefinite;
``-Shat_c`` is Cholesky-factorized cellwise (eq. 4.6).  The inner
preconditioner therefore applies ``-(Shat_c)^-1``, the inverse of ``S`` up to
the hpG cellwise approximation of ``B^T A_alpha^-1 B``.

``E_beta`` is always ``0`` in the operator
------------------------------------------
The paper puts ``E_beta`` in the operator; this implementation does not.
``E_beta != 0`` is a regularization, and regularizations belong on ``Jp`` only
(the Jp-only rule, ``RESEARCH.md`` Finding 1: an operator-level floor fails at
every mesh level).  This is the obstacle problem, where the paper also runs
``beta = 0`` in its refinement study, and the sweep ``{0, 1e-5, 1e-4, 1e-3}``
is iteration-identical -- ``beta`` enters *only* ``Shat``.  ``HPGTwoStage``
adds no ``jacobian_correction``.

The cached alpha-independent primal factorization
-------------------------------------------------
With Dirichlet rows replaced by identity rows ``E``, the assembled primal
block is ``A(alpha) = alpha * K0 + (1 - alpha) * E`` with ``K0 = stiffness +
E``, which is alpha-free (verified at machine precision, 3.6e-15), so

    A(alpha)^-1 b = K0^-1 (D_alpha b),   D_alpha = diag(1/alpha interior, 1 bc)

``K0`` is factored once at the start of the nonlinear solve (the cache of the
paper's section 4.4) and reused across every proximal iteration and every
alpha.  All that is alpha-dependent is the cheap vector scaling ``D_alpha``,
which :meth:`HPGTwoStage.setUp` rebuilds from the live ``alpha`` of the saddle
view it was installed with
(:class:`~lvpp.preconditioners.base.SaddleView`).

The matrix-free option is ``a_action="gamg"``: CG with AMG on the same
alpha-free ``K0`` in place of the cached factorization, so that no global
factorization appears in the apply path.  It is tuned with ``a_rtol`` /
``a_maxit`` / ``a_pc``, and measured bit-consistent in iteration counts with
the LU factorization at p=2/p=4 uniform.

Inner solver
------------
GMRES (not CG -- the older ``promob.py`` proximal stall) on the true Schur,
preconditioned by the batched cellwise Cholesky of ``-Shat_c``.  The default
``inner_rtol=1e-4`` / ``inner_maxit=40`` is the capped-inexact discipline that
keeps the ``p=4`` case from wedging: with the loosest published settings the
inner solver hit the 500-iteration cap on nearly every apply once alpha
deepened, and never finished.  The recorded uniform L0-L2 ``p=2`` table used
the paper-literal settings ``inner_rtol=1e-6, inner_maxit=500``, which remain
selectable: ``HPGTwoStage(inner_rtol=1e-6, inner_maxit=500)``.

The ``cell_blocks`` type contract
---------------------------------
:meth:`SpectralGalerkin.cell_blocks` takes a PETSc ``Mat`` (it reads
``getValuesCSR``), so the latent block is kept as a PETSc ``Mat`` for that
call and a separate scipy ``csr_matrix`` view of the same values is held for
the matvecs.  One documented type per purpose, and no dispatch on the input
type.
"""

import numpy as np
import scipy.sparse as sp
from firedrake.petsc import PETSc

from lvpp.hpg.spectral import SpectralGalerkin
from lvpp.preconditioners.base import PreconditionerBase

__all__ = ["HPGTwoStage", "SP_TWOSTAGE"]


# outer solver of the two-stage configuration: matrix-free action, assembled
# preconditioner matrix, FGMRES restart 250, Python PC.  The same settings as
# SP_TWOSTAGE in the hpG experiment scripts.
SP_TWOSTAGE = {
    "mat_type": "matfree",
    "pmat_type": "aij",
    "snes_type": "newtonls",
    "snes_linesearch_type": "l2",
    "snes_rtol": 1e-6,
    "snes_max_it": 100,
    "ksp_type": "fgmres",
    "ksp_rtol": 1e-6,
    "ksp_max_it": 1000,
    "ksp_gmres_restart": 250,
    "pc_type": "python",
}


def _csr(mat):
    """scipy ``csr_matrix`` view of a PETSc ``Mat``'s values, for the cheap
    matvecs the apply path needs."""
    indptr, indices, data = mat.getValuesCSR()
    return sp.csr_matrix((data, indices, indptr), shape=mat.getSize())


def _split_blocks(P, n0, n1):
    """``(K, B, D)`` PETSc submatrices of a monolithic mixed matrix: the primal
    block, the coupling and the latent block.

    Primal dofs come first and are contiguous (the lvpp ordering contract), so
    the split is two index sets: ``[0, n0)`` and ``[n0, n0 + n1)``.
    """
    is0 = PETSc.IS().createGeneral(np.arange(n0, dtype=np.int32), comm=P.comm)
    is1 = PETSc.IS().createGeneral(np.arange(n0, n0 + n1, dtype=np.int32),
                                   comm=P.comm)
    try:
        return (P.createSubMatrix(is0, is0), P.createSubMatrix(is0, is1),
                P.createSubMatrix(is1, is1))
    finally:
        is0.destroy()
        is1.destroy()


def _primal_degree(space):
    """``p`` of the hpG primal space ``CG_p``, i.e. its element degree."""
    degree = space.ufl_element().degree()
    if isinstance(degree, tuple):        # tensor-product spellings
        degree = degree[0]
    return int(degree)


class _TrueSchur:
    """PETSc MatShell context holding the action of the true Schur complement
    ``S = D - B^T A^-1 B``, the operator the inner GMRES solves with.

    ``D`` and ``B`` are the per-``setUp`` blocks and ``A^-1`` is the cached (or
    capped CG) primal inverse; all three are read live from the owning
    :class:`HPGTwoStage`, so a MatShell built once in ``install`` sees the
    blocks of each linearization without being rebuilt.
    """

    def __init__(self, owner):
        self._owner = owner

    def mult(self, mat, v, w):
        o = self._owner
        vv = v.getArray(readonly=True).copy()
        w.setArray(o._D.dot(vv) - o._BT.dot(o._Ainv(o._B.dot(vv))))


class _ShatPC:
    """Inner PC: batched cellwise Cholesky solves of ``-Shat_c``.

    ``shat_apply`` is ``-(Shat_nodal)^-1`` (``V (-Shat_modal)^-1 V^T``), so
    the apply is direct: it is the inverse of the negative-definite Schur up to
    the cellwise approximation, and nothing is iterated here.
    """

    def __init__(self, owner):
        self._owner = owner

    def apply(self, pc, x, y):
        sg = self._owner._sg
        xr = x.getArray(readonly=True).copy()
        X = xr[sg.perm_can].reshape(sg.ncells, sg.k)
        Y = sg.shat_apply(X)
        out = np.empty(self._owner._n1)
        out[sg.perm_can] = Y.reshape(-1)
        y.setArray(out)


class HPGTwoStage(PreconditionerBase):
    """The sequential ``P_F`` preconditioner of hpG section 4.4, eq. (4.5), on
    the mixed Newton system.

    Constructed with no arguments it uses the shipped defaults, which are the
    ones measured on the hpG benchmarks.  ``inner_rtol`` and ``inner_maxit``
    govern the inner GMRES on the true Schur: 1e-4 and 40, the capped-inexact
    discipline, against the paper-literal 1e-6 / 500 of the recorded ``p=2``
    table.  ``beta`` regularizes ``Shat`` alone (``beta * M_c``) and never
    enters the operator; the measured-viable default is 0.  ``a_action``
    selects how ``A(alpha)^-1`` is applied: ``"lu"`` (the default) factors the
    alpha-free ``K0`` once with MUMPS, while ``"gamg"`` is the matrix-free
    option -- capped CG with AMG on the same ``K0``, tuned by ``a_rtol``,
    ``a_maxit`` and ``a_pc`` (``"gamg"`` or ``"hypre"``).

    There is no module-global state: ``install`` wires the PETSc objects onto
    the instance, PETSc calls the preconditioner once per linear solve through
    ``setUp`` and once per outer Krylov iteration through ``apply``, and
    ``finalize`` releases everything.  ``inner_its`` records the inner-GMRES
    iteration count of every apply and ``a_its`` the CG count of every
    ``"gamg"`` A-solve; both are diagnostics for callers.
    """

    def __init__(self, inner_rtol=1e-4, inner_maxit=40, beta=0.0,
                 a_action="lu", a_rtol=1e-8, a_maxit=60, a_pc="gamg"):
        if a_action not in ("lu", "gamg"):
            raise ValueError(
                f"a_action must be 'lu' or 'gamg', got {a_action!r}")
        self.inner_rtol = float(inner_rtol)
        self.inner_maxit = int(inner_maxit)
        self.beta = float(beta)
        self.a_action = str(a_action)
        self.a_rtol = float(a_rtol)
        self.a_maxit = int(a_maxit)
        self.a_pc = str(a_pc)

        # --- wired by install() ---------------------------------------------
        self._view = None
        self._sg = None
        self._n0 = 0
        self._n1 = 0
        self._bc_nodes = np.zeros(0, dtype=np.int64)
        self._interior = None
        self._comm = None
        # --- refreshed per linearization -------------------------------------
        self._B = None
        self._BT = None
        self._D = None
        self._Dmat = None
        self._scale = None
        self._alpha = 1.0
        # --- the primal action and the inner solver --------------------------
        self._K0 = None
        self._kspA = None
        self._av = None
        self._ax = None
        self._matS = None
        self._inner = None
        self._wb = None
        self._wx = None
        # --- diagnostics ------------------------------------------------------
        self.inner_its = []
        self.a_its = []
        self.a_calls = 0
        self.a_nonconv = 0
        self.a_maxits_seen = 0
        self.n_applies = 0
        self.bad = False

    def parameters(self):
        return dict(SP_TWOSTAGE)

    # ---------------------------------------------------------------- wiring

    def install(self, snes, view):
        """Build the context and attach this object as the KSP's Python PC.

        The sizes, the true-Schur MatShell, the inner GMRES and the scratch
        vectors for the applies are built here.  The alpha-free ``K0``
        factorization and the per-linearization blocks follow in the first
        ``setUp``, which is the first point at which the assembled Pmat
        exists.
        """
        if len(view.latent_spaces) != 1:
            raise ValueError(
                "HPGTwoStage implements the single-constraint hpG P_F; got "
                f"{len(view.latent_spaces)} latent blocks")
        self._view = view
        self._n0 = int(view.n_primal)
        W = view.latent_spaces[0]
        self._n1 = int(W.dim())
        self._comm = snes.comm
        self._sg = SpectralGalerkin(
            view.mesh, W, _primal_degree(view.primal_spaces[0]),
            alpha=float(view.alpha))
        self._bc_nodes = self._collect_bc_nodes(view)
        self._interior = np.ones(self._n0, dtype=bool)
        self._interior[self._bc_nodes] = False

        self._matS = PETSc.Mat().createPython((self._n1, self._n1),
                                              comm=self._comm)
        self._matS.setPythonContext(_TrueSchur(self))
        self._matS.setUp()
        self._inner = PETSc.KSP().create(comm=self._comm)
        self._inner.setType("gmres")
        self._inner.setTolerances(rtol=self.inner_rtol, max_it=self.inner_maxit)
        self._inner.setGMRESRestart(self.inner_maxit)
        self._inner.setOperators(self._matS)
        pc_in = self._inner.getPC()
        pc_in.setType("python")
        pc_in.setPythonContext(_ShatPC(self))
        self._inner.setUp()
        self._wb = self._matS.createVecLeft()
        self._wx = self._matS.createVecRight()

        pc = snes.getKSP().getPC()
        pc.setType(PETSc.PC.Type.PYTHON)
        pc.setPythonContext(self)

    def _collect_bc_nodes(self, view):
        """Nodes of the essential BCs that fall in the primal block.

        A lifted BC on unknown ``i`` lives on ``Z.sub(i)``; its node indices
        are global mixed-space dofs, so those of unknown 0 (the first
        subspace) are exactly ``[0, n_primal)``.  A constraint outside the
        primal block is not an identity row of ``K`` and is dropped here.
        """
        chunks = []
        for bc in view.bcs:
            nodes = np.asarray(bc.nodes, dtype=np.int64)
            nodes = nodes[nodes < self._n0]
            if nodes.size:
                chunks.append(nodes)
        if not chunks:
            return np.zeros(0, dtype=np.int64)
        return np.unique(np.concatenate(chunks))

    # ------------------------------------------------- per-linearization hook

    def setUp(self, pc):
        """PETSc calls this once per linear solve, i.e. once per linearization.

        Reads the live ``alpha``, refreshes ``D_alpha``, re-splits the
        assembled Pmat into its three blocks, rebuilds the cellwise ``Shat``
        from the current latent block, and factors ``K0`` the first time it is
        called.
        """
        alpha = float(self._view.alpha)
        self._alpha = alpha
        self._refresh_scale(alpha)

        _, P = pc.getOperators()
        Km, Bm, Dm = _split_blocks(P, self._n0, self._n1)
        try:
            self._set_blocks(Bm, Dm)
        finally:
            Bm.destroy()
        try:
            if self._kspA is None:
                self._build_a(Km)
        finally:
            Km.destroy()
        self._sg.build_shat(self._sg.cell_blocks(self._Dmat), beta=self.beta)

    def _set_blocks(self, Bm, Dm):
        """Keep the latent block both as a PETSc ``Mat``, which ``cell_blocks``
        wants, and as a scipy ``csr_matrix`` view, which the matvecs want."""
        self._B = _csr(Bm)
        self._BT = self._B.T.tocsr()
        if self._Dmat is not None:
            self._Dmat.destroy()
        self._Dmat = Dm
        self._D = _csr(Dm)

    def _refresh_scale(self, alpha):
        """``D_alpha = diag(1/alpha interior, 1 on bc rows)``: the whole
        alpha-dependence of the primal solve, applied as a vector scaling."""
        scale = np.ones(self._n0)
        scale[self._interior] = 1.0 / alpha
        self._scale = scale

    def _build_a(self, Km):
        """Factor the alpha-free ``K0 = diag(1/alpha interior, 1 bc) @ Km``.

        On interior rows ``diag(scale) A(alpha)`` is ``K``, the ``alpha``
        cancelling, and on BC rows it is the identity ``E``; ``K0`` therefore
        does not depend on the alpha it was built from, and one factorization
        serves the whole nonlinear solve.  ``scale`` is the ``D_alpha``
        refreshed for this linearization; it is used only to build ``K0`` and
        is never read again.
        """
        K0 = Km.copy()
        v = K0.createVecLeft()
        v.setArray(self._scale)
        K0.diagonalScale(L=v)
        v.destroy()

        ksp = PETSc.KSP().create(comm=K0.comm)
        ksp.setOperators(K0)
        if self.a_action == "lu":
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.getPC().setFactorSolverType("mumps")
        else:
            ksp.setType("cg")
            ksp.setTolerances(rtol=self.a_rtol, max_it=self.a_maxit)
            ksp.getPC().setType(self.a_pc)
            if self.a_pc == "hypre":
                ksp.getPC().setHYPREType("boomeramg")
        ksp.setUp()
        self._K0 = K0
        self._kspA = ksp
        self._av = K0.createVecRight()
        self._ax = K0.createVecRight()

    # ------------------------------------------------------------------ apply

    def apply(self, pc, x, y):
        """One sequential ``P_F`` apply, the three steps of eq. (4.5)."""
        xr = x.getArray(readonly=True).copy()
        b_u, b_psi = xr[:self._n0], xr[self._n0:]
        t = self._Ainv(b_u)                       # A_alpha^-1 b_u
        y_schur = b_psi - self._BT.dot(t)         # y = b_psi - B^T A^-1 b_u
        dpsi = self._inner_solve(y_schur)         # S^-1 y
        du = self._Ainv(b_u - self._B.dot(dpsi))  # A^-1 (b_u - B dpsi)
        out = np.concatenate([du, dpsi])
        if not np.all(np.isfinite(out)):
            self.bad = True
        y.setArray(out)
        self.n_applies += 1

    def _Ainv(self, b):
        """``A(alpha)^-1 b = K0^-1 (D_alpha b)``.

        One solve with the cached ``K0`` and a scaling by ``D_alpha``; for
        ``a_action="gamg"`` it is only as accurate as ``a_rtol`` / ``a_maxit``
        allow, which the outer FGMRES tolerates by design.  ``a_its`` records
        its cost and ``a_nonconv`` whether it ever hit the cap.
        """
        self._av.setArray(np.ascontiguousarray(b * self._scale))
        self._kspA.solve(self._av, self._ax)
        self.a_calls += 1
        if self.a_action != "lu":
            its = int(self._kspA.getIterationNumber())
            self.a_its.append(its)
            self.a_maxits_seen = max(self.a_maxits_seen, its)
            if int(self._kspA.getConvergedReason()) < 0:
                self.a_nonconv += 1
        return self._ax.getArray(readonly=True).copy()

    def _inner_solve(self, y_schur):
        """Inner GMRES: solve ``S dpsi = y_schur`` with the cellwise Shat PC,
        i.e. the ``S^-1`` of eq. (4.5)."""
        self._wb.setArray(np.ascontiguousarray(y_schur))
        self._inner.solve(self._wb, self._wx)
        self.inner_its.append(int(self._inner.getIterationNumber()))
        return self._wx.getArray(readonly=True).copy()

    # --------------------------------------------------------------- teardown

    def finalize(self):
        """Release every PETSc object this preconditioner created, and drop the
        references to the saddle view and the spectral data."""
        for attr in ("_inner", "_matS", "_kspA", "_K0", "_Dmat",
                     "_av", "_ax", "_wb", "_wx"):
            obj = getattr(self, attr)
            if obj is not None:
                obj.destroy()
                setattr(self, attr, None)
        self._view = None
        self._sg = None

    def __repr__(self):
        return (f"HPGTwoStage(inner_rtol={self.inner_rtol!r}, "
                f"inner_maxit={self.inner_maxit!r}, beta={self.beta!r}, "
                f"a_action={self.a_action!r}, a_rtol={self.a_rtol!r}, "
                f"a_maxit={self.a_maxit!r}, a_pc={self.a_pc!r})")
