"""Per-cell spectral-Galerkin machinery for the hpG Shat preconditioner.

Paper (arXiv:2412.13733v4, sec 4.5): the per-cell spectral-Galerkin basis
    Y_n = P_n - P_{n+2}   (vanishes at the cell endpoints; = Jacobi P^{(1,1)}
    up to scaling) has DIAGONAL 1D stiffness:
    Y_n' = -(2n+3) P_{n+1}  =>  int Y_n' Y_m' = 2(2n+3) delta_nm,
but its 1D mass is NOT diagonal in dx (the Y family is orthogonal w.r.t. the
weight (1-x^2), not dx): band at +-2.  In 2D (tensor product) the per-cell
stiffness Ahat_c in the Y basis is therefore per-cell BLOCK-diagonal with 4
parity blocks (the paper's "4N blocks in 2D"); at p <= 4 the blocks are
k x k dense with k = (p-1)^2 <= 9, kept dense here.  The Gram Bhat between
Phi (Y basis) and Psi (Legendre-modal DG_{p-2}) is a Kronecker product of
the 1D upper-triangular coupling matrix
    G[n,m] = int Y_n P_m = 2/(2n+1) delta_nm - 2/(2n+5) delta_{m,n+2}.
Shat per cell (modal Psi basis):
    -Shat_c = D_c + beta * M_c + Bhat_c^T Ahat_c^-1 Bhat_c,
factorized cellwise (batched Cholesky of the SPD matrix -Shat_c).

Psi space in Firedrake: DQ (variant "spectral") = tensor-product Legendre
per cell (basis functions), with a NODAL dual (point evaluation at the
Gauss-Legendre points).  So Function.dat holds nodal values; the modal <->
nodal maps are the Vandermonde transforms V (k x k), DISCOVERED numerically
(probe interpolation on a one-cell mesh) and verified against
Firedrake-assembled mass/Gram/stiffness blocks:
    nodal = V^T @ modal,   modal = (V^T)^-1 @ nodal.
Canonical 2D ordering: flat = a*(q+1) + b for 1D indices (a, b), q = p-2
(both Psi and Phi 1D dims = q+1 = p-1).
"""
import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_triangular


# ------------------------------------------------------------------ 1D algebra
def legendre_mass_1d(q):
    """int P_m^2 on [-1,1], m = 0..q."""
    return 2.0 / (2.0 * np.arange(q + 1) + 1.0)


def y_mass_full_1d(q):
    """Full 1D mass of Y: diag 2/(2n+1)+2/(2n+5) plus +-2 band -2/(2n+5)."""
    n = np.arange(q + 1)
    M = np.diag(2.0 / (2.0 * n + 1) + 2.0 / (2.0 * n + 5))
    for i in range(q - 1):
        M[i, i + 2] = M[i + 2, i] = -2.0 / (2 * i + 5)
    return M


def y_stiff_full_1d(q):
    """int Y_n' Y_m' = 2(2n+3) delta_nm (Y_n' = -(2n+3) P_{n+1})."""
    return np.diag(2.0 * (2.0 * np.arange(q + 1) + 3))


def y_legendre_gram_1d(q):
    """G[n, m] = int Y_n P_m ds, n, m = 0..q (upper triangular)."""
    G = np.diag(2.0 / (2.0 * np.arange(q + 1) + 1))
    for i in range(q - 1):
        G[i, i + 2] = -2.0 / (2 * i + 5)
    return G


def y_to_legendre_1d(q):
    """cy[n, k]: coefficient of P_k in Y_n, n = 0..q, k = 0..q+2."""
    cy = np.zeros((q + 1, q + 3))
    for i in range(q + 1):
        cy[i, i] = 1.0
        cy[i, i + 2] = -1.0
    return cy


def numeric_1d_matrices(q, nq=None):
    """1D reference matrices by Gauss-Legendre quadrature on [-1,1]:
    (M_PP, M_YY, S_YY, G_YP); used to CHECK the closed forms."""
    nq = nq if nq is not None else q + 12
    xg, wg = np.polynomial.legendre.leggauss(nq)
    P = np.polynomial.legendre.legval(xg, np.eye(q + 3))
    dP = np.zeros((q + 3, nq))
    for k in range(1, q + 3):
        dP[k] = np.polynomial.legendre.legval(
            xg, np.polynomial.legendre.legder(np.eye(q + 3)[k]))
    Y = P[:q + 1] - P[2:q + 3]
    M_PP = (P[:q + 1] * wg) @ P[:q + 1].T
    M_YY = (Y * wg) @ Y.T
    S_YY = ((dP[0:q + 1] - dP[2:q + 3]) * wg) \
        @ (dP[0:q + 1] - dP[2:q + 3]).T
    G_YP = (Y * wg) @ P[:q + 1].T
    return M_PP, M_YY, S_YY, G_YP


def legendre_ufl_1d(a, s):
    """UFL expression for P_a(s), a <= 4."""
    import ufl
    if a == 0:
        return 0 * s + 1.0
    if a == 1:
        return s + 0 * s
    if a == 2:
        return 0.5 * (3 * s * s - 1.0)
    if a == 3:
        return 0.5 * (5 * s ** 3 - 3 * s)
    if a == 4:
        return 0.125 * (35 * s ** 4 - 30 * s * s + 3)
    raise ValueError("a <= 4 supported")


def discover_vandermonde(q, halfwidth=1.0, center=1.0):
    """Discovered modal<->nodal transform of the DQ_q spectral element.

    Interpolates the probe P_a(sx) P_b(sy) into a ONE-cell mesh; dat_j =
    probe value at the dual node of firedrake-local dof j (the DQ spectral
    variant has a point-evaluation dual at the Gauss-Legendre nodes).
    Returns V with V[(a,b), j] = probe value => nodal = V^T @ modal.
    """
    from firedrake import (Function, FunctionSpace, RectangleMesh,
                           SpatialCoordinate)
    mesh = RectangleMesh(1, 1, 2 * halfwidth, 2 * halfwidth,
                         quadrilateral=True)
    W1 = FunctionSpace(mesh, "DQ", q, variant="spectral")
    x, y = SpatialCoordinate(mesh)
    sx = (x - center) / halfwidth
    sy = (y - center) / halfwidth
    k = (q + 1) ** 2
    V = np.zeros((k, k))
    for a in range(q + 1):
        for b in range(q + 1):
            f = Function(W1)
            f.interpolate(legendre_ufl_1d(a, sx) * legendre_ufl_1d(b, sy))
            V[a * (q + 1) + b, :] = f.dat.data_ro
    xg, _ = np.polynomial.legendre.leggauss(q + 1)
    V1 = np.kron(np.polynomial.legendre.legval(xg, np.eye(q + 1)),
                 np.polynomial.legendre.legval(xg, np.eye(q + 1)))
    dv, d1 = abs(np.linalg.det(V)), abs(np.linalg.det(V1))
    assert abs(dv - d1) < 1e-6 * max(1.0, d1), \
        f"discovered Vandermonde det {dv} vs analytic {d1}"
    return V


class CellGeometry:
    """Per-cell (hx, hy) for an axis-aligned quadrilateral mesh."""

    def __init__(self, mesh):
        coords = mesh.coordinates.dat.data_ro
        vm = mesh.coordinates.function_space().cell_node_map().values
        corners = coords[vm]                          # (ncells, 4, 2)
        self.hx = corners[:, :, 0].max(axis=1) - corners[:, :, 0].min(axis=1)
        self.hy = corners[:, :, 1].max(axis=1) - corners[:, :, 1].min(axis=1)
        self.ncells = len(self.hx)


def blocks_from_petsc(mat, perm_r, perm_c, krow, kcol):
    """Per-cell dense blocks of a cell-block-diagonal PETSc Mat, given
    cell-major row/column permutations: (ncells, krow, kcol)."""
    indptr, indices, data = mat.getValuesCSR()
    M = sp.csr_matrix((data, indices, indptr), shape=mat.getSize())
    Mp = M[perm_r][:, perm_c].tocoo()
    assert np.all(Mp.row // krow == Mp.col // kcol), \
        "matrix couples different cells"
    ncells = (Mp.row.max() // krow) + 1 if Mp.row.size else 0
    out = np.zeros((ncells, krow, kcol))
    out[Mp.row // krow, Mp.row % krow, Mp.col % kcol] = Mp.data
    return out


class SpectralGalerkin:
    """Per-cell hpG spectral-Galerkin data on Psi = DQ_{p-2} spectral.

    Per-cell blocks: (ncells, k, k), cells cell-major in FIREDRAKE's local
    dof order (nodal dual).  Modal conversion via the discovered Vandermonde
    V: nodal = V^T modal, modal = solve(V^T, nodal).
    """

    def __init__(self, mesh, W, p, alpha):
        q = p - 2
        self.q, self.p = q, p
        self.W, self.mesh = W, mesh
        self.alpha = float(alpha)
        geo = CellGeometry(mesh)
        self.hx, self.hy = geo.hx, geo.hy
        self.ncells = geo.ncells
        cmap = W.cell_node_map().values
        self.cmap = cmap
        k = (q + 1) ** 2
        self.k = k
        assert W.dim() == self.ncells * k, (W.dim(), self.ncells, k)
        # cell-major permutation: (cell, firedrake-local) -> global dof
        self.perm_can = cmap.reshape(-1)
        self.V = discover_vandermonde(q)
        self.Vinv = np.linalg.inv(self.V)
        # 1D pieces (closed forms, quadrature-verified)
        self.dp = legendre_mass_1d(q)
        self.MY1 = y_mass_full_1d(q)
        self.SY1 = y_stiff_full_1d(q)
        self.G1 = y_legendre_gram_1d(q)
        # per-cell Ahat in the Y basis, canonical (n outer, m inner):
        # dense k x k; block-diagonal with 4 parity blocks
        self.Afull = (self.alpha
                      * ((self.hy / self.hx)[:, None, None]
                         * np.kron(self.SY1, self.MY1)[None]
                         + (self.hx / self.hy)[:, None, None]
                         * np.kron(self.MY1, self.SY1)[None]))
        # Bhat (modal): rows Y (n,m), cols modal Psi (a,b)
        self.Bhat = ((self.hx * self.hy / 4.0)[:, None, None]
                     * np.kron(self.G1, self.G1)[None, :, :])
        # modal latent mass diagonal per cell
        dp = 2.0 / (2.0 * np.arange(q + 1) + 1)
        self.Mdiag = ((self.hx * self.hy / 4.0)[:, None, None]
                      * (dp[None, :, None] * dp[None, None, :])
                      ).reshape(self.ncells, k)
        self.shat_chol = None       # modal Cholesky factor of -Shat_c
        self.beta = 0.0
        self.min_eig_negS = None

    # ------------------------------------------------------------- assembly
    def cell_blocks(self, Dmat):
        """Per-cell dense NODAL blocks of a cell-block-diagonal latent-block
        matrix (PETSc Mat on W): (ncells, k, k)."""
        return blocks_from_petsc(Dmat, self.perm_can, self.perm_can,
                                 self.k, self.k)

    def to_modal(self, NodalBlocks):
        """modal_c = V Nodal_c V^T (batched)."""
        return np.einsum("ij,cjk,lk->cil", self.V, NodalBlocks, self.V)

    def build_shat(self, D_nodal, beta=0.0, check=False):
        """Batched Cholesky of -Shat_c (modal) = -D_jac + beta M_c + Bhat^T
        Ahat^-1 Bhat, where D_nodal is the latent block OF THE JACOBIAN
        (= -D_psi in the lvpp/lower-bound sign convention; hpG: -Shat =
        D_psi + beta M + coupling).  SPD for D_psi PSD."""
        self.beta = float(beta)
        D_modal = self.to_modal(D_nodal)
        X = np.linalg.solve(self.Afull, self.Bhat)     # Ahat^-1 Bhat
        coupling = np.einsum("cik,ckj->cij", self.Bhat, X)
        negS = -D_modal + coupling
        if beta:
            negS = negS + beta * self.Mdiag[:, :, None] * np.eye(self.k)[None]
        if check or self.min_eig_negS is None:
            ev = np.linalg.eigvalsh(negS)
            self.min_eig_negS = float(ev.min())
        self.shat_chol = np.linalg.cholesky(negS)
        return self.shat_chol

    # ------------------------------------------------------------- applies
    def shat_apply(self, X_nodal):
        """Batched solve -(Shat_nodal_c)^-1 X: (ncells, k) -> (ncells, k).
        -(Shat_nodal)^-1 = V (-(Shat_modal))^-1 V^T."""
        Xmod = np.einsum("ji,cj->ci", self.V, X_nodal)     # V^T x (modal)
        L = self.shat_chol
        y = np.stack([
            solve_triangular(L[c], Xmod[c], lower=True)
            for c in range(L.shape[0])])
        y = np.stack([
            solve_triangular(L[c].T, y[c], lower=False)
            for c in range(L.shape[0])])
        return np.einsum("ij,cj->ci", self.V, y)           # V y (nodal)

    def gather(self, vec_np):
        return vec_np[self.perm_can]

    def scatter(self, vec_np, vals):
        vec_np[self.perm_can] = vals

    # --------------------------------------------------------- verification
    def verify_vs_firedrake(self):
        """Check closed-form per-cell structures against Firedrake-assembled
        references.  Returns dict of max abs deviations (see stage0_gate)."""
        from firedrake import (FunctionSpace, TestFunction, TrialFunction,
                               assemble, dx, grad, inner)
        q = self.q
        k, kp = self.k, (self.p + 1) ** 2
        Vp = discover_vandermonde(self.p)     # DQ_p nodal<->modal
        Wp = FunctionSpace(self.mesh, "DQ", self.p, variant="spectral")
        perm_p = Wp.cell_node_map().values.reshape(-1)
        out = {}

        # Psi mass (nodal assembly -> modal)
        Mfd = assemble(inner(TrialFunction(self.W), TestFunction(self.W)) * dx,
                       mat_type="aij").M.handle
        Mn = blocks_from_petsc(Mfd, self.perm_can, self.perm_can,
                               self.k, self.k)
        Mm = self.to_modal(Mn)
        out["psi_mass_modal_offdiag"] = float(np.abs(
            Mm - Mm.diagonal(axis1=1, axis2=2)[:, :, None]
            * np.eye(self.k)[None]).max())
        out["psi_mass_modal_diag"] = float(np.abs(
            Mm.diagonal(axis1=1, axis2=2) - self.Mdiag).max())
        Mn_rt = np.einsum("ij,cjk,lk->cil", self.Vinv, Mm, self.Vinv)
        out["psi_mass_nodal_match"] = float(np.abs(Mn_rt - Mn).max())
        Mfd.destroy()

        # Gram(DQ_p, Psi), nodal blocks; modal-ize both sides
        Gfd = assemble(inner(TrialFunction(Wp), TestFunction(self.W)) * dx,
                       mat_type="aij").M.handle
        Gn = blocks_from_petsc(Gfd, self.perm_can, perm_p, self.k, kp)
        Gm = np.einsum("ij,cjk,lk->cil", self.V, Gn, Vp)  # Psi x DQ_p modal
        dp_p = 2.0 / (2.0 * np.arange(self.p + 1) + 1)
        Gexp = np.zeros((self.ncells, self.k, kp))
        for a in range(q + 1):
            for b in range(q + 1):
                Gexp[:, a * (q + 1) + b, a * (self.p + 1) + b] = \
                    (self.hx * self.hy / 4.0) * self.dp[a] * self.dp[b]
        out["gram_modal_vs_kron"] = float(np.abs(Gm - Gexp).max())
        Gfd.destroy()

        # Bhat == J2L @ G_modal (rows Y, cols modal Psi)
        J2L = np.kron(y_to_legendre_1d(q), y_to_legendre_1d(q))  # (k, kp)
        out["bhat_vs_injection"] = float(np.abs(
            np.einsum("ik,cjk->cij", J2L, Gm) - self.Bhat).max())
        # DQ_p stiffness (modal) vs quadrature 1D derivative Grams
        Afd = assemble(inner(grad(TrialFunction(Wp)),
                             grad(TestFunction(Wp))) * dx,
                       mat_type="aij").M.handle
        An = blocks_from_petsc(Afd, perm_p, perm_p, kp, kp)
        Am = np.einsum("ij,cjk,lk->cil", Vp, An, Vp)
        # 1D Legendre-derivative Gram: dPP[a,c] = int P_a' P_c' (NOT
        # diagonal -- only the Y-basis derivative Gram is diagonal)
        _, M_YYn, S_YYn, _ = numeric_1d_matrices(self.p)
        _, M_Pn, _, _ = numeric_1d_matrices(self.p)
        xg2, wg2 = np.polynomial.legendre.leggauss(self.p + 10)
        PPn = np.polynomial.legendre.legval(xg2, np.eye(self.p + 1))
        dPPn = np.zeros((self.p + 1, self.p + 1))
        for a in range(1, self.p + 1):
            da = np.polynomial.legendre.legval(
                xg2, np.polynomial.legendre.legder(np.eye(self.p + 1)[a]))
            for c in range(self.p + 1):
                dc = np.polynomial.legendre.legval(
                    xg2, np.polynomial.legendre.legder(np.eye(self.p + 1)[c]))
                dPPn[a, c] = float(np.sum(da * dc * wg2))
        Aexp = ((self.hy / self.hx)[:, None, None] * np.kron(dPPn, M_YYn * 0
                + np.diag(dp_p))[None]
                + (self.hx / self.hy)[:, None, None]
                * np.kron(np.diag(dp_p), dPPn)[None])
        out["aleg_modal_vs_closed"] = float(np.abs(Am - Aexp).max())
        Afd.destroy()
        # Y check: J2L Ahat_Leg_modal J2L^T == closed-form dense Afull
        AY = np.einsum("ia,cab,bj->cij", J2L, Am, J2L.T)
        out["yfull_vs_closed"] = float(np.abs(AY - self.Afull).max())
        # parity off-block: entries between different (n%2, m%2) classes
        idx = np.arange(self.k)
        par = (((idx // (q + 1)) % 2) * 2 + (idx % (q + 1)) % 2)
        ri, ci = np.where(par[:, None] != par[None, :])
        out["yparity_offblock"] = (float(np.abs(AY[:, ri, ci]).max())
                                   if ri.size else 0.0)
        return out
