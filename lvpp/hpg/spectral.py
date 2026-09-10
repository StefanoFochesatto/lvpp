"""Per-cell spectral-Galerkin machinery for the hpG Shat preconditioner.

Dimension-generic replacement for ``experiments/hpg/hpg_shat.py``: everything
is expressed with ``d = mesh.topological_dimension()`` in (1, 2, 3) on
tensor-product cells, and the 2D reduction is exactly the old ``Afull`` /
``Bhat`` / ``Mdiag``.

The Y basis (paper arXiv:2412.13733v4, sec 4.5)
----------------------------------------------
    Y_n = P_n - P_{n+2}          (vanishes at the cell endpoints; = Jacobi
                                  P^{(1,1)} up to scaling)
    Y_n' = -(2n+3) P_{n+1}   =>   int Y_n' Y_m' = 2(2n+3) delta_nm

so the 1D **stiffness** of Y is diagonal.  Its 1D **mass** is NOT diagonal in
dx -- the Y family is orthogonal w.r.t. the weight (1-x^2), not dx -- and has a
+-2 band (``y_mass_full_1d``).  Hence the per-cell stiffness ``Ahat_c`` in the
Y basis is *diagonal when d = 1*, but for d >= 2 only BLOCK-diagonal with
``2^d`` parity classes per cell (the paper's "4N blocks in 2D").  The d = 3
count of 8 is an inference from the same parity argument [INFERENCE].  We keep
the dense k x k block (k = (q+1)^d; k <= 9 at p <= 4 in 2D, k <= 27 at p = 4
in 3D) and *verify* the structure rather than storing it.

The Gram ``Bhat`` between Phi (Y basis) and Psi (Legendre-modal DG_{p-2}) is a
tensor product of the 1D upper-triangular coupling matrix
    G[n, m] = int Y_n P_m = 2/(2n+1) delta_nm - 2/(2n+5) delta_{m,n+2}.
Shat per cell (modal Psi basis):
    -Shat_c = D_c + beta * M_c + Bhat_c^T Ahat_c^-1 Bhat_c,
factorized cellwise (batched Cholesky of the SPD matrix -Shat_c).

Psi space in Firedrake: ``DQ`` (variant "spectral") = tensor-product Legendre
per cell (basis functions) with a NODAL dual (point evaluation at the
Gauss-Legendre points).  (On intervals FInAT rejects the ``DQ`` alias
outright -- "DQ is supported, but handled incorrectly" -- so 1D uses the
``DG`` spectral alias, the same element; see ``_latent_family``.)  So
``Function.dat`` holds nodal values; it is NOT
modal coefficients, and the modal <-> nodal maps are the Vandermonde
transforms V (k x k), DISCOVERED numerically (probe interpolation on a
one-cell mesh) and verified against Firedrake-assembled mass/Gram/stiffness
blocks:
    nodal = V^T @ modal,   modal = (V^T)^-1 @ nodal.
Canonical flat ordering: ``flat = sum_a idx_a * (q+1)**(d-1-a)``, i.e. the
first axis is the outermost index; for d = 2 this is the old ``a*(q+1) + b``.

Per-cell blocks (cell extents h_1..h_d, ``hprod = prod_j h_j``):
    Ahat_c  = alpha * (hprod / 2**d) * sum_i (2/h_i)**2
              * kron(... SY1 on axis i, MY1 on axis j != i ...)
            = alpha * sum_i (hprod / h_i**2) * 2**(2-d) * kron(...)
    Bhat_c  = (hprod / 2**d) * kron(G1, ..., G1)
    Mdiag_c = (hprod / 2**d) * kron(diag(dp), ..., diag(dp)), flattened

The ``(hprod / 2**d) * (2/h_i)**2`` factor is the *physical* cell stiffness
(reference measure times the affine pullback of the i-th derivative); at
d = 2 it is exactly the old ``hy/hx`` / ``hx/hy``.  The naive generalisation
of that 2D expression, ``prod_{j != i} h_j / h_i``, agrees with it only at
d = 2 and is a factor ``2**(d-2)`` too large elsewhere -- it is not the
operator that ``Bhat^T Ahat^-1 Bhat`` requires (both matrix and Gram must
carry the same reference measure), and the assembled cell stiffness in
:meth:`SpectralGalerkin.verify_vs_firedrake` pins the physical one.
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_triangular

__all__ = [
    "legendre_mass_1d", "y_mass_full_1d", "y_stiff_full_1d",
    "y_legendre_gram_1d", "y_to_legendre_1d", "numeric_1d_matrices",
    "legendre_ufl_1d", "discover_vandermonde", "CellGeometry",
    "blocks_from_petsc", "SpectralGalerkin",
]

# Tensor-product cells: the whole module assumes a Cartesian reference cell.
_TENSOR_PRODUCT_CELLS = ("interval", "quadrilateral", "hexahedron")


def _topological_dimension(mesh):
    """``d`` of ``mesh``; Firedrake exposes it as an int or as a method."""
    td = mesh.topological_dimension
    return int(td()) if callable(td) else int(td)


def _latent_family(d):
    """Firedrake family name of the DQ spectral element.

    On intervals FInAT rejects the ``DQ`` alias outright ("DQ is supported,
    but handled incorrectly"); ``DG`` there is the same tensor-product
    discontinuous Legendre element with a nodal dual.
    """
    return "DG" if d == 1 else "DQ"


def _cell_name(mesh):
    """Cell name of ``mesh``, or None if the attribute is absent.

    Across Firedrake versions this is a method, a property, or missing
    entirely; fall back to the UFL cell, which has the same two spellings.
    """
    for obj, attr in ((mesh, "cell_name"), (mesh.ufl_cell(), "cellname")):
        getter = getattr(obj, attr, None)
        name = getter() if callable(getter) else getter
        if name is not None:
            return name
    return None


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


# ------------------------------------------------------- canonical indexing
def _multi_indices(q, d):
    """(k, d) multi-index array in canonical flat order.

    ``np.indices`` ravels C-order, so the last axis varies fastest -- exactly
    ``flat = sum_a idx_a * (q+1)**(d-1-a)``.  For d = 2 this is ``a*(q+1)+b``.
    """
    return np.indices((q + 1,) * d).reshape(d, -1).T


def _flatten(multi, base, d):
    """Canonical flat index of a single multi-index, on a ``base``-ary grid."""
    out = 0
    for a in range(d):
        out = out * base + int(multi[a])
    return out


def _kron_chain(mats):
    """Tensor product with the FIRST factor outermost (matches the flat order)."""
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _one_cell_mesh(d, halfwidth, center):
    """One-cell tensor-product mesh with coordinates spanning
    ``[center - halfwidth, center + halfwidth]^d``."""
    from firedrake import BoxMesh, IntervalMesh, RectangleMesh
    length = 2.0 * halfwidth
    if d == 1:
        mesh = IntervalMesh(1, length)
    elif d == 2:
        mesh = RectangleMesh(1, 1, length, length, quadrilateral=True)
    elif d == 3:
        mesh = BoxMesh(1, 1, 1, length, length, length, hexahedral=True)
    else:
        raise ValueError(f"d = {d} not supported (1, 2 or 3)")
    shift = center - halfwidth
    if shift != 0.0:
        mesh.coordinates.dat.data[:] += shift
    return mesh


def discover_vandermonde(q, d=2, halfwidth=1.0, center=1.0):
    """Discovered modal<->nodal transform of the DQ_q spectral element.

    Interpolates the tensor-product probe ``prod_a P_{idx_a}(s_a)`` into a
    ONE-cell mesh; ``dat_j`` = probe value at the dual node of firedrake-local
    dof j (the DQ spectral variant has a point-evaluation dual at the
    Gauss-Legendre nodes).  Returns V with ``V[flat(idx), j] = probe value``,
    so ``nodal = V^T @ modal``; the flat index is canonical
    (``sum_a idx_a * (q+1)**(d-1-a)``).  The determinant is cross-checked
    against the analytic ``kron`` of ``legval(gauss, I)``.
    """
    from firedrake import Function, FunctionSpace, SpatialCoordinate
    mesh = _one_cell_mesh(d, halfwidth, center)
    W1 = FunctionSpace(mesh, _latent_family(d), q, variant="spectral")
    X = SpatialCoordinate(mesh)
    s = [(X[i] - center) / halfwidth for i in range(d)]
    k = (q + 1) ** d
    V = np.zeros((k, k))
    for multi in _multi_indices(q, d):
        f = Function(W1)
        expr = None
        for a, sa in zip(multi, s):
            term = legendre_ufl_1d(int(a), sa)
            expr = term if expr is None else expr * term
        f.interpolate(expr)
        V[_flatten(multi, q + 1, d), :] = f.dat.data_ro
    xg, _ = np.polynomial.legendre.leggauss(q + 1)
    V1 = np.polynomial.legendre.legval(xg, np.eye(q + 1))
    Van = _kron_chain([V1] * d)
    dv, d1 = abs(np.linalg.det(V)), abs(np.linalg.det(Van))
    assert abs(dv - d1) < 1e-6 * max(1.0, d1), \
        f"discovered Vandermonde det {dv} vs analytic {d1}"
    return V


class CellGeometry:
    """Per-cell extents of an axis-aligned tensor-product mesh.

    ``h`` is ``(ncells, d)``; ``hx``/``hy`` are the per-axis views kept for
    back-compatibility with the 2D harness (``d >= 2``).
    """

    def __init__(self, mesh):
        d = _topological_dimension(mesh)
        self.d = d
        coords = mesh.coordinates.dat.data_ro
        if coords.ndim == 1:            # 1D meshes store coordinates flat
            coords = coords.reshape(-1, 1)
        vm = mesh.coordinates.function_space().cell_node_map().values
        corners = coords[vm]                          # (ncells, ncorners, d)
        self.h = corners.max(axis=1) - corners.min(axis=1)
        self.ncells = self.h.shape[0]
        self.hx = self.h[:, 0]
        if d >= 2:
            self.hy = self.h[:, 1]


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

    Per-cell blocks: (ncells, k, k) with ``k = (q+1)**d``, cells cell-major in
    FIREDRAKE's local dof order (nodal dual).  Modal conversion via the
    discovered Vandermonde V: ``nodal = V^T modal``, ``modal = solve(V^T,
    nodal)``.
    """

    def __init__(self, mesh, W, p, alpha):
        d = _topological_dimension(mesh)
        if d not in (1, 2, 3):
            raise ValueError(f"d = {d} not supported (1, 2 or 3)")
        cell = _cell_name(mesh)
        if cell is not None and cell not in _TENSOR_PRODUCT_CELLS:
            raise ValueError(
                f"hpG needs tensor-product cells; got {cell!r}")
        q = p - 2
        self.q, self.p, self.d = q, p, d
        self.W, self.mesh = W, mesh
        self.alpha = float(alpha)
        geo = CellGeometry(mesh)
        self.h, self.ncells = geo.h, geo.ncells
        self.hprod = np.prod(self.h, axis=1)
        self.hx = geo.hx
        if d >= 2:                       # 2D harness back-compat
            self.hy = geo.hy
        cmap = W.cell_node_map().values
        self.cmap = cmap
        k = (q + 1) ** d
        self.k = k
        assert W.dim() == self.ncells * k, (W.dim(), self.ncells, k)
        # cell-major permutation: (cell, firedrake-local) -> global dof
        self.perm_can = cmap.reshape(-1)
        self.V = discover_vandermonde(q, d)
        self.Vinv = np.linalg.inv(self.V)
        # 1D pieces (closed forms, quadrature-verified)
        self.dp = legendre_mass_1d(q)
        self.MY1 = y_mass_full_1d(q)
        self.SY1 = y_stiff_full_1d(q)
        self.G1 = y_legendre_gram_1d(q)
        # per-cell Ahat in the Y basis, canonical order (axis 0 outermost):
        # dense k x k; diagonal for d = 1, block-diagonal (2^d parity
        # classes) for d >= 2.  The weight is the physical reference-cell
        # one, (hprod / 2**d) * (2 / h_i)**2 = prod_{j != i} h_j / h_i at
        # d = 2 (the old hy/hx, hx/hy) -- the value the assembled cell
        # stiffness pins in every d.
        self.Afull = np.zeros((self.ncells, k, k))
        ref_measure = self.hprod / 2.0 ** d
        for i in range(d):
            mats = [self.SY1 if j == i else self.MY1 for j in range(d)]
            weight = self.alpha * ref_measure * (2.0 / self.h[:, i]) ** 2
            self.Afull += weight[:, None, None] * _kron_chain(mats)[None]
        # Bhat (modal): rows Y, cols modal Psi
        self.Bhat = (self.hprod / 2.0 ** d)[:, None, None] \
            * _kron_chain([self.G1] * d)[None, :, :]
        # modal latent mass diagonal per cell, flattened in canonical order
        # (the diagonal of the tensor product of the 1D Legendre masses)
        dp_flat = np.prod(self.dp[_multi_indices(q, d)], axis=1)
        self.Mdiag = (self.hprod / 2.0 ** d)[:, None] * dp_flat[None, :]
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
    def _parity_classes(self):
        """Parity class of each canonical flat index (``2**d`` classes).

        Only used for d >= 2; at d = 1 there is a single class and the
        verification instead demands that EVERY off-diagonal entry of Ahat
        vanishes (the paper's "Ahat is diagonal when d = 1").
        """
        d, q = self.d, self.q
        idx = _multi_indices(q, d)
        par = np.zeros(self.k, dtype=int)
        for a in range(d):
            par = par * 2 + (idx[:, a] % 2)
        return par

    def verify_vs_firedrake(self):
        """Check closed-form per-cell structures against Firedrake-assembled
        references.  Returns dict of max abs deviations (see stage0_gate)."""
        from firedrake import (FunctionSpace, TestFunction, TrialFunction,
                               assemble, dx, grad, inner)
        d, q = self.d, self.q
        k, kp = self.k, (self.p + 1) ** d
        Vp = discover_vandermonde(self.p, d)     # DQ_p nodal<->modal
        Wp = FunctionSpace(self.mesh, _latent_family(d), self.p,
                           variant="spectral")
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
        for multi in _multi_indices(q, d):
            Gexp[:, _flatten(multi, q + 1, d), _flatten(multi, self.p + 1, d)] \
                = (self.hprod / 2.0 ** d) * np.prod(self.dp[multi])
        out["gram_modal_vs_kron"] = float(np.abs(Gm - Gexp).max())
        Gfd.destroy()

        # Bhat == J2L @ G_modal (rows Y, cols modal Psi)
        J2L = _kron_chain([y_to_legendre_1d(q)] * d)      # (k, kp)
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
        xg2, wg2 = np.polynomial.legendre.leggauss(self.p + 10)
        dPPn = np.zeros((self.p + 1, self.p + 1))
        der = [np.polynomial.legendre.legval(
            xg2, np.polynomial.legendre.legder(np.eye(self.p + 1)[a]))
            for a in range(self.p + 1)]
        for a in range(1, self.p + 1):
            for c in range(self.p + 1):
                dPPn[a, c] = float(np.sum(der[a] * der[c] * wg2))
        Mleg = np.diag(dp_p)
        Aexp = np.zeros((self.ncells, kp, kp))
        ref_measure = self.hprod / 2.0 ** d
        for i in range(d):
            mats = [dPPn if j == i else Mleg for j in range(d)]
            weight = ref_measure * (2.0 / self.h[:, i]) ** 2
            Aexp += weight[:, None, None] * _kron_chain(mats)[None]
        out["aleg_modal_vs_closed"] = float(np.abs(Am - Aexp).max())
        Afd.destroy()
        # Y check: J2L Ahat_Leg_modal J2L^T == closed-form dense Afull
        AY = np.einsum("ia,cab,bj->cij", J2L, Am, J2L.T)
        out["yfull_vs_closed"] = float(np.abs(AY - self.Afull).max())
        # parity off-block: entries between different parity classes; for
        # d = 1 there is a single class, so EVERY off-diagonal must vanish.
        if d == 1:
            ri, ci = np.where(~np.eye(self.k, dtype=bool))
        else:
            par = self._parity_classes()
            ri, ci = np.where(par[:, None] != par[None, :])
        out["yparity_offblock"] = (float(np.abs(AY[:, ri, ci]).max())
                                   if ri.size else 0.0)
        out["shape"] = ("diagonal",) if d == 1 else ("block-diagonal", 2 ** d)
        out["d"] = int(d)
        out["k"] = int(k)
        return out
