"""Per-cell spectral-Galerkin algebra for the hpG Shat preconditioner.

These routines assemble and apply the cellwise factors of the two-stage
preconditioner for the latent variable proximal point (LVPP) method of
Dokken, Farrell, Keith, Papadopoulos, & Surowiec (2025), in the spectral
cell solver of Papadopoulos (2026) = hpG.  The construction is dimension
generic: the only thing it asks of the mesh is
``d = mesh.topological_dimension()``, taken to be 1, 2 or 3, and the cells
are Cartesian tensor products.  The two- and three-dimensional cases carry
the weight of the code; one dimension is the base of the tensor product,
where every closed form collapses to a diagonal.  The two-dimensional blocks
agree with the older 2D scripts in experiments/archive_2026-09/.

The Y basis
-----------
The latent variable is expanded in the hierarchical basis
    Y_n = P_n - P_{n+2},
with P_n the Legendre polynomial, as in section 4.5 of hpG.  Each Y_n
vanishes at both endpoints of the reference interval -- Y_n is the Jacobi
polynomial P^{(1,1)}_n up to scaling -- so the family spans the same modal
space as DG while carrying the cell trace to zero.  Differentiating gives
    Y_n' = -(2n+3) P_{n+1},
and therefore
    int Y_n' Y_m' = 2(2n+3) delta_nm,
so the one-dimensional stiffness of Y is diagonal.  The mass is a different
story: the Y family is orthogonal with respect to the weight 1 - x^2, not
with respect to dx, so its mass carries a band at offset +/-2 (see
``y_mass_full_1d``).  That asymmetry reaches every corner of the module.  In one
dimension the per-cell stiffness Ahat_c in the Y basis is exactly diagonal.
For d >= 2 it is only block diagonal, split into 2^d classes indexed by the
parities of the multi-indices: the "4N blocks" of section 4.5 of hpG in two
dimensions, and eight such classes in three.  The code keeps the dense k x k
block of every cell (k = (q+1)^d, at most nine at p <= 4 in 2D and
twenty-seven at p = 4 in 3D) and checks the structure numerically rather
than storing it.

The Gram and the coupling
-------------------------
The Gram matrix Bhat between Phi, the Y basis, and Psi, the Legendre-modal
discontinuous space DG_{p-2}, is a tensor product of the one-dimensional
upper-triangular coupling
    G[n, m] = int Y_n P_m = 2/(2n+1) delta_nm - 2/(2n+5) delta_{m,n+2}.
With D_c the latent block of the Jacobian and M_c the latent mass, the
cellwise factor of the second stage is
    -Shat_c = D_c + beta * M_c + Bhat_c^T Ahat_c^-1 Bhat_c,
the discrete form of hpG (4.6); it is factorized cell by cell with a batched
Cholesky of the symmetric positive definite -Shat_c.

The Psi space in Firedrake
--------------------------
``DQ`` with ``variant="spectral"`` is the tensor-product Legendre basis with
a nodal dual: its functionals are point evaluations at the Gauss-Legendre
nodes, and Firedrake uses that dual as the basis functions.  Hence
``Function.dat`` holds nodal values rather than modal coefficients.  The two
are linked by the Vandermonde transforms V (k x k),
    nodal = V^T @ modal,   modal = (V^T)^-1 @ nodal,
which are found numerically by probing interpolation on a one-cell mesh (see
``discover_vandermonde``) and checked against the analytic tensor product of
``legval`` at the Gauss nodes and against Firedrake-assembled mass, Gram and
stiffness blocks.  On intervals FInAT rejects the ``DQ`` alias -- it reports
"DQ is supported, but handled incorrectly" -- so one dimension asks for the
``DG`` spectral alias, which is the same element in everything that matters
here; see ``_latent_family``.

Canonical flat ordering
-----------------------
For a multi-index (idx_0, ..., idx_{d-1}) the flat index is
    flat = sum_a idx_a * (q+1)**(d-1-a),
so the first axis is the outermost index; in two dimensions this is the
familiar a*(q+1) + b.

Per-cell blocks
---------------
With h_1..h_d the cell extents and hprod = prod_j h_j,
    Ahat_c  = alpha * (hprod / 2**d) * sum_i (2/h_i)**2
              * kron(MY1, ..., SY1 on axis i, ..., MY1)
    Bhat_c  = (hprod / 2**d) * kron(G1, ..., G1)
    Mdiag_c = (hprod / 2**d) * kron(diag(dp), ..., diag(dp)), flattened.
The factor (hprod / 2**d) * (2/h_i)**2 is the physical cell stiffness: the
reference measure times the affine pullback of the i-th derivative, which in
two dimensions is the hy/hx and hx/hy of the older scripts.  The naive
generalisation of that 2D expression, prod_{j != i} h_j / h_i, agrees with
it only at d = 2 and is a factor 2**(d-2) too large elsewhere; it is not the
operator that Bhat^T Ahat^-1 Bhat requires, since the matrix and the Gram
must carry the same reference measure.  The cell stiffness assembled in
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
    """The topological dimension d of ``mesh``.

    Firedrake exposes it either as a plain attribute or as a method, with
    both spellings occurring across releases, so accept whichever is there.
    """
    td = mesh.topological_dimension
    return int(td()) if callable(td) else int(td)


def _latent_family(d):
    """Firedrake family name of the spectral DQ element in dimension d.

    On intervals FInAT rejects the ``DQ`` alias outright -- it reports "DQ is
    supported, but handled incorrectly" -- so one dimension asks for ``DG``,
    which is the same tensor-product discontinuous Legendre element with a
    nodal dual.
    """
    return "DG" if d == 1 else "DQ"


def _cell_name(mesh):
    """Cell name of ``mesh``, or None when no spelling of it is available.

    Across Firedrake releases the name is a method, a property, or absent
    altogether; fall back to the UFL cell, which offers the same two
    spellings.
    """
    for obj, attr in ((mesh, "cell_name"), (mesh.ufl_cell(), "cellname")):
        getter = getattr(obj, attr, None)
        name = getter() if callable(getter) else getter
        if name is not None:
            return name
    return None


# ------------------------------------------------------------------ 1D algebra
def legendre_mass_1d(q):
    """int P_m^2 dx = 2/(2m+1) on [-1,1], for m = 0..q."""
    return 2.0 / (2.0 * np.arange(q + 1) + 1.0)


def y_mass_full_1d(q):
    """Full 1D mass matrix of the Y family.

    The diagonal is 2/(2n+1) + 2/(2n+5) and the only off-diagonal entries
    form a band at offset +/-2 with value -2/(2n+5).  The band is the price
    of orthogonality with respect to the weight 1 - x^2 instead of dx, and
    it is what forces the 2D stiffness to split into parity blocks.
    """
    n = np.arange(q + 1)
    M = np.diag(2.0 / (2.0 * n + 1) + 2.0 / (2.0 * n + 5))
    for i in range(q - 1):
        M[i, i + 2] = M[i + 2, i] = -2.0 / (2 * i + 5)
    return M


def y_stiff_full_1d(q):
    """int Y_n' Y_m' = 2(2n+3) delta_nm, from Y_n' = -(2n+3) P_{n+1}.

    The derivative Gram of the Y family is diagonal, in contrast with the
    mass just above.
    """
    return np.diag(2.0 * (2.0 * np.arange(q + 1) + 3))


def y_legendre_gram_1d(q):
    """G[n, m] = int Y_n P_m, for n, m = 0..q, upper triangular.

    Expanding Y_n = P_n - P_{n+2} and using int P_n P_m = 2/(2n+1) delta_nm
    leaves the diagonal 2/(2n+1) together with the band -2/(2n+5) two
    columns to its right.
    """
    G = np.diag(2.0 / (2.0 * np.arange(q + 1) + 1))
    for i in range(q - 1):
        G[i, i + 2] = -2.0 / (2 * i + 5)
    return G


def y_to_legendre_1d(q):
    """Coefficients cy[n, k] of P_k in Y_n, for n = 0..q.

    Each row holds a 1 at k = n and a -1 at k = n+2, which is the expansion
    Y_n = P_n - P_{n+2}; the columns run out to k = q+2.
    """
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
    """UFL expression for P_a(s) when a <= 4, written out explicitly.

    The low-order Legendre polynomials are spelled out so they can be
    interpolated as UFL expressions; any larger degree raises.
    """
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
    """(k, d) array of multi-indices in canonical flat order.

    ``np.indices`` ravels in C order, so the last axis varies fastest, which
    is exactly flat = sum_a idx_a * (q+1)**(d-1-a); in two dimensions that
    is a*(q+1) + b.
    """
    return np.indices((q + 1,) * d).reshape(d, -1).T


def _flatten(multi, base, d):
    """Canonical flat index of one multi-index on a base-ary grid.

    The first axis is the outermost digit, matching ``_multi_indices``.
    """
    out = 0
    for a in range(d):
        out = out * base + int(multi[a])
    return out


def _kron_chain(mats):
    """Tensor product of matrices with the FIRST factor outermost.

    The ordering matches the canonical flat index, so a Kronecker product of
    1D matrices acts axis by axis on the flattened multi-index.
    """
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
    """Discover the modal<->nodal transform of the DQ_q spectral element.

    The tensor-product probe prod_a P_{idx_a}(s_a) is interpolated into a
    one-cell mesh, and ``dat_j`` then holds the probe value at the dual node
    of the Firedrake-local dof j -- the DQ spectral variant has a
    point-evaluation dual at the Gauss-Legendre nodes.  The returned V has
    V[flat(idx), j] = probe value, so that nodal = V^T @ modal, with the
    canonical flat index.  The determinant of the discovered V is checked
    against the analytic tensor product of ``legval`` at the Gauss nodes.
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

    ``h`` has shape (ncells, d), and ``hx`` and ``hy`` are per-axis views
    kept for the older 2D scripts in experiments/archive_2026-09/; the
    latter is present only when d >= 2.
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
    """Dense per-cell blocks of a cell-block-diagonal PETSc Mat.

    The row and column permutations list the dofs cell-major, so that dof
    (c, i) sits at c*krow + i; reordering the assembled matrix accordingly
    lets each cell's krow x kcol block be copied out.  An assertion rejects
    any matrix that couples two different cells.
    """
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
    """Per-cell spectral-Galerkin data for the hpG latent block on the space
    Psi = DQ_{p-2} with ``variant="spectral"``.

    The object holds the cell geometry, the closed-form per-cell blocks in
    the Y basis, and the modal conversion through the discovered Vandermonde
    matrix V.  Per-cell arrays have shape (ncells, k, k) with k = (q+1)^d
    and are ordered cell-major in Firedrake's local dof order, which is
    nodal because of the point-evaluation dual.  Nodal and modal values are
    related by nodal = V^T modal and modal = solve(V^T, nodal).

    Its public methods are:

      cell_blocks():  per-cell nodal blocks of a cell-block-diagonal PETSc Mat

      to_modal():  batched conversion of nodal cell blocks to modal ones

      build_shat():  batched Cholesky of the modal -Shat_c, the cellwise factor of the two-stage preconditioner

      shat_apply():  apply -(Shat_nodal)^-1 cellwise, on nodal values

      gather(), scatter():  move a global vector to and from cell-major order

      verify_vs_firedrake():  check the closed-form cellwise structures against assembled references

    A typical use builds the data once on the latent space W and reuses the
    factorization over several solves:

    .. code-block:: python3

      sg = SpectralGalerkin(mesh, W, p, alpha)
      sg.build_shat(D_nodal, beta=beta)
      y = sg.shat_apply(x_nodal)

    The class is dimension generic: all of the above holds for d = 1, 2 and
    3 on tensor-product cells, with the Y structure collapsing to a diagonal
    in one dimension.
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
        if d >= 2:                       # 2D scripts in experiments/archive_2026-09/
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
        # 1D pieces, closed form and checked by quadrature
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
        """Per-cell dense nodal blocks of a cell-block-diagonal latent matrix.

        ``Dmat`` is a PETSc Mat on W, assembled in Firedrake's global dof
        order; the result is (ncells, k, k) in the same cell-major order
        used everywhere else in the class.
        """
        return blocks_from_petsc(Dmat, self.perm_can, self.perm_can,
                                 self.k, self.k)

    def to_modal(self, NodalBlocks):
        """Convert per-cell nodal blocks to modal ones.

        Applies the similarity transform modal_c = V Nodal_c V^T to every
        cell at once, through the discovered Vandermonde matrix.
        """
        return np.einsum("ij,cjk,lk->cil", self.V, NodalBlocks, self.V)

    def build_shat(self, D_nodal, beta=0.0, check=False):
        """Build and factor the modal -Shat_c of every cell.

        The cellwise block is
            -Shat_c = -D_jac + beta M_c + Bhat_c^T Ahat_c^-1 Bhat_c
        (hpG (4.6)), where ``D_nodal`` is the latent block of the Jacobian;
        that is -D_psi in the lvpp/lower-bound sign convention, the same
        quantity hpG writes as -Shat = D_psi + beta M + coupling.  The modal
        Cholesky is stored for ``shat_apply``.  The block is symmetric
        positive definite whenever D_psi is positive semi-definite.  With
        ``check`` the smallest eigenvalue is computed as well.
        """
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
        """Apply -(Shat_nodal)^-1 to one nodal vector per cell.

        Uses the identity -(Shat_nodal)^-1 = V (-Shat_modal)^-1 V^T: the
        input is mapped to modal values with V^T, the two triangular systems
        of the stored Cholesky factor are solved cellwise, and the result is
        mapped back with V.
        """
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
        """Parity class of each canonical flat index, one of 2**d values.

        Used only for d >= 2.  In one dimension there is a single class, and
        the verification instead demands that every off-diagonal entry of
        Ahat vanishes -- the diagonal structure hpG records for d = 1.
        """
        d, q = self.d, self.q
        idx = _multi_indices(q, d)
        par = np.zeros(self.k, dtype=int)
        for a in range(d):
            par = par * 2 + (idx[:, a] % 2)
        return par

    def verify_vs_firedrake(self):
        """Check the closed-form cellwise structures against assembled
        references.  Returns dict of max abs deviations (see the caller)."""
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