"""Shared harness for the hpG (hierarchical Proximal Galerkin, arXiv:2412.13733)
port into the Firedrake LVPP ecosystem.

Everything here runs under the env guard:
  PETSC_DIR=/home/stefano/firedrake/petsc PETSC_ARCH=arch-firedrake-default \
  OMP_NUM_THREADS=1 /home/stefano/firedrake/venv-firedrake/bin/python <script>

Serial only.  No existing files are modified; this folder (lvpp/experiments/hpg/)
is the only write target.

Benchmark (shared with the archived P1 experiments, so errors are comparable
ACROSS discretizations): sphere obstacle on RectangleMesh(2.0, 2.0) with the
analytic psi/u of eps_schedule.py.
"""
import time

import numpy as np
import scipy.sparse as sp

from firedrake import (DirichletBC, Function, FunctionSpace, RectangleMesh,
                       SpatialCoordinate, TestFunction, TrialFunction,
                       assemble, conditional, dx, errornorm, grad, inner, le,
                       ln, sqrt)
from firedrake.petsc import PETSc
from lvpp import LVPP

r0 = 0.9
AFREE = 0.697965148223374
A_, B_ = 0.680259411891719, 0.471519893402112

FCP_QUAD = 20          # quadrature degree for exp(psi) forms (>= 3p for p<=4)
LU_REFS_P1 = {0: (8, 21, 1.553e-2, 545), 1: (11, 23, 3.599e-3, 2113),
              2: (8, 19, 8.375e-4, 8321)}   # our floorless P1 references

REASON_NAMES = {
    2: "RTOL", 3: "ATOL", 4: "ITS",
    -3: "DIVERGED_ITS", -4: "DIVERGED_DTOL", -5: "DIVERGED_BREAKDOWN",
    -6: "DIVERGED_BICG", -7: "DIVERGED_NONSYMM", -8: "DIVERGED_INDEF_PC",
    -9: "DIVERGED_INDEF_MAT", -11: "DIVERGED_PCSETUP",
}


# --------------------------------------------------------------- analytic data
def psiUFL(r):
    psi0 = np.sqrt(1.0 - r0 * r0)
    dpsi0 = -r0 / psi0
    return conditional(le(r, r0), sqrt(1.0 - r * r), psi0 + dpsi0 * (r - r0))


def uexactUFL(r):
    return conditional(le(r, AFREE), psiUFL(r), -A_ * ln(r) + B_)


# ---------------------------------------------------------------------- meshes
def uniform_quad(n):
    """n x n uniform quadrilateral mesh, 2.0 x 2.0 (the archived benchmark)."""
    return RectangleMesh(n, n, 2.0, 2.0, quadrilateral=True)


def graded_axis(n_total, x_lo, x_hi, band_center, band_halfwidth, ratio):
    """1D graded axis: a uniform fine band of n_band cells centered on
    band_center, geometric growth away from the band on both sides with
    coarsest/finest ~ `ratio`; n_total cells overall.  The band width adapts
    so the geometric chains fit exactly.  Returns (coords, hmin, hmax).
    """
    n_band = max(2, int(round(0.12 * n_total)))
    n_side = n_total - n_band
    nl = n_side // 2
    nr = n_side - nl
    lo0 = band_center - band_halfwidth
    hi0 = band_center + band_halfwidth

    def geo_sizes(n, length, R):
        # sizes h*g^k (k = 0..n-1), h*g^(n-1)/h = R, sum == length
        if n == 0:
            return np.zeros(0), 1.0, 1.0
        g = R ** (1.0 / n)
        h = length / float((g ** np.arange(n)).sum())
        return h * g ** np.arange(n), g, h

    _, _, h_l = geo_sizes(nl, lo0 - x_lo, ratio)
    _, _, h_r = geo_sizes(nr, x_hi - hi0, ratio)
    h_band = min(h_l, h_r)
    w = h_band * n_band                      # adapted band width
    lo, hi = band_center - w / 2, band_center + w / 2
    s_l, _, _ = geo_sizes(nl, lo - x_lo, ratio)
    s_r, _, _ = geo_sizes(nr, x_hi - hi, ratio)
    coords = np.concatenate((
        [x_lo], x_lo + np.cumsum(s_l[::-1]),          # coarse -> fine
        lo + h_band * (1 + np.arange(n_band)),        # band (exact fit)
        hi + np.cumsum(s_r)[:-1], [x_hi]))
    sizes = np.diff(coords)
    return coords, float(sizes.min()), float(sizes.max())


def graded_quad(n_total, band_center=0.9, band_halfwidth=0.10, ratio=20.0):
    """Quad mesh on the same 2x2 domain as uniform_quad, geometrically graded
    toward the free-boundary band |r - band_center| < halfwidth on BOTH axes
    (Cartesian tensor-product cells, as hpG requires).  Returns (mesh,
    grading_ratio hmax/hmin)."""
    coords_x, hmin, hmax = graded_axis(
        n_total, 0.0, 2.0, band_center, band_halfwidth, ratio)
    mesh = RectangleMesh(n_total, n_total, 2.0, 2.0, quadrilateral=True)
    cd = mesh.coordinates.dat.data
    # rank-match the structured vertex ordering to the graded axis
    xs = np.unique(cd[:, 0])
    ys = np.unique(cd[:, 1])
    assert len(xs) == len(coords_x) and len(ys) == len(coords_x), \
        (len(xs), len(ys), len(coords_x))
    xmap = dict(zip(xs, coords_x))
    ymap = dict(zip(ys, coords_x))
    cd[:, 0] = [xmap[v] for v in cd[:, 0]]
    cd[:, 1] = [ymap[v] for v in cd[:, 1]]
    return mesh, hmax / hmin


# ---------------------------------------------------------------------- spaces
def build_spaces(mesh, p):
    """hpG spaces: u in CG_p, latent psi in DG_{p-2} with the Legendre-modal
    (spectral) variant.  The CG_p space IS the hierarchical basis's polynomial
    space (P1 hats + Jacobi bubbles span exactly P_p); the hierarchy is a basis
    choice, so standard nodal CG_p is the same discretization.  The DQ spectral
    variant makes the psi mass matrix diagonal (Lemma 3.2) -- verified
    numerically in stage0."""
    V = FunctionSpace(mesh, "CG", p)
    W = FunctionSpace(mesh, "DQ", max(p - 2, 0), variant="spectral")
    return V, W


def make_lvpp(mesh, p, solver_parameters, name, tol=1e-4, max_prox=100,
              psi_spaces=None, verbose=False, monitor=None):
    """LVPP instance on the hpG spaces with the sphere-obstacle benchmark."""
    V, W = build_spaces(mesh, p)
    u = Function(V, name="u")
    x, y = SpatialCoordinate(mesh)
    r = sqrt(x * x + y * y)
    lb = Function(V, name="psi_lb").interpolate(psiUFL(r))
    bc = DirichletBC(V, uexactUFL(r), "on_boundary")
    energy = 0.5 * inner(grad(u), grad(u)) * dx  # f = 0
    lv = LVPP(energy=energy, u=u, bounds=(lb, None), bcs=bc,
              psi_spaces=[W],
              alpha_rule="double_exponential",
              alpha_parameters={"alpha_max": 10.0},
              increment_norm="H1", verbose=verbose,
              psi_floor=0.0,                      # floorless: our Finding-1 rule
              form_compiler_parameters={"quadrature_degree": FCP_QUAD},
              solver_parameters=solver_parameters, name=name)
    if monitor is not None:
        lv._solver.snes.ksp.setMonitor(monitor)
    return lv


# ------------------------------------------------------------ matrix utilities
def csr(m):
    indptr, indices, data = m.getValuesCSR()
    return sp.csr_matrix((data, indices, indptr), shape=m.getSize())


def field_ises(Jm, n0, n1):
    is0 = PETSc.IS().createGeneral(np.arange(n0, dtype=np.int32), comm=Jm.comm)
    is1 = PETSc.IS().createGeneral(np.arange(n0, n0 + n1, dtype=np.int32),
                                   comm=Jm.comm)
    return is0, is1


def split_blocks(Jm, n0, n1):
    """Split the monolithic mixed Jacobian into (K, B, D) PETSc matrices.
    Ordering: primal dofs first (contiguous), latent last (lvpp convention)."""
    import numpy as np
    is0, is1 = field_ises(Jm, n0, n1)
    Km = Jm.createSubMatrix(is0, is0)
    Bm = Jm.createSubMatrix(is0, is1)
    Dm = Jm.createSubMatrix(is1, is1)
    is0.destroy()
    is1.destroy()
    return Km, Bm, Dm


field_ises_petsc = field_ises
split_blocks_pet = split_blocks


def mass_structure(W):
    """Assemble the latent mass matrix; return (max|off-diag|, min|diag|,
    max|off-cell-block|)."""
    Mt = assemble(inner(TrialFunction(W), TestFunction(W)) * dx, mat_type="aij")
    M = Mt.M.handle
    indptr, indices, data = M.getValuesCSR()
    d = np.abs(data)
    rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    offd = d[indices != rows]
    maxoff = float(offd.max()) if offd.size else 0.0
    diag = d[indices == rows]
    mindg = float(diag.min()) if diag.size else 0.0
    # off-cell-block: entries coupling dofs owned by different cells
    cmap = W.cell_node_map().values
    dof2cell = np.empty(W.dim(), dtype=np.int64)
    for c in range(cmap.shape[0]):
        dof2cell[cmap[c]] = c
    cross = dof2cell[rows] != dof2cell[indices]
    maxcross = float(d[cross].max()) if cross.any() else 0.0
    M.destroy()
    return maxoff, mindg, maxcross


class SolveMonitor:
    """Captures outer-GMRES its per Newton solve + the RHS of each solve
    (archive ksp_mon pattern; per_prox grouping included)."""

    def __init__(self):
        self.groups = []
        self.rhss = []
        self.reasons = []

    def __call__(self, ksp, its, rnorm):
        if its == 1:
            self.groups.append(1)
            self.rhss.append(ksp.getRhs().copy())
            self.reasons.append(None)
        elif its > 1:
            if self.groups:
                self.groups[-1] = its
            else:
                self.groups.append(its)

    def per_prox(self, newton_its):
        out, i = [], 0
        for n in newton_its:
            out.append(self.groups[i:i + n])
            i += n
        return out, self.groups[i:]


def run_lvpp(lv, tol=1e-4, max_prox=100):
    t0 = time.time()
    lv.solve(tol=tol, max_proximal_iterations=max_prox)
    return time.time() - t0
