"""Matrix-free LVPP saddle solves: constant vs drift-aware preconditioner floor.

Falsification check (see RESEARCH.md, Finding 1) that the LVPP mixed system
solves with a matrix-free operator + block preconditioner, and measurement of
the degeneracy-aware floor

    eps(x) = psi_floor + psi_floor_drift * alpha * |lambda(x)|

(lambda = the proximal drift = the discrete multiplier, in-place-updated)
against constant floors, across sphere-obstacle mesh levels.

All floors live on Jp (preconditioner Jacobian) only: the Newton direction is
exact, so every converged config must reproduce the LU baseline error.

Run:  python experiments/matfree_fieldsplit.py
"""
import time

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.linalg import lu_factor, lu_solve
import ufl
from firedrake import (Constant, DirichletBC, Function, FunctionSpace,
                       MeshHierarchy, RectangleMesh, SpatialCoordinate,
                       assemble, conditional, errornorm, grad, inner, le, ln,
                       sqrt, dx)
from firedrake.petsc import PETSc
from lvpp import LVPP

r0 = 0.9
AFREE = 0.697965148223374
A, B = 0.680259411891719, 0.471519893402112


def psiUFL(r):
    psi0 = np.sqrt(1.0 - r0 * r0)
    dpsi0 = -r0 / psi0
    return conditional(le(r, r0), sqrt(1.0 - r * r), psi0 + dpsi0 * (r - r0))


def uexactUFL(r):
    return conditional(le(r, AFREE), psiUFL(r), -A * ln(r) + B)


SP_LU = {
    "mat_type": "aij",
    "snes_linesearch_type": "l2",
    "snes_linesearch_maxlambda": 1.0,
    "snes_rtol": 1e-6,
    "snes_max_it": 100,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

SP_MF = {
    "mat_type": "matfree",
    "pmat_type": "aij",
    # fieldsplit must extract its blocks from the assembled Pmat: the matfree
    # operator has no submatrix extraction (this was the level-0 breakthrough)
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
    "pc_fieldsplit_type": "additive",
    "fieldsplit_0_ksp_type": "cg",
    "fieldsplit_0_pc_type": "gamg",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "jacobi",
}


class TScalePC:
    """Right-preconditioner M^{-1} = diag(e^{-phi}) on latent dofs (1 on
    primal).  GMRES on J*M^{-1} is similar to the symmetric congruence
    T J T with T = diag(e^{-phi/2}): the latent block becomes
    (lambda/alpha) M and the coupling O(1) -- the e^phi degeneracy is
    repaired at the operator level.  delta = M^{-1} y is the exact Newton
    direction; no form, root, or path change.  Unlike the Jacobi
    equilibration variant, the primal rows are left unscaled."""

    def __init__(self, lvpp, latent_field):
        self.lvpp = lvpp
        self.k = latent_field
        self._sc = None

    def _scaling(self, nloc):
        z = self.lvpp._z
        if self._sc is None or self._sc.shape[0] != nloc:
            self._sc = np.ones(nloc)
        phi = z.subfunctions[self.k].dat.data_ro
        off = nloc - phi.size  # latent field last (single-latent problems)
        self._sc[off:] = np.exp(np.minimum(-phi, 700.0))
        return self._sc

    def apply(self, pc, x, y):
        xa = x.getArray(readonly=True)
        ya = y.getArray()
        ya[:] = xa * self._scaling(xa.size)
        y.setArray(ya)


class DeflationPC:
    """Two-level deflation with the exact compensated near-null basis.

    Near-null modes (measured): v = (-K^-1 B e_j, e_j) for latent dofs j on
    the deep-contact set, eigenvalues ~ h^4/alpha (Schur collapse).  The
    deep-latent Newton component is PHYSICAL: it carries the contact-force
    (drift) update -- latent_increment diverges 0.2 -> 1390 across the prox
    loop while the primal increment converges -- so the coarse space must
    contain the FULL compensated mode, not just the local latent part
    (truncating the global K^-1-smoothed u-part injects a huge residual
    through the huge flat-direction coefficient).

    Coarse space: Z = [-K^-1 B E, E] built per refresh (m u-block CG solves
    against the unfloored assembled Jacobian's (1,1) block); coarse
    operator H = Z^T J Z (near-singular, well-conditioned relatively);
    direct LU on H reproduces the LU baseline's flat-direction handling.

    y = Q x + (I - Q J) M_fs^-1 (I - Q J) x,   Q = Z H^-1 Z^T.
"""

    def __init__(self, lvpp, latent_field, thresh=-50.0, refresh=25,
                 cg_rtol=1e-8):
        self.lvpp = lvpp
        self.k = latent_field
        self.thresh = thresh
        self.refresh = refresh
        self.cg_rtol = cg_rtol
        self.inner = None
        self.ndeep = -1
        self.ncalls = 0
        self._since = 10 ** 9
        self._Z = None
        self._lu = None
        self._deep = None

    def setUp(self, pc):
        self.comm = pc.getComm()
        self.A, self.P = pc.getOperators()
        pre = "defl_"
        opts = PETSc.Options()
        opts[pre + "pc_type"] = "fieldsplit"
        opts[pre + "pc_fieldsplit_type"] = "additive"
        opts[pre + "fieldsplit_0_ksp_type"] = "cg"
        opts[pre + "fieldsplit_0_pc_type"] = "gamg"
        opts[pre + "fieldsplit_1_ksp_type"] = "preonly"
        opts[pre + "fieldsplit_1_pc_type"] = "jacobi"
        self.inner = PETSc.PC()
        self.inner.create(comm=self.comm)
        self.inner.setOptionsPrefix(pre)
        dm = pc.getDM()
        if dm is not None:
            self.inner.setDM(dm)
        self.inner.setOperators(self.P, self.P)
        self.inner.setFromOptions()
        self.inner.setUp()
        nloc = self.P.getSize()[0]
        self._t1 = PETSc.Vec().createWithArray(np.zeros(nloc), comm=self.comm)
        self._t2 = PETSc.Vec().createWithArray(np.zeros(nloc), comm=self.comm)

    def _refresh(self):
        z = self.lvpp._z
        phi = z.subfunctions[self.k].dat.data_ro
        nloc = self.P.getSize()[0]
        off = nloc - phi.size          # latent field offset (last field)
        deep = np.where(phi < self.thresh)[0]
        m = deep.size
        self.ndeep = m
        self._deep = deep + off        # global latent dof ids
        if m == 0:
            self._Z = None
            self._lu = None
            self._since = 0
            return
        # unfloored assembled Jacobian for this refresh
        self._Jm = assemble(self.lvpp._J, mat_type="aij")
        Jop = self._Jm.M.handle
        self._Jop = Jop
        # u-block K: (1,1) principal block on all u dofs
        ISu = PETSc.IS().createGeneral(np.arange(off, dtype=np.int32),
                                       comm=self.comm)
        Kuu = Jop.createSubMatrix(ISu, ISu)
        kspu = PETSc.KSP().create(self.comm)
        kspu.setOperators(Kuu, Kuu)
        kspu.setType("cg")
        kspu.getPC().setType("gamg")
        kspu.setTolerances(rtol=self.cg_rtol, max_it=300)
        kspu.setUp()
        # compensated basis: Z = [-K^-1 B E, E]
        Z = np.zeros((nloc, m))
        bvec = PETSc.Vec().createWithArray(np.zeros(off), comm=self.comm)
        zvec = PETSc.Vec().createWithArray(np.zeros(off), comm=self.comm)
        for i, jcol in enumerate(deep + off):
            rows, vals = Jop.getColumn(int(jcol))
            keep = rows < off
            bj = np.zeros(off)
            bj[rows[keep]] = vals[keep]
            bvec.setArray(-bj)
            kspu.solve(bvec, zvec)
            Z[:off, i] = zvec.getArray(readonly=True)
        Z[deep + off, np.arange(m)] = 1.0
        # H = Z^T J Z
        WZ = np.zeros((nloc, m))
        for i in range(m):
            self._t1.setArray(Z[:, i])
            Jop.mult(self._t1, self._t2)
            WZ[:, i] = self._t2.getArray(readonly=True)
        self._Z = Z
        self._lu = lu_factor(Z.T @ WZ)
        self._since = 0

    def _Q(self, v):
        """Q v = Z H^-1 Z^T v."""
        c = lu_solve(self._lu, self._Z.T @ v)
        return self._Z @ c

    def apply(self, pc, x, y):
        self.ncalls += 1
        self._since += 1
        xa = x.getArray(readonly=True)
        if self._Z is None or self._since > self.refresh:
            self._refresh()
        if self._Z is None:            # empty deep set: inner only
            self._t1.setArray(xa)
            self.inner.apply(self._t1, y)
            return
        p1 = self._Q(xa)
        self._t1.setArray(p1)
        self._Jop.mult(self._t1, self._t2)
        r = xa - self._t2.getArray(readonly=True)
        self._t1.setArray(r)
        self.inner.apply(self._t1, self._t2)
        w = self._t2.getArray(readonly=True).copy()
        self._t1.setArray(w)
        self._Jop.mult(self._t1, self._t2)
        aw = self._t2.getArray(readonly=True)
        p2 = self._Q(aw)
        y.setArray(p1 + w - p2)


def run(mesh, sp, tag, psi_floor, psi_floor_drift, psi_floor_operator=False,
        mode=None):
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name="u")
    x, y = SpatialCoordinate(mesh)
    r = sqrt(x * x + y * y)
    psi = Function(V, name="psi").interpolate(psiUFL(r))
    bc = DirichletBC(V, uexactUFL(r), "on_boundary")
    energy = 0.5 * inner(grad(u), grad(u)) * dx  # f = 0

    lvpp = LVPP(energy=energy, u=u, bounds=(psi, None), bcs=bc,
                alpha_rule="double_exponential",
                alpha_parameters={"alpha_max": 10.0},
                increment_norm="H1", verbose=False,
                psi_floor=psi_floor, psi_floor_drift=psi_floor_drift,
                psi_floor_operator=psi_floor_operator,
                form_compiler_parameters={"quadrature_degree": 6},
                solver_parameters=sp, name=tag)
    if mode == "tscale":
        snes = lvpp._solver.snes
        latent_field = len(lvpp._spaces)  # single-latent problems
        pc = snes.ksp.getPC()
        pc.setType("python")
        pc.setPythonContext(TScalePC(lvpp, latent_field))
    if mode == "deflate":
        snes = lvpp._solver.snes
        latent_field = len(lvpp._spaces)  # single-latent problems
        pc = snes.ksp.getPC()
        pc.setType("python")
        pc.setPythonContext(DeflationPC(lvpp, latent_field))
    t0 = time.time()
    lvpp.solve(tol=1e-4, max_proximal_iterations=500)
    wall = time.time() - t0
    err = errornorm(uexactUFL(r), lvpp.u_out[0])
    return (lvpp.proximal_iterations, sum(lvpp.newton_iterations), err, wall)


base = RectangleMesh(16, 16, 2.0, 2.0, originX=-2.0, originY=-2.0,
                     diagonal="crossed")
hier = MeshHierarchy(base, 2)
import sys as _sys
if _sys.argv[1] == "diag":
    configs = [("LU", SP_LU, 0.0, 0.0), ("mf eps0 0.1", SP_MF, 0.1, 0.0)]
    hier = MeshHierarchy(base, 1)
    SP_MF["snes_monitor"] = None
    SP_MF["ksp_monitor"] = None

# (label, solver params, psi_floor, psi_floor_drift); LU baseline keeps
# Jp = J (with ksp_type=preonly the PC matrix *is* the solve matrix, so an
# epsilon there would change the system -- the floor only makes sense with a
# Krylov inner solve, where Jp is a true preconditioner)
import sys
sel = sys.argv[1] if len(sys.argv) > 1 else "all"
configs = [("LU", SP_LU, 0.0, 0.0)]
if sel in ("all", "eps0"):
    for eps0 in (1e-3, 1e-2, 3e-2, 1e-1):
        configs.append((f"mf eps0 {eps0:g}", SP_MF, eps0, 0.0))
if sel in ("all", "drift"):
    for c in (0.03, 0.1, 0.3, 1.0):
        configs.append((f"mf drift c {c:g}", SP_MF, 1e-3, c))
if sel == "probe":
    # Column-equilibration test (right-preconditioned Jacobi): the Krylov
    # operator J*diag(J)^-1 is similar to the congruence diag(e^-phi/2) J
    # diag(e^-phi/2), whose latent block is (lambda/alpha)M -- the e^phi
    # degeneracy cancels exactly.  Prediction: tight rtol converges at
    # depth with NO floor.  Jp must stay unfloored (psi_floor=0) so the
    # Jacobi diagonal carries the true e^phi scale.
    SP_EQ = {
        "mat_type": "matfree", "pmat_type": "aij",
        "snes_linesearch_type": "l2", "snes_linesearch_maxlambda": 1.0,
        "snes_rtol": 1e-6, "snes_max_it": 100,
        "ksp_type": "gmres", "ksp_rtol": 1e-6, "ksp_max_it": 1000,
        "ksp_gmres_restart": 250,
        "pc_type": "jacobi", "ksp_pc_side": "right",
    }
    SP_FSR = dict(SP_MF, ksp_pc_side="right")
    configs = [("eq right tight", SP_EQ, 0.0, 0.0),
               ("fs right tight", SP_FSR, 0.0, 0.0),
               ("eq right rtol2", dict(SP_EQ, ksp_rtol=1e-2), 0.0, 0.0),
               ("tscale tight", {"mat_type": "matfree",
                                 "snes_linesearch_type": "l2",
                                 "snes_linesearch_maxlambda": 1.0,
                                 "snes_rtol": 1e-6, "snes_max_it": 100,
                                 "ksp_type": "gmres", "ksp_pc_side": "right",
                                 "ksp_rtol": 1e-6, "ksp_max_it": 1000,
                                 "ksp_gmres_restart": 250,
                                 "pc_type": "python"},
                0.0, 0.0, "tscale"),
               ("tscale rtol2", {"mat_type": "matfree",
                                 "snes_linesearch_type": "l2",
                                 "snes_linesearch_maxlambda": 1.0,
                                 "snes_rtol": 1e-6, "snes_max_it": 100,
                                 "ksp_type": "gmres", "ksp_pc_side": "right",
                                 "ksp_rtol": 1e-2, "ksp_max_it": 1000,
                                 "ksp_gmres_restart": 250,
                                 "pc_type": "python"},
                0.0, 0.0, "tscale")]
if sel == "defl":
    # deflation target: tight rtol 1e-6 at ALL levels (was impossible:
    # sqrt(kappa)*eps floor).  SP_MF provides the floored Pmat (1e-2);
    # the outer python PC projects and wraps the fieldsplit child.
    SP_DEF = dict(SP_MF, pc_type="python")
    configs = [("LU", SP_LU, 0.0, 0.0),
               ("mf defl tight", SP_DEF, 1e-2, 0.0, "deflate"),
               ("mf defl rtol2", dict(SP_DEF, ksp_rtol=1e-2), 1e-2, 0.0,
                "deflate")]

if sel == "thresh":
    # kappa*eps stagnation-floor prediction: convergence threshold should
    # track kappa(J) ~ 1e8/1e9/3e10 across levels and be restart-independent
    def _sp(rtol, restart):
        return dict(SP_MF, ksp_rtol=rtol, ksp_gmres_restart=restart)
    configs = [("floor 1e-4 r250", _sp(1e-4, 250), 1e-2, 0.0),
               ("floor 1e-3 r250", _sp(1e-3, 250), 1e-2, 0.0),
               ("floor 1e-4 r1000", _sp(1e-4, 1000), 1e-2, 0.0),
               ("floor 1e-6 r1000", _sp(1e-6, 1000), 1e-2, 0.0)]

print(f"{'level':>5} {'dofs':>7} {'config':>16} {'prox':>5} {'newton':>7} "
      f"{'err(u_h)':>10} {'wall(s)':>8}")
ref_err = {}
for lev, mesh in enumerate(hier):
    V = FunctionSpace(mesh, "CG", 1)
    for cfg in configs:
        label, sp, eps0, cdrift = cfg[:4]
        opflag = cfg[4] if len(cfg) > 4 else False
        try:
            tag = "".join(ch if ch.isalnum() else "_" for ch in label)
            mode = cfg[4] if len(cfg) > 4 else None
            prox, newton, err, wall = run(mesh, sp, f"{tag}{lev}",
                                          eps0, cdrift,
                                          psi_floor_operator=mode in ("opfloor",),
                                          mode=mode)
            print(f"{lev:>5} {V.dim():>7} {label:>16} {prox:>5} {newton:>7} "
                  f"{err:>10.3e} {wall:>8.2f}")
            if label == "LU":
                ref_err[lev] = err
        except Exception as e:
            reason = ""
            try:
                snes = lvpp._solver.snes
                ksp = snes.ksp
                reason = (f" [snes={snes.getConvergedReason()} "
                          f"ksp={ksp.getConvergedReason()}]")
            except Exception:
                pass
            print(f"{lev:>5} {V.dim():>7} {label:>16}  FAILED: "
                  f"{type(e).__name__}{reason}")
print("\n(LU reference errors:", {k: f"{v:.3e}" for k, v in ref_err.items()},
      ")")
