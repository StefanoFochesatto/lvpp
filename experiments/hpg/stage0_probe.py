"""Stage-0 pre-flight probe: (a) Legendre-modal DG variant support (FIAT),
(b) viamr SBR on quadrilateral meshes, (c) LVPP with a DG_{p-2} latent space.

Env guard REQUIRED:
PETSC_DIR=/home/stefano/firedrake/petsc PETSC_ARCH=arch-firedrake-default \
OMP_NUM_THREADS=1 venv-firedrake/bin/python stage0_probe.py
"""
import numpy as np
from firedrake import (Constant, DirichletBC, Function, FunctionSpace,
                       RectangleMesh, SpatialCoordinate, TestFunction,
                       TrialFunction, assemble, dx, errornorm, exp, grad,
                       inner, sqrt)
from firedrake import VectorFunctionSpace  # noqa: F401
from firedrake.petsc import PETSc
from lvpp import LVPP

print("=== (a) DG variant probe ===", flush=True)
m = RectangleMesh(4, 4, 2.0, 2.0, quadrilateral=True)
for var in ("equispaced", "spectral", "moment"):
    try:
        W = FunctionSpace(m, "DQ", 2, variant=var)
        print(f"  DQ degree2 variant={var}: OK dim={W.dim()}", flush=True)
    except Exception as e:
        print(f"  DQ degree2 variant={var}: FAIL {type(e).__name__}: "
              f"{str(e)[:120]}", flush=True)
# try FIAT Gauss-Legendre element registration through Firedrake
try:
    import FIAT  # noqa: F401
    from FIAT import reference_element as ref_el
    print("  FIAT imported", flush=True)
except Exception as e:
    print(f"  FIAT import failed: {e}", flush=True)
try:
    from firedrake import TensorProductElement, FiniteElement, interval
    te = FiniteElement("DQ", "interval", 2, variant="spectral")
    print("  interval DQ spectral:", te, flush=True)
except Exception as e:
    print(f"  interval DQ spectral FAIL: {str(e)[:160]}", flush=True)

print("=== (b) mass matrix structure of DG latent ===", flush=True)
W = FunctionSpace(m, "DQ", 1)
Mt = assemble(inner(TrialFunction(W), TestFunction(W)) * dx, mat_type="aij")
M = Mt.M.handle
indptr, indices, data = M.getValuesCSR()
d = np.abs(data)
rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
maxoff = float(d[indices != rows].max()) if (indices != rows).any() else 0.0
mindg = float(d[indices == rows].min()) if (indices == rows).any() else 0.0
print(f"  DQ1 mass nnz={len(data)}, dim={W.dim()}, "
      f"max|off-diag|={maxoff:.3e}, min|diag|={mindg:.3e}",
      flush=True)

print("=== (c) LVPP with DG latent on quads ===", flush=True)
mesh = RectangleMesh(16, 16, 2.0, 2.0, quadrilateral=True)
V = FunctionSpace(mesh, "CG", 2)
Wdg = FunctionSpace(mesh, "DG", 0)
u = Function(V, name="u")
x, y = SpatialCoordinate(mesh)
r = sqrt(x * x + y * y)
r0, AFREE = 0.9, 0.697965148223374
A_, B_ = 0.680259411891719, 0.471519893402112
psi0 = np.sqrt(1.0 - r0 * r0)
dpsi0 = -r0 / psi0
from firedrake import conditional, le, ln  # noqa: E402


def psiUFL(rr):
    return conditional(le(rr, r0), sqrt(1.0 - rr * rr),
                       psi0 + dpsi0 * (rr - r0))


def uexactUFL(rr):
    return conditional(le(rr, AFREE), psiUFL(rr), -A_ * ln(rr) + B_)


lb = Function(V, name="psi_lb").interpolate(psiUFL(r))
bc = DirichletBC(V, uexactUFL(r), "on_boundary")
energy = 0.5 * inner(grad(u), grad(u)) * dx
lv = LVPP(energy=energy, u=u, bounds=(lb, None), bcs=bc,
          psi_spaces=[Wdg],
          alpha_rule="double_exponential",
          alpha_parameters={"alpha_max": 10.0},
          increment_norm="H1", verbose=False,
          form_compiler_parameters={"quadrature_degree": 6},
          solver_parameters={"snes_type": "newtonls",
                             "snes_linesearch_type": "l2",
                             "snes_rtol": 1e-6,
                             "ksp_type": "preonly",
                             "pc_type": "lu",
                             "pc_factor_mat_solver_type": "mumps",
                             "snes_max_it": 100},
          name="probe_")
import time
t0 = time.time()
lv.solve(tol=1e-4, max_proximal_iterations=100)
wall = time.time() - t0
err = float(errornorm(uexactUFL(r), lv.u_out[0]))
print(f"  OK prox={lv.proximal_iterations} "
      f"newton={sum(lv.newton_iterations)} err={err:.4e} wall={wall:.1f}s "
      f"dofs={V.dim() + Wdg.dim()}", flush=True)
ut = lv.u_tilde[0]
print(f"  u_tilde space={ut.function_space().ufl_element_type()} "
      f"min-ub margin={float(ut.dat.data.min()):.3e}", flush=True)
print("PROBE DONE", flush=True)
