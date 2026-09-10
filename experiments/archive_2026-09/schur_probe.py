"""Schur-complement fieldsplit probe for the matrix-free LVPP saddle system.

Problem setup copied verbatim from matfree_fieldsplit.py (sphere obstacle
problem, mixed (u, phi) with phi the latent variable, matfree Amat + floored
aij Pmat).

Design ladder at level 0 over PCFIELDSPLIT schur variants:

  pc_fieldsplit_use_amat: False    # MANDATORY: extract blocks from the
                                   # assembled Pmat (matfree Amat has no
                                   # submatrix extraction)
  pc_fieldsplit_type: schur        # captures S = D - B^T K^{-1} B, which the
                                   # additive PC ignores
  pc_fieldsplit_schur_fact_type:   # upper / lower / diag
  pc_fieldsplit_schur_precondition: a11 (default) / selfp / user
  fieldsplit_0: preonly + lu       # exact K (small at these sizes); also used
                                   # inside the Schur matvec for K^{-1}
  fieldsplit_1: gmres (NOT cg: S = D - B^T K^{-1} B is indefinite/negative on
                        the degenerate latent tail) + jacobi, rtol variants

psi_floor = 1e-2 (LVPP kwarg) floors the Pmat only.

Usage:
  schur_probe.py            # LU baseline + level-0 ladder, then best config
                            # at levels 1-2 (growth pattern)
  schur_probe.py ladder     # level-0 ladder only
  schur_probe.py view       # one proximal iteration of the headliner with a
                            # single explicit PC view (setup-failure autopsy)
  schur_probe.py <label>    # single ladder config at levels 0-2

LU references (LU prox/newton, err): level0 8/21 1.553e-2, level1 11/23
3.599e-3, level2 8/19 8.375e-4.
"""
import sys
import time

import numpy as np
from firedrake import (DirichletBC, Function, FunctionSpace, MeshHierarchy,
                       RectangleMesh, SpatialCoordinate, conditional, dx,
                       errornorm, grad, inner, le, ln, sqrt)
from lvpp import LVPP

r0 = 0.9
AFREE = 0.697965148223374
A, B = 0.680259411891719, 0.471519893402112

_LAST = None  # most recent LVPP object (reachable when run() raises)


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


def schur_sp(fact="upper", pre=None, inner="gmres", inner_pc="jacobi",
             inner_rtol=1e-4, snes_max_it=100):
    """Schur fieldsplit: field 0 = u (K), field 1 = phi (latent)."""
    sp = {
        "mat_type": "matfree",
        "pmat_type": "aij",
        # fieldsplit must extract its blocks from the assembled Pmat
        "pc_fieldsplit_use_amat": False,
        "snes_linesearch_type": "l2",
        "snes_linesearch_maxlambda": 1.0,
        "snes_rtol": 1e-6,
        "snes_max_it": snes_max_it,
        "ksp_type": "gmres",
        "ksp_rtol": 1e-6,
        "ksp_max_it": 1000,
        "ksp_gmres_restart": 250,
        "ksp_converged_reason": None,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": fact,
        # exact K: block solve AND the K^{-1} inside the Schur matvec
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "lu",
        "fieldsplit_0_pc_factor_mat_solver_type": "mumps",
        # S = D - B^T K^{-1} B is indefinite on the degenerate tail: gmres,
        # never cg
        "fieldsplit_1_ksp_type": inner,
        "fieldsplit_1_pc_type": inner_pc,
        "fieldsplit_1_ksp_rtol": inner_rtol,
        "fieldsplit_1_ksp_gmres_restart": 250,
        "fieldsplit_1_pc_factor_mat_solver_type": "mumps",
    }
    if pre is not None:
        sp["pc_fieldsplit_schur_precondition"] = pre
    return sp


# (label, sp, psi_floor); psi_floor 1e-2 floors the Pmat latent diagonal
PSI_FLOOR = 1e-2
LADDER = [
    ("LU", SP_LU, 0.0),
    ("schur upper a11 r4", schur_sp(), PSI_FLOOR),
    ("schur upper selfp r4", schur_sp(pre="selfp"), PSI_FLOOR),
    ("schur upper user r4", schur_sp(pre="user"), PSI_FLOOR),
    ("schur lower selfp r4", schur_sp(fact="lower", pre="selfp"), PSI_FLOOR),
    ("schur diag selfp r4", schur_sp(fact="diag", pre="selfp"), PSI_FLOOR),
    ("schur upper selfp r2", schur_sp(pre="selfp", inner_rtol=1e-2),
     PSI_FLOOR),
    ("schur upper selfp r6", schur_sp(pre="selfp", inner_rtol=1e-6),
     PSI_FLOOR),
    # near-exact-Schur diagnostic: LU on the assembled approx-Schur Pmat
    ("schur upper selfp LUS", schur_sp(pre="selfp", inner="preonly",
                                       inner_pc="lu"), PSI_FLOOR),
]


def run(mesh, sp, tag, psi_floor, max_prox=500):
    global _LAST
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name="u")
    x, y = SpatialCoordinate(mesh)
    r = sqrt(x * x + y * y)
    energy = 0.5 * inner(grad(u), grad(u)) * dx  # f = 0
    psi = Function(V, name="psi").interpolate(psiUFL(r))
    bc = DirichletBC(V, uexactUFL(r), "on_boundary")
    lvpp = LVPP(energy=energy, u=u, bounds=(psi, None), bcs=bc,
                alpha_rule="double_exponential",
                alpha_parameters={"alpha_max": 10.0},
                increment_norm="H1", verbose=False,
                psi_floor=psi_floor,
                form_compiler_parameters={"quadrature_degree": 6},
                solver_parameters=sp, name=tag)
    _LAST = lvpp
    t0 = time.time()
    lvpp.solve(tol=1e-4, max_proximal_iterations=max_prox)
    wall = time.time() - t0
    err = errornorm(uexactUFL(r), lvpp.u_out[0])
    return (lvpp, lvpp.proximal_iterations, sum(lvpp.newton_iterations),
            err, wall)


def failure_reason(exc, lvpp=None):
    if lvpp is None:
        lvpp = _LAST
    reason = ""
    if lvpp is not None:
        try:
            snes = lvpp._solver.snes
            ksp = snes.ksp
            reason = (f" [snes={snes.getConvergedReason()} "
                      f"ksp={ksp.getConvergedReason()}]")
        except Exception:
            pass
    msg = str(exc).strip().replace("\n", " ")
    return f"{type(exc).__name__}: {msg[:220]}{reason}"


def probe_view(mesh, sp, tag, psi_floor):
    """One proximal iteration, then a single explicit PC view."""
    sp = dict(sp, snes_max_it=1)
    print(f"--- PC view probe: {tag} ---")
    try:
        run(mesh, sp, tag, psi_floor, max_prox=1)
    except Exception as e:
        print(f"probe run exception: {failure_reason(e)}")
    if _LAST is not None:
        try:
            pc = _LAST._solver.snes.ksp.getPC()
            print("--- PCView (fieldsplit schur) ---")
            pc.view()
        except Exception as e:
            print(f"PC view failed: {failure_reason(e)}")


base = RectangleMesh(16, 16, 2.0, 2.0, originX=-2.0, originY=-2.0,
                     diagonal="crossed")
hier = MeshHierarchy(base, 2)

sel = sys.argv[1] if len(sys.argv) > 1 else "auto"

if sel == "view":
    probe_view(hier[0], schur_sp(pre="selfp"), "viewselfp", PSI_FLOOR)
    sys.exit(0)

if sel in ("auto", "ladder"):
    configs = LADDER
else:
    configs = [c for c in LADDER if c[0] == sel]

print(f"{'level':>5} {'dofs':>7} {'config':>22} {'prox':>5} {'newton':>7} "
      f"{'err(u_h)':>10} {'wall(s)':>8}")
results = {}   # (level, label) -> (prox, newton, err, wall) or None
levels = [0, 1, 2] if sel not in ("auto", "ladder") else [0]
# ladder always runs at level 0; best config goes to 1-2 below
for lev, mesh in enumerate(hier):
    if lev not in levels:
        continue
    V = FunctionSpace(mesh, "CG", 1)
    for label, sp, eps0 in configs:
        tag = "".join(ch if ch.isalnum() else "_" for ch in label) + str(lev)
        try:
            lvpp, prox, newton, err, wall = run(mesh, sp, tag, eps0)
            print(f"{lev:>5} {V.dim():>7} {label:>22} {prox:>5} {newton:>7} "
                  f"{err:>10.3e} {wall:>8.2f}")
            results[(lev, label)] = (prox, newton, err, wall)
        except Exception as e:
            print(f"{lev:>5} {V.dim():>7} {label:>22}  FAILED: "
                  f"{failure_reason(e)}")
            results[(lev, label)] = None

if sel in ("auto", "ladder"):
    # pick the best converged schur config at level 0: fewest newton its,
    # wall-clock tiebreak
    ok = [(k[1], v) for k, v in results.items()
          if k[0] == 0 and k[1] != "LU" and v is not None]
    if not ok:
        print("\n(no schur config converged at level 0 -- PC view probe)")
        probe_view(hier[0], schur_sp(pre="selfp"), "viewselfp", PSI_FLOOR)
    elif sel == "auto":
        ok.sort(key=lambda kv: (kv[1][1], kv[1][3]))
        best = ok[0][0]
        print(f"\nbest level-0 schur config: {best}; running levels 1-2")
        for lev, mesh in enumerate(hier):
            if lev == 0:
                continue
            V = FunctionSpace(mesh, "CG", 1)
            for label, sp, eps0 in ([("LU", SP_LU, 0.0)]
                                    + [c for c in LADDER if c[0] == best]):
                tag = ("".join(ch if ch.isalnum() else "_" for ch in label)
                       + str(lev))
                try:
                    lvpp, prox, newton, err, wall = run(mesh, sp, tag, eps0)
                    print(f"{lev:>5} {V.dim():>7} {label:>22} {prox:>5} "
                          f"{newton:>7} {err:>10.3e} {wall:>8.2f}")
                    results[(lev, label)] = (prox, newton, err, wall)
                except Exception as e:
                    print(f"{lev:>5} {V.dim():>7} {label:>22}  FAILED: "
                          f"{failure_reason(e)}")
                    results[(lev, label)] = None

print("\nSummary (LU refs: 8/21 1.553e-2, 11/23 3.599e-3, 8/19 8.375e-4):")
for (lev, label), v in sorted(results.items()):
    if v is None:
        print(f"  level {lev}  {label:>22}  FAILED")
    else:
        prox, newton, err, wall = v
        print(f"  level {lev}  {label:>22}  prox={prox:<4d} "
              f"newton={newton:<4d} err={err:.3e} wall={wall:.1f}s")
