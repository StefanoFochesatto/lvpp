"""Sp-upgrade design ladder for the Schur fieldsplit LVPP solver.

Follow-up to schur_probe.py / Finding 6 in RESEARCH.md: the validated matfree
configuration (schur/upper/selfp + LU on the assembled approx-Schur Sp) has
LU-identical Newton counts but its outer-GMRES iteration count grows ~45 (L0)
-> 130-560 (L1) -> 400-926 (L2) because Sp only uses inv(diag(A00)) instead
of inv(A00).  This script upgrades Sp:

  1. mat_schur_complement_ainv_type (prefix fieldsplit_1_):
     diag (baseline) | lump | blockdiag | full.
     PETSc >= 3.20; 'full' assembles Sp = A11 - A10 A00^{-1} A01 exactly
     (MatSchurComplementComputeExplicitOperator, one inner-KSP solve per
     A00 column per PC setup); 'blockdiag' degenerates to 'diag' for the
     scalar CG1 u-block (block size 1) but is run as a control.
  2. PCLSC: NOT available in this PETSc (removed upstream; pc_type pclsc
     fails at setType).  Checked and documented at startup.
  3. Iterative S-solve: fieldsplit_1 gmres + gamg on the assembled Sp
     (rtol 1e-4 / 1e-6) instead of preonly+lu.
  4. AMG-on-K: fieldsplit_0 cg+gamg (rtol 1e-8) instead of preonly+lu(MUMPS),
     keeping the LUS Schur block; changes the OTHER exact block.

Instrumentation: a KSP monitor on the outer ksp (lvpp._solver.snes.ksp)
records the outer-GMRES iteration count of EVERY Newton solve; the growth
curve per (level, config) is reported as per-proximal-solve outer-its lists.

Usage:
  schur_sp_upgrade.py                    # level-0 ladder (all configs)
  schur_sp_upgrade.py ladder             # same
  schur_sp_upgrade.py promote l1 l2 ...  # baseline + named configs at L1-L2
  schur_sp_upgrade.py single <label> <levels>   # one config at comma levels
  schur_sp_upgrade.py view <label>       # one prox it + PCView (setup autopsy)
  schur_sp_upgrade.py features           # version/feature probe only

LU references (prox/newton, err): L0 8/21 1.553e-2, L1 11/23 3.599e-3,
L2 8/19 8.375e-4.  Baseline outer-GMRES its/solve: ~45 (L0), 130-560 (L1),
400-926 (L2).
"""
import statistics
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


def schur_sp(fact="upper", pre="selfp", inner="preonly", inner_pc="lu",
             inner_rtol=1e-6, ainv=None, field0=None, sp_solver="mumps"):
    """Schur fieldsplit: field 0 = u (K), field 1 = phi (latent).

    Baseline = schur_probe.py's winner 'schur upper selfp LUS'.
    """
    sp = {
        "mat_type": "matfree",
        "pmat_type": "aij",
        # fieldsplit must extract its blocks from the assembled Pmat
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
        "pc_fieldsplit_schur_fact_type": fact,
        "pc_fieldsplit_schur_precondition": pre,
        # exact K: block solve AND the K^{-1} inside the Schur matvec
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "lu",
        "fieldsplit_0_pc_factor_mat_solver_type": "mumps",
        # S = D - B^T K^{-1} B is indefinite on the degenerate tail: gmres
        # (or preonly+lu) on the assembled Sp, never cg
        "fieldsplit_1_ksp_type": inner,
        "fieldsplit_1_pc_type": inner_pc,
        "fieldsplit_1_ksp_rtol": inner_rtol,
    }
    if sp_solver is not None:
        sp["fieldsplit_1_pc_factor_mat_solver_type"] = sp_solver
    if inner == "gmres":
        sp["fieldsplit_1_ksp_gmres_restart"] = 250
        sp["fieldsplit_1_ksp_max_it"] = 250
    if ainv is not None:
        # the MATSCHURCOMPLEMENT object carries the second field's prefix
        sp["fieldsplit_1_mat_schur_complement_ainv_type"] = ainv
    if field0:
        for k, v in field0.items():
            sp["fieldsplit_0_" + k] = v
    return sp


PSI_FLOOR = 1e-2

LADDER = [
    ("base_lus", schur_sp(), PSI_FLOOR),
    ("ainv_blkdiag", schur_sp(ainv="blockdiag"), PSI_FLOOR),
    ("ainv_full", schur_sp(ainv="full", sp_solver=None), PSI_FLOOR),
    ("ainv_full_gamg", schur_sp(ainv="full", sp_solver=None, inner="gmres",
                                inner_pc="gamg"), PSI_FLOOR),
    ("diag_gamg_r4", schur_sp(inner="gmres", inner_pc="gamg", inner_rtol=1e-4),
     PSI_FLOOR),
    ("diag_gamg_r6", schur_sp(inner="gmres", inner_pc="gamg", inner_rtol=1e-6),
     PSI_FLOOR),
    # floor-magnitude probes: is the outer-GMRES growth set by the eps floor
    # on the degenerate latent block rather than the Sp approximation?
    ("full_floor1e1", schur_sp(ainv="full", sp_solver=None), 1e-1),
    ("full_floor1e3", schur_sp(ainv="full", sp_solver=None), 1e-3),
    ("amg_on_k_lus", schur_sp(field0={"ksp_type": "cg", "ksp_rtol": 1e-8,
                                      "ksp_max_it": 500, "pc_type": "gamg"}),
     PSI_FLOOR),
]
BY_LABEL = {label: (label, sp, eps0) for label, sp, eps0 in LADDER}


def instrument(lvpp):
    """Attach monitors recording outer-GMRES its of every Newton solve.

    state['groups'][i] = final outer iteration count of the i-th outer KSP
    solve (one per Newton iteration, in order).  state['solves'] counts
    SNES solves (prox steps + alpha-halving retries).
    """
    state = {"groups": [], "solves": 0}

    def ksp_mon(ksp, its, rnorm):
        if its == 1:
            state["groups"].append(1)
        elif its > 1:
            if state["groups"]:
                state["groups"][-1] = its
            else:
                state["groups"].append(its)

    def snes_mon(snes, its, norm):
        if its == 1:
            state["solves"] += 1

    lvpp._solver.snes.ksp.setMonitor(ksp_mon)
    lvpp._solver.snes.setMonitor(snes_mon)
    return state


def per_prox_its(state, newton_its):
    """Slice the recorded outer-KSP counts per proximal solve."""
    groups = state["groups"]
    out, i = [], 0
    for n in newton_its:
        out.append(groups[i:i + n])
        i += n
    return out, groups[i:]  # leftovers = alpha-halving retry solves


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
    state = instrument(lvpp)
    t0 = time.time()
    lvpp.solve(tol=1e-4, max_proximal_iterations=max_prox)
    wall = time.time() - t0
    per_prox, retries = per_prox_its(state, lvpp.newton_iterations)
    err = errornorm(uexactUFL(r), lvpp.u_out[0])
    return (lvpp, lvpp.proximal_iterations, sum(lvpp.newton_iterations),
            err, wall, per_prox, retries)


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


def flat(per_prox):
    return [its for chunk in per_prox for its in chunk]


def its_stats(per_prox, retries):
    rec = flat(per_prox) + flat(retries)
    if not rec:
        return "n/a", "n/a", "n/a"
    mx = max(rec)
    med = statistics.median(rec)
    curve = ",".join(str(max(c)) if c else "0" for c in per_prox)
    if retries:
        curve += f" (+{len(retries)} retry)"
    return mx, med, curve

    return f"{type(exc).__name__}: {msg[:900]}{reason}"
def print_header(dofs):
    print(f"{'level':>5} {'dofs':>7} {'config':>16} {'prox':>5} {'newton':>7} "
          f"{'err(u_h)':>10} {'wall(s)':>8} {'oits_max':>9} {'oits_med':>9} "
          f"outer-its curve (max per prox solve)")


def print_row(lev, dofs, label, v):
    if v is None:
        print(f"{lev:>5} {dofs:>7} {label:>16}  FAILED")
        return
    prox, newton, err, wall, per_prox, retries = v
    mx, med, curve = its_stats(per_prox, retries)
    print(f"{lev:>5} {dofs:>7} {label:>16} {prox:>5} {newton:>7} "
          f"{err:>10.3e} {wall:>8.2f} {mx:>9} {med:>9}  {curve}")


def probe_view(mesh, sp, tag, psi_floor):
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


def features():
    import petsc4py
    from petsc4py import PETSc
    print(f"petsc4py/PETSc version: {petsc4py.__version__}")
    for pc_t in ("pclsc", "gamg", "hypre", "lu"):
        try:
            pc = PETSc.PC().create()
            pc.setType(pc_t)
            pc.destroy()
            print(f"  pc_type {pc_t}: available")
        except Exception as e:
            print(f"  pc_type {pc_t}: NOT available ({str(e).splitlines()[0]})")
    try:
        opts = PETSc.Options()
        opts.setValue("-mat_schur_complement_ainv_type", "full")
        got = opts.getString("-mat_schur_complement_ainv_type")
        print(f"  mat_schur_complement_ainv_type option accepted: {got!r} "
              "(values diag|lump|blockdiag|full)")
        opts.delValue("-mat_schur_complement_ainv_type")
    except Exception as e:
        print(f"  ainv_type option probe failed: {e}")


def run_levels(configs, levels):
    base = RectangleMesh(16, 16, 2.0, 2.0, originX=-2.0, originY=-2.0,
                         diagonal="crossed")
    hier = MeshHierarchy(base, 2)
    results = {}
    for lev in levels:
        mesh = hier[lev]
        V = FunctionSpace(mesh, "CG", 1)
        print_header(V.dim())
        for label, sp, eps0 in configs:
            tag = "".join(ch if ch.isalnum() else "_" for ch in
                          label) + str(lev)
            try:
                (_, prox, newton, err, wall, per_prox,
                 retries) = run(mesh, sp, tag, eps0)
                print_row(lev, V.dim(), label,
                          (prox, newton, err, wall, per_prox, retries))
                results[(lev, label)] = (prox, newton, err, wall, per_prox,
                                         retries)
            except Exception as e:
                print_row(lev, V.dim(), label, None)
                print(f"    failure: {failure_reason(e)}")
                results[(lev, label)] = None
    return results


def summarize(results):
    print("\nSummary (LU refs 8/21 1.553e-2 | 11/23 3.599e-3 | 8/19 "
          "8.375e-4; baseline outer-its/solve ~45 L0, 130-560 L1, 400-926 "
          "L2):")
    for (lev, label), v in sorted(results.items()):
        if v is None:
            print(f"  level {lev}  {label:>16}  FAILED")
        else:
            prox, newton, err, wall, per_prox, retries = v
            mx, med, curve = its_stats(per_prox, retries)
            print(f"  level {lev}  {label:>16}  prox={prox:<3d} "
                  f"newton={newton:<4d} err={err:.3e} wall={wall:6.1f}s "
                  f"outer-its max={mx} med={med}")
            print(f"      per-prox outer-its: {curve}")


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = args[0] if args else "ladder"
    features()
    if mode == "features":
        sys.exit(0)
    if mode == "view":
        label = args[1] if len(args) > 1 else "ainv_full"
        _, sp, eps0 = BY_LABEL[label]
        base = RectangleMesh(16, 16, 2.0, 2.0, originX=-2.0, originY=-2.0,
                             diagonal="crossed")
        probe_view(base, sp, "view_" + label, eps0)
        sys.exit(0)
    if mode in ("ladder", "auto"):
        results = run_levels(LADDER, [0])
    elif mode == "promote":
        labels = ["base_lus"] + args[1:]
        configs = [BY_LABEL[l] for l in labels]
        results = run_levels(configs, [1, 2])
    elif mode == "single":
        configs = [BY_LABEL[l] for l in args[1].split(",")]
        levels = [int(l) for l in args[2].split(",")] if len(args) > 2 \
            else [0, 1, 2]
        results = run_levels(configs, levels)
    else:
        configs = [BY_LABEL[mode]]
        levels = [int(l) for l in args[1].split(",")] if len(args) > 1 \
            else [0, 1, 2]
        results = run_levels(configs, levels)
    summarize(results)
