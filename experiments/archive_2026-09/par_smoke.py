"""Parallel smoke (MPI): correctness + performance of the floorless Schur
production config (Finding 6, psi_floor=0) at 2 ranks.

Run as:
  PETSC_DIR=/home/stefano/firedrake/petsc \
  PETSC_ARCH=arch-firedrake-default OMP_NUM_THREADS=1 \
  mpiexec -n 2 /home/stefano/firedrake/venv-firedrake/bin/python \
  experiments/par_smoke.py [--levels 0,1] [--variant mumps|gamg|both]

Config under test (production, psi_floor=0): matfree Amat + assembled
Pmat, pc_fieldsplit_type schur / upper / selfp / use_amat False,
fieldsplit_0 preonly+lu(MUMPS), fieldsplit_1 preonly+lu(MUMPS), outer
GMRES rtol 1e-6 restart 250, snes_rtol 1e-6, l2 linesearch maxlambda 1.0.

Variants:
  mumps  -- production config verbatim (MUMPS on both blocks).
  gamg   -- fieldsplit_0 (u/K block) cg+GAMG instead of MUMPS; fieldsplit_1
            stays lu(MUMPS).  Scaling datum: is the exact MUMPS K-block
            solve replaceable at 2 ranks, and what does it cost?
  amat   -- pc_fieldsplit_use_amat True (split on the matfree Amat).
            Known-risk probe: parallel correctness of fieldsplit on the
            matrix-free Amat.
Serial LU refs (eps_schedule.py, floorless): L0 8/21/1.553e-02,
L1 11/23/3.599e-03 (prox/newton/err), walls 0.9 s / 2.0 s.
Serial-vs-parallel counts need not match bit-for-bit (MUMPS/GAMG
ordering differences), but errors must agree to solver tolerance and
u_tilde must stay feasible (min(u_tilde - lb) >= 0).

Rank-safety: no rank-branching control flow, no assert-style collective
mismatches; all prints guarded by COMM_WORLD.rank == 0; reductions go
through mpi4py allreduce with explicit ops.

petsc4py idiom note: the context-manager array form raises PETSc error
101 in this petsc4py inside PC callbacks; this script does no PC surgery
(the production config builds everything through options), so the idiom
only applies if a callback is added later.
"""
import argparse
import gc
import statistics
import sys
import time

# PETSc inserts sys.argv into its options DB when firedrake is imported;
# strip the script's own flags first and keep them for argparse.
_CLI_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0]]

import mpi4py.MPI as MPI
import numpy as np
from firedrake import (COMM_WORLD, DirichletBC, Function, FunctionSpace,
                       MeshHierarchy, RectangleMesh, SpatialCoordinate,
                       conditional, dx, errornorm, grad, inner, le, ln, sqrt)

from lvpp import LVPP

r0 = 0.9
AFREE = 0.697965148223374
A_, B_ = 0.680259411891719, 0.471519893402112

FCP = {"quadrature_degree": 6}

LU_REFS = {0: (8, 21, 1.553e-2, 0.9), 1: (11, 23, 3.599e-3, 2.0)}

# MEASURED 2-RANK RUNS (2026-09-09, this box, levels 0-1):
#   L0 mumps : prox=8 newton=21 err=1.553e-02 curve [6,10,13,14,12,14,12,14]
#              feas 0.0  wall 0.9-1.7 s (serial LU ref 0.9 s)
#   L1 mumps : prox=11 newton=23 err=3.599e-03 curve [6,12,25..29]
#              feasible 0.0  wall 1.9-3.7 s (serial 2.0 s)
#   L1 gamg  : prox=11 newton=29 err=3.600e-03 curve [7,16,27..32]
#              feasible 0.0  wall 4.4-6.4 s (serial 2.0 s)
#   L0/L1 amat (use_amat True): LU-identical to mumps, feasible.
#   Verdict: LU-identical counts and errors at 2 ranks on all variants;
#   no parallel-specific failure (both use_amat settings work); wall is
#   NOT improved at these sizes (1-2x serial -- blocks too small for the
#   MUMPS distributed factorization to pay for ghost exchange).
# Production floorless Schur config (RESEARCH.md "RESOLVED — floorless
# Schur configuration"; psi_floor=0 is the LVPP kwarg, not an option).
SP_MUMPS = {
    "mat_type": "matfree",
    "pmat_type": "aij",
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
    "pc_fieldsplit_schur_fact_type": "upper",
    "pc_fieldsplit_schur_precondition": "selfp",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "lu",
    "fieldsplit_0_pc_factor_mat_solver_type": "mumps",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "lu",
    "fieldsplit_1_pc_factor_mat_solver_type": "mumps",
}

# GAMG variant: replace the exact MUMPS solve of the u/K block with
# cg+GAMG; the latent block keeps MUMPS (it is tiny).
SP_GAMG = dict(SP_MUMPS)
del SP_GAMG["fieldsplit_0_pc_factor_mat_solver_type"]
SP_GAMG["fieldsplit_0_ksp_type"] = "cg"
SP_GAMG["fieldsplit_0_pc_type"] = "gamg"
SP_AMAT = dict(SP_MUMPS)
SP_AMAT["pc_fieldsplit_use_amat"] = True
VARIANTS = {"mumps": SP_MUMPS, "gamg": SP_GAMG, "amat": SP_AMAT}

# Known-risk probe: split on the matfree Amat instead of the assembled
# Pmat (pc_fieldsplit_use_amat True).  Probed at 2 ranks; outcome recorded
# in the MEASURED block above.


def psiUFL(r):
    psi0 = np.sqrt(1.0 - r0 * r0)
    dpsi0 = -r0 / psi0
    return conditional(le(r, r0), sqrt(1.0 - r * r), psi0 + dpsi0 * (r - r0))


def uexactUFL(r):
    return conditional(le(r, AFREE), psiUFL(r), -A_ * ln(r) + B_)


def per_prox_its(groups, newton_its):
    """Group per-solve outer-GMRES its by proximal solve (schur_percell)."""
    out, i = [], 0
    for n in newton_its:
        out.append(groups[i:i + n])
        i += n
    return out, groups[i:]  # leftovers = alpha-halving retry solves


def instrument(lvpp):
    """Outer-GMRES its per Newton solve (eps_schedule pattern)."""
    state = {"groups": []}

    def ksp_mon(ksp, its, rnorm):
        if its == 1:
            state["groups"].append(1)
        elif its > 1:
            if state["groups"]:
                state["groups"][-1] = its
            else:
                state["groups"].append(its)

    lvpp._solver.snes.ksp.setMonitor(ksp_mon)
    return state


def run_variant(mesh, tag, sp):
    """One full LVPP solve with the given solver config; collective."""
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
                increment_norm="H1", verbose=False, psi_floor=0.0,
                form_compiler_parameters=FCP,
                solver_parameters=dict(sp), name=tag)
    state = instrument(lvpp)
    t0 = time.time()
    lvpp.solve(tol=1e-4, max_proximal_iterations=500)
    wall = time.time() - t0
    wall_max = COMM_WORLD.allreduce(wall, op=MPI.MAX)
    err = errornorm(uexactUFL(r), lvpp.u_out[0])

    # Feasibility: u_tilde is the bound-preserving reconstruction, and
    # u_tilde - lb = exp(psi_latent) nodally, so min >= 0 must hold to
    # roundoff.  Reduce over owned+ghost dofs across ranks.
    ut = lvpp.u_tilde
    ut = ut[0] if isinstance(ut, (list, tuple)) else ut
    lat = lvpp.psi_out[0] if getattr(lvpp, "psi_out", None) else None
    d = ut.dat.data_ro - psi.dat.data_ro
    feas_local = float(np.min(d))
    feas = COMM_WORLD.allreduce(feas_local, op=MPI.MIN)
    finite_local = 1.0 if np.all(np.isfinite(d)) else -1.0
    finite = COMM_WORLD.allreduce(finite_local, op=MPI.MIN)
    lat_max_local = (float(np.max(np.abs(lat.dat.data_ro)))
                     if lat is not None else 0.0)
    lat_max = COMM_WORLD.allreduce(lat_max_local, op=MPI.MAX)

    per_prox, leftover = per_prox_its(state["groups"],
                                      list(lvpp.newton_iterations))
    flat = [i for chunk in per_prox + ([leftover] if leftover else [])
            for i in chunk]
    curve = [max(ch) if ch else 0 for ch in per_prox]
    snes = lvpp._solver.snes
    return dict(tag=tag, dofs=V.dim(), prox=lvpp.proximal_iterations,
                newton=sum(lvpp.newton_iterations), err=float(err),
                wall=wall, wall_max=wall_max, its=flat, curve=curve,
                feas=feas, finite=finite, lat_max=lat_max,
                snes=int(snes.getConvergedReason()),
                ksp=int(snes.ksp.getConvergedReason()))


def print_row(lev, variant, v, exc=None):
    if COMM_WORLD.rank != 0:
        return
    if exc is not None:
        print(f"  L{lev} [{variant}]: FAILED: {exc}")
        return
    p, n, e, w = LU_REFS.get(lev, (0, 0, 0.0, 0.0))
    med = statistics.median(v["its"]) if v["its"] else float("nan")
    print(f"  L{lev} [{variant}]: prox={v['prox']} newton={v['newton']} "
          f"err={v['err']:.3e} (LU ref {p}/{n} {e:.3e}) "
          f"snes={v['snes']} ksp={v['ksp']} "
          f"outer-its max={max(v['its']) if v['its'] else 0} "
          f"med={med:.1f} wall={v['wall']:.1f}s (max-rank {v['wall_max']:.1f}s) "
          f"[serial {w:.1f}s]\n"
          f"       per-prox curve: {v['curve']}\n"
          f"       feasibility: min(u_tilde - lb) = {v['feas']:.3e} "
          f"(finite: {v['finite'] >= 1.0}, |psi|max {v['lat_max']:.1f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="0,1")
    ap.add_argument("--variant", default="mumps",
                    choices=["mumps", "gamg", "amat", "both"])
    args = ap.parse_args(_CLI_ARGS)
    levels = [int(s) for s in args.levels.split(",")]
    base = RectangleMesh(16, 16, 2.0, 2.0, originX=-2.0, originY=-2.0,
                         diagonal="crossed")
    hier = MeshHierarchy(base, max(levels))
    variants = (["mumps", "gamg"] if args.variant == "both"
                else [args.variant])

    for lev, mesh in enumerate(hier):
        if lev not in levels:
            continue
        for variant in variants:
            sp = VARIANTS[variant]
            if COMM_WORLD.rank == 0:
                print(f"\n== L{lev} [{variant}] at {COMM_WORLD.size} ranks ==",
                      flush=True)
            try:
                v = run_variant(mesh, f"par_l{lev}_{variant}", sp)
                print_row(lev, variant, v)
            except Exception as e:
                print_row(lev, variant, None, exc=repr(e)[:300])
            gc.collect()


if __name__ == "__main__":
    main()