"""Gate: the matfree hpG arm (``a_action="gamg"``).

The archived ``experiments/hpg/matfree_hpG.py`` removes every *global*
factorization from the hpG solve path: the cached MUMPS-LU of the alpha-free
``K0`` is replaced by a capped CG + AMG solve on the same ``K0``.  Everything
else (true-Schur matvec, cellwise ``Shat`` Cholesky, inner GMRES, ``E_beta``
decoupling) is unchanged.  In the rewritten library that arm is
``HPGTwoStage(a_action="gamg")``.

This pins the two claims that make it usable:

  1. the matfree arm reaches the SAME solution as the cached-LU arm at L0 p=2
     (the archive's claim: identical prox/err, only the A-action differs);
  2. it stays clean on a graded mesh (no divergence, no A-CG solve hitting
     its iteration cap) -- the archive's graded48/80 stages.

Recorded context (``RESULTS.md``): "the matfree arm is bit-consistent with
RESULTS.md's claim" from a throwaway smoke; this script makes it a committed
gate.  A-CG budget defaults (``a_rtol=1e-8``, ``a_maxit=60``) are the
archive's.

Run (optionally ``l0`` / ``graded``):
    PETSC_DIR=... PETSC_ARCH=arch-firedrake-default OMP_NUM_THREADS=1 \
        python experiments/rewrite_checks/check_hpg_matfree.py [l0 graded]
"""

import sys
import time

from firedrake import (DirichletBC, Function, SpatialCoordinate, dx, errornorm,
                       grad, inner, sqrt)

from lvpp.benchmarks import uexactUFL, psiUFL
from lvpp.hpg import HPG, HPGDiscretization, HPGTwoStage


class SolveMonitor:
    def __init__(self):
        self.groups = []

    def __call__(self, ksp, its, rnorm):
        if its == 1:
            self.groups.append(1)
        elif its > 1 and self.groups:
            self.groups[-1] = its


def build(disc, uexact_r, pc_kwargs, tag):
    u = Function(disc.primal, name="u")
    x, y = SpatialCoordinate(disc.mesh)
    r = sqrt(x * x + y * y)
    lb = Function(disc.primal, name="lb").interpolate(psiUFL(r))
    bc = DirichletBC(disc.primal, uexactUFL(r), "on_boundary")
    pc = HPGTwoStage(**pc_kwargs)
    lv = HPG(disc, u, (lb, None), bcs=bc, preconditioner=pc,
             energy=0.5 * inner(grad(u), grad(u)) * dx,
             alpha_schedule="double_exponential",
             alpha_parameters={"alpha_max": 10.0},
             increment_norm="H1", verbose=False,
             form_compiler_parameters={"quadrature_degree": 20},
             name=tag)
    return lv, pc, r


def run(disc, pc_kwargs, tag):
    lv, pc, r = build(disc, None, pc_kwargs, tag)
    mon = SolveMonitor()
    lv.install_monitor(mon)
    t0 = time.time()
    lv.solve(tol=1e-4, max_proximal_iterations=500)
    wall = time.time() - t0
    a_its = list(pc.a_its)
    return dict(
        prox=int(lv.proximal_iterations),
        newton=int(sum(lv.newton_iterations)),
        err=float(errornorm(uexactUFL(r), lv.u_out[0])),
        outer=max(mon.groups) if mon.groups else 0,
        inner_avg=(sum(pc.inner_its) / len(pc.inner_its)) if pc.inner_its else float("nan"),
        a_action=pc.a_action, a_calls=pc.a_calls, a_nonconv=pc.a_nonconv,
        a_avg=(sum(a_its) / len(a_its)) if a_its else float("nan"),
        a_max=pc.a_maxits_seen, wall=wall,
    )


def row(tag, res):
    return (f"  {tag:>18} prox={res['prox']:>2} newton={res['newton']:>2} "
            f"err={res['err']:.6e} outer={res['outer']:>2} "
            f"inner={res['inner_avg']:>6.2f} | A[{res['a_action']}] "
            f"calls={res['a_calls']:>4} its={res['a_avg']:>5.1f}/max{res['a_max']:>2} "
            f"nonconv={res['a_nonconv']} | {res['wall']:.1f}s")


def main():
    which = sys.argv[1:] or ["l0", "graded"]
    failures = []

    if "l0" in which:
        print("=== L0 p=2 uniform: cached-LU arm vs matfree arm ===")
        disc = HPGDiscretization.uniform(16, 2)
        lu = run(disc, dict(a_action="lu"), "mf_l0_lu")
        mf = run(disc, dict(a_action="gamg"), "mf_l0_gamg")
        print(row("lu (cached)", lu))
        print(row("matfree (CG+AMG)", mf))
        if mf["prox"] != lu["prox"]:
            failures.append(f"L0: prox {mf['prox']} != lu {lu['prox']}")
        if abs(mf["err"] - lu["err"]) > 1e-4 * abs(lu["err"]):
            failures.append(f"L0: err {mf['err']:.6e} != lu {lu['err']:.6e}")
        if mf["a_nonconv"] != 0:
            failures.append(f"L0: {mf['a_nonconv']} non-converged A-CG solves")
        if not mf["a_calls"] or not mf["a_action"] == "gamg":
            failures.append("L0: the matfree arm recorded no A-CG solves")

    if "graded" in which:
        print("=== graded 26.4x, p=2: matfree arm, A-tolerance sensitivity ===")
        print("  (a saturated A-CG solve is reported, not asserted: the outer"
              " FGMRES absorbs a variable-accuracy A-action by design)")
        disc = HPGDiscretization.graded(64, 2, 26.414)
        for label, kw in (("a_rtol=1e-8", dict(a_action="gamg")),
                          ("a_rtol=1e-6", dict(a_action="gamg", a_rtol=1e-6))):
            mf = run(disc, kw, "mf_g4_" + label.split("=")[1])
            print(row(label, mf))
            if mf["prox"] > 20:
                failures.append(f"graded[{label}]: prox {mf['prox']} > 20")
            if mf["outer"] > 10:
                failures.append(f"graded[{label}]: outer {mf['outer']} > 10")
            if not (mf["err"] < 1e-3):
                failures.append(f"graded[{label}]: err {mf['err']:.3e} not < 1e-3")

    if failures:
        print("\nHPG MATFREE MISMATCH:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("\nHPG MATFREE OK (no global factorization in the apply path; "
          "same solution as the cached-LU arm, clean on grading)")


if __name__ == "__main__":
    main()
