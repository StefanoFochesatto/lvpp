"""The hpG two-stage solver on a strongly graded chain.

``experiments/hpg/RESULTS.md`` reports the hpG two-stage solver converging
without divergence on a 2D graded chain out to 226x grading and 25.7k dofs,
where the P1 Schur-fieldsplit configuration diverged (``ksp=-5`` at 14.2k).
This script re-runs that chain through :mod:`lvpp.hpg`.

Note on the meshes: the older ``graded_quad(n, ratio)`` in
``experiments/hpg/`` produced meshes measuring ~1.21x the ratio it was asked
for, so its recorded chain is ``(64, 21.8) -> 26.41x``, ``(80, 45.3) ->
55.33x``, ``(96, 93.5) -> 114.78x``, ``(112, 187.0) -> 226.06x``.
:meth:`HPGDiscretization.graded` calibrates the growth so the *measured* ratio
equals the requested one, so the recorded grade strengths are passed directly
below.  The meshes therefore differ slightly from the older ones (same band,
~10% weaker chains), so what is checked is *no divergence with bounded
iterations*, not digit-identical counts; the recorded counts are printed
alongside and must not blow up (proximal iterations <= 20, outer Krylov <= 10,
err(u_h) < 1e-3, measured grading within 1% of the request).

Run from the repository root (optionally with a subset of levels, e.g. ``g4
g5``):

    PETSC_DIR=/home/stefano/firedrake/petsc PETSC_ARCH=arch-firedrake-default \
    OMP_NUM_THREADS=1 /home/stefano/firedrake/venv-firedrake/bin/python \
    experiments/checks/check_hpg_graded.py [g4 g5 g6 g7]
"""

import sys
import time

from firedrake import (DirichletBC, Function, SpatialCoordinate, dx, errornorm,
                       grad, inner, sqrt)

from lvpp.benchmarks import uexactUFL, psiUFL
from lvpp.hpg import HPG, HPGDiscretization, HPGTwoStage

# (level, cells per axis, measured grading of the archive's mesh,
#  recorded (prox, newton, err) from RESULTS.md at p = 2)
LEVELS = {
    "g4": (64, 26.414, (6, 20, 5.892e-5)),
    "g5": (80, 55.330, (6, 17, 4.960e-5)),
    "g6": (96, 114.775, (7, 19, 7.748e-5)),
    "g7": (112, 226.056, (6, 20, 4.226e-5)),
}


class SolveMonitor:
    def __init__(self):
        self.groups = []

    def __call__(self, ksp, its, rnorm):
        if its == 1:
            self.groups.append(1)
        elif its > 1 and self.groups:
            self.groups[-1] = its


def run(tag, n, ratio, p=2, published=False):
    disc = HPGDiscretization.graded(n, p, ratio)
    u = Function(disc.primal, name="u")
    x, y = SpatialCoordinate(disc.mesh)
    r = sqrt(x * x + y * y)
    lb = Function(disc.primal, name="lb").interpolate(psiUFL(r))
    bc = DirichletBC(disc.primal, uexactUFL(r), "on_boundary")
    pc = HPGTwoStage() if not published else HPGTwoStage(inner_rtol=1e-6,
                                                         inner_maxit=500)
    lv = HPG(disc, u, (lb, None), bcs=bc, preconditioner=pc,
             energy=0.5 * inner(grad(u), grad(u)) * dx,
             alpha_schedule="double_exponential",
             alpha_parameters={"alpha_max": 10.0},
             increment_norm="H1", verbose=False,
             form_compiler_parameters={"quadrature_degree": 20},
             name=tag)
    mon = SolveMonitor()
    lv.install_monitor(mon)
    t0 = time.time()
    lv.solve(tol=1e-4, max_proximal_iterations=500)
    wall = time.time() - t0
    err = float(errornorm(uexactUFL(r), lv.u_out[0]))
    its = list(pc.inner_its) if hasattr(pc, "inner_its") else []
    return dict(prox=int(lv.proximal_iterations),
                newton=int(sum(lv.newton_iterations)),
                err=err, outer=max(mon.groups) if mon.groups else 0,
                inner_avg=(sum(its) / len(its)) if its else float("nan"),
                measured=disc.grading_ratio(), wall=wall)


def main():
    which = sys.argv[1:] or list(LEVELS)
    print(f"{'tag':>4} {'cells':>6} {'ratio':>8} {'prox':>4} {'newton':>6} "
          f"{'err(u_h)':>10} {'outer':>5} {'inner/apply':>11} {'wall':>7}  "
          f"recorded")
    failures = []
    for tag in which:
        n, ratio, (r_prox, r_newton, r_err) = LEVELS[tag]
        res = run(tag, n, ratio)
        print(f"{tag:>4} {n * n:>6} {res['measured']:>8.1f} {res['prox']:>4} "
              f"{res['newton']:>6} {res['err']:>10.3e} {res['outer']:>5} "
              f"{res['inner_avg']:>11.2f} {res['wall']:>6.1f}s  "
              f"({r_prox}, {r_newton}, {r_err:.3e})")
        if res["prox"] > 20:
            failures.append(f"{tag}: prox {res['prox']} > 20")
        if res["outer"] > 10:
            failures.append(f"{tag}: outer {res['outer']} > 10")
        if not (res["err"] < 1e-3):
            failures.append(f"{tag}: err {res['err']:.3e} not < 1e-3")
        if abs(res["measured"] - ratio) > 0.01 * ratio:
            failures.append(f"{tag}: measured grading {res['measured']} != {ratio}")

    if failures:
        print("\nHPG GRADED CHAIN MISMATCH:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("\nHPG GRADED CHAIN OK (no divergence; bounded iterations and outer Krylov)")


if __name__ == "__main__":
    main()
