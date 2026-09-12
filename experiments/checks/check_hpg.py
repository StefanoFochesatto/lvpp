"""End-to-end run of the hpG two-stage solver on the uniform L0 p = 2 problem.

Runs the uniform L0 p = 2 row of ``experiments/hpg/RESULTS.md`` with the
:class:`lvpp.hpg.HPG` preset and the
:class:`lvpp.hpg.HPGTwoStage` preconditioner, under both inner solver settings
that are of interest:

  paper-literal  ``HPGTwoStage(inner_rtol=1e-6, inner_maxit=500)``
  default        ``HPGTwoStage()``  (rtol 1e-4, cap 40)

Reported target (``experiments/hpg/RESULTS.md``, uniform L0 p2, paper-literal
inner settings): prox 8, newton 23, err(u_h) 1.193e-3, outer FGMRES max 2,
inner its/apply ~12.8 (first 5.4/7 -> deep 18/20).  Only the paper-literal
numbers are checked: prox exactly, err to a relative 1e-3 (the reported errors
carry four significant figures) and the outer FGMRES maximum bounded by 4;
everything else is printed for comparison.

Env guard REQUIRED:

  PETSC_DIR=/home/stefano/firedrake/petsc PETSC_ARCH=arch-firedrake-default \
  OMP_NUM_THREADS=1 /home/stefano/firedrake/venv-firedrake/bin/python \
  experiments/checks/check_hpg.py

Serial only.
"""

import sys

import numpy as np
from firedrake import (DirichletBC, Function, SpatialCoordinate, dx, errornorm,
                       grad, inner, sqrt)

from lvpp.benchmarks import psiUFL, uexactUFL
from lvpp.hpg.solver import HPG
from lvpp.hpg.spaces import HPGDiscretization
from lvpp.hpg.twostage import HPGTwoStage

# RESULTS.md uniform L0 p=2, two-stage with the paper-literal inner settings.
RECORDED = {"prox": 8, "newton": 23, "err": 1.193e-3, "outer": 2, "inner": 12.8}
ERR_REL_TOL = 1e-3          # recorded errors are quoted to 4 significant figs
OUTER_MAX_TOL = 4           # recorded max is 2


class SolveMonitor:
    """Outer KSP monitor: one group per linear solve, split on ``its == 1``."""

    def __init__(self):
        self.groups = []

    def __call__(self, ksp, its, rnorm):
        if its == 1:
            self.groups.append(1)
        elif its > 1 and self.groups:
            self.groups[-1] = its


def build():
    """The sphere-obstacle L0 p=2 problem on the hpG discretization."""
    disc = HPGDiscretization.uniform(16, 2)
    mesh = disc.mesh
    x, y = SpatialCoordinate(mesh)
    r = sqrt(x * x + y * y)
    u = Function(disc.primal, name="u")
    lb = Function(disc.primal, name="psi_lb").interpolate(psiUFL(r))
    bc = DirichletBC(disc.primal, uexactUFL(r), "on_boundary")
    energy = 0.5 * inner(grad(u), grad(u)) * dx
    return disc, u, (lb, None), bc, energy, r


def run(preconditioner, name):
    """One end-to-end solve; returns (err, prox, newton, outer its, PC)."""
    disc, u, bounds, bc, energy, r = build()
    hp = HPG(disc, u, bounds, bcs=bc, preconditioner=preconditioner,
             energy=energy,
             alpha_schedule="double_exponential",
             alpha_parameters={"alpha_max": 10.0},
             increment_norm="H1", verbose=False,
             form_compiler_parameters={"quadrature_degree": 20},
             name=name)
    mon = SolveMonitor()
    hp.install_monitor(mon)
    hp.solve(tol=1e-4, max_proximal_iterations=100)
    err = float(errornorm(uexactUFL(r), hp.u_out[0]))
    prox = int(hp.proximal_iterations)
    newton = int(sum(hp.newton_iterations))
    outer = max(mon.groups) if mon.groups else 0
    return err, prox, newton, outer, hp.preconditioner


def report(label, err, prox, newton, outer, pc):
    inner = np.asarray(pc.inner_its, dtype=float)
    inner_avg = float(inner.mean()) if inner.size else float("nan")
    print(f"  {label:<14} prox={prox:3d} newton={newton:3d} "
          f"err={err:.6e} outer_max={outer:3d} "
          f"inner={inner_avg:6.2f}/apply ({inner.size} applies)")
    return inner_avg


def main():
    disc, u, bounds, bc, energy, r = build()
    print("=== hpG L0 p=2 uniform (16x16 quads, CG2 x DQ0 spectral) ===")
    print(f"  primal.dim() = {disc.primal.dim()} (expect 1089)")
    print(f"  latent.dim() = {disc.latent.dim()} (expect 256)")
    assert disc.primal.dim() == 1089, disc.primal.dim()
    assert disc.latent.dim() == 256, disc.latent.dim()

    print(f"\n  recorded (RESULTS.md, paper-literal inner settings): "
          f"prox={RECORDED['prox']} "
          f"newton={RECORDED['newton']} err={RECORDED['err']:.3e} "
          f"outer_max={RECORDED['outer']} inner={RECORDED['inner']}")

    literal = HPGTwoStage(inner_rtol=1e-6, inner_maxit=500)
    err_l, prox_l, newton_l, outer_l, pc_l = run(literal, "hpg_literal")
    inner_l = report("paper-literal", err_l, prox_l, newton_l, outer_l, pc_l)

    default = HPGTwoStage()
    err_d, prox_d, newton_d, outer_d, pc_d = run(default, "hpg_default")
    inner_d = report("default", err_d, prox_d, newton_d, outer_d, pc_d)

    print("\n=== assertions (on the recorded paper-literal numbers) ===")
    failures = []
    if prox_l != RECORDED["prox"]:
        failures.append(f"prox {prox_l} != {RECORDED['prox']}")
    if abs(err_l - RECORDED["err"]) > ERR_REL_TOL * RECORDED["err"]:
        failures.append(f"err {err_l:.6e} not within {ERR_REL_TOL:g} relative "
                        f"of {RECORDED['err']:.6e}")
    if outer_l > OUTER_MAX_TOL:
        failures.append(f"outer max {outer_l} > {OUTER_MAX_TOL}")
    for f in failures:
        print(f"  FAIL: {f}")
    if failures:
        print("\nHPG EXAMPLE FAILED")
        sys.exit(1)
    print(f"  PASS: prox={prox_l}, err within {ERR_REL_TOL:g} rel, "
          f"outer {outer_l} <= {OUTER_MAX_TOL}")
    print(f"  (default settings: prox={prox_d}, newton={newton_d}, "
          f"err={err_d:.6e}, outer={outer_d}, inner={inner_d:.2f}/apply, "
          f"literal inner={inner_l:.2f}/apply)")
    print("\nHPG EXAMPLE OK")


if __name__ == "__main__":
    main()
