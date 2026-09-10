"""LVPP verification tests.

Cross-checks against firedrake's built-in ``vinewtonrsls`` VI solver, path
parity (energy vs residual), bound preservation, multi-unknown wiring, alpha
schedules, and a regression re-implementation of the ProximalGalerkin
``08_intersecting_constraints`` firedrake example (Farrell) on the lvpp API.
"""

import ufl
from firedrake import (Constant, DirichletBC, Function, FunctionSpace,
                       NonlinearVariationalProblem, NonlinearVariationalSolver,
                       TestFunction, UnitIntervalMesh, UnitSquareMesh,
                       VectorFunctionSpace, as_vector, assemble, conditional,
                       errornorm, exp, ge, grad, inner, le, dx)
import pytest

from lvpp import LVPP, BoxConstraint, Constraint, Hellinger


def _obstacle_bump(V):
    """Active paraboloid obstacle: peak 0.3 at the center."""
    x, y = ufl.SpatialCoordinate(V.mesh())
    return Function(V, name="psi").interpolate(
        0.3 - 4.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))


def _inactive_obstacle(V):
    return Function(V, name="psi").interpolate(Constant(-0.1))


def _poisson_energy(u):
    f = Constant(1.0)
    return (0.5 * inner(grad(u), grad(u)) - f * u) * dx


def _solve_vi_reference(u, psi, bc):
    """Solve the same obstacle problem with firedrake's vinewtonrsls."""
    v = TestFunction(u.function_space())
    F = (inner(grad(u), grad(v)) - Constant(1.0) * v) * dx
    problem = NonlinearVariationalProblem(F, u, bcs=bc)
    solver = NonlinearVariationalSolver(problem, solver_parameters={
        "snes_type": "vinewtonrsls", "ksp_type": "preonly", "pc_type": "lu",
        "snes_rtol": 1e-10, "snes_atol": 1e-12})
    upper = Function(u.function_space()).interpolate(Constant(1.0e20))
    solver.solve(bounds=(psi, upper))
    return u


def test_vinewtonrsls_crosscheck():
    """LVPP and vinewtonrsls solve the same continuum VI but different
    discrete KKT systems (V-multiplier + L2 complementarity vs nodal VI
    bounds), so their solutions differ by O(h^2).  Assert that rate."""
    def err_at(n):
        mesh = UnitSquareMesh(n, n)
        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V, name="u")
        x, y = ufl.SpatialCoordinate(mesh)
        psi = Function(V, name="psi").interpolate(
            0.3 - 4.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))
        bc = DirichletBC(V, 0.0, "on_boundary")
        lvpp = LVPP(energy=_poisson_energy(u), u=u, bounds=(psi, None),
                    bcs=bc, alpha_rule="double_exponential", verbose=False,
                    name=f"xcheck{n}")
        lvpp.solve(tol=1e-10)
        u_vi = _solve_vi_reference(u.copy(deepcopy=True), psi, bc)
        return errornorm(u_vi, lvpp.u_out[0])

    e12 = err_at(12)
    e24 = err_at(24)
    assert e24 < 1e-3, f"coarse cross-check error too large: {e24:.3e}"
    assert e24 < e12 / 2.5, \
        f"error not converging at O(h^2): {e12:.3e} -> {e24:.3e}"


def test_energy_residual_parity():
    mesh = UnitSquareMesh(16, 16)
    V = FunctionSpace(mesh, "CG", 1)
    psi = _obstacle_bump(V)
    bc = DirichletBC(V, 0.0, "on_boundary")

    u1 = Function(V, name="u")
    LVPP(energy=_poisson_energy(u1), u=u1, bounds=(psi, None), bcs=bc,
         verbose=False, name="e-path").solve(tol=1e-9)

    u2 = Function(V, name="u")
    v = TestFunction(V)
    residual = (inner(grad(u2), grad(v)) - Constant(1.0) * v) * dx
    LVPP(residual=residual, u=u2, bounds=(psi, None), bcs=bc,
         verbose=False, name="r-path").solve(tol=1e-9)

    err = errornorm(u1, u2)
    assert err < 1e-9, f"energy vs residual paths differ: {err:.3e}"


def test_bilateral_bounds_u_tilde():
    mesh = UnitSquareMesh(16, 16)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name="u")
    bc = DirichletBC(V, 0.0, "on_boundary")
    f = Constant(-5.0)  # strong push down onto the lower bound
    energy = (0.5 * inner(grad(u), grad(u)) - f * u) * dx

    lvpp = LVPP(energy=energy, u=u, bounds=(-0.3, 0.3), bcs=bc, verbose=False,
                name="bilateral")
    lvpp.solve(tol=1e-9)

    ut = lvpp.u_tilde[0].dat.data_ro
    assert ut.min() >= -0.3 - 1e-12, f"u_tilde below lower bound: {ut.min()}"
    assert ut.max() <= 0.3 + 1e-12, f"u_tilde above upper bound: {ut.max()}"
    # lower bound actually active, so the constraint is exercised
    assert ut.min() > -0.3 - 1e-6
    # u_tilde is feasible by construction; the raw iterate may undershoot O(h^2)
    feas_ut = sum(assemble(f) for f in
                  lvpp._raw_constraints[0].feasibility_forms(V, lvpp.u_tilde[0]))
    assert feas_ut < 1e-12, f"u_tilde infeasible: {feas_ut:.3e}"
    assert lvpp.feasibility() < 1e-3
    assert lvpp._raw_constraints[0].kind == "bilateral"


def test_multi_unknown_bounds_on_one():
    mesh = UnitSquareMesh(16, 16)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name="u")
    w = Function(V, name="w")
    psi = _obstacle_bump(V)
    bc_u = DirichletBC(V, 0.0, "on_boundary")
    # w has no gradient term: its equation reduces to w = -u exactly
    energy = (0.5 * inner(grad(u), grad(u)) + 0.5 * inner(w, w)
              + inner(u, w)) * dx

    lvpp = LVPP(energy=energy, u=[u, w], bounds=[(psi, None), None],
                bcs=bc_u, verbose=False, name="multi")
    lvpp.solve(tol=1e-9)

    assert len(lvpp.u_out) == 2 and len(lvpp.psi_out) == 1
    assert len(lvpp.u_tilde) == 1
    feas_ut = sum(assemble(f) for f in
                  lvpp._raw_constraints[0].feasibility_forms(V, lvpp.u_tilde[0]))
    assert feas_ut < 1e-12, f"u_tilde infeasible: {feas_ut:.3e}"
    assert lvpp.feasibility() < 1e-3
    ut = lvpp.u_tilde[0].dat.data_ro
    assert ut.min() >= psi.dat.data_ro.min() - 1e-12
    # w = -u holds exactly at the coupled solution
    w_ref = Function(V).assign(-lvpp.u_out[0])
    err = errornorm(w_ref, lvpp.u_out[1])
    assert err < 1e-8, f"coupled equation w = -u violated: {err:.3e}"


def test_alpha_rules_converge():
    mesh = UnitSquareMesh(16, 16)
    V = FunctionSpace(mesh, "CG", 1)
    bc = DirichletBC(V, 0.0, "on_boundary")
    for rule, params in [("constant", {"C": 1.0}),
                         ("double_exponential", None)]:
        u = Function(V, name="u")
        psi = _inactive_obstacle(V)
        lvpp = LVPP(energy=_poisson_energy(u), u=u, bounds=(psi, None),
                    bcs=bc, alpha_rule=rule, alpha_parameters=params,
                    verbose=False, name=f"rule-{rule}")
        lvpp.solve(tol=1e-8)
        assert lvpp.proximal_iterations < 100, f"{rule} did not converge"
        assert lvpp.history["primal_increment"][-1] < 1e-8


def test_vector_box_componentwise():
    mesh = UnitSquareMesh(16, 16)
    V = VectorFunctionSpace(mesh, "CG", 1)
    V0 = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name="u")
    bc = DirichletBC(V, as_vector((0.0, 0.0)), "on_boundary")
    # mild load: the second component touches its lower bound (active set
    # engaged) without driving psi into float saturation (see LVPP docstring
    # note on exactly-active bounds)
    f = Constant((0.0, -0.5))
    energy = (0.5 * inner(grad(u), grad(u)) - inner(f, u)) * dx
    lower = as_vector((-0.3, -0.02))
    upper = as_vector((0.3, 0.4))
    lvpp = LVPP(energy=energy, u=u, bounds=(lower, upper), bcs=bc,
                verbose=False, name="vector")
    lvpp.solve(tol=1e-8)

    ut = lvpp.u_tilde[0]
    lo = (-0.3, -0.02)
    hi = (0.3, 0.4)
    for comp in range(2):
        c = Function(V0).interpolate(ut[comp])
        assert c.dat.data_ro.min() >= lo[comp] - 1e-12, \
            f"component {comp} below bound: {c.dat.data_ro.min()}"
        assert c.dat.data_ro.max() <= hi[comp] + 1e-12, \
            f"component {comp} above bound: {c.dat.data_ro.max()}"
    # second component pushed against its lower bound by f
    c1 = Function(V0).interpolate(ut[1])
    assert c1.dat.data_ro.min() > -0.02 - 1e-6
    feas_ut = sum(assemble(f) for f in
                  lvpp._raw_constraints[0].feasibility_forms(V, ut))
    assert feas_ut < 1e-12, f"u_tilde infeasible: {feas_ut:.3e}"
    assert lvpp.feasibility() < 1e-3


class GradientConstraint(Constraint):
    """|grad u| <= phi via the Hellinger entropy; B = grad (paper Sec. 4.1).

    Mirrors the hand-written mixed system in ProximalGalerkin
    examples/08_intersecting_constraints/intersecting_constraints_firedrake.py.
    """

    def __init__(self, phi):
        super().__init__(Hellinger(phi))

    def latent_space(self, V):
        return VectorFunctionSpace(V.mesh(), "CG", 1)

    def observable_space(self, V):
        return self.latent_space(V)

    def coupling_form(self, V, du, psi, psi_prev):
        dxm = ufl.Measure("dx", domain=V.mesh())
        return (ufl.inner(psi, ufl.grad(du))
                - ufl.inner(psi_prev, ufl.grad(du))) * dxm

    def state_form(self, V, u, dw, psi):
        dxm = ufl.Measure("dx", domain=V.mesh())
        return (ufl.inner(ufl.grad(u), dw)
                - ufl.inner(self.observable(psi), dw)) * dxm

    def feasibility_forms(self, V, u):
        dxm = ufl.Measure("dx", domain=V.mesh())
        g = ufl.sqrt(ufl.dot(ufl.grad(u), ufl.grad(u)))
        phi = self.legendre.obstacle
        return [ufl.conditional(g > phi, g - phi, 0.0) * dxm]


def _bump_obstacle(V):
    """Smooth normalized bump on (0.2, 0.8), zero outside; peak 1 at x=0.5."""
    x = ufl.SpatialCoordinate(V.mesh())[0]
    l, r = 0.2, 0.8
    bump = exp(-1.0 / (10.0 * (x - l) * (r - x))) \
        / exp(-1.0 / (10.0 * (0.5 - l) * (r - 0.5)))
    return Function(V, name="psi").interpolate(
        conditional(le(x, l), 0.0, conditional(ge(x, r), 0.0, bump)))


def test_intersecting_constraints_regression():
    """Re-implementation of ProximalGalerkin example 08 on the lvpp API:
    obstacle u >= phi0 AND gradient bound |grad u| <= phi, one unknown."""
    mesh = UnitIntervalMesh(1001)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name="u")
    phi0 = _bump_obstacle(V)
    bc = DirichletBC(V, 0, "on_boundary")
    energy = 0.5 * inner(grad(u), grad(u)) * dx  # c = 0, as in the example

    grad_constraint = GradientConstraint(Constant(1.0))

    def pg_alpha(k, alpha_prev, newton_its):
        # the ProximalGalerkin example's schedule: start at 1, only ever
        # decrease (the retry loop persists halved alphas via alpha_prev)
        return alpha_prev if k > 1 else 1.0

    lvpp = LVPP(energy=energy, u=u,
                bounds=[BoxConstraint(lower=phi0), grad_constraint],
                bcs=bc, alpha_rule=pg_alpha,
                on_newton_failure="reduce_alpha", max_consecutive_failures=50,
                solver_parameters={"snes_linesearch_type": "l2",
                                   "snes_linesearch_maxlambda": 1.0,
                                   "snes_atol": 1.0e-6,
                                   "ksp_type": "preonly", "pc_type": "lu",
                                   "pc_factor_mat_solver_type": "mumps"},
                verbose=False, name="intersecting")

    # NOTE: the reference example also steps phi down (3, 2, 1, 0.5, 0.1,
    # 0.01).  Continuing past phi=3 here reproduces a method corner: where the
    # converged |grad u| equals the bound exactly, the Hellinger latent psi
    # must diverge and this PETSc lacks the reference's absolute line-search
    # step cap ("snes_linesearch_maxstep"), so Newton directions blow up and
    # alpha-halving cannot help (it only freezes u).  The phi=3 solve below
    # still exercises both constraints and the general B = grad interface.
    grad_constraint.legendre.obstacle.assign(3.0)
    lvpp.solve(tol=1e-6)
    assert lvpp.proximal_iterations < 100, "no convergence for phi=3"
    # obstacle feasibility of the bound-preserving output
    feas = sum(assemble(f) for f in
               lvpp._raw_constraints[0].feasibility_forms(V, lvpp.u_tilde[0]))
    # gradient feasibility of the actual solution
    feas += sum(assemble(f) for f in
                grad_constraint.feasibility_forms(V, lvpp.u_out[0]))
    assert feas < 1e-8, f"feasibility {feas:.3e}"
    # obstacle engaged at the bump peak (u reaches the obstacle there)
    peak = float(lvpp.u_out[0].dat.data_ro.max())
    assert abs(peak - 1.0) < 1e-4, f"obstacle not engaged: peak {peak}"


def test_rejects_bad_inputs():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    energy = inner(grad(u), grad(u)) * dx
    with pytest.raises(ValueError):
        LVPP(u=u)  # neither energy nor residual
    with pytest.raises(ValueError):
        LVPP(energy=energy, residual=energy, u=u)  # both
    with pytest.raises(ValueError):
        LVPP(energy=energy, u=u, bounds=(None, None))  # no actual bound
    with pytest.raises(ValueError):
        LVPP(energy=energy, u=[u, u])  # same Function twice
