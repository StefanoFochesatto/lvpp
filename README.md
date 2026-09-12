# lvpp: latent variable proximal point for variational inequalities

`lvpp` solves finite-element variational problems with pointwise inequality
constraints in Firedrake.  The constrained quantities may be obstacles, box
bounds, gradient bounds, or simplex constraints; several constraints may act on
one unknown, and a problem may have several unknowns.  The algorithm is the
latent variable proximal point (LVPP) method of J. S. Dokken, P. E. Farrell,
B. Keith, I. P. A. Papadopoulos, and T. M. Surowiec, "The latent variable
proximal point algorithm for variational problems with inequality
constraints", CMAME 445:118181 (2025); arXiv:2503.05672.

Every constrained unknown `u` carries a latent variable `psi` on a space `W` of
its own.  The pair is advanced by the mixed proximal iteration (2.7) of LVPP:
at step `k` one semismooth Newton solve of

    alpha_k <J'(u^k), v> + (psi^k, Bv) - (psi^{k-1}, Bv) = 0    for all v   (2.7a)
    (B u^k, w) - (grad R*(psi^k), w)                     = 0    for all w   (2.7b)

is taken, starting from `psi^0 = 0`, until the stopping rule fires.  Because
`grad R*(psi)` lies in the interior of the admissible set for every `psi`, the
reconstruction `u_tilde = grad R*(psi)` is feasible pointwise by construction
whatever `alpha` is, so the iteration needs no `alpha -> infinity` to return an
admissible answer.

## Install

Inside a Firedrake virtual environment:

    pip install -e .

## The classical obstacle problem

The smallest complete problem is minimising

    J(u) = int 0.5 |grad u|^2 - f u dx

on the unit square subject to `u >= psi` and `u = 0` on the boundary.  The
energy goes in as a UFL 0-form, the unknown as a `Function`, and the constraint
as a `(lower, upper)` pair with either side possibly `None`.  The obstacle here
is the smooth bump `psi = 0.1 - (x - 0.5)^2 - (y - 0.5)^2`.

```python
from firedrake import *
from lvpp import LVPP

mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "CG", 1)
u = Function(V, name="u")
x, y = SpatialCoordinate(mesh)
psi = Function(V, name="psi").interpolate(
    0.1 - (x - 0.5)**2 - (y - 0.5)**2)
f = Constant(1.0)
energy = (0.5 * inner(grad(u), grad(u)) - f * u) * dx
bc = DirichletBC(V, 0.0, "on_boundary")

lvpp = LVPP(energy=energy, u=u, bounds=(psi, None), bcs=bc)
lvpp.solve(tol=1e-8)

u_tilde = lvpp.u_tilde[0]
feas_ut = sum(assemble(F) for F in
              lvpp.constraints[0].feasibility_forms(V, u_tilde))
gap = Function(V).interpolate(u_tilde - psi)

print("proximal iterations :", lvpp.proximal_iterations)
print("newton iterations   :", sum(lvpp.newton_iterations))
print("last primal inc     :", lvpp.history["primal_increment"][-1])
print("feasibility(u_h)    :", lvpp.feasibility())
print("feasibility(u_tilde):", feas_ut)
print("min(u_tilde - psi)  :", gap.dat.data_ro.min())
print("energy()            :", lvpp.energy())
print("active nodes u_tilde==psi:", (gap.dat.data_ro < 1e-10).sum(), "/", V.dim())
```

A run of it, with the default double-exponential schedule and `verbose=True`:

```
lvpp[lvpp] alpha rule: double_exponential(C=1, alpha_max=10, q=1.5, r=1.5), stopping=primal_increment(tol=1e-08), max_proximal_iterations=100
lvpp[lvpp]   1 alpha=1.837e+00 |du|=9.188e-02 |dpsi|=9.193e-02 feas=0.000e+00 comp=9.453e-02 dual=0.000e+00 newton=6
lvpp[lvpp]   2 alpha=1.000e+00 |du|=2.910e-02 |dpsi|=2.903e-02 feas=0.000e+00 comp=2.687e-02 dual=7.412e-09 newton=5
lvpp[lvpp]   3 alpha=2.929e+00 |du|=1.229e-02 |dpsi|=1.220e-02 feas=2.154e-08 comp=9.341e-03 dual=4.075e-10 newton=6
lvpp[lvpp]   4 alpha=4.859e+00 |du|=2.187e-03 |dpsi|=2.155e-03 feas=4.767e-07 comp=7.175e-03 dual=4.319e-11 newton=6
lvpp[lvpp]   5 alpha=1.000e+01 |du|=3.766e-04 |dpsi|=3.696e-04 feas=3.626e-07 comp=6.860e-03 dual=3.428e-12 newton=4
lvpp[lvpp]   6 alpha=1.000e+01 |du|=4.056e-05 |dpsi|=3.962e-05 feas=3.528e-07 comp=6.828e-03 dual=3.519e-13 newton=3
lvpp[lvpp]   7 alpha=1.000e+01 |du|=5.447e-06 |dpsi|=5.282e-06 feas=3.620e-07 comp=6.824e-03 dual=4.563e-14 newton=3
lvpp[lvpp]   8 alpha=1.000e+01 |du|=8.835e-07 |dpsi|=8.486e-07 feas=3.643e-07 comp=6.823e-03 dual=7.270e-15 newton=2
lvpp[lvpp]   9 alpha=1.000e+01 |du|=1.672e-07 |dpsi|=1.587e-07 feas=3.648e-07 comp=6.823e-03 dual=1.363e-15 newton=2
lvpp[lvpp]  10 alpha=1.000e+01 |du|=3.536e-08 |dpsi|=3.324e-08 feas=3.650e-07 comp=6.823e-03 dual=2.863e-16 newton=2
lvpp[lvpp]  11 alpha=1.000e+01 |du|=8.022e-09 |dpsi|=7.487e-09 feas=3.650e-07 comp=6.823e-03 dual=6.465e-17 newton=2
proximal iterations : 11
newton iterations   : 41
last primal inc     : 8.02177139186498e-09
feasibility(u_h)    : 3.64996739241523e-07
feasibility(u_tilde): 0.0
min(u_tilde - psi)  : 0.0
energy()            : -0.016685011403379905
active nodes u_tilde==psi: 21 / 1089
```

`solve()` returns `self` and leaves everything on the object.  It reports
`proximal_iterations`, the per-step `newton_iterations` list and their sum,
`alpha_history`, `history` (the primal and latent increments, the diagnostics
of every accepted iterate), the final `alpha`, and the multiplier field
`drift`.  The verbose line prints the parameter of the step, the primal and
latent increments (`|du|`, `|dpsi|`), the three variational-inequality
diagnostics, and the Newton count.

`solve()` also builds `u_tilde`: one `Function` per constraint, holding
`grad R*(psi)` at the accepted iterate.  For a lower bound the Shannon entropy
gives `u_tilde = psi + exp(psi)`, so `u_tilde >= psi` pointwise by
construction.  In this run the obstacle binds on 21 of the 1089 nodes, in a
disk around the centre where the reconstructed unknown sits exactly on the
obstacle (`min(u_tilde - psi) = 0.0`); the constraint is inactive outside it.
The total violation of `u_tilde` is exactly zero, while the raw iterate `u_h`
undershoots the obstacle by `3.65e-7` in integral: `u_h` is a Newton iterate on
the mixed system and carries the projection error of the state row, and
`u_tilde` is the output that is admissible.

The initial guess for `u` is the user's `Function`, and the latent blocks start
at zero.  `warm_start=True` keeps the current latent block and carries it into
the proximal shift instead of resetting it, so a re-solve continues from the
seeded `psi` (prolonged from a coarser mesh, for instance).

## Choosing an entropy (Legendre) function

Each constraint family is a Legendre function `R`, a convex function whose
domain is the admissible set `C`.  Its conjugate gradient `grad R*` maps the
whole latent space onto the interior of `C`, and that map, `grad R*(psi)`, is
the only thing the state equation (2.7b) needs.  The library ships five:

| class | feasible set | reconstruction `grad R*(psi)` |
|---|---|---|
| `ShannonLower(phi)` | `u >= phi` | `phi + exp(psi)` |
| `ShannonUpper(phi)` | `u <= phi` | `phi - exp(-psi)` |
| `FermiDirac(phi1, phi2)` | `phi1 <= u <= phi2` | `phi1 + (phi2 - phi1) (1 + tanh(psi/2))/2` |
| `Hellinger(phi)` | `|grad u| <= phi` | `phi psi / sqrt(1 + |psi|^2)` |
| `GibbsSimplex()` | `sum u_i = 1`, `u_i >= 0` | `exp(psi_i) / sum_j exp(psi_j)` |

`ShannonLower` is eq. (3.4) of LVPP, `FermiDirac` is eq. (3.5) written in the
overflow-safe `tanh` form, `Hellinger` is eq. (4.2), and `GibbsSimplex` is
eq. (3.17).  `GibbsSimplex` and `Hellinger` act on vector-valued latent
variables: the gradient constraint uses the geometric dimension, the simplex
constraint one entry per phase.  For the box constraints the latent variable
lives on a copy of the unknown's space, the equal-order pairing admitted by the
compatibility condition of Keith & Surowiec (2024, Section 4.7).

`BoxConstraint` selects the entropy from the bounds it is given: a lower bound
alone gives `ShannonLower`, an upper bound alone gives `ShannonUpper`, and both
give `FermiDirac`.  The selection is visible on the object, and a Legendre
function can be passed explicitly with `legendre=`:

```python
import ufl
from firedrake import *
from lvpp import BoxConstraint, LVPP, ShannonLower

mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, "CG", 1)
u = Function(V, name="u")
x, y = SpatialCoordinate(mesh)
phi = Function(V, name="phi").interpolate(
    0.1 - (x - 0.5)**2 - (y - 0.5)**2)
bc = DirichletBC(V, 0.0, "on_boundary")
energy = (0.5 * inner(grad(u), grad(u)) - Constant(1.0) * u) * dx

print("auto, lower bound:", BoxConstraint(lower=phi).legendre)
print("auto, upper bound:", BoxConstraint(upper=phi).legendre)
print("auto, two-sided  :", BoxConstraint(lower=phi, upper=0.3).legendre)


class SquaredShannon:
    """Lower-bound entropy with grad R*(psi) = phi + exp(2 psi)."""

    def __init__(self, phi):
        self.phi = phi

    def reconstruction(self, psi):
        return self.phi + ufl.exp(2.0 * psi)


mine = LVPP(energy=energy, u=u,
            bounds=BoxConstraint(lower=phi, legendre=SquaredShannon(phi)),
            bcs=bc, verbose=False, name="mine")
mine.solve(tol=1e-8)
gap = Function(V).interpolate(mine.u_tilde[0] - phi)
print("custom Legendre: prox", mine.proximal_iterations,
      "newton", sum(mine.newton_iterations),
      "min(u_tilde - phi)", gap.dat.data_ro.min())

u2 = Function(V, name="u2")
stock = LVPP(energy=(0.5 * inner(grad(u2), grad(u2)) - Constant(1.0) * u2) * dx,
            u=u2, bounds=BoxConstraint(lower=phi, legendre=ShannonLower(phi)),
            bcs=bc, verbose=False, name="stock")
stock.solve(tol=1e-8)
print("ShannonLower   : prox", stock.proximal_iterations,
      "newton", sum(stock.newton_iterations),
      "err(custom, stocked)", errornorm(mine.u_tilde[0], stock.u_tilde[0]))
```

```
auto, lower bound: ShannonLower(phi)
auto, upper bound: ShannonUpper(phi)
auto, two-sided  : FermiDirac(phi1, phi2)
custom Legendre: prox 8 newton 31 min(u_tilde - phi) 0.0
ShannonLower   : prox 9 newton 37 err(custom, stocked) 5.314209308078293e-10
```

Writing a new Legendre function is one method.  `reconstruction(psi)` returns
the UFL expression for `grad R*(psi)`, and it must land in the interior of the
admissible set for every `psi`; `SquaredShannon` above is `phi + exp(2 psi)`,
the conjugate map of `R(a) = 0.5 ((a - phi) log(a - phi) - (a - phi))`, which
has the same feasible set as `ShannonLower` and reaches the same reconstruction
to `5.3e-10` in the `u_tilde` norm.  The latent argument may be a `Function` or
a `split()` piece of the mixed unknown, and for a vector unknown the map has to
be written componentwise (see `ShannonLower.reconstruction`, which does this).

A new *family* of constraints, as opposed to a new map for the same family,
also needs a `Constraint` subclass, which says what the operator `B` is and how
it is assembled.  `BoxConstraint` has `B = id`; the gradient constraint
`|grad u| <= phi` has `B = grad`, and `tests/test_lvpp.py` works one out in
full: `GradientConstraint` implements `latent_space` (a vector space for
`psi`), `coupling_form` (the term `(psi, grad v) - (psi_prev, grad v)` for
row (2.7a)), `state_form` (`(grad u, dw) - (grad R*(psi), dw)` for row (2.7b)),
`observable_space` and `feasibility_forms`, and takes its `grad R*` from
`Hellinger`.  Section 4 below uses it as the second constraint on one unknown.

## Upper and lower constraints

A bound is any UFL scalar expression: a float, a `Constant`, a `Function`, or a
coordinate expression, and either side of the pair may be `None`.  For a
vector-valued unknown the bounds act componentwise, and `as_vector(...)` gives
each component its own bound.  A bound may also depend on the solution itself,
which is how obstacle-type quasi-variational inequalities are written: the
bound is remapped to the current iterate by `Constraint.remap`, so the mixed
system sees the live `u` while the feasibility diagnostics are evaluated on the
user's data.

A two-sided problem, solved and then checked against both bounds:

```python
from firedrake import *
from lvpp import LVPP

mesh = UnitSquareMesh(16, 16)
V = FunctionSpace(mesh, "CG", 1)
u = Function(V, name="u")
bc = DirichletBC(V, 0.0, "on_boundary")
f = Constant(-5.0)          # strong push down onto the lower bound
energy = (0.5 * inner(grad(u), grad(u)) - f * u) * dx

lvpp = LVPP(energy=energy, u=u, bounds=(-0.3, 0.3), bcs=bc, verbose=False)
lvpp.solve(tol=1e-9)

u_tilde = lvpp.u_tilde[0]
feas = sum(assemble(F) for F in
           lvpp.constraints[0].feasibility_forms(V, u_tilde))
print("proximal iterations :", lvpp.proximal_iterations)
print("newton iterations   :", sum(lvpp.newton_iterations))
print("min(u_tilde)        :", u_tilde.dat.data_ro.min())
print("max(u_tilde)        :", u_tilde.dat.data_ro.max())
print("min(u_h)            :", lvpp.u_out[0].dat.data_ro.min())
print("feasibility(u_tilde):", feas)
print("constraint          :", lvpp.constraints[0])
```

```
proximal iterations : 10
newton iterations   : 45
min(u_tilde)        : -0.3
max(u_tilde)        : 0.00042134801526594356
min(u_h)            : -0.3015884428852818
feasibility(u_tilde): 0.0
constraint          : BoxConstraint(bilateral: FermiDirac(phi1, phi2))
```

The reconstruction sits exactly on the lower bound (`-0.3`) where the load
presses the solution onto it, and the raw iterate goes `1.6e-3` below that
bound.  `bounds=(psi, None)` is the obstacle problem of the previous section
with `psi` the obstacle; `bounds=(None, phi)` is the mirror image, with the
solution clamped from above:

```python
from firedrake import *
from lvpp import LVPP

mesh = UnitSquareMesh(16, 16)
V = FunctionSpace(mesh, "CG", 1)
u = Function(V, name="u")
x, y = SpatialCoordinate(mesh)
phi = Function(V, name="phi").interpolate(0.15 + (x - 0.5)**2 + (y - 0.5)**2)
bc = DirichletBC(V, 0.0, "on_boundary")
f = Constant(8.0)           # push u up onto phi
energy = (0.5 * inner(grad(u), grad(u)) - f * u) * dx

lvpp = LVPP(energy=energy, u=u, bounds=(None, phi), bcs=bc, verbose=False)
lvpp.solve(tol=1e-9)

gap = Function(V).interpolate(lvpp.u_tilde[0] - phi)
print("constraint          :", lvpp.constraints[0])
print("proximal iterations :", lvpp.proximal_iterations)
print("max(u_tilde - phi)  :", gap.dat.data_ro.max())
print("max(u_h - phi)      :",
      Function(V).interpolate(lvpp.u_out[0] - phi).dat.data_ro.max())
print("feasibility(u_tilde):", sum(
    assemble(F) for F in lvpp.constraints[0].feasibility_forms(V, lvpp.u_tilde[0])))
```

```
constraint          : BoxConstraint(upper: ShannonUpper(phi))
proximal iterations : 14
max(u_tilde - phi)  : 0.0
max(u_h - phi)      : 0.004945123861799627
feasibility(u_tilde): 0.0
```

Here `max(u_tilde - phi) = 0.0` where the load presses the solution against the
ceiling, and the raw iterate exceeds `phi` by `4.9e-3`.

For a vector unknown, `as_vector` gives componentwise bounds:

```python
from firedrake import *
from lvpp import LVPP

mesh = UnitSquareMesh(16, 16)
V = VectorFunctionSpace(mesh, "CG", 1)
V0 = FunctionSpace(mesh, "CG", 1)
u = Function(V, name="u")
bc = DirichletBC(V, as_vector((0.0, 0.0)), "on_boundary")
f = Constant((-0.5, -0.5))
energy = (0.5 * inner(grad(u), grad(u)) - inner(f, u)) * dx
lower = as_vector((-0.3, -0.02))
upper = as_vector((0.3, 0.4))

lvpp = LVPP(energy=energy, u=u, bounds=(lower, upper), bcs=bc, verbose=False,
            alpha_parameters={"alpha_max": 100.0})
lvpp.solve(tol=1e-8)

ut = lvpp.u_tilde[0]
for c in range(2):
    comp = Function(V0).interpolate(ut[c])
    print(f"component {c}: min {comp.dat.data_ro.min():.6f} "
          f"max {comp.dat.data_ro.max():.6f}")
print("proximal iterations :", lvpp.proximal_iterations)
print("constraint          :", lvpp.constraints[0])
print("feasibility(u_tilde):", sum(
    assemble(F) for F in lvpp.constraints[0].feasibility_forms(V, ut)))
```

```
component 0: min -0.036723 max 0.000001
component 1: min -0.020000 max 0.000311
proximal iterations : 18
constraint          : BoxConstraint(bilateral: FermiDirac(phi1, phi2))
feasibility(u_tilde): 0.0
```

Component 0 stays strictly inside its bounds and component 1 is driven exactly
onto its own lower bound `-0.02`.  The schedule is widened to `alpha_max = 100`
here because the two-sided problem has a component whose latent variable stays
in the interior, and a small cap makes the interior converge slowly (the same
solve takes 69 proximal iterations with the default `alpha_max = 10`).

Several constraints may act on one unknown: the entry for that unknown becomes
a list.  The next script has the smooth bump as an obstacle and a gradient
bound `|grad u| <= 3`, which is the `GradientConstraint` of
`tests/test_lvpp.py`:

```python
import numpy as np
import ufl
from firedrake import *
from lvpp import BoxConstraint, Constraint, Hellinger, LVPP


class GradientConstraint(Constraint):
    """|grad u| <= phi, via the Hellinger entropy (paper Sec. 4.1)."""

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


mesh = UnitIntervalMesh(201)
V = FunctionSpace(mesh, "CG", 1)
u = Function(V, name="u")
x = SpatialCoordinate(mesh)[0]
lo, hi = 0.2, 0.8
bump = exp(-1.0 / (10.0 * (x - lo) * (hi - x))) \
    / exp(-1.0 / (10.0 * (0.5 - lo) * (hi - 0.5)))
obstacle = Function(V, name="phi0").interpolate(
    conditional(le(x, lo), 0.0, conditional(ge(x, hi), 0.0, bump)))
bc = DirichletBC(V, 0.0, "on_boundary")
energy = 0.5 * inner(grad(u), grad(u)) * dx

lvpp = LVPP(energy=energy, u=u,
            bounds=[BoxConstraint(lower=obstacle), GradientConstraint(Constant(3.0))],
            bcs=bc, alpha_schedule="constant", alpha_parameters={"C": 1.0},
            on_newton_failure="reduce_alpha", max_consecutive_failures=50,
            solver_parameters={"snes_linesearch_type": "l2",
                               "snes_linesearch_maxlambda": 1.0,
                               "snes_atol": 1.0e-6},
            verbose=False)
lvpp.solve(tol=1e-6)

print("proximal iterations:", lvpp.proximal_iterations)
print("latent blocks      :", len(lvpp.psi_out), "on",
      len(lvpp.u_out), "unknown(s)")
feas = sum(assemble(F) for F in
           lvpp.constraints[0].feasibility_forms(V, lvpp.u_tilde[0]))
feas += sum(assemble(F) for F in
            GradientConstraint(Constant(3.0)).feasibility_forms(V, lvpp.u_out[0]))
print("feasibility        :", feas)
print("peak(u_h)          :", lvpp.u_out[0].dat.data_ro.max())
xs = V.mesh().coordinates.dat.data_ro
du = np.diff(lvpp.u_out[0].dat.data_ro) / np.diff(xs)
print("max|du/dx|         :", abs(du).max())
```

```
proximal iterations: 16
latent blocks      : 2 on 1 unknown(s)
feasibility        : 0.0
peak(u_h)          : 0.999923603027891
max|du/dx|         : 2.187031603793071
```

The unknown has two latent blocks, one per constraint.  The obstacle is
engaged at the bump peak (`u_h = 0.99992`), and the gradient bound is satisfied
(`2.19 < 3`) though not active at this bound; the example reproduces the
`ProximalGalerkin` intersecting-constraints problem on the `lvpp` API.

For several unknowns the bounds list carries one entry per unknown, and an
entry of `None` leaves that unknown unconstrained and without a latent block.
The second unknown below has no gradient term, so its equation is `w = -u`:

```python
from firedrake import *
from lvpp import LVPP

mesh = UnitSquareMesh(16, 16)
V = FunctionSpace(mesh, "CG", 1)
u = Function(V, name="u")
w = Function(V, name="w")
x, y = SpatialCoordinate(mesh)
psi = Function(V, name="psi").interpolate(
    0.3 - 4.0 * ((x - 0.5)**2 + (y - 0.5)**2))
bc = DirichletBC(V, 0.0, "on_boundary")
energy = (0.5 * inner(grad(u), grad(u)) + 0.5 * w * w + u * w) * dx

lvpp = LVPP(energy=energy, u=[u, w], bounds=[(psi, None), None],
            bcs=bc, verbose=False)
lvpp.solve(tol=1e-9)

print("unknowns / latent blocks:", len(lvpp.u_out), "/", len(lvpp.psi_out))
print("u_tilde blocks           :", len(lvpp.u_tilde))
print("proximal iterations      :", lvpp.proximal_iterations)
print("feasibility(u_tilde)     :", sum(
    assemble(F) for F in lvpp.constraints[0].feasibility_forms(V, lvpp.u_tilde[0])))
print("norm(w + u)              :", norm(lvpp.u_out[1] + lvpp.u_out[0]))
```

```
unknowns / latent blocks: 2 / 1
u_tilde blocks           : 1
proximal iterations      : 11
feasibility(u_tilde)     : 0.0
norm(w + u)              : 1.3677477840656154e-25
```

Two unknowns, one latent block because only `u` is constrained, and one
`u_tilde` because there is one constraint.

## The alpha parameter

`alpha` is the parameter of the proximal term.  A small `alpha` makes the
subproblem easy and the step short, so the iteration is close to a proximal
step of the original problem; a large `alpha` makes the subproblem close to the
original problem and the step long.  The sequence `alpha_k` is the paper's
freedom, and the library supplies five rules, all in `lvpp.schedules`:

| name | rule | parameters (defaults) |
|---|---|---|
| `constant` | `alpha_k = C` | `C` (1.0) |
| `geometric` | `alpha_k = C r^k` | `C` (1.0), `r` (1.5) |
| `linear` | `alpha_1 = alpha0`, then `min(c alpha_{k-1}, C_max)` | `alpha0` (1.0), `c` (1.5), `C_max` (1e10) |
| `double_exponential` | eq. (3.8) of LVPP | `C` (1.0), `r` (1.5), `q` (1.5), `alpha_max` (10.0) |
| `newton_adaptive` | `2 alpha_prev` after at most 4 Newton iterations, `alpha_prev/2` after 10 or more, held otherwise | `alpha0` (1.0), `alpha_max` (1e6) |

`double_exponential` is `min(max(C r^(q^k) - alpha_prev, C), alpha_max)`, and it
is evaluated in log space: the direct expression overflows a Python float near
`k = 19` for the default `r = q = 1.5`.  `newton_adaptive` scales by the Newton
count of the previous proximal solve, which is the fracture heuristic.

A rule is either a name, tuned through `alpha_parameters`, or any callable
`f(k, alpha_prev, newton_its) -> alpha`, in which case `alpha_parameters` is
ignored.  `k` is the 1-based proximal iteration, `alpha_prev` is the parameter
of the previous step (0.0 before the first), and `newton_its` is the Newton
count of the previous proximal solve.  `make_schedule(name, params)` resolves a
name or a callable and `describe(schedule)` reproduces the verbose line.

```python
from math import sqrt

from firedrake import *
from lvpp import LVPP, AlphaPlateau, make_schedule
from lvpp.schedules import ALPHA_RULE_DEFAULTS

for name, params in ALPHA_RULE_DEFAULTS.items():
    print(f"{name:18s} {params}")

sched = make_schedule("linear", {"alpha0": 2**-7, "c": sqrt(2), "C_max": 2**-3})
a, seq = 0.0, []
for k in range(1, 9):
    a = sched(k, a, 0)
    seq.append(a)
print("linear values:", [f"{v:.5f}" for v in seq])

mesh = UnitSquareMesh(16, 16)
V = FunctionSpace(mesh, "CG", 1)
u = Function(V, name="u")
x, y = SpatialCoordinate(mesh)
psi = Function(V, name="psi").interpolate(
    0.3 - 4.0 * ((x - 0.5)**2 + (y - 0.5)**2))
bc = DirichletBC(V, 0.0, "on_boundary")
energy = (0.5 * inner(grad(u), grad(u)) - Constant(1.0) * u) * dx


def ramp(k, alpha_prev, newton_its):
    return min(2.0 * alpha_prev, 5.0) if k > 1 else 2.0**-7


lvpp = LVPP(energy=energy, u=u, bounds=(psi, None), bcs=bc,
            alpha_schedule=ramp, increment_norm="H1", verbose=False)
lvpp.solve(tol=1e-8)
print("custom callable: prox", lvpp.proximal_iterations,
      "alpha values", [f"{v:.5f}" for v in lvpp.alpha_history[:6]])

u2 = Function(V, name="u2")
plateau = LVPP(energy=(0.5 * inner(grad(u2), grad(u2)) - Constant(1.0) * u2) * dx,
               u=u2, bounds=(psi, None), bcs=bc,
               alpha_schedule="linear",
               alpha_parameters={"alpha0": 2**-7, "c": sqrt(2), "C_max": 2**-3},
               stopping=AlphaPlateau(), increment_norm="H1", verbose=False)
plateau.solve(tol=1e-8)
print("Linear + AlphaPlateau: prox", plateau.proximal_iterations,
      "stopping", plateau.stopping_description,
      "last alpha", f"{plateau.alpha:.5f}",
      "last |du|_H1", f"{plateau.history['primal_increment'][-1]:.3e}")
```

```
constant           {'C': 1.0}
geometric          {'C': 1.0, 'r': 1.5}
linear             {'alpha0': 1.0, 'c': 1.5, 'C_max': 10000000000.0}
double_exponential {'C': 1.0, 'r': 1.5, 'q': 1.5, 'alpha_max': 10.0}
newton_adaptive    {'alpha0': 1.0, 'alpha_max': 1000000.0}
linear values: ['0.00781', '0.01105', '0.01563', '0.02210', '0.03125', '0.04419', '0.06250', '0.08839']
custom callable: prox 16 alpha values ['0.00781', '0.01562', '0.03125', '0.06250', '0.12500', '0.25000']
Linear + AlphaPlateau: prox 10 stopping default last alpha 0.12500 last |du|_H1 6.087e-02
```

In practice the choice is driven by the contact set.  Where the solution sits
on the constraint over a region, the latent variable drifts towards minus
infinity at the rate `alpha` times the local multiplier, and the latent
diagonal of the Jacobian loses those rows; once `|psi|` approaches floating
point saturation the Newton system becomes hard.  Keeping `alpha_max` modest
for such problems, terminating on a moderate `tol`, or stabilizing Newton on
the preconditioner side (Section 7) are the three remedies the library
supports.

The examples of the hpG paper use a capped geometric ramp rather than the
double-exponential rule, with `alpha_1 = 2^-7` and
`alpha_{k+1} = min(sqrt(2) alpha_k, 2^-3)` and termination on the alpha
plateau.  That is `alpha_schedule="linear"` with `alpha0 = 2**-7`,
`c = sqrt(2)`, `C_max = 2**-3`, together with `stopping=AlphaPlateau()`, and
`increment_norm="H1"` for the exit test.  The run above stops at the plateau
(step 10, `alpha = 0.125`) with the H1 primal increment still at `6.1e-2`:
`AlphaPlateau` terminates on the parameter alone, which is what that scheme
intends.

## Solving

`solve(tol=1e-8, max_proximal_iterations=100, warm_start=False)` runs the
proximal iteration until the stopping rule fires and returns `self`.  `tol`
builds the default `PrimalIncrement` rule, which stops when the norm of
`u^k - u^{k-1}` falls below `tol`; `increment_norm` (chosen at construction)
selects the `L2` or the `H1` norm for that quantity and for the reported
increments.  `stopping=` replaces the rule with any object having
`converged(k, alpha, alpha_prev, history)`; `AlphaPlateau` is the other one
shipped.  Exhausting `max_proximal_iterations` raises `LVPPConvergenceError`.

`solver_parameters` is merged over the defaults, and the default is a direct
factorization:

```
{'snes_type': 'newtonls', 'snes_linesearch_type': 'basic', 'snes_rtol': 1e-10,
 'snes_atol': 1e-12, 'snes_stol': 0.0, 'snes_max_it': 200,
 'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
```

The merge order is user `solver_parameters` over the preconditioner's options
over this dict, so an explicit choice is never silently overridden.
`on_newton_failure="reduce_alpha"` restores the previous iterate, halves
`alpha`, and retries the same proximal iteration; the retries per iteration are
bounded, and `max_consecutive_failures` bounds the failed iterations in a row
before `LVPPConvergenceError`.  The default `"raise"` propagates the Newton
failure.  Very tight bounds under strong forcing are what needs
`reduce_alpha`.

A problem may be given by its energy or by its residual.  `residual=` takes one
weak 1-form per unknown, or one form whose test function lives on a
`MixedFunctionSpace` with one subspace per unknown in the order of `u`.  The
two paths produce the same iterate:

```python
from firedrake import *
from lvpp import DEFAULT_SOLVER_PARAMETERS, LVPP

print("defaults:", DEFAULT_SOLVER_PARAMETERS)

mesh = UnitSquareMesh(16, 16)
V = FunctionSpace(mesh, "CG", 1)
x, y = SpatialCoordinate(mesh)
psi = Function(V, name="psi").interpolate(
    0.3 - 4.0 * ((x - 0.5)**2 + (y - 0.5)**2))
bc = DirichletBC(V, 0.0, "on_boundary")

u = Function(V, name="u")
energy = (0.5 * inner(grad(u), grad(u)) - Constant(1.0) * u) * dx
lvpp = LVPP(energy=energy, u=u, bounds=(psi, None), bcs=bc, verbose=False)
lvpp.solve(tol=1e-9)

print("proximal_iterations :", lvpp.proximal_iterations)
print("newton_iterations   :", lvpp.newton_iterations)
print("history keys        :", sorted(lvpp.history))
print("alpha history tail  :", [f"{v:.3e}" for v in lvpp.alpha_history[-3:]])
print("energy()            :", lvpp.energy())
print("feasibility()       :", lvpp.feasibility())
print("complementarity()   :", lvpp.complementarity())
print("dual_feasibility()  :", lvpp.dual_feasibility())
print("|drift| max         :", max(abs(lvpp.drift[0].dat.data_ro)))
print("matrix size         :", lvpp.matrix().petscmat.getSize())
print("snes/ksp/pc         :", type(lvpp.snes).__name__,
      type(lvpp.ksp).__name__, type(lvpp.pc).__name__)

u2 = Function(V, name="u2")
v = TestFunction(V)
residual = (inner(grad(u2), grad(v)) - Constant(1.0) * v) * dx
lvpp2 = LVPP(residual=residual, u=u2, bounds=(psi, None), bcs=bc,
             verbose=False, solver_parameters={"snes_rtol": 1e-11})
lvpp2.solve(tol=1e-9)
print("energy vs residual u_tilde:", errornorm(lvpp.u_tilde[0], lvpp2.u_tilde[0]))
```

```
defaults: {'snes_type': 'newtonls', 'snes_linesearch_type': 'basic', 'snes_rtol': 1e-10, 'snes_atol': 1e-12, 'snes_stol': 0.0, 'snes_max_it': 200, 'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
proximal_iterations : 9
newton_iterations   : [12, 6, 4, 3, 2, 2, 1, 1, 1]
history keys        : ['alpha', 'complementarity', 'dual_feasibility', 'energy', 'feasibility', 'latent_increment', 'newton_iterations', 'primal_increment']
alpha history tail  : ['1.000e+01', '1.000e+01', '1.000e+01']
energy()            : 0.05226436711190467
feasibility()       : 4.249804484540823e-05
complementarity()   : -6.400177603221353e-18
dual_feasibility()  : 0.0
|drift| max         : 25.960797344586535
matrix size         : (578, 578)
snes/ksp/pc         : SNES KSP PC
energy vs residual u_tilde: 8.926153759970253e-15
```

The diagnostics are `energy()` (the user's energy, energy path only),
`feasibility()` (the integral of the pointwise constraint violation, zero
exactly when the iterate is admissible), `complementarity()`, and
`dual_feasibility()`; the last two vanish at the solution of the variational
inequality.  `history` holds one entry per accepted iterate under the keys
`alpha`, `newton_iterations`, `primal_increment`, `latent_increment`,
`energy`, `feasibility`, `complementarity`, `dual_feasibility`;
`proximal_iterations`, `newton_iterations` and `alpha_history` are the same
records as attributes.  The `drift` field is `(psi_prev - psi)/alpha` captured
before the proximal shift, the discrete multiplier of LVPP (2.1); at the fixed
point it persists on the contact set and converges to the exact multiplier, so
it is the field to read a contact pressure from.

`snes`, `ksp` and `pc` expose the PETSc objects of the nonlinear solve,
`matrix()` assembles the true Jacobian `J` at the current iterate (with any
preconditioner-side correction left out), and `install_monitor(fn)` attaches a
KSP monitor to the outer Krylov solver.  The monitor is called once per Krylov
iteration, and a monitor can count the iterations of each linear solve:

```python
from firedrake import *
from lvpp import LVPP


class KrylovCounter:
    """Outer KSP monitor: one entry per linear solve, holding its count."""

    def __init__(self):
        self.per_solve = []

    def __call__(self, ksp, its, rnorm):
        if its == 0:                 # PETSc marks the start of a solve with its = 0
            self.per_solve.append(0)
        else:
            self.per_solve[-1] = its


mesh = UnitSquareMesh(16, 16)
V = FunctionSpace(mesh, "CG", 1)
u = Function(V, name="u")
x, y = SpatialCoordinate(mesh)
psi = Function(V, name="psi").interpolate(
    0.3 - 4.0 * ((x - 0.5)**2 + (y - 0.5)**2))
bc = DirichletBC(V, 0.0, "on_boundary")
energy = (0.5 * inner(grad(u), grad(u)) - Constant(1.0) * u) * dx

lvpp = LVPP(energy=energy, u=u, bounds=(psi, None), bcs=bc,
            preconditioner="schur", verbose=False)
counter = KrylovCounter()
lvpp.install_monitor(counter)
lvpp.solve(tol=1e-9)

print("proximal iterations :", lvpp.proximal_iterations)
print("linear solves       :", len(counter.per_solve))
print("krylov its per solve:", counter.per_solve)
print("max krylov its      :", max(counter.per_solve))
```

```
proximal iterations : 9
linear solves       : 21
krylov its per solve: [3, 3, 3, 4, 6, 5, 6, 6, 8, 7, 8, 9, 6, 8, 9, 6, 9, 9, 9, 9, 9]
max krylov its      : 9
```

The solve took 21 outer Krylov steps, 3 to 9 iterations each; with the default
direct factorization every linear solve is a single `preonly` step.  The
`"schur"` preconditioner appears in this example because its iteration counts
are the ones a monitor is usually installed to watch; it is described next.

## Preconditioning

The default preconditioner is a monolithic sparse direct factorization
(`preonly` + MUMPS LU).  The latent block of the mixed Jacobian is exactly
singular on the contact set, and a direct factorization is the configuration
that is unconditionally correct at every mesh level.

The interface is `lvpp.preconditioners.SaddlePreconditioner`.  A preconditioner
supplies PETSc solver options from `parameters()`; those are merged over the
defaults as `{**DEFAULT_SOLVER_PARAMETERS, **parameters(), **user}`.  When
options cannot express what it needs, it may return a UFL 2-form from
`jacobian_correction(view)`, which is added to the preconditioner Jacobian
`Jp` **only**, never to the operator `J`: the outer proximal iteration is a
contraction that needs exact Newton directions, and an operator-level
correction perturbs those directions.  `install(snes, view)` is the hook for a
Python PC.  The `view` is a read-only `SaddleView` carrying the live mixed
unknown `z`, the live `alpha` constant, the multiplier fields `drift`, the
constraint-to-unknown map, the primal and latent spaces, the lifted Dirichlet
conditions, the true Jacobian, the mesh, and `n_primal`, the offset at which
the latent dofs begin.  Its fields are live objects, so a preconditioner can
cache the reference in `install` and read current values in its own `setUp`,
which PETSc calls once per linear solve.

`preconditioner="schur"` selects the floorless Schur fieldsplit used for large
two-dimensional problems: an upper block factorization whose two diagonal
blocks are solved exactly, with the Schur complement approximated by
`Sp = D - B diag(K) B^T` from PETSc's `selfp`.  The dead rows of `D` are
carried by the coupling term, so no floor is needed.  On the obstacle problem
`u >= 0.3 - 4 ((x - 0.5)^2 + (y - 0.5)^2)` on a `16 x 16` mesh:

```python
from firedrake import *
from lvpp import LVPP

mesh = UnitSquareMesh(16, 16)
V = FunctionSpace(mesh, "CG", 1)
x, y = SpatialCoordinate(mesh)
psi = Function(V, name="psi").interpolate(
    0.3 - 4.0 * ((x - 0.5)**2 + (y - 0.5)**2))
bc = DirichletBC(V, 0.0, "on_boundary")


def solve(**kw):
    u = Function(V, name="u")
    lv = LVPP(energy=(0.5 * inner(grad(u), grad(u)) - Constant(1.0) * u) * dx,
              u=u, bounds=(psi, None), bcs=bc, verbose=False, **kw)
    lv.solve(tol=1e-9)
    return lv


lu = solve()
schur = solve(preconditioner="schur")
print("direct LU : prox", lu.proximal_iterations,
      "newton", sum(lu.newton_iterations),
      "feasibility", f"{lu.feasibility():.3e}")
print("schur     : prox", schur.proximal_iterations,
      "newton", sum(schur.newton_iterations),
      "feasibility", f"{schur.feasibility():.3e}")
print("errornorm(u_tilde LU, u_tilde schur):",
      errornorm(lu.u_tilde[0], schur.u_tilde[0]))
```

```
direct LU : prox 9 newton 32 feasibility 4.250e-05
schur     : prox 9 newton 21 feasibility 4.250e-05
errornorm(u_tilde LU, u_tilde schur): 3.977863295313254e-11
```

The proximal count and the reconstructed solution agree to `4e-11`; the outer
GMRES replaces some of the Newton iterations, which is the point of the
configuration.  A dict is taken verbatim as raw PETSc options, with no
interpretation, and merged over the defaults exactly as given:

```python
from firedrake import *
from lvpp import LVPP

mesh = UnitSquareMesh(16, 16)
V = FunctionSpace(mesh, "CG", 1)
u = Function(V, name="u")
x, y = SpatialCoordinate(mesh)
psi = Function(V, name="psi").interpolate(
    0.3 - 4.0 * ((x - 0.5)**2 + (y - 0.5)**2))
bc = DirichletBC(V, 0.0, "on_boundary")
energy = (0.5 * inner(grad(u), grad(u)) - Constant(1.0) * u) * dx

options = {"ksp_type": "gmres", "ksp_rtol": 1e-8, "ksp_max_it": 100,
           "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
lvpp = LVPP(energy=energy, u=u, bounds=(psi, None), bcs=bc,
            preconditioner=options, verbose=False)
lvpp.solve(tol=1e-9)
print("raw options: prox", lvpp.proximal_iterations,
      "newton", sum(lvpp.newton_iterations),
      "feasibility", f"{lvpp.feasibility():.3e}")
```

```
raw options: prox 9 newton 32 feasibility 4.250e-05
```

`DegeneracyFloor` adds

    sum_j int eps_j(x) psi_j_trial psi_j_test dx,
    eps_j(x) = constant + drift * alpha * |lambda_j(x)|,

to the latent block of `Jp` only, with `lambda = drift` the live multiplier.
It exists for the configurations that invert the latent block, where
`psi -> -infinity` on the contact set makes that block singular; the constant
part removes the exact singularity, and the `drift` part tracks the contact
force so the inactive region is not flooded.  For the Schur configuration the
recommended setting is `constant = 0`, because the coupling term already
carries the dead rows, and the floor is a no-op there.  `on_operator=True` puts
the floor on `J` instead and is a diagnostic, not a knob: it makes the Newton
directions inexact:

```python
from firedrake import *
from lvpp import LVPP, DegeneracyFloor, LVPPConvergenceError, SchurFieldsplit

mesh = UnitSquareMesh(16, 16)
V = FunctionSpace(mesh, "CG", 1)
x, y = SpatialCoordinate(mesh)
psi = Function(V, name="psi").interpolate(
    0.3 - 4.0 * ((x - 0.5)**2 + (y - 0.5)**2))
bc = DirichletBC(V, 0.0, "on_boundary")


def energy(w):
    return (0.5 * inner(grad(w), grad(w)) - Constant(1.0) * w) * dx


u1 = Function(V, name="u1")
lvpp = LVPP(energy=energy(u1), u=u1, bounds=(psi, None), bcs=bc,
            preconditioner=DegeneracyFloor(
                constant=1e-2, solver_parameters=SchurFieldsplit().parameters()),
            verbose=False)
lvpp.solve(tol=1e-9)
print("floor on Jp (schur options): prox", lvpp.proximal_iterations,
      "newton", sum(lvpp.newton_iterations),
      "feasibility", f"{lvpp.feasibility():.3e}")

u2 = Function(V, name="u2")
bad = LVPP(energy=energy(u2), u=u2, bounds=(psi, None), bcs=bc,
           preconditioner=DegeneracyFloor(constant=1e-2, on_operator=True),
           verbose=False)
try:
    bad.solve(tol=1e-4, max_proximal_iterations=10)
    print("floor on the operator: converged (unexpected)")
except (LVPPConvergenceError, ConvergenceError) as exc:
    print("floor on the operator: failed as", type(exc).__name__)
```

```
floor on Jp (schur options): prox 9 newton 21 feasibility 4.250e-05
floor on the operator: failed as ConvergenceError
```

A preconditioner of your own is an object with a callable `parameters()`;
subclassing `PreconditionerBase` supplies no-op defaults for the other hooks,
so an options-only preconditioner is one method:

```python
from firedrake import *
from lvpp import LVPP, PreconditionerBase


class GMRESWithLU(PreconditionerBase):
    """Outer GMRES with a direct factorization as the preconditioner."""

    def parameters(self):
        return {"ksp_type": "gmres", "ksp_rtol": 1e-8, "ksp_max_it": 100,
                "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}


mesh = UnitSquareMesh(16, 16)
V = FunctionSpace(mesh, "CG", 1)
u = Function(V, name="u")
x, y = SpatialCoordinate(mesh)
psi = Function(V, name="psi").interpolate(
    0.3 - 4.0 * ((x - 0.5)**2 + (y - 0.5)**2))
bc = DirichletBC(V, 0.0, "on_boundary")
energy = (0.5 * inner(grad(u), grad(u)) - Constant(1.0) * u) * dx

lvpp = LVPP(energy=energy, u=u, bounds=(psi, None), bcs=bc,
            preconditioner=GMRESWithLU(), verbose=False)
lvpp.solve(tol=1e-9)
print("custom preconditioner:", lvpp.preconditioner.__class__.__name__)
print("prox", lvpp.proximal_iterations,
      "newton", sum(lvpp.newton_iterations),
      "feasibility", f"{lvpp.feasibility():.3e}")
```

```
custom preconditioner: GMRESWithLU
prox 9 newton 32 feasibility 4.250e-05
```

The three hooks are independent: `parameters()`, `jacobian_correction(view)`
for a term on `Jp` alone, and `install(snes, view)` for a Python PC;
`DegeneracyFloor` and `HPGTwoStage` are the two shipped examples that use more
than the first.  `close()` releases a preconditioner's PETSc objects.

## The hpG discretization

`lvpp.hpg` holds the hierarchical proximal Galerkin (hpG) preset of
I. P. A. Papadopoulos, "Hierarchical proximal Galerkin: a fast hp-FEM solver
for variational problems with pointwise inequality constraints",
arXiv:2412.13733 (2026).  It is the same proximal loop and the same saddle
system on a `CG_p x DQ_{p-2}` spectral hierarchical discretization, factored
by the two-stage `P_F` preconditioner of that paper (eq. 4.5).

`HPGDiscretization.uniform(n, p, length=2.0, dim=2)` builds `n` cells per axis
on `[0, length]^dim`; `graded(n, p, ratio, band_center=0.9,
band_halfwidth=0.10, dim=2, length=2.0)` redistributes the same cells
geometrically toward a fine band so that the mesh measures `hmax/hmin = ratio`
(`grading_ratio()` reports it), and `dim=1` or `dim=3` builds interval or
hexahedral cells.  The cells must be tensor products of 1D bases --
intervals, quadrilaterals or hexahedra -- because the latent operator is
factored cell by cell.  The pair is `CG_p` for the unknown and `DQ_{p-2}` in
the spectral (Legendre-modal) variant for the multiplier, which is the inf-sup
stable pairing of hpG Section 4.1; the degree pairing is the one admitted by
the inf-sup stability result of B. Keith and T. M. Surowiec, "Proximal
Galerkin: a structure-preserving finite element method for pointwise bound
constraints", Found. Comput. Math. (2024) (Lemma B.3).  `CG_p` is the same
polynomial space as the paper's hierarchical basis, so the hierarchy is a
choice of basis rather than a different space.

A worked high-order obstacle solve, on the sphere obstacle of
`lvpp/benchmarks.py` (the domain is `[0, 2]^2`, `p = 2`):

```python
import numpy as np
from firedrake import (DirichletBC, Function, SpatialCoordinate, dx, errornorm,
                       grad, inner, sqrt)
from lvpp.benchmarks import psiUFL, uexactUFL
from lvpp.hpg import HPG, HPGDiscretization, HPGTwoStage

FCP = {"quadrature_degree": 20}


def sphere_problem(disc):
    x, y = SpatialCoordinate(disc.mesh)
    r = sqrt(x * x + y * y)
    u = Function(disc.primal, name="u")
    lb = Function(disc.primal, name="lb").interpolate(psiUFL(r))
    bc = DirichletBC(disc.primal, uexactUFL(r), "on_boundary")
    return u, (lb, None), bc, 0.5 * inner(grad(u), grad(u)) * dx, r


def run(disc, **kw):
    u, bounds, bc, energy, r = sphere_problem(disc)
    hp = HPG(disc, u, bounds, bcs=bc, energy=energy,
             alpha_schedule="double_exponential",
             alpha_parameters={"alpha_max": 10.0}, increment_norm="H1",
             verbose=False, form_compiler_parameters=FCP, **kw)
    hp.solve(tol=1e-4, max_proximal_iterations=100)
    err = float(errornorm(uexactUFL(r), hp.u_out[0]))
    its = np.asarray(hp.preconditioner.inner_its, dtype=float)
    return hp, err, its


for name, disc in [("uniform(16,2)", HPGDiscretization.uniform(16, 2)),
                   ("graded(16,2,ratio=25)",
                    HPGDiscretization.graded(16, 2, ratio=25.0))]:
    hp, err, its = run(disc)
    print(f"{name}: grading {disc.grading_ratio():.2f}x, "
          f"primal {disc.primal.dim()} dofs, latent {disc.latent.dim()} dofs")
    print(f"    prox {hp.proximal_iterations}, newton "
          f"{sum(hp.newton_iterations)}, err(u_h) {err:.4e}, "
          f"inner its/apply {its.mean():.2f}")

hp, err, its = run(HPGDiscretization.uniform(16, 2),
                   preconditioner=HPGTwoStage(a_action="gamg"))
print(f"uniform(16,2), a_action='gamg': prox {hp.proximal_iterations}, "
      f"newton {sum(hp.newton_iterations)}, err(u_h) {err:.4e}, "
      f"A-CG its/call {np.mean(hp.preconditioner.a_its):.1f}")

for dim in (1, 3):
    d = HPGDiscretization.uniform(8, 2, dim=dim)
    print(f"dim={dim}: cell {d.mesh.ufl_cell().cellname}, "
          f"primal {d.primal.dim()}, latent {d.latent.dim()}")
```

```
uniform(16,2): grading 1.00x, primal 1089 dofs, latent 256 dofs
    prox 8, newton 22, err(u_h) 1.1919e-03, inner its/apply 8.77
graded(16,2,ratio=25): grading 25.00x, primal 1089 dofs, latent 256 dofs
    prox 6, newton 17, err(u_h) 1.7074e-03, inner its/apply 5.00
uniform(16,2), a_action='gamg': prox 8, newton 22, err(u_h) 1.1919e-03, A-CG its/call 14.7
dim=1: cell interval, primal 17, latent 8
dim=3: cell hexahedron, primal 4913, latent 512
```

`HPG` is a preset, not a second solver: it chooses the spaces and installs
`HPGTwoStage` unless another preconditioner is passed, and every algorithmic
step is `LVPP`'s.  The two choices are independent -- any `LVPP` accepts
`preconditioner=HPGTwoStage()`, and `HPGDiscretization` can be driven with a
direct factorization or any other preconditioner.

`HPGTwoStage(inner_rtol=1e-4, inner_maxit=40, beta=0.0, a_action="lu",
a_rtol=1e-8, a_maxit=60, a_pc="gamg")` implements the sequential `P_F`:
`y = b_psi - B^T A(alpha)^-1 b_u`, an inner GMRES solve
`dpsi = S^-1 y` on the true Schur complement preconditioned by the cellwise
`Shat` Cholesky, and `du = A(alpha)^-1 (b_u - B dpsi)`.  `inner_rtol` and
`inner_maxit` are the inner GMRES tolerance and cap.  `beta` regularizes
`Shat` alone and never enters the operator; the measured-viable default is 0.
The alpha-free primal block is factored once per nonlinear solve and reused at
every step, which is the cached Cholesky of hpG Section 4.4.  `a_action="gamg"`
replaces that factorization with capped CG + AMG on the same alpha-free block,
so no global factorization appears in the apply step; `a_rtol`, `a_maxit` and
`a_pc` tune it.  On the uniform `16 x 16` problem the matrix-free action is
identical to the cached one -- both give `prox 8`, `newton 22` and
`err = 1.1919e-3`, with the A-CG taking 14.7 iterations per call -- which is
the equality the run above reports.  On strongly graded meshes the A-solve
saturates its cap, because AMG is weak on a strongly graded stiffness matrix;
`experiments/checks/check_hpg_matfree.py` records all 569 A-CG calls hitting
the 60-iteration cap at `a_rtol = 1e-8` on a `26.4x` graded mesh, and 60 of 602
at `a_rtol = 1e-6`, with the outer FGMRES still converging in 2-3 iterations.

The two-stage factorization itself is robust under grading.
`experiments/checks/check_hpg_graded.py` takes the chain to `226x` grading and
roughly 12.5 thousand cells with no divergence and bounded iteration counts,
while the Schur fieldsplit configuration of Section 7 diverges on that chain at
about 14.2 thousand degrees of freedom.

High order earns its keep through the error per degree of freedom.  Section 6.2
of hpG reports the obstacle problem's Newton count independent of both `h` and
`p` (24 iterations over its refinement study), and the design notes record the
same solve at `p = 4` on 4225 dofs at error `9.542e-5` against `p = 2` on 1089
dofs at `1.1919e-3` -- roughly a factor of twelve in error for a factor of four
in degrees of freedom.  The discretization is worth its cellwise factorization
when the solution is smooth away from a lower-dimensional free boundary, which
is exactly the situation the graded-mesh builder targets.

## Where the pieces live

| module | contents |
|---|---|
| `lvpp/solver.py` | `LVPP`, the proximal loop (schedule, stopping, SNES, diagnostics) |
| `lvpp/assembly.py` | `ProblemSpec` and `MixedSystem`: spaces, residual, Jacobians, diagnostic forms |
| `lvpp/schedules.py` | alpha schedules and stopping rules (`PrimalIncrement`, `AlphaPlateau`) |
| `lvpp/constraints.py` | `Constraint` interface and `BoxConstraint` |
| `lvpp/legendre.py` | Legendre functions (Shannon, Fermi-Dirac, Hellinger, Gibbs simplex) |
| `lvpp/preconditioners/` | the saddle-point preconditioner interface: `DirectFactorization`, `DegeneracyFloor`, `SchurFieldsplit`, `RawOptions` |
| `lvpp/hpg/` | the hpG preset: spaces, per-cell spectral-Galerkin algebra, two-stage preconditioner |
| `lvpp/benchmarks.py` | analytic sphere-obstacle data |

`examples/` holds two complete drivers: `sphere_lvpp.py`, the uniform
refinement study of the sphere obstacle, and `signorini_lvpp.py`, a contact
problem whose boundary constraint is a `Constraint` subclass.  `tests/` holds
the verification suite.

The design notes are in `LVPP_DESIGN.md`: the measured choices behind the
schedules, the preconditioners and the hpG preset, and the point-by-point
comparison against the hpG paper.  The scripts under `experiments/checks/`
reproduce the measurements reported there.
