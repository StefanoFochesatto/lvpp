"""Signorini contact: an elastic block pressed onto a rigid plane.

2-d linear elasticity with unilateral frictionless contact on the bottom
edge, solved with the latent variable proximal point (LVPP) algorithm.
On the potential contact boundary Gamma_C the displacement must satisfy
the Signorini conditions

    u . n_g <= g        (no penetration into the rigid foundation)
    lambda  >= 0        (compressive contact pressure only)
    lambda (g - u.n_g) = 0   (complementarity),

with n_g = (0, -1) and gap g(x) = y - gap_0.  The problem mirrors the
reference implementation ProximalGalerkin/examples/02_signorini
(Figure 4 of arXiv:2503.05672): unit square, E = 2e4, nu = 0.3, top edge
clamped with prescribed downward displacement disp = -0.10, foundation at
gap = 0, alpha_0 = 0.005 with a doubling schedule.

lvpp v1 ships box (volume) constraints only, so the boundary constraint
is implemented below as a small `Constraint` subclass -- the extension
point promised by the lvpp.constraints module docstring ("Signorini
boundary constraints ... follow the same interface").  The class is
self-contained and could be lifted into lvpp.constraints verbatim.

Formulation notes (single-mesh variant of the dolfinx reference, which
puts psi on a submesh of the contact facets -- not available in this
Firedrake build):

  * B u = dot(u, n_g) on Gamma_C, Shannon entropy: the reconstruction
    grad R*(psi) = g - exp(-psi) is < g for every psi, so the latent
    output is feasible by construction and psi -> +inf on the contact
    set (upper-type bound).  The contact pressure is
    lambda = (psi - psi_prev)/alpha >= 0.
  * The latent psi lives on the same scalar space as u's trace.  Its
    state/coupling equations are integrated over Gamma_C only, so
    interior psi dofs receive no equation; a small ridge
    eps*(psi, dw)*dx (eps = 1e-10) is folded into the state form to keep
    the mixed system nonsingular.  On Gamma_C the ridge perturbs the
    state equation by O(eps * psi * h) -- negligible, and it vanishes on
    refinement.  Only the trace of psi (and of u_tilde) on Gamma_C is
    physical.

Run from anywhere except a directory containing an unrelated `lvpp`
folder (the editable install must resolve):

    python signorini_lvpp.py
"""

from firedrake import (Constant, DirichletBC, Function, FunctionSpace,
                       RectangleMesh, SpatialCoordinate, VectorFunctionSpace,
                       as_vector, assemble, conditional, dx, ds, inner,
                       min_value, VTKFile)
import ufl
from lvpp import Constraint, LVPP, ShannonUpper

# --------------------------------------------------------------- parameters
E = 2.0e4          # Young's modulus (reference example default)
NU = 0.3           # Poisson's ratio
DISP = -0.10       # prescribed vertical displacement of the clamped top edge
GAP0 = 0.0         # rigid foundation height; g(x) = y - GAP0
NX = 16            # quadrilateral mesh, NX x NX
DEGREE = 2         # equal-order displacement / latent (Q_k)
CONTACT_ID = 3     # RectangleMesh facet marker for y = 0 (bottom)
TOP_ID = 4         # RectangleMesh facet marker for y = 1 (top)
RIDGE = 1e-10      # interior psi pinning (see module docstring)

mu = E / (2.0 * (1.0 + NU))
lmbda = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))


def epsilon(w):
    return ufl.sym(ufl.grad(w))


def sigma(w, mu, lmbda):
    gdim = w.ufl_shape[0]
    return 2.0 * mu * epsilon(w) + lmbda * ufl.tr(ufl.grad(w)) * ufl.Identity(gdim)


class SignoriniContact(Constraint):
    """Unilateral frictionless contact dot(u, n_g) <= g on a boundary part.

    Boundary analogue of BoxConstraint: the operator is B u = dot(u, n_g)
    evaluated on the potential-contact facets, the feasible image is
    (-inf, g(x)] per point, and the Shannon entropy gives the
    bound-preserving reconstruction grad R*(psi) = g - exp(-psi).

    n_g : UFL constant vector, the obstacle-side normal (e.g. (0, -1)
          for a rigid plane below the body; the body's outward normal
          there points the same way).
    gap : UFL scalar expression g(x) on the mesh.
    contact_id : integer facet marker of the potential contact boundary.
    ridge : weight of the interior psi pinning term, see module docstring.
    """

    def __init__(self, n_g, gap, contact_id, ridge=RIDGE):
        super().__init__(ShannonUpper(gap))
        self.n_g = n_g
        self.gap = gap
        self.contact_id = contact_id

    def _ds_contact(self, V):
        return ufl.Measure("ds", domain=V.mesh(), subdomain_id=self.contact_id)

    def latent_space(self, V):
        """Scalar space matching the trace discretization of V."""
        return FunctionSpace(V.mesh(), "CG", V.ufl_element().degree())

    def observable_space(self, V):
        return self.latent_space(V)

    def observable(self, psi):
        # clamp so exp() cannot overflow on transient Newton undershoots
        # (psi -> +inf on the contact set is the intended direction, and
        # exp(-psi) underflows harmlessly to 0 there)
        return self.legendre.reconstruction(min_value(psi, 700.0))

    def coupling_form(self, V, du, psi, psi_prev):
        """(psi, B du) - (psi_prev, B du) with B du = dot(du, n_g) on Gamma_C."""
        ds_c = self._ds_contact(V)
        return (ufl.inner(psi - psi_prev, ufl.dot(du, self.n_g)) * ds_c)

    def state_form(self, V, u, dw, psi):
        """(Bu, dw) - (grad R*(psi), dw) on Gamma_C, plus the psi ridge."""
        ds_c = self._ds_contact(V)
        return (ufl.inner(ufl.dot(u, self.n_g), dw) * ds_c
                - ufl.inner(self.observable(psi), dw) * ds_c
                + RIDGE * ufl.inner(psi, dw) * dx)

    def dual_feasibility_form(self, V, psi, psi_prev, alpha):
        """Integral of the negative part of dpsi/alpha: psi must climb on
        the contact set (upper-type bound), so a decreasing psi is infeasible."""
        ds_c = self._ds_contact(V)
        dneg = -(psi - psi_prev) / alpha
        return conditional(dneg > 0.0, dneg, 0.0) * ds_c

    def feasibility_forms(self, V, u):
        """Integral of max(0, dot(u, n_g) - g) over Gamma_C (penetration).

        ``u`` is the vector-valued unknown (or its mixed split piece).
        """
        ds_c = self._ds_contact(V)
        pen = ufl.dot(u, self.n_g) - self.gap
        return [conditional(pen > 0.0, pen, 0.0) * ds_c]

    def complementarity_form(self, V, u, psi, psi_prev, alpha):
        """Integral of lambda * (g - dot(u, n_g)) over Gamma_C; >= 0 and
        vanishes at the solution (lambda = contact pressure >= 0)."""
        ds_c = self._ds_contact(V)
        lam = (psi - psi_prev) / alpha
        return (conditional(lam > 0.0, lam, 0.0)
                * (self.gap - ufl.dot(u, self.n_g)) * ds_c)

    def dual_feasibility_form(self, V, psi, psi_prev, alpha):
        """Integral of the negative part of dpsi/alpha: psi must climb on
        the contact set (upper-type bound), so a decreasing psi is infeasible."""
        ds_c = self._ds_contact(V)
        dneg = -(psi - psi_prev) / alpha
        return conditional(dneg > 0.0, dneg, 0.0) * ds_c

    def __repr__(self):
        return f"SignoriniContact(contact_id={self.contact_id})"


# --------------------------------------------------------------- problem
mesh = RectangleMesh(NX, NX, 1.0, 1.0, quadrilateral=True)
V = VectorFunctionSpace(mesh, "CG", DEGREE)
u = Function(V, name="displacement")

x, y = SpatialCoordinate(mesh)
n_g = Constant((0.0, -1.0))          # body's outward normal on Gamma_C
gap = y - GAP0                       # rigid foundation height

u_top = Function(V).interpolate(as_vector([0.0, DISP]))
bc = DirichletBC(V, u_top, TOP_ID)   # top edge clamped, pressed down

energy = 0.5 * inner(sigma(u, mu, lmbda), epsilon(u)) * dx
constraint = SignoriniContact(n_g=n_g, gap=gap, contact_id=CONTACT_ID)

lvpp = LVPP(energy=energy, u=u, bounds=constraint, bcs=bc,
            alpha_rule="linear",
            alpha_parameters={"alpha0": 0.005, "c": 2.0, "C_max": 1e3},
            on_newton_failure="reduce_alpha",
            form_compiler_parameters={"quadrature_degree": 6},
            solver_parameters={"snes_rtol": 1e-8, "snes_max_it": 100},
            name="signorini")

lvpp.solve(tol=1e-6, max_proximal_iterations=60)

# --------------------------------------------------------------- diagnostics
u_h = lvpp.u_out[0]
u_tilde = lvpp.u_tilde[0]            # reconstructed u.n_g = g - exp(-psi); its
# trace on Gamma_C is the physical gap
psi = lvpp.psi_out[0]

# contact pressure lambda = (psi - psi_prev)/alpha >= 0 on Gamma_C.
# lvpp.drift holds (psi_prev - psi)/alpha captured BEFORE the proximal shift
# (post-solve psi_prev equals psi, so the raw difference is identically zero);
# with the ShannonUpper sign convention psi climbs on the contact set, hence
# lambda = -drift.  For the same reason the complementarity/dual values are
# read from the last pre-shift history entry, not from the post-solve methods.
lam = Function(lvpp.psi_out[0].function_space(), name="contact_pressure")
lam.interpolate(-lvpp.drift[0])

ds_c = ds(domain=mesh, subdomain_id=CONTACT_ID)
ds_t = ds(domain=mesh, subdomain_id=TOP_ID)

penetration = sum(assemble(f) for f in constraint.feasibility_forms(V, u_h))
penetration_tilde = assemble(
    conditional(u_tilde - gap > 0.0, u_tilde - gap, 0.0) * ds_c)
complementarity = lvpp.history["complementarity"][-1]
dual = lvpp.history["dual_feasibility"][-1]

# vertical equilibrium: contact force vs. reaction on the clamped top edge
# (lateral sides are traction-free only approximately in the discrete sense)
contact_force = assemble(lam * ds_c)
top_reaction = assemble(sigma(u_h, mu, lmbda)[1, 1] * ds_t)

contact_measure = assemble(Constant(1.0) * ds_c)
contact_fraction = assemble(conditional(lam > 1e-8 * max(contact_force, 1e-30),
                                        Constant(1.0), Constant(0.0)) * ds_c) \
    / contact_measure

print(f"penetration(u_h)    : {penetration:.3e}")
print(f"penetration(u_tilde): {penetration_tilde:.3e}")
print(f"complementarity     : {complementarity:.3e}")
print(f"dual feasibility    : {dual:.3e}")
print(f"contact fraction    : {contact_fraction:.3f}")
print(f"contact force       : {contact_force:.6e}")
print(f"top reaction        : {top_reaction:.6e}")
print(f"equilibrium residual: {contact_force + top_reaction:.3e}")

VTKFile("signorini_lvpp.pvd").write(u_h, lam, u_tilde)

# --------------------------------------------------------------- assertions
assert lvpp.proximal_iterations < 60, "no proximal convergence"
# headline LVPP property: the reconstruction is feasible BY CONSTRUCTION
assert penetration_tilde < 1e-14, f"u_tilde penetrates: {penetration_tilde:.3e}"
# the raw iterate satisfies the constraint up to the L2-projection artifact
assert penetration < 1e-3, f"u_h penetrates too much: {penetration:.3e}"
assert complementarity < 1e-6, f"complementarity residual: {complementarity:.3e}"
assert dual < 1e-6, f"dual feasibility residual: {dual:.3e}"
assert contact_force > 0, "contact force must be compressive"
assert abs(contact_force + top_reaction) < 5e-2 * contact_force, (
    f"vertical equilibrium residual too large: {contact_force + top_reaction:.3e}")

print("SIGNORINI EXAMPLE OK")
