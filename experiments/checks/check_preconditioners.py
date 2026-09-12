"""Self-check for :mod:`lvpp.preconditioners`.

(a) prints ``parameters()`` for :class:`DirectFactorization` and
    :class:`SchurFieldsplit`, with and without the AMG option on the (0,0)
    block, and checks the option dicts: preonly + MUMPS-LU by default; schur
    type, upper factorization, ``use_amat=False``, self-preconditioned Schur
    complement; outer GMRES at rtol 1e-6 restarting every 250; no
    ``psi_floor`` entry; and the AMG variant differing from the base dict on
    the (0,0) block only.
(b) exercises :func:`resolve_preconditioner` for None, a name, a raw option
    dict, an instance and an unknown name.
(c) assembles the degeneracy floor on a real mixed quadrilateral problem and
    checks its block structure against ``eps * mass``: the correction touches
    the latent block only and never Jp, ``eps`` is
    ``constant + drift * alpha * |lambda|``, and the ``on_operator`` variant
    hands the same form back for the operator.  The Jp-only rule behind this
    is ``RESEARCH.md`` Finding 1.

Run from the repository root:

  PETSC_DIR=/home/stefano/firedrake/petsc PETSC_ARCH=arch-firedrake-default \
  OMP_NUM_THREADS=1 /home/stefano/firedrake/venv-firedrake/bin/python \
  experiments/checks/check_preconditioners.py
"""

import numpy as np

from firedrake import (Constant, Function, FunctionSpace, MixedFunctionSpace,
                       RectangleMesh, TestFunction, TrialFunction, assemble,
                       derivative, dx, grad, inner, replace, split)

from lvpp.preconditioners import (DegeneracyFloor, DirectFactorization,
                                 RawOptions, SaddleView, SchurFieldsplit,
                                 resolve_preconditioner)

FAILURES = []


def check(label, condition):
    if condition:
        print(f"ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILURES.append(label)


def dense(mat):
    """Dense copy of a firedrake Matrix / petsc4py Mat (CSR -> numpy)."""
    mat = getattr(mat, "petscmat", mat)
    n = mat.getSize()[0]
    ptr, idx, val = mat.getValuesCSR()
    rows = np.repeat(np.arange(n), np.diff(ptr))
    out = np.zeros((n, n))
    out[rows, idx] = val
    return out


# --- (a) parameters -------------------------------------------------------
print("(a) parameters()")
direct = DirectFactorization()
schur = SchurFieldsplit()
schur_gamg = SchurFieldsplit(gamg=True)
print(f"DirectFactorization():        {direct.parameters()}")
print(f"SchurFieldsplit():            {schur.parameters()}")
print(f"SchurFieldsplit(gamg=True):   {schur_gamg.parameters()}")

check("DirectFactorization default is preonly+LU(MUMPS)",
      direct.parameters() == {"ksp_type": "preonly", "pc_type": "lu",
                              "pc_factor_mat_solver_type": "mumps"})
check("DirectFactorization(solver_type=...) is honoured",
      DirectFactorization("superlu_dist").parameters()
      ["pc_factor_mat_solver_type"] == "superlu_dist")

base_params, gamg_params = schur.parameters(), schur_gamg.parameters()
check("schur: schur/upper/use_amat False/selfp",
      (base_params["pc_fieldsplit_type"] == "schur"
       and base_params["pc_fieldsplit_schur_fact_type"] == "upper"
       and base_params["pc_fieldsplit_use_amat"] is False
       and base_params["pc_fieldsplit_schur_precondition"] == "selfp"))
check("schur: both blocks preonly+MUMPS-LU",
      all(base_params[k] == v for k, v in {
          "fieldsplit_0_ksp_type": "preonly",
          "fieldsplit_0_pc_type": "lu",
          "fieldsplit_0_pc_factor_mat_solver_type": "mumps",
          "fieldsplit_1_ksp_type": "preonly",
          "fieldsplit_1_pc_type": "lu",
          "fieldsplit_1_pc_factor_mat_solver_type": "mumps"}.items()))
check("schur: outer GMRES rtol 1e-6 restart 250",
      (base_params["ksp_type"] == "gmres" and base_params["ksp_rtol"] == 1e-6
       and base_params["ksp_gmres_restart"] == 250))
check("schur: floorless config carries no psi_floor option",
      "psi_floor" not in base_params)
check("gamg changes only fieldsplit_0",
      set(gamg_params) - set(base_params)
      == {"fieldsplit_0_ksp_rtol", "fieldsplit_0_ksp_max_it",
          "fieldsplit_0_ksp_converged_reason", "fieldsplit_0_pc_gamg_type",
          "fieldsplit_0_pc_gamg_agg_nsmooths",
          "fieldsplit_0_mg_levels_ksp_type", "fieldsplit_0_mg_levels_pc_type"}
      and {k: v for k, v in gamg_params.items()
           if not k.startswith("fieldsplit_0")}
      == {k: v for k, v in base_params.items()
          if not k.startswith("fieldsplit_0")})
check("gamg: fieldsplit_0 is cg+gamg with inner_rtol/inner_maxit",
      (gamg_params["fieldsplit_0_ksp_type"] == "cg"
       and gamg_params["fieldsplit_0_pc_type"] == "gamg"
       and gamg_params["fieldsplit_0_ksp_rtol"] == 1e-6
       and gamg_params["fieldsplit_0_ksp_max_it"] == 1000
       and SchurFieldsplit(inner_rtol=1e-8, inner_maxit=7,
                           gamg=True).parameters()
       ["fieldsplit_0_ksp_rtol"] == 1e-8))
check("gamg: fieldsplit_1 still MUMPS-LU",
      (gamg_params["fieldsplit_1_pc_type"] == "lu"
       and gamg_params["fieldsplit_1_pc_factor_mat_solver_type"] == "mumps"))
p = schur.parameters()
p["pc_type"] = "none"
check("parameters() returns a fresh dict each call",
      SchurFieldsplit().parameters()["pc_type"] == "fieldsplit")

# --- (b) resolve_preconditioner ------------------------------------------
print("(b) resolve_preconditioner")
check("None -> DirectFactorization",
      isinstance(resolve_preconditioner(None), DirectFactorization))
check("'lu'/'direct' -> DirectFactorization",
      isinstance(resolve_preconditioner("lu"), DirectFactorization)
      and isinstance(resolve_preconditioner("direct"), DirectFactorization))
check("'schur' -> SchurFieldsplit",
      isinstance(resolve_preconditioner("schur"), SchurFieldsplit))
raw = resolve_preconditioner({"ksp_type": "cg", "pc_type": "jacobi"})
check("dict -> RawOptions",
      isinstance(raw, RawOptions)
      and raw.parameters() == {"ksp_type": "cg", "pc_type": "jacobi"})
instance = DegeneracyFloor(constant=1e-2)
check("duck-typed instance returned as-is",
      resolve_preconditioner(instance) is instance)
try:
    resolve_preconditioner("nope")
except ValueError as exc:
    check("unknown name raises ValueError listing valid names",
          all(name in str(exc) for name in ("direct", "lu", "schur")))
    print(f"     ValueError: {exc}")
else:
    check("unknown name raises ValueError", False)

# --- (c) DegeneracyFloor on a real mixed problem --------------------------
print("(c) DegeneracyFloor on a real quad mixed problem")
mesh = RectangleMesh(4, 4, 1.0, 1.0, quadrilateral=True)
V = FunctionSpace(mesh, "CG", 1)
u = Function(V)
energy = 0.5 * inner(grad(u), grad(u)) * dx
bounds = (Function(V).interpolate(0.1), None)   # the lower-bound obstacle case

Z = MixedFunctionSpace([V, V])
z = Function(Z)
alpha = Constant(3.0)
drift = (Function(V),)
# The true Jacobian of the (mixed, current-iterate) energy -- the same
# derivative(F, z, ztr) construction LVPP does; the floor never touches it.
jacobian = derivative(replace(energy, {u: split(z)[0]}), z, TestFunction(Z))
view = SaddleView(z=z, alpha=alpha, drift=drift, constraint_unknowns=(0,),
                  primal_spaces=(V,), latent_spaces=(V,), bcs=(),
                  n_primal=V.dim(), mesh=mesh, jacobian=jacobian)
n0, ndof = V.dim(), Z.dim()
print(f"     V.dim()={n0}, Z.dim()={ndof}, mesh={mesh.num_cells()} quads, "
      f"energy={energy}, bounds={bounds}")

check("disabled floor (0, 0) is a true no-op on Jp",
      DegeneracyFloor(0.0, 0.0).jacobian_correction(view) is None)
check("operator_correction is None unless on_operator=True",
      DegeneracyFloor(0.0, 0.0).operator_correction(view) is None
      and DegeneracyFloor(1e-2).operator_correction(view) is None
      and DegeneracyFloor(1e-2, on_operator=True)
      .operator_correction(view) is not None)
check("floor with alpha/|lambda| both present is not a no-op",
      DegeneracyFloor(1e-2).jacobian_correction(view) is not None)
check("parameters() empty by default (LU default stands)",
      DegeneracyFloor().parameters() == {}
      and DegeneracyFloor(solver_parameters={"ksp_rtol": 1e-8}).parameters()
      == {"ksp_rtol": 1e-8})

mass = dense(assemble(inner(TrialFunction(V), TestFunction(V)) * dx))

form = DegeneracyFloor(constant=1e-2).jacobian_correction(view)
A = dense(assemble(form).petscmat)
check("floor assembles to the full mixed shape",
      A.shape == (ndof, ndof))
check("floor leaves the primal block exactly untouched",
      np.all(A[:n0, :n0] == 0.0) and np.all(A[:n0, n0:] == 0.0)
      and np.all(A[n0:, :n0] == 0.0))
check("latent block == constant * mass matrix",
      np.allclose(A[n0:, n0:], 1e-2 * mass, rtol=1e-12, atol=1e-16))

# lambda(x) = 2 -> eps = 0 + 1.0 * alpha(3) * 2 = 6
drift[0].interpolate(2.0)
B = dense(assemble(DegeneracyFloor(constant=0.0, drift=1.0)
                   .jacobian_correction(view)).petscmat)
check("latent block == (drift*alpha*|lambda|) * mass matrix",
      np.allclose(B[n0:, n0:], 6.0 * mass, rtol=1e-12, atol=1e-16))

# the on_operator branch hands back the same form
C = dense(assemble(DegeneracyFloor(constant=1e-2, on_operator=True)
                   .operator_correction(view)).petscmat)
check("on_operator correction is the same form",
      np.allclose(C, A, rtol=1e-12, atol=1e-16))

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    raise SystemExit(1)
print("all checks passed")
