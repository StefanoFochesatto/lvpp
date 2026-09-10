"""Self-check: ``lvpp.schedules`` must reproduce the pre-rewrite alpha rules
bit-for-bit.

Compares every resolved schedule against ``lvpp.lvpp._make_alpha_rule`` (the
original closures) over ``k = 1..30``, several ``alpha_prev`` and
``newton_its``, and also compares the descriptions and the
``ALPHA_RULE_DEFAULTS`` table.  Prints the per-case maximum absolute difference
(all must be 0.0) and ``SCHEDULES OK`` on success.

Run:
    PETSC_DIR=... PETSC_ARCH=arch-firedrake-default OMP_NUM_THREADS=1 \
        python lvpp/experiments/rewrite_checks/check_schedules.py
"""

import importlib.util
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from lvpp.schedules import (
    ALPHA_RULE_DEFAULTS,
    AlphaPlateau,
    Constant,
    DoubleExponential,
    Geometric,
    PrimalIncrement,
    describe,
    make_schedule,
)


def _load_legacy():
    """Materialize the pre-rewrite ``lvpp.lvpp`` module from the git baseline.

    That module is deleted on this branch (its content now lives in
    ``lvpp/schedules.py``, ``lvpp/assembly.py``, ``lvpp/solver.py``), but it is
    the golden reference for this check, so read it out of the ``main`` commit.
    It uses relative imports, so it is reassembled as a throwaway package
    alongside the (unchanged) ``constraints`` and ``legendre`` modules.
    """
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(["git", "show", "main:lvpp/lvpp.py"], cwd=root,
                          capture_output=True, text=True, check=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="_legacy_lvpp_"))
    pkg = tmpdir / "legacy_lvpp"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "lvpp.py").write_text(proc.stdout)
    for name in ("constraints.py", "legendre.py"):
        (pkg / name).write_text((root / "lvpp" / name).read_text())
    sys.path.insert(0, str(tmpdir))
    return importlib.import_module("legacy_lvpp.lvpp")


_legacy = _load_legacy()
_ALPHA_RULE_DEFAULTS = _legacy._ALPHA_RULE_DEFAULTS
_make_alpha_rule = _legacy._make_alpha_rule

# --- the comparison grid ---------------------------------------------------
KS = range(1, 31)
ALPHA_PREVS = [0.0, 0.25, 1.0, 3.7, 1e3, 1e10]
NEWTON_ITS = [1, 4, 7, 11]

# --- (rule name, parameter overrides) --------------------------------------
# The default double_exponential case is the one that exercises the overflow
# guard: log_term >= 700 for k >= 19 with r = q = 1.5.
CASES = [
    ("constant", {}),
    ("geometric", {}),
    ("linear", {}),
    ("double_exponential", {}),
    ("double_exponential", {"alpha_max": 1e300, "C": 2.0}),
    ("double_exponential", {"r": 1.25, "q": 2.0}),
    ("newton_adaptive", {}),
    ("geometric", {"C": 0.5, "r": 2.0}),
    ("linear", {"alpha0": 2.0**-7, "c": math.sqrt(2.0), "C_max": 2.0**-3}),
    ("newton_adaptive", {"alpha0": 0.5, "alpha_max": 8.0}),
]


def main():
    assert ALPHA_RULE_DEFAULTS == _ALPHA_RULE_DEFAULTS, "default tables differ"
    print(f"ALPHA_RULE_DEFAULTS == lvpp.lvpp._ALPHA_RULE_DEFAULTS: "
          f"{ALPHA_RULE_DEFAULTS == _ALPHA_RULE_DEFAULTS}")

    total = 0
    for name, params in CASES:
        old_fn, old_desc = _make_alpha_rule(name, params)
        new_fn = make_schedule(name, params)
        worst = 0.0
        for k in KS:
            for alpha_prev in ALPHA_PREVS:
                for newton_its in NEWTON_ITS:
                    old = old_fn(k, alpha_prev, newton_its)
                    new = new_fn(k, alpha_prev, newton_its)
                    assert new == old, (
                        f"{name}{params} k={k} alpha_prev={alpha_prev} "
                        f"newton_its={newton_its}: {new!r} != {old!r}")
                    assert math.isfinite(new), f"{name}{params}: non-finite {new!r}"
                    worst = max(worst, abs(new - old))
                    total += 1
        assert describe(new_fn) == old_desc, (
            f"{name}{params}: describe={describe(new_fn)!r} != {old_desc!r}")
        print(f"{name}{params}  n={len(KS) * len(ALPHA_PREVS) * len(NEWTON_ITS):4d}"
              f"  max|diff|={worst!r}  describe={describe(new_fn)!r}")

    # --- the overflow guard really is the live path at default parameters ---
    default = make_schedule("double_exponential")
    saturated = [default(k, 0.0, 1) for k in range(1, 31)]
    assert saturated[18] == 10.0, saturated[18]          # k = 19: log_term >= 700
    assert all(math.isfinite(v) for v in saturated)
    try:
        naive = 1.0 * 1.5 ** (1.5**19)
    except OverflowError:
        naive = None
    assert naive is None, "naive evaluation unexpectedly survived k=19"
    print("double_exponential default params: naive 1.5**(1.5**19) -> OverflowError, "
          f"log-space value k=19 -> {saturated[18]!r}")

    # --- the `y > 700.0` branch needs k >= 1727 at the default q = 1.5 ------
    worst_big = 0.0
    big_ks = [(1726, 0.0), (1727, 0.0), (2000, 3.7), (10**4, 1e10)]
    for name, params in [("double_exponential", {}),
                         ("double_exponential", {"alpha_max": 1e300})]:
        old_fn, _ = _make_alpha_rule(name, params)
        new_fn = make_schedule(name, params)
        for k, alpha_prev in big_ks:
            for newton_its in NEWTON_ITS:
                old = old_fn(k, alpha_prev, newton_its)
                new = new_fn(k, alpha_prev, newton_its)
                assert new == old, (name, params, k, alpha_prev, old, new)
                assert math.isfinite(new), (name, params, k, alpha_prev, new)
                worst_big = max(worst_big, abs(new - old))
    print("double_exponential y>700.0 branch (k = 1726, 1727, 2000, 10000): "
          f"max|diff|={worst_big!r}")

    # --- pass-through, error message, describe, stopping rules --------------
    custom = lambda k, alpha_prev, newton_its: 1.0 / k    # noqa: E731
    assert make_schedule(custom) is custom
    assert describe(custom) == "custom(<lambda>)"
    assert describe(describe) == "custom(describe)"
    assert make_schedule(describe) is describe
    instance = Geometric(C=0.25, r=1.25)
    assert make_schedule(instance) is instance
    assert make_schedule(instance, {"C": 99.0}) is instance, "params must be ignored"
    assert describe(instance) == "geometric(C=0.25, r=1.25)"
    # `sorted()` is plain ASCII, so 'C' (67) sorts before the lowercase keys.
    assert describe(Constant()) == _make_alpha_rule("constant")[1] == "constant(C=1)"
    assert describe(DoubleExponential()) == _make_alpha_rule("double_exponential")[1] == (
        "double_exponential(C=1, alpha_max=10, q=1.5, r=1.5)")
    assert describe(make_schedule("linear")) == _make_alpha_rule("linear")[1]
    try:
        make_schedule("no_such_rule")
    except ValueError as exc:
        message = str(exc)
        assert "no_such_rule" in message and "callable" in message
        assert all(name in message for name in ALPHA_RULE_DEFAULTS)
        print(f"unknown name -> ValueError({message!r})")
    else:
        raise AssertionError("unknown rule name did not raise ValueError")

    primal = PrimalIncrement(1e-3)
    assert primal.converged(3, 1.0, 0.5, {"primal_increment": [1e-1, 1e-4]}) is True
    assert primal.converged(3, 1.0, 0.5, {"primal_increment": [1e-3]}) is False
    assert primal.converged(3, 1.0, 0.5, {"primal_increment": []}) is False
    assert primal.converged(3, 1.0, 0.5, {}) is False
    assert primal.converged(3, 1.0, 0.5, None) is False
    assert primal.converged(3, 1.0, 0.5, {"primal_increment": None}) is False

    plateau = AlphaPlateau()
    assert plateau.converged(2, 2.0, 2.0, {}) is True
    assert plateau.converged(1, 2.0, 2.0, {}) is False
    assert plateau.converged(5, 2.0, 1.0, {}) is False
    assert AlphaPlateau(target=2.0).converged(4, 2.0, 2.0, {}) is True
    assert AlphaPlateau(target=2.0).converged(4, 1.0, 1.0, {}) is False

    print(f"comparisons: {total}")
    print("SCHEDULES OK")


if __name__ == "__main__":
    main()
