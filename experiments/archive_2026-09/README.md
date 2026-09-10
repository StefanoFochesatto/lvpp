# Session experiments of record (2026-09-03 → 09-10)

All scripts run under the env guard:
`PETSC_DIR=/home/stefano/firedrake/petsc PETSC_ARCH=arch-firedrake-default OMP_NUM_THREADS=1`
with `/home/stefano/firedrake/venv-firedrake/bin/python`.
Findings lab notebook: `../RESEARCH.md`. Rendered reports: `../report.html`, `../pipeline.html`.

| script | question it answers | verdict (measured) |
|---|---|---|
| `matfree_fieldsplit.py` | Is the matfree operator exact? Does an ε-floor on Jp vs J matter? Does additive fieldsplit scale? | Matfree action bit-exact (LU-identical loop). Additive floor sweep: ε ∈ [1e-2, 1e-1] works at L0 only; ε=1e-3 too small; floor on the operator fails everywhere. Growth mechanism first observed: GMRES its 12→20 as ψ deepens, then breakdown. |
| `schur_probe.py` | Does a Schur-fieldsplit PC solve at depth? | YES — 8-point design ladder; winner `schur/upper/selfp/LUS` (MUMPS both blocks, ψ_floor=1e-2): LU-identical at L0-L2. Failure modes mapped (diag→breakdown ksp=-5; loose inner S→DTOL; CG→PC setup fails). |
| `schur_sp_upgrade.py` | Do PETSc-native Sp upgrades cut the outer-GMRES growth? | PCLSC absent in PETSc 3.25; ainv blockdiag==diag (scalar u-block); ladder inconclusive at L0 — superseded by per-cell and eps-schedule experiments. |
| `schur_percell.py` | Does the exact per-cell broken-basis Sp (hierarchical-pG recipe) cut the growth? | REFUTED — bit-identical solves; coupling |C0| ~ 7.3e-4 × |D_floor| (O(h⁴) vs O(h²), subdominant). Coupling-kill control bit-identical proves the Sp was attached. |
| `spectrum_anatomy.py` (+ `spectrum_results.md`) | What does the true Schur spectrum look like? Where does the degeneracy live? | S strictly negative, λ_min −5.6e-2/−1.4e-2/−3.5e-3 (h²-scaled, α-independent), no near-null cluster at any depth. Dead rows 37/177/749. D = −M⊙e^ψ (scaled mass, not diagonal). Floor shifts S by +ε·M. |
| `growth_driver.py` | Floor vs dead rows vs coupling — which drives outer-GMRES growth? | Exact block-LU PC (exact K⁻¹ + exact UNfloored S⁻¹): its = 1 FLAT at every depth → 100% PC-fixable. Floored exact-S reproduces the entire growth curve → the ε-floor is the driver. Naive active-set identity rows: worse (825 its L0, diverges L2). |
| `rankk_correction.py` | Is the floored PC's deficiency low-rank-curable? | REFUTED — deficiency band (σ > 0.1) has rank ≥ 70/241/829 ≫ dead count 45/177/765, σ_k still O(1). Correction exact on span(V) to 1e-17 yet fails at depth. Includes complete PCShell promotion machinery for future variants. |
| `eps_schedule.py` | Is ψ_floor magnitude a tunable knob? | YES, and the optimum is ZERO: ψ_floor=0 on the Schur config is LU-identical with max its 14/29/48 (floored baseline 45/130-560/400-926). Safety band: ε ≤ 1e-5 safe everywhere, 1e-4 knife-edge (non-monotone), ≥1e-3 kills L2. Drift-concentrated floor REJECTED (lands in the bad band). |
| `scale_up.py` | Does the floorless config scale? Is CG+GAMG-on-K viable? | L3 (33k) and L4 (132k) solve with clean O(h²) errors; outer its ~N^0.8 (14/29/48/105/329); MUMPS 14% of L4 wall; RSS 1.7 GB. CG+GAMG-on-K: correct but wall loser (1.6→3.8×; Schur apply invokes K-solve per outer iteration). Saturation corner OK at L2. |
| `amr_health.py` | Does free-boundary AMR reduce degeneracy (Finding 1 prediction)? | REFUTED — band-refined meshes are dead-row FACTORIES: 5169 dead rows at 6861 dofs vs uniform 765 at 8321 (6.8×); max its 323 vs 48; chain breaks at 14k dofs. Dead fraction 7%→50%. Also: raw \|diag(D)\| census is h-unfair on graded meshes (h-fair census: ψ < log tol). |
| `par_smoke.py` | Is the floorless config correct and viable at 2 ranks? | PASS — LU-identical L0/L1, feasible, both use_amat settings safe in parallel. Wall-neutral at these sizes. |
| `grade_pc.py` | What fixes graded-mesh outer-GMRES growth? | diag(K)⁻¹ mismatch hypothesis REFUTED (κ 3.3e3→8.9e3 across 1×→45.6×; ‖Sp−S‖ stays ~1.6e-3). Variant A (graded diag) and C (per-cell coupling) worse than baseline. **Variant B (matfree true-S action + capped inner Krylov, rtol 1e-4, cap 40) WINS: 6/10/15 its on uni/band_l4/band_l5** — nearly flat under 45.6× grading. |
| `promob.py` | Does variant B survive end-to-end promotion? | PASS with honest costs: uniform LU-identical (L2 8/19/8.365e-04); band_l4 161 max-its vs selfp 297; inner-GMRES arm solves band_l5 (8/17, 188 vs 323) and band_l6 records all linear solves converged where selfp diverged (ksp=-5). Costs: ~4.5× slower than selfp on uniform meshes (coldest solve pays 62-70 its); inner CG cap-40 STALLED the band_l5 proximal iteration (GMRES arm solved it) → production inner solver: GMRES. |
| `_tmp_pc_test.py` | scratch (getArray idiom test) | superseded — keep as repro of the PETSc-101 bug |

Harness notes: petsc4py PC callbacks must use `x.getArray(readonly=True).copy()` +
`y.setArray(arr)` (context-manager form raises PETSc error 101 in this petsc4py);
`rel-resid` must be computed as ‖Jx−b‖/‖b‖ (two harnesses initially computed ‖Jx‖/‖b‖).
`scale_up.py` imports `eps_schedule` — both live here, so sibling imports keep working.