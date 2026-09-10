# LVPP saddle spectrum anatomy (sphere obstacle)

Measured on the assembled monolithic Jacobian J = [[K, B], [B^T, D]] with K = alpha*stiffness+bc, B = M (mass), D = -M*exp(psi) (ShannonLower map; psi -> -inf on the contact set).

Schur S = D - B^T K^{-1} B applied matrix-free (K via PETSc LU, one factorization per state); extremes reported as the outer bound of scipy ARPACK eigsh 'BE' (partial sets kept when one end stalls in a cluster) and a 120-step full-reorthogonalization Lanczos run; the near-zero count n(|lambda_S| < 1e-8 * t_scale) uses the Lanczos Ritz values (inertia-free estimate), t_scale = median K diagonal.

Floored columns: D' = D + 1e-2 (diagonal approximation of the psi_floor regularization; the lvpp floor form eps*inner(trial_phi,test_phi)dx is assembled per level and its off-diagonal norm reported in the log -- it is a full mass matrix, not lumped).

| level | state | note | phi_min | phi_max | alpha | d_min (|d| closest 0) | d_med | n(|d|<1e-8) | n(|d|<1e-12) | K diag med | lambda_S_min | lambda_S_max | n(|lambda_S|<1e-8 t) | floored d_min' | floored lambda_S_min' | floored lambda_S_max' |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | early | latent := psiUFL(r) | -3.54581 | 1 | 1 | 1.739e-04 | 0.00538732 | 0 | 0 | 4 | -0.0660023 | -1.755e-04 | 0 | 3.509e-05 | -0.0560023 | 0.00982455 |
| 0 | 3prox | 3 proximal its (LVPPConvergenceError) | -32.5895 | 1.19816 | 2.92923 | 2.033e-08 | 0.0165828 | 0 | 0 | 11.7169 | -0.0558271 | -3.361e-06 | 0 | 6.422e-04 | -0.0458271 | 0.00999664 |
| 0 | converged | converged | -425.009 | 5.98596 | 10 | 6.363e-28 | 0.0165827 | 45 | 37 | 40 | -0.0557552 | -1.962e-06 | 0 | 6.050e-04 | -0.0457552 | 0.00999803 |
| 1 | early | latent := psiUFL(r) | -3.54581 | 1 | 1 | 4.041e-05 | 0.00135113 | 0 | 0 | 4 | -0.0150233 | -4.051e-05 | 0 | 2.245e-05 | -0.00502335 | 0.00995949 |
| 1 | 3prox | 3 proximal its (LVPPConvergenceError) | -25.6087 | 1.19726 | 2.92923 | 7.535e-11 | 0.00437527 | 72 | 0 | 11.7169 | -0.0139276 | -6.426e-07 | 0 | 5.084e-05 | -0.00392761 | 0.00999936 |
| 1 | converged | converged | -763.277 | 1.19726 | 10 | 4.929e-80 | 0.00436383 | 177 | 177 | 40 | -0.0139191 | -2.862e-07 | 1 | 5.225e-05 | -0.00391905 | 0.00999971 |
| 2 | early | latent := psiUFL(r) | -3.54581 | 1 | 1 | 9.739e-06 | 3.497e-04 | 0 | 0 | 4 | -0.00359436 | -9.746e-06 | 0 | 0.00646265 | 0.00640564 | 0.00999025 |
| 2 | 3prox | 3 proximal its (LVPPConvergenceError) | -29.9132 | 1.19704 | 2.92923 | 1.605e-11 | 0.00110882 | 656 | 0 | 11.7169 | -0.00347931 | -6.859e-08 | 1 | 0.00652157 | 0.00652069 | 0.00999993 |
| 2 | converged | converged | -317.556 | 1.19704 | 10 | 1.685e-69 | 0.00110603 | 765 | 749 | 40 | -0.00347831 | -3.669e-08 | 1 | 0.00652194 | 0.00652169 | 0.00999996 |
