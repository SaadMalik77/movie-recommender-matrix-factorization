"""
Simulation test of all three methods: recover the known truth.

Data: the report's Simulation Check models (simulate.py).
  Factor model  a true rank-3 matrix of 100 users × 80 movies, noise sd 0.1, 30% of
                the entries observed. The hidden entries of R_true are the truth.
  Bias model    r_ui = µ + b_u + c_i + noise on the same observed entries.
  Error model   N = 20,000 prediction errors from N(0, 0.9²), repeated 2,000 times.

Checks:
  1. Theorem 4     ALS: J never increases, over every half-step.
  2. Truth         ALS: the error on the hidden entries is small (RMSE < 0.3), and
                   the error on the known entries is close to the noise sd.
  3. Library       ALS: every row update solved by our Gaussian elimination agrees
                   with numpy.linalg.solve on the same system (Proposition 1), to 1e-10.
  4. Method 3      truncated SVD by power iteration: singular values and vectors agree
                   with numpy.linalg.svd to 1e-8, the error equals the tail sum
                   (Proposition 5), and the filled SVD predicts the hidden entries much
                   worse than ALS.
  5. Sample size   ALS: the error on the hidden entries shrinks as the matrix grows.
  6. Stress test   ALS: lower the observed fraction until ALS stops recovering R_true.
  7. Method 1      bias baseline: agrees with numpy.linalg.lstsq on the stacked system
                   (Proposition 6), B never increases (Remark 7), biases recovered.
  8. Prop. 8       the 95% interval contains the true MSE (and RMSE) about 95% of the time.
  9. Assumptions   all seven assumptions (assumptions.py). They hold here by
                   construction, so this shows what "passing" looks like before the
                   same checks run on MovieLens (run_real.py).
At the end it prints the rows of the verification table.

Run from the repository root:   python src/check_simulation.py
Figures are saved to figures/.

Owners: Muhammad Saad Malik (ALS: 1, 2, 3, 5, 6; Assumptions 1, 3, 4, 5)
        Jagadeesh Aravind Kalyan Shetti (truncated SVD: 4; Assumption 7)
        Emmanuel Adjei Danso (bias baseline and error bars: 7, 8; Assumptions 2, 6)
"""

# %% Setup
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from methods import (my_centre, my_normal_equations, my_gauss_solve, my_als,
                     my_predict, my_rmse, report,
                     my_fill, my_svd_power, my_svd_predict,
                     my_fit_biases, my_predict_biases, my_rmse_ci)
from simulate import simulate_ratings, simulate_bias_ratings, simulate_test_errors
from assumptions import (min_eig_G, print_checks, print_residual_table, plot_checks,
                         popularity_check, ci_vs_spread, singular_gaps,
                         print_extra_checks, plot_extra_checks)

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

K = 3            # true rank, and the rank we fit
LAM = 1.0        # ALS regularization strength λ
LAM_B = 1.0      # bias regularization strength λ_b
NOISE_SD = 0.1   # noise sd used by simulate_ratings
N_SWEEPS = 50    # ALS sweeps
SVD_TOL = 1e-10  # power iteration stops when ‖Δx‖ < SVD_TOL
SEED = 0         # same seed as the report's Simulation Check

table = []       # rows of the verification table


def fit(R, mask, k=K, lam=LAM, iters=N_SWEEPS, seed=SEED):
    """Method 2, Steps 1–7: centre, run ALS, predict µ + U Vᵀ."""
    mu, R_c = my_centre(R, mask)                                       # Step 1
    U, V, J_hist = my_als(R_c, mask, k, lam, iters=iters, seed=seed)  # Steps 2–6
    return mu, R_c, U, V, J_hist, my_predict(U, V, mu)                 # Step 7


# %% Data with known truth
R, R_true, mask = simulate_ratings(seed=SEED)
m, n = R.shape
print(f"R is {m} × {n}, true rank {K}, noise sd {NOISE_SD}")
print(f"Known entries |Ω| = {mask.sum()} of {m * n} ({mask.mean():.1%}); "
      f"fewest per user {mask.sum(axis=1).min()}, per movie {mask.sum(axis=0).min()}\n")

mu, R_c, U, V, J_hist, P = fit(R, mask)
print(f"µ = {mu:.4f};  J went from {J_hist[0]:.2f} to {J_hist[-1]:.2f} "
      f"over {N_SWEEPS} sweeps ({len(J_hist) - 1} half-steps)\n")


# %% 1. Theorem 4: J never increases
changes = np.diff(J_hist)
n_up = int(np.sum(changes > 1e-9 * J_hist[:-1]))
print(f"1. Theorem 4   half-steps where J increased: {n_up} of {len(changes)};"
      f"  largest change {changes.max():.2e}")
table.append(("Simulation", "J after each half-step", "previous J",
              f"{n_up} increases", n_up == 0))

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(np.arange(len(J_hist)) / 2, J_hist, marker=".", color="steelblue")
ax.set_yscale("log")
ax.set_xlabel("ALS sweep")
ax.set_ylabel("Objective J(U, V) (log scale)")
ax.set_title("The ALS objective never increases")
fig.tight_layout()
fig.savefig(FIG_DIR / "als_objective_simulation.png", dpi=150)


# %% 2. Truth: error on the hidden and the known entries
rmse_hidden = my_rmse(P, R_true, ~mask)
rmse_known = my_rmse(P, R, mask)
print(f"\n2. Truth       RMSE on hidden entries vs R_true: {rmse_hidden:.4f}  (pass: < 0.3)")
print(f"               RMSE on known entries vs R:       {rmse_known:.4f}  (noise sd {NOISE_SD})")
table.append(("Simulation", "ALS r̂ on hidden entries", "R_true", f"RMSE = {rmse_hidden:.3f}",
              rmse_hidden < 0.3))


# %% 3. Library: our Gaussian elimination against numpy.linalg.solve
G_0, h_0 = my_normal_equations(V[mask[0]], R_c[0, mask[0]], LAM)
print("\n3. Library (ALS)")
report("   user 1 update vs numpy.linalg.solve",
       my_gauss_solve(G_0, h_0), np.linalg.solve(G_0, h_0))
diff_all = 0.0
for u in range(m):
    G, h = my_normal_equations(V[mask[u]], R_c[u, mask[u]], LAM)
    diff_all = max(diff_all, np.abs(my_gauss_solve(G, h) - np.linalg.solve(G, h)).max())
for i in range(n):
    G, h = my_normal_equations(U[mask[:, i]], R_c[mask[:, i], i], LAM)
    diff_all = max(diff_all, np.abs(my_gauss_solve(G, h) - np.linalg.solve(G, h)).max())
print(f"   all {m} user and {n} movie updates      max|diff| = {diff_all:.2e}  (pass: < 1e-10)")
table.append(("Simulation", "every ALS row update", "numpy.linalg.solve",
              f"max|·| = {diff_all:.0e}", diff_all < 1e-10))


# %% 4. Method 3: truncated SVD of the filled matrix, by power iteration
# Step 1 fills each missing entry with its movie's mean known rating; Step 2 finds
# the top K singular triples (one extra triple is kept for Assumption 7).
R_fill = my_fill(R, mask, how="movie")                                    # Step 1
sig, W, Z, svd_iters = my_svd_power(R_fill, K + 1, tol=SVD_TOL, seed=SEED)  # Step 2
W_lib, s_lib, Zt_lib = np.linalg.svd(R_fill, full_matrices=False)       # check only
sgn = np.sign(np.sum(Z[:, :K] * Zt_lib[:K].T, axis=0))                  # align signs
print("\n4. Method 3 (truncated SVD by power iteration)")
d_sig = report(f"   σ_1..σ_{K + 1} vs numpy.linalg.svd", sig, s_lib[:K + 1])
d_z = report("   z_j (signs aligned)", Z[:, :K] * sgn, Zt_lib[:K].T)
d_w = report("   w_j (signs aligned)", W[:, :K] * sgn, W_lib[:, :K])
P_svd = my_svd_predict(sig, W, Z, K)                                     # Step 4
err_k = np.sum((R_fill - P_svd) ** 2)
tail = np.sum(s_lib[K:] ** 2)
print(f"   ‖R̃ − R̃_{K}‖_F² = {err_k:.4f},  Σ_(j>{K}) σ_j² = {tail:.4f}  (Proposition 5)")
print(f"   power-iteration steps per factor: {[int(t) for t in svd_iters[:K]]}")
rmse_svd = my_rmse(P_svd, R_true, ~mask)
P_svd_zero = my_svd_predict(*my_svd_power(my_fill(R, mask, how="zero"), K, tol=SVD_TOL, seed=SEED)[:3])
rmse_svd_zero = my_rmse(P_svd_zero, R_true, ~mask)
print(f"   RMSE on hidden entries: ALS {rmse_hidden:.4f}, SVD movie-mean fill {rmse_svd:.4f} "
      f"({rmse_svd / rmse_hidden:.1f}× worse), SVD zero fill {rmse_svd_zero:.4f}")
table.append(("Simulation", "σ_j (power iteration)", "numpy.linalg.svd",
              f"max|·| = {d_sig:.0e}", d_sig < 1e-8))
table.append(("Simulation", "z_j, w_j (signs aligned)", "numpy.linalg.svd",
              f"max|·| = {max(d_z, d_w):.0e}", max(d_z, d_w) < 1e-8))
table.append(("Simulation", "‖R̃ − R̃_k‖_F²", "Σ_(j>k) σ_j²",
              f"rel. {abs(err_k - tail) / tail:.0e}", abs(err_k - tail) / tail < 1e-8))
table.append(("Simulation", "filled-SVD on hidden", "ALS on hidden",
              f"{rmse_svd:.3f} vs {rmse_hidden:.3f}", rmse_svd > 3 * rmse_hidden))


# %% 5. Sample size: error on the hidden entries as the matrix grows
sizes = [(50, 40), (100, 80), (200, 160), (400, 320)]
n_obs, errs = [], []
print("\n5. Sample size")
print(f"   {'m × n':>10} {'|Ω|':>8} {'hidden RMSE':>12}")
for (ms, ns) in sizes:
    Rs, Rs_true, Ms = simulate_ratings(m=ms, n=ns, seed=SEED)
    *_, Ps = fit(Rs, Ms)
    n_obs.append(int(Ms.sum()))
    errs.append(my_rmse(Ps, Rs_true, ~Ms))
    print(f"   {f'{ms} × {ns}':>10} {n_obs[-1]:>8} {errs[-1]:>12.4f}")
slope = np.polyfit(np.log(n_obs), np.log(errs), 1)[0]
print(f"   log–log slope of error against |Ω|: {slope:.2f}")

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.loglog(n_obs, errs, marker="o", color="steelblue")
ax.set_xlabel("Known ratings |Ω| (log scale)")
ax.set_ylabel("RMSE on hidden entries (log scale)")
ax.set_title(f"Error against sample size (slope {slope:.2f})")
fig.tight_layout()
fig.savefig(FIG_DIR / "als_error_vs_size.png", dpi=150)
table.append(("Simulation (4 sizes)", "ALS hidden RMSE", "|Ω|", f"slope {slope:.2f}", slope < 0))


# %% 6. Stress test: fewer and fewer known ratings
# A rank-3 100 × 80 matrix has k(m + n − k) = 531 free numbers, so with fewer
# known ratings than that it cannot be pinned down. "Mean only" predicts µ
# everywhere: an ALS error above it means ALS has stopped recovering R_true.
fracs = [0.3, 0.2, 0.15, 0.1, 0.07, 0.05]
dof = K * (m + n - K)
stress = []
print(f"\n6. Stress test   (free numbers in a rank-{K} {m} × {n} matrix: {dof})")
print(f"   {'observed':>8} {'|Ω|':>6} {'min per user':>13} {'ALS':>8} {'mean only':>10}  recovers?")
for f in fracs:
    Rs, Rs_true, Ms = simulate_ratings(frac=f, seed=SEED)
    mu_s, *_, Ps = fit(Rs, Ms)
    e_als = my_rmse(Ps, Rs_true, ~Ms)
    e_mean = my_rmse(np.full_like(Rs_true, mu_s), Rs_true, ~Ms)
    stress.append((f, int(Ms.sum()), int(Ms.sum(axis=1).min()), e_als, e_mean))
    print(f"   {f:>8.0%} {stress[-1][1]:>6} {stress[-1][2]:>13} {e_als:>8.3f} {e_mean:>10.3f}"
          f"  {'yes' if e_als < 0.3 else 'no'}")

fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.plot([s[0] * 100 for s in stress], [s[3] for s in stress], marker="o",
        color="steelblue", label="ALS")
ax.plot([s[0] * 100 for s in stress], [s[4] for s in stress], marker="s",
        color="darkorange", label="mean only")
ax.axhline(0.3, color="gray", linestyle="--", linewidth=1, label="RMSE 0.3")
ax.set_xlabel("Observed entries (%)")
ax.set_ylabel("RMSE on hidden entries")
ax.set_title("ALS stops recovering when too few ratings are known")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "als_stress_test.png", dpi=150)


# %% 7. Method 1: the bias baseline on the bias model
# Stack the biases into x = (b, c). Each known rating gives one row of D, with a 1
# in column u and a 1 in column m + i, and the response y_ui = r_ui − µ. Prop. 6 says
# the updates solve (DᵀD + λ_b I) x = Dᵀy; numpy.linalg.lstsq minimizes ‖Dx − y‖²
# only, so we append the rows √λ_b I (response 0) to include the penalty.
Rb, Rb_true, b_true, c_true = simulate_bias_ratings(mask, seed=SEED)
mu_b, b_hat, c_hat, B_hist = my_fit_biases(Rb, mask, LAM_B)
rows, cols = np.nonzero(mask)
D = np.zeros((len(rows), m + n))
D[np.arange(len(rows)), rows] = 1.0
D[np.arange(len(rows)), m + cols] = 1.0
D_aug = np.vstack([D, np.sqrt(LAM_B) * np.eye(m + n)])
y_aug = np.concatenate([Rb[rows, cols] - mu_b, np.zeros(m + n)])
x_lib = np.linalg.lstsq(D_aug, y_aug, rcond=None)[0]                     # check only
n_up_b = int(np.sum(np.diff(B_hist) > 1e-9 * B_hist[:-1]))
P_bias = my_predict_biases(mu_b, b_hat, c_hat)
rmse_bias = my_rmse(P_bias, Rb_true, ~mask)
rmse_mu = my_rmse(np.full_like(Rb_true, mu_b), Rb_true, ~mask)
print(f"\n7. Method 1 (bias baseline, λ_b = {LAM_B:g}): converged in {(len(B_hist) - 1) // 2} sweeps")
d_bias = report("   biases vs numpy.linalg.lstsq", np.concatenate([b_hat, c_hat]), x_lib)
print(f"   half-steps where B increased: {n_up_b} of {len(B_hist) - 1}  (Remark 7)")
print(f"   RMSE on hidden entries vs the truth: µ + b_u + c_i {rmse_bias:.4f}, µ only {rmse_mu:.4f}")
table.append(("Simulation (bias model)", "biases b, c", "numpy.linalg.lstsq",
              f"max|·| = {d_bias:.0e}", d_bias < 1e-8))
table.append(("Simulation (bias model)", "B after each half-step", "previous B",
              f"{n_up_b} increases", n_up_b == 0))


# %% 8. Proposition 8: coverage of the 95% confidence interval
# Each experiment draws N = 20,000 errors from N(0, 0.9²), so the true MSE is 0.81
# and the true RMSE 0.9. Proposition 8 says the interval contains them about 95% of
# the time; my_rmse_ci builds it from the squared errors.
N_TEST, TRUE_RMSE, N_REPEATS = 20_000, 0.9, 2_000
rng = np.random.default_rng(SEED)
covered = 0
zeros = np.zeros(N_TEST)
all_pairs = np.ones(N_TEST, dtype=bool)
for _ in range(N_REPEATS):
    e = simulate_test_errors(rng, N_TEST, TRUE_RMSE)
    _, lo, hi = my_rmse_ci(e, zeros, all_pairs)
    covered += lo <= TRUE_RMSE <= hi
coverage = covered / N_REPEATS
mc_se = np.sqrt(0.95 * 0.05 / N_REPEATS)
print(f"\n8. Proposition 8   the RMSE interval contained the true RMSE in {covered} of "
      f"{N_REPEATS} experiments ({coverage:.1%}); 3 Monte Carlo s.e. = {3 * mc_se:.1%}")
table.append(("Simulation (error model)", "coverage of 95% interval", "0.95",
              f"{coverage:.3f} (±{3 * mc_se:.3f})", abs(coverage - 0.95) < 3 * mc_se))


# %% 9. Assumption checks (Section 3 of the report)
# Assumption 1 uses the held-out RMSE: the hidden entries against the noisy R,
# the same quantity that the test set gives on real data. Assumptions 3 and 5
# use the residuals r − r̂ on the hidden entries of the main fit (k = 3, λ = 1).
print("\n9. Assumption checks (the simulated data satisfy them by construction)")
ks_check = [1, 2, 3, 4, 5, 6]
held_out = []
for kk in ks_check:
    *_, P_k = fit(R, mask, k=kk)
    held_out.append(my_rmse(P_k, R, ~mask))
curves = {f"λ = {LAM:g}": (ks_check, held_out)}

lowest_eig = min_eig_G(R_c, mask, U, V, LAM)
pred = P[~mask]
resid = R[~mask] - pred
print_checks(curves, mask, K, LAM, lowest_eig, pred, resid, true_rank=K)
plot_checks(curves, pred, resid, "Assumption checks on simulated data (k = 3, λ = 1)",
            FIG_DIR / "assumptions_simulation.png")
assert lowest_eig >= LAM - 1e-9, "Proposition 2 violated"
table.append(("Simulation", "smallest eigenvalue of any G", f"λ = {LAM:g}",
              f"{lowest_eig:.2f} ≥ {LAM:g}", lowest_eig >= LAM - 1e-9))

# The residuals of a fitted model mix two things: the noise in the data (normal,
# and unrelated to r̂, by construction) and the estimator's own error. To separate
# them: predicting with R_true itself leaves only the noise, so Assumptions 3 and
# 5 must pass exactly; a much smaller λ removes most of the shrinkage towards µ.
*_, P_small = fit(R, mask, lam=0.01)
print("   Assumptions 3 and 5, the data against the estimator:")
print_residual_table([
    ("R_true itself (residual = noise)", R_true[~mask], R[~mask] - R_true[~mask]),
    (f"ALS, k = {K}, λ = {LAM:g}", pred, resid),
    (f"ALS, k = {K}, λ = 0.01", P_small[~mask], R[~mask] - P_small[~mask]),
])

# Assumption 2. Here every entry is observed with the same probability, whatever its
# value, so popular and rarely rated movies should have the same mean. With the truth
# known we can also compare the true values that were observed with those that were not.
pop = popularity_check(R, mask)
print(f"   (Assumption 2 with the truth: mean of R_true on observed entries "
      f"{R_true[mask].mean():+.4f}, on hidden entries {R_true[~mask].mean():+.4f})")

# Assumption 6. Twenty different random 30% masks of the same matrix: refit ALS each
# time and compare the spread of the held-out RMSE with one interval's standard error.
rmses_rep, cis_rep = [], []
for s in range(20):
    M_s = np.random.default_rng(100 + s).random((m, n)) < 0.3
    *_, P_s = fit(R, M_s)
    ci = my_rmse_ci(P_s, R, ~M_s)
    rmses_rep.append(ci[0])
    cis_rep.append(ci)
spread = ci_vs_spread(rmses_rep, cis_rep)

# Assumption 7. The singular values of the filled matrix from section 4 (K + 1 of them).
gaps = singular_gaps(sig, svd_iters, SVD_TOL)
print_extra_checks(pop, spread, gaps)
plot_extra_checks(pop, spread, gaps, "Assumptions 2, 6 and 7 on simulated data",
                  FIG_DIR / "assumptions_extra_simulation.png")


# %% Verification table
print("\nVerification table")
print(f"{'Data':<26}{'Quantity':<30}{'Compared with':<22}{'Difference':<24}Pass?")
for row in table:
    print(f"{row[0]:<26}{row[1]:<30}{row[2]:<22}{row[3]:<24}{'✓' if row[4] else '✗'}")

assert all(row[4] for row in table), "a check failed"
print("\nAll simulation checks passed.")
