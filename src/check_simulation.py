"""
Simulation test of ALS (Method 2): recover the known truth.

Data: the report's Simulation Check model (simulate.py), a true rank-3 matrix
of 100 users × 80 movies with noise sd 0.1 and 30% of the entries observed.
The hidden entries of R_true are the truth.

Checks:
  1. Theorem 4    J never increases, over every half-step.
  2. Truth        the error on the hidden entries is small (RMSE < 0.3),
                  and the error on the known entries is close to the noise sd.
  3. Library      every row update solved by our Gaussian elimination agrees with
                  numpy.linalg.solve on the same system (Proposition 1), to 1e-10.
  4. Baseline     a mean-filled truncated SVD does much worse on the hidden entries.
  5. Sample size  the error on the hidden entries shrinks as the matrix grows.
  6. Stress test  lower the observed fraction until ALS stops recovering R_true.
  7. Assumptions  the checks of Assumptions 1, 3, 4 and 5 (assumptions.py). They
                  hold here by construction, so this shows what "passing" looks
                  like before the same checks run on MovieLens (run_real.py).
At the end it prints the rows of the verification table.

Run from the repository root:   python src/check_simulation.py
Figures are saved to figures/.

Owner: Muhammad Saad Malik (ALS)
"""

# %% Setup
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from methods import (my_centre, my_normal_equations, my_gauss_solve, my_als,
                     my_predict, my_rmse, report)
from simulate import simulate_ratings
from assumptions import min_eig_G, print_checks, print_residual_table, plot_checks

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

K = 3            # true rank, and the rank we fit
LAM = 1.0        # regularization strength λ
NOISE_SD = 0.1   # noise sd used by simulate_ratings
N_SWEEPS = 50    # ALS sweeps
SEED = 0         # same seed as the report's Simulation Check

table = []       # rows of the verification table


def fit(R, mask, k=K, lam=LAM, iters=N_SWEEPS, seed=SEED):
    """Method 2, Steps 1–7: centre, run ALS, predict µ + U Vᵀ."""
    mu, R_c = my_centre(R, mask)                                   # Step 1
    U, V, J_hist = my_als(R_c, mask, k, lam, iters=iters, seed=seed)  # Steps 2–6
    return mu, R_c, U, V, J_hist, my_predict(U, V, mu)             # Step 7


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
table.append(("Simulation", "r̂ on hidden entries", "R_true", f"RMSE = {rmse_hidden:.3f}",
              rmse_hidden < 0.3))


# %% 3. Library: our Gaussian elimination against numpy.linalg.solve
# Rebuild one user's system G_u u_u = h_u with the final V, then every user
# and every movie system, and solve each one both ways.
G_0, h_0 = my_normal_equations(V[mask[0]], R_c[0, mask[0]], LAM)
print("\n3. Library")
diff_one = report("   user 1 update vs numpy.linalg.solve",
                  my_gauss_solve(G_0, h_0), np.linalg.solve(G_0, h_0))
diff_all = 0.0
for u in range(m):
    G, h = my_normal_equations(V[mask[u]], R_c[u, mask[u]], LAM)
    diff_all = max(diff_all, np.abs(my_gauss_solve(G, h) - np.linalg.solve(G, h)).max())
for i in range(n):
    G, h = my_normal_equations(U[mask[:, i]], R_c[mask[:, i], i], LAM)
    diff_all = max(diff_all, np.abs(my_gauss_solve(G, h) - np.linalg.solve(G, h)).max())
print(f"   all {m} user and {n} movie updates      max|diff| = {diff_all:.2e}  (pass: < 1e-10)")
table.append(("Simulation", "every row update", "numpy.linalg.solve",
              f"max|·| = {diff_all:.0e}", diff_all < 1e-10))


# %% 4. Baseline: truncated SVD of the mean-filled matrix
# Reference baseline only: fill the missing entries with µ, keep the top K
# singular triples (numpy.linalg.svd), and score the hidden entries.
R_fill = np.where(mask, R, mu)
W, s, Zt = np.linalg.svd(R_fill, full_matrices=False)
R_svd = (W[:, :K] * s[:K]) @ Zt[:K]
rmse_svd = my_rmse(R_svd, R_true, ~mask)
print(f"\n4. Baseline    RMSE on hidden entries: ALS {rmse_hidden:.4f}, "
      f"mean-filled SVD {rmse_svd:.4f}  ({rmse_svd / rmse_hidden:.1f}× worse)")
table.append(("Simulation", "filled-SVD on hidden", "ALS on hidden",
              f"{rmse_svd:.3f} vs {rmse_hidden:.3f}", rmse_svd > 3 * rmse_hidden))


# %% 5. Sample size: error on the hidden entries as the matrix grows
# Same model, same 30% observed, same seed; m and n double each time.
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
table.append(("Simulation (4 sizes)", "hidden RMSE", "|Ω|", f"slope {slope:.2f}", slope < 0))


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


# %% 7. Assumption checks (Section 3 of the report)
# Assumption 1 uses the held-out RMSE: the hidden entries against the noisy R,
# the same quantity that the test set gives on real data. Assumptions 3 and 5
# use the residuals r − r̂ on the hidden entries of the main fit (k = 3, λ = 1).
print("\n7. Assumption checks (they hold here by construction)")
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


# %% Verification table
print("\nVerification table")
print(f"{'Data':<22}{'Quantity':<26}{'Compared with':<22}{'Difference':<22}Pass?")
for row in table:
    print(f"{row[0]:<22}{row[1]:<26}{row[2]:<22}{row[3]:<22}{'✓' if row[4] else '✗'}")

assert all(row[4] for row in table), "a check failed"
print("\nAll simulation checks passed.")
