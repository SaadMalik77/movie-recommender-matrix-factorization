"""
ALS (Method 2) on MovieLens 100K, verified against a library.

Data: the provided split u1.base (training) / u1.test (test), so that results
are comparable across teams. This replaces the seeded 80/20 split made by
load_data.py for the data description.

Sections:
  1. Data and cleaning record (what was checked, what was removed).
  2. Baselines: the global mean µ, and each movie's mean training rating.
  3. ALS for k ∈ {2, 5, 10, 20} and λ ∈ {0.1, 1, 10}: Method 2, Steps 1–7,
     with training and test RMSE. my_als stops with an error if J ever rises.
  4. The best model: J after every half-step (Theorem 4 on real data).
  5. Library check: every row update of the best model, solved by our Gaussian
     elimination and by numpy.linalg.solve; the size of the difference is
     explained by the condition number of G.
  6. Assumption checks: Assumptions 1, 3, 4 and 5 (assumptions.py), the same
     checks that check_simulation.py runs on simulated data.
  7. Rows of the verification table.

Run from the repository root:   python src/run_real.py
Figures are saved to figures/.

Owner: Muhammad Saad Malik (ALS)
"""

# %% Setup
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from methods import (my_centre, my_normal_equations, my_gauss_solve, my_als,
                     my_predict, my_rmse, report)
from assumptions import min_eig_G, print_checks, plot_checks

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
DATA_DIR = ROOT / "data" / "ml-100k"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

KS = [2, 5, 10, 20]          # numbers of latent factors
LAMS = [0.1, 1.0, 10.0]      # regularization strengths
MAX_SWEEPS = 50              # Step 6: at most this many sweeps ...
TOL = 1e-5                   # ... or stop when one sweep lowers J by less than TOL × J
SEED = 0                     # Step 2: np.random.default_rng(0)
CLIP = (1, 5)                # Step 7: predictions clipped to the rating scale

table = []                   # rows of the verification table


def read_split(name):
    """One rating per row: user, movie, rating, timestamp (tab-separated)."""
    return pd.read_csv(DATA_DIR / name, sep="\t",
                       names=["user", "movie", "rating", "timestamp"])


def to_matrix(df, m, n):
    """Rating matrix R (m × n) and mask of known ratings. Ids start at 1."""
    R = np.zeros((m, n))
    mask = np.zeros((m, n), dtype=bool)
    R[df["user"] - 1, df["movie"] - 1] = df["rating"]
    mask[df["user"] - 1, df["movie"] - 1] = True
    return R, mask


# %% 1. Data and cleaning record
train, test = read_split("u1.base"), read_split("u1.test")
both = pd.concat([train, test])
m, n = both["user"].max(), both["movie"].max()

print("1. Data and cleaning record")
print(f"   u1.base: {len(train)} ratings,  u1.test: {len(test)} ratings,  matrix {m} × {n}")
print(f"   missing values: train {train.isna().sum().sum()}, test {test.isna().sum().sum()}")
print(f"   duplicate (user, movie) pairs: train {train.duplicated(['user', 'movie']).sum()}, "
      f"test {test.duplicated(['user', 'movie']).sum()}")
print(f"   ratings outside 1–5: {(~both['rating'].between(1, 5)).sum()}")

R_tr, mask_tr = to_matrix(train, m, n)
R_te, mask_te = to_matrix(test, m, n)
print(f"   pairs in both train and test: {(mask_tr & mask_te).sum()}")
print("   rows removed: none")

per_user = mask_tr.sum(axis=1)
per_movie = mask_tr.sum(axis=0)
cold_users = ~test["user"].isin(train["user"])
cold_movies = ~test["movie"].isin(train["movie"])
print(f"   users with training ratings: {(per_user > 0).sum()} of {m};  "
      f"movies with training ratings: {(per_movie > 0).sum()} of {n}")
print(f"   test ratings whose user has no training ratings:  {cold_users.sum()}")
print(f"   test ratings whose movie has no training ratings: {cold_movies.sum()}  (predicted at µ)")
print(f"   fewest training ratings: user {per_user.min()}, movie {per_movie.min()}")
print(f"   users with fewer than k = {max(KS)} training ratings: {(per_user < max(KS)).sum()};  "
      f"movies: {(per_movie < max(KS)).sum()}")
print("   (for these rows V_uᵀV_u or U_iᵀU_i is singular, so λ > 0 is needed: Proposition 2)\n")


# %% 2. Baselines
# Global mean:  r̂_ui = µ.   Movie mean:  r̂_ui = mean of movie i's training
# ratings, or µ for a movie with no training ratings.
mu, R_c = my_centre(R_tr, mask_tr)                                  # Step 1
movie_mean = np.where(per_movie > 0,
                      (R_tr * mask_tr).sum(axis=0) / np.maximum(per_movie, 1), mu)
rmse_global = my_rmse(np.full((m, n), mu), R_te, mask_te)
rmse_movie = my_rmse(np.tile(movie_mean, (m, 1)), R_te, mask_te)
print("2. Baselines")
print(f"   µ = {mu:.4f}")
print(f"   test RMSE, global mean µ:  {rmse_global:.4f}")
print(f"   test RMSE, movie mean:     {rmse_movie:.4f}\n")


# %% 3. ALS for every (k, λ)
print("3. ALS (Method 2, Steps 1–7)")
print(f"   {'k':>3} {'λ':>6} {'sweeps':>7} {'train RMSE':>11} {'test RMSE':>10} {'seconds':>8}")
results = {}
for lam in LAMS:
    for k in KS:
        t0 = time.time()
        U, V, J_hist = my_als(R_c, mask_tr, k, lam,                  # Steps 2–6
                              iters=MAX_SWEEPS, seed=SEED, tol=TOL)
        P = my_predict(U, V, mu, clip=CLIP)                          # Step 7
        res = dict(U=U, V=V, J=J_hist, sweeps=(len(J_hist) - 1) // 2,
                   train=my_rmse(P, R_tr, mask_tr), test=my_rmse(P, R_te, mask_te))
        results[(k, lam)] = res
        print(f"   {k:>3} {lam:>6g} {res['sweeps']:>7} {res['train']:>11.4f} "
              f"{res['test']:>10.4f} {time.time() - t0:>8.1f}")

print("\n   Test RMSE (rows k, columns λ); movie-mean baseline "
      f"{rmse_movie:.4f}")
print("   " + f"{'k':>4}" + "".join(f"{f'λ = {lam:g}':>11}" for lam in LAMS))
for k in KS:
    print("   " + f"{k:>4}" + "".join(f"{results[(k, lam)]['test']:>11.4f}" for lam in LAMS))

# The lowest test RMSE over the grid. Choosing k and λ this way uses the test
# set; tuning them without touching it would need a validation split of u1.base.
best = min(results, key=lambda key: results[key]["test"])
k_best, lam_best = best
rb = results[best]
print(f"\n   Lowest test RMSE: k = {k_best}, λ = {lam_best:g}: {rb['test']:.4f} "
      f"(movie mean {rmse_movie:.4f}, {100 * (1 - rb['test'] / rmse_movie):.1f}% lower)")
table.append(("Real data", f"ALS test RMSE (k={k_best}, λ={lam_best:g})", "movie-mean baseline",
              f"{rb['test']:.3f} vs {rmse_movie:.3f}", rb["test"] < rmse_movie))

fig, ax = plt.subplots(figsize=(6, 3.8))
for lam, marker in zip(LAMS, "osd"):
    ax.plot(KS, [results[(k, lam)]["test"] for k in KS], marker=marker, label=f"ALS, λ = {lam:g}")
ax.axhline(rmse_movie, color="gray", linestyle="--", linewidth=1, label="movie mean")
ax.set_xscale("log")
ax.set_xticks(KS, [str(k) for k in KS])
ax.set_xlabel("Number of factors k")
ax.set_ylabel("Test RMSE (u1.test)")
ax.set_title("Test RMSE of ALS on MovieLens 100K")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "als_test_rmse_real.png", dpi=150)


# %% 4. Theorem 4 on real data
n_up = sum(int(np.sum(np.diff(r["J"]) > 1e-9 * r["J"][:-1])) for r in results.values())
n_steps = sum(len(r["J"]) - 1 for r in results.values())
print(f"\n4. Theorem 4   half-steps where J increased, over all {len(results)} runs: "
      f"{n_up} of {n_steps}")
table.append((f"Real data ({len(results)} runs)", "J after each half-step", "previous J",
              f"{n_up} increases", n_up == 0))

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(np.arange(len(rb["J"])) / 2, rb["J"], marker=".", color="steelblue")
ax.set_yscale("log")
ax.set_xlabel("ALS sweep")
ax.set_ylabel("Objective J(U, V) (log scale)")
ax.set_title(f"ALS objective on MovieLens (k = {k_best}, λ = {lam_best:g})")
fig.tight_layout()
fig.savefig(FIG_DIR / "als_objective_real.png", dpi=150)


# %% 5. Library check: our Gaussian elimination against numpy.linalg.solve
# Every user system G_u u_u = h_u (with the final V) and every movie system
# (with the final U) of the best model, solved both ways. A solver with
# rounding error ε ≈ 1.1e-16 can be off by roughly cond(G) × ε (relative).
U, V = rb["U"], rb["V"]
G_1, h_1 = my_normal_equations(V[mask_tr[0]], R_c[0, mask_tr[0]], lam_best)
print("\n5. Library")
report("   user 1 update vs numpy.linalg.solve", my_gauss_solve(G_1, h_1),
       np.linalg.solve(G_1, h_1))
diff_all, cond_max = 0.0, 0.0
for F, Rows, M in ((V, R_c, mask_tr), (U, R_c.T, mask_tr.T)):
    for row in range(M.shape[0]):
        G, h = my_normal_equations(F[M[row]], Rows[row, M[row]], lam_best)
        diff_all = max(diff_all, np.abs(my_gauss_solve(G, h) - np.linalg.solve(G, h)).max())
        cond_max = max(cond_max, np.linalg.cond(G))
print(f"   all {m} user and {n} movie updates      max|diff| = {diff_all:.2e}  (pass: < 1e-10)")
print(f"   largest cond(G) = {cond_max:.1f}, so rounding of order "
      f"{cond_max * np.finfo(float).eps:.1e} is expected")
table.append(("Real data", "every row update", "numpy.linalg.solve",
              f"max|·| = {diff_all:.0e}", diff_all < 1e-10))


# %% 6. Assumption checks (Section 3 of the report)
# Assumption 1 uses the test RMSE of the whole grid. Assumptions 3 and 5 use the
# test residuals r − r̂ of the best model, with predictions clipped as in Step 7.
print("\n6. Assumption checks")
curves = {f"λ = {lam:g}": (KS, [results[(k, lam)]["test"] for k in KS]) for lam in LAMS}
lowest_eig = min_eig_G(R_c, mask_tr, rb["U"], rb["V"], lam_best)
P_best = my_predict(rb["U"], rb["V"], mu, clip=CLIP)
pred = P_best[mask_te]
resid = R_te[mask_te] - pred
print_checks(curves, mask_tr, k_best, lam_best, lowest_eig, pred, resid)
plot_checks(curves, pred, resid,
            f"Assumption checks on MovieLens (k = {k_best}, λ = {lam_best:g}, test residuals)",
            FIG_DIR / "assumptions_real.png")
assert lowest_eig >= lam_best - 1e-9, "Proposition 2 violated"


# %% 7. Verification table
print("\nVerification table")
print(f"{'Data':<22}{'Quantity':<30}{'Compared with':<22}{'Difference':<22}Pass?")
for row in table:
    print(f"{row[0]:<22}{row[1]:<30}{row[2]:<22}{row[3]:<22}{'✓' if row[4] else '✗'}")
