"""
All three methods on MovieLens 100K, verified against libraries.

Data: the provided split u1.base (training) / u1.test (test), so that results
are comparable across teams. This replaces the seeded 80/20 split made by
load_data.py for the data description. u.data is used only for the repeated
random splits of Assumption 6.

Sections:
  1. Data and cleaning record (what was checked, what was removed).
  2. Baselines: the global mean, each movie's mean, and Method 1 (µ + b_u + c_i) for
     λ_b ∈ {1, 10, 25}; the biases are checked against numpy.linalg.solve.
  3. Method 2 (ALS) for k ∈ {2, 5, 10, 20} and λ ∈ {0.1, 1, 10}, Steps 1–7, with
     training and test RMSE. my_als stops with an error if J ever rises.
  4. Method 3 (truncated SVD by power iteration) of the movie-mean-filled and the
     zero-filled training matrix, for the same k; checked against numpy.linalg.svd.
  5. Theorem 4 on real data: J after every half-step of every ALS run.
  6. Library check of ALS: every row update of the best model, by our Gaussian
     elimination and by numpy.linalg.solve; the difference explained by cond(G).
  7. 95% confidence intervals (Proposition 8) for every method, and for the
     differences between the best ALS model and its competitors.
  8. Assumption checks: all seven assumptions (assumptions.py), the same checks that
     check_simulation.py runs on simulated data.
  9. Rows of the verification table.

Run from the repository root:   python src/run_real.py   (about 3–5 minutes)
Figures are saved to figures/.

Owners: Muhammad Saad Malik (ALS: 3, 5, 6; Assumptions 1, 3, 4, 5)
        Jagadeesh Aravind Kalyan Shetti (truncated SVD: 4; Assumption 7)
        Emmanuel Adjei Danso (bias baseline and error bars: 2, 7; Assumptions 2, 6)
"""

# %% Setup
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from methods import (my_centre, my_normal_equations, my_gauss_solve, my_als,
                     my_predict, my_rmse, report,
                     my_fit_biases, my_predict_biases,
                     my_fill, my_svd_power, my_svd_predict,
                     my_rmse_ci, my_mse_diff_ci)
from assumptions import (min_eig_G, print_checks, plot_checks,
                         popularity_check, ci_vs_spread, singular_gaps,
                         print_extra_checks, plot_extra_checks)

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
DATA_DIR = ROOT / "data" / "ml-100k"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

KS = [2, 5, 10, 20]          # numbers of latent factors (ALS and SVD)
LAMS = [0.1, 1.0, 10.0]      # ALS regularization strengths λ
LAMS_B = [1.0, 10.0, 25.0]   # bias regularization strengths λ_b
MAX_SWEEPS = 50              # ALS Step 6: at most this many sweeps ...
TOL = 1e-5                   # ... or stop when one sweep lowers J by less than TOL × J
SVD_TOL = 1e-10              # power iteration stops when ‖Δx‖ < SVD_TOL
SEED = 0                     # Step 2: np.random.default_rng(0)
CLIP = (1, 5)                # predictions clipped to the rating scale
N_SPLITS = 10                # repeated random splits for Assumption 6

table = []                   # rows of the verification table
t_start = time.time()


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


def fit_als(R, mask, k, lam):
    """Method 2, Steps 1–7. Returns µ, U, V, J after every half-step, and predictions."""
    mu, R_c = my_centre(R, mask)                                              # Step 1
    U, V, J_hist = my_als(R_c, mask, k, lam, iters=MAX_SWEEPS, seed=SEED, tol=TOL)  # Steps 2–6
    return mu, R_c, U, V, J_hist, my_predict(U, V, mu, clip=CLIP)             # Step 7


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
print(f"   users with training ratings: {(per_user > 0).sum()} of {m};  "
      f"movies with training ratings: {(per_movie > 0).sum()} of {n}")
print(f"   test ratings whose user has no training ratings:  {(~test['user'].isin(train['user'])).sum()}")
print(f"   test ratings whose movie has no training ratings: "
      f"{(~test['movie'].isin(train['movie'])).sum()}  (predicted at µ)")
print(f"   fewest training ratings: user {per_user.min()}, movie {per_movie.min()}")
print(f"   users with fewer than k = {max(KS)} training ratings: {(per_user < max(KS)).sum()};  "
      f"movies: {(per_movie < max(KS)).sum()}")
print("   (for these rows V_uᵀV_u or U_iᵀU_i is singular, so λ > 0 is needed: Proposition 2)\n")


# %% 2. Baselines, including Method 1
# Global mean:  r̂_ui = µ.   Movie mean:  r̂_ui = mean of movie i's training ratings
# (µ for a movie with none), which is exactly how Method 3 fills a missing entry.
mu, R_c = my_centre(R_tr, mask_tr)
P_global = np.full((m, n), mu)
P_movie = my_fill(R_tr, mask_tr, how="movie")
rmse_global = my_rmse(P_global, R_te, mask_te)
rmse_movie = my_rmse(P_movie, R_te, mask_te)
print("2. Baselines")
print(f"   µ = {mu:.4f}")
print(f"   test RMSE, global mean µ:  {rmse_global:.4f}")
print(f"   test RMSE, movie mean:     {rmse_movie:.4f}")

bias = {}
print("   Method 1, r̂_ui = µ + b_u + c_i (Steps 1–4):")
print(f"   {'λ_b':>6} {'sweeps':>7} {'train RMSE':>11} {'test RMSE':>10}")
for lam_b in LAMS_B:
    mu_b, b, c, B_hist = my_fit_biases(R_tr, mask_tr, lam_b)
    P_b = my_predict_biases(mu_b, b, c, clip=CLIP)
    bias[lam_b] = dict(b=b, c=c, B=B_hist, P=P_b, sweeps=(len(B_hist) - 1) // 2,
                       train=my_rmse(P_b, R_tr, mask_tr), test=my_rmse(P_b, R_te, mask_te))
    r = bias[lam_b]
    print(f"   {lam_b:>6g} {r['sweeps']:>7} {r['train']:>11.4f} {r['test']:>10.4f}")
lam_b_best = min(bias, key=lambda key: bias[key]["test"])
rb_bias = bias[lam_b_best]
n_up_B = sum(int(np.sum(np.diff(r["B"]) > 1e-9 * r["B"][:-1])) for r in bias.values())
print(f"   lowest test RMSE at λ_b = {lam_b_best:g}: {rb_bias['test']:.4f};  "
      f"half-steps where B increased: {n_up_B}  (Remark 7)")

# Library check (Proposition 6): the normal equations (DᵀD + λ_b I) x = Dᵀy, built
# from the counts and the mask without forming D, and solved by numpy.linalg.solve.
#   DᵀD = [[diag(|I_u|), M], [Mᵀ, diag(|𝒰_i|)]],   Dᵀy = (Σ_{i∈I_u} (r_ui − µ), Σ_{u∈𝒰_i} (r_ui − µ)).
Mf = mask_tr.astype(float)
NE = np.block([[np.diag(per_user + lam_b_best), Mf], [Mf.T, np.diag(per_movie + lam_b_best)]])
Y = np.where(mask_tr, R_tr - mu, 0.0)
x_lib = np.linalg.solve(NE, np.concatenate([Y.sum(axis=1), Y.sum(axis=0)]))   # check only
d_bias = report(f"   biases (λ_b = {lam_b_best:g}) vs numpy.linalg.solve",
                np.concatenate([rb_bias["b"], rb_bias["c"]]), x_lib)
table.append(("Real data", f"biases b, c (λ_b = {lam_b_best:g})", "numpy.linalg.solve",
              f"max|·| = {d_bias:.0e}", d_bias < 1e-8))
table.append((f"Real data ({len(bias)} bias runs)", "B after each half-step", "previous B",
              f"{n_up_B} increases", n_up_B == 0))
print()


# %% 3. Method 2: ALS for every (k, λ)
print("3. Method 2, ALS (Steps 1–7)")
print(f"   {'k':>3} {'λ':>6} {'sweeps':>7} {'train RMSE':>11} {'test RMSE':>10} {'seconds':>8}")
results = {}
for lam in LAMS:
    for k in KS:
        t0 = time.time()
        _, _, U, V, J_hist, P = fit_als(R_tr, mask_tr, k, lam)
        results[(k, lam)] = dict(U=U, V=V, J=J_hist, sweeps=(len(J_hist) - 1) // 2,
                                 train=my_rmse(P, R_tr, mask_tr), test=my_rmse(P, R_te, mask_te))
        res = results[(k, lam)]
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
P_als = my_predict(rb["U"], rb["V"], mu, clip=CLIP)
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
print()


# %% 4. Method 3: truncated SVD of the filled training matrix
# Step 1 fills the gaps (movie means, or zeros for comparison); Step 2 finds the top
# max(KS) + 1 singular triples once (the extra one is for Assumption 7); Step 4
# predicts with the first k of them for every k, because the top-k triples are nested.
print("4. Method 3, truncated SVD by power iteration (Steps 1–4)")
k_svd = max(KS)
svd = {}
for how in ("movie", "zero"):
    t0 = time.time()
    A = my_fill(R_tr, mask_tr, how=how)                                       # Step 1
    sig, W, Z, iters = my_svd_power(A, k_svd + 1, tol=SVD_TOL, seed=SEED)      # Step 2
    tests = {k: my_rmse(my_svd_predict(sig, W, Z, k, clip=CLIP), R_te, mask_te) for k in KS}  # Step 4
    svd[how] = dict(A=A, sig=sig, W=W, Z=Z, iters=iters, test=tests)
    print(f"   {how}-filled: test RMSE " + "  ".join(f"k={k}: {tests[k]:.4f}" for k in KS)
          + f"   ({time.time() - t0:.0f} s, at most {iters.max()} steps per factor)")
k_svd_best = min(KS, key=lambda k: svd["movie"]["test"][k])
P_svd = my_svd_predict(svd["movie"]["sig"], svd["movie"]["W"], svd["movie"]["Z"], k_svd_best, clip=CLIP)
k_zero_best = min(KS, key=lambda k: svd["zero"]["test"][k])
P_svd_zero = my_svd_predict(svd["zero"]["sig"], svd["zero"]["W"], svd["zero"]["Z"], k_zero_best, clip=CLIP)
print(f"   lowest test RMSE, movie-mean fill: k = {k_svd_best}: {svd['movie']['test'][k_svd_best]:.4f};  "
      f"zero fill: k = {k_zero_best}: {svd['zero']['test'][k_zero_best]:.4f}")

# Library check on the movie-mean-filled matrix: singular values, singular vectors
# after sign alignment, and the error of the rank-k truncation (Proposition 5).
sv = svd["movie"]
W_lib, s_lib, Zt_lib = np.linalg.svd(sv["A"], full_matrices=False)          # check only
sgn = np.sign(np.sum(sv["Z"][:, :k_svd] * Zt_lib[:k_svd].T, axis=0))
d_sig = report(f"   σ_1..σ_{k_svd + 1} vs numpy.linalg.svd", sv["sig"], s_lib[:k_svd + 1])
d_z = report(f"   z_1..z_{k_svd} (signs aligned)", sv["Z"][:, :k_svd] * sgn, Zt_lib[:k_svd].T)
d_w = report(f"   w_1..w_{k_svd} (signs aligned)", sv["W"][:, :k_svd] * sgn, W_lib[:, :k_svd])
err_k = np.sum((sv["A"] - my_svd_predict(sv["sig"], sv["W"], sv["Z"], k_svd)) ** 2)
tail = np.sum(s_lib[k_svd:] ** 2)
print(f"   ‖R̃ − R̃_{k_svd}‖_F² = {err_k:.2f},  Σ_(j>{k_svd}) σ_j² = {tail:.2f}  "
      f"(relative difference {abs(err_k - tail) / tail:.1e}, Proposition 5)")
table.append(("Real data", f"σ_1..σ_{k_svd + 1} (power iteration)", "numpy.linalg.svd",
              f"max|·| = {d_sig:.0e}", d_sig < 1e-8 * s_lib[0]))
table.append(("Real data", "z_j, w_j (signs aligned)", "numpy.linalg.svd",
              f"max|·| = {max(d_z, d_w):.0e}", max(d_z, d_w) < 1e-6))
table.append(("Real data", f"‖R̃ − R̃_{k_svd}‖_F²", f"Σ_(j>{k_svd}) σ_j²",
              f"rel. {abs(err_k - tail) / tail:.0e}", abs(err_k - tail) / tail < 1e-8))

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(KS, [results[(k, lam_best)]["test"] for k in KS], marker="o", label=f"ALS, λ = {lam_best:g}")
ax.plot(KS, [svd["movie"]["test"][k] for k in KS], marker="s", label="SVD, movie-mean fill")
ax.axhline(rb_bias["test"], color="darkorange", linestyle=":", linewidth=1.2,
           label=f"bias baseline, λ_b = {lam_b_best:g}")
ax.axhline(rmse_movie, color="gray", linestyle="--", linewidth=1, label="movie mean")
ax.set_xscale("log")
ax.set_xticks(KS, [str(k) for k in KS])
ax.set_xlabel("Number of factors k")
ax.set_ylabel("Test RMSE (u1.test)")
ax.set_title("ALS against the truncated SVD and the baselines")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "test_rmse_methods_real.png", dpi=150)
print()


# %% 5. Theorem 4 on real data
n_up = sum(int(np.sum(np.diff(r["J"]) > 1e-9 * r["J"][:-1])) for r in results.values())
n_steps = sum(len(r["J"]) - 1 for r in results.values())
print(f"5. Theorem 4   half-steps where J increased, over all {len(results)} ALS runs: "
      f"{n_up} of {n_steps}")
table.append((f"Real data ({len(results)} ALS runs)", "J after each half-step", "previous J",
              f"{n_up} increases", n_up == 0))

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(np.arange(len(rb["J"])) / 2, rb["J"], marker=".", color="steelblue")
ax.set_yscale("log")
ax.set_xlabel("ALS sweep")
ax.set_ylabel("Objective J(U, V) (log scale)")
ax.set_title(f"ALS objective on MovieLens (k = {k_best}, λ = {lam_best:g})")
fig.tight_layout()
fig.savefig(FIG_DIR / "als_objective_real.png", dpi=150)


# %% 6. Library check of ALS: our Gaussian elimination against numpy.linalg.solve
# Every user system G_u u_u = h_u (with the final V) and every movie system
# (with the final U) of the best model, solved both ways. Two stable solvers
# can differ by roughly cond(G) × ε (relative), with ε ≈ 2.2e-16 the machine epsilon.
U, V = rb["U"], rb["V"]
G_1, h_1 = my_normal_equations(V[mask_tr[0]], R_c[0, mask_tr[0]], lam_best)
print("\n6. Library (ALS)")
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
table.append(("Real data", "every ALS row update", "numpy.linalg.solve",
              f"max|·| = {diff_all:.0e}", diff_all < 1e-10))


# %% 7. Confidence intervals (Proposition 8)
# A 95% interval for each method's true RMSE on new ratings, and for the difference
# in MSE between the best ALS model and each competitor on the same test ratings.
# An interval for the difference that excludes 0 means the gap is real.
P_als_21 = my_predict(results[(2, 1.0)]["U"], results[(2, 1.0)]["V"], mu, clip=CLIP)
methods = [("global mean µ", P_global), ("movie mean", P_movie),
           (f"Method 1, λ_b = {lam_b_best:g}", rb_bias["P"]),
           (f"Method 3, movie fill, k = {k_svd_best}", P_svd),
           (f"Method 3, zero fill, k = {k_zero_best}", P_svd_zero),
           ("Method 2, k = 2, λ = 1", P_als_21),
           (f"Method 2, k = {k_best}, λ = {lam_best:g}", P_als)]
print("\n7. Confidence intervals (Proposition 8)")
print(f"   {'method':<34} {'test RMSE':>10}   95% interval")
for name, P in methods:
    r, lo, hi = my_rmse_ci(P, R_te, mask_te)
    print(f"   {name:<34} {r:>10.4f}   [{lo:.4f}, {hi:.4f}]")
best_name = f"Method 2, k = {k_best}, λ = {lam_best:g}"
print(f"   Difference in test MSE, {best_name} minus each competitor:")
diffs = {}
for name, P in methods[:-1]:
    d, lo, hi = my_mse_diff_ci(P_als, P, R_te, mask_te)
    diffs[name] = (d, lo, hi)
    verdict = "ALS lower" if hi < 0 else ("ALS higher" if lo > 0 else "not distinguishable")
    print(f"   {name:<34} {d:>+10.4f}   [{lo:+.4f}, {hi:+.4f}]   {verdict}")


# %% 8. Assumption checks (Section 3 of the report)
# Assumption 1 uses the test RMSE of the whole grid. Assumptions 3 and 5 use the
# test residuals r − r̂ of the best model, with predictions clipped as in Step 7.
print("\n8. Assumption checks")
curves = {f"λ = {lam:g}": (KS, [results[(k, lam)]["test"] for k in KS]) for lam in LAMS}
lowest_eig = min_eig_G(R_c, mask_tr, rb["U"], rb["V"], lam_best)
pred = P_als[mask_te]
resid = R_te[mask_te] - pred
print_checks(curves, mask_tr, k_best, lam_best, lowest_eig, pred, resid)
plot_checks(curves, pred, resid,
            f"Assumption checks on MovieLens (k = {k_best}, λ = {lam_best:g}, test residuals)",
            FIG_DIR / "assumptions_real.png")
assert lowest_eig >= lam_best - 1e-9, "Proposition 2 violated"
table.append(("Real data", "smallest eigenvalue of any G", f"λ = {lam_best:g}",
              f"{lowest_eig:.4f} ≥ {lam_best:g}", lowest_eig >= lam_best - 1e-9))

# Assumption 2: popularity against mean rating, on the training ratings.
pop = popularity_check(R_tr, mask_tr)

# Assumption 6: N_SPLITS random 80/20 splits of u.data, the best ALS model refitted on
# each; the spread of the test RMSE against one interval's standard error.
ratings = read_split("u.data")
rmses_rep, cis_rep = [], []
for s in range(N_SPLITS):
    idx = np.random.default_rng(100 + s).permutation(len(ratings))
    n_te = len(ratings) // 5
    R_a, M_a = to_matrix(ratings.iloc[idx[n_te:]], m, n)
    R_b, M_b = to_matrix(ratings.iloc[idx[:n_te]], m, n)
    *_, P_s = fit_als(R_a, M_a, k_best, lam_best)
    ci = my_rmse_ci(P_s, R_b, M_b)
    rmses_rep.append(ci[0])
    cis_rep.append(ci)
spread = ci_vs_spread(rmses_rep, cis_rep)

# Assumption 7: the singular values of the movie-mean-filled matrix from section 4.
gaps = singular_gaps(svd["movie"]["sig"], svd["movie"]["iters"], SVD_TOL)
print_extra_checks(pop, spread, gaps)
plot_extra_checks(pop, spread, gaps, "Assumptions 2, 6 and 7 on MovieLens",
                  FIG_DIR / "assumptions_extra_real.png")


# %% 9. Verification table
print("\nVerification table")
print(f"{'Data':<24}{'Quantity':<34}{'Compared with':<22}{'Difference':<24}Pass?")
for row in table:
    print(f"{row[0]:<24}{row[1]:<34}{row[2]:<22}{row[3]:<24}{'✓' if row[4] else '✗'}")
print(f"\nTotal time: {time.time() - t_start:.0f} s")
