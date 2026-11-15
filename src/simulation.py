"""Simulation with known truth for the matrix-factorization recommender.

We generate data whose true structure we choose, so the correct answers are
known in advance, and check every derived result of the report numerically:

    Prop 1 (Q2)  Each ALS row update solves its own normal equations.
    Prop 2 (Q3)  G_u = V_u^T V_u + lambda*I is positive definite, and every
                 eigenvalue is at least lambda.
    Thm 4  (Q4)  The ALS objective J(U, V) never increases.
    Prop 5       The truncated SVD A_k is the best rank-k fit of a full matrix,
                 with error equal to the sum of the dropped sigma_j^2
                 (Eckart-Young).
    Q5           After the missing entries are filled in, the truncated SVD
                 predicts the hidden ratings much worse than ALS.
    Prop 6       The alternating bias updates solve the stacked regularized
                 least-squares system, and the bias objective never increases.
    Prop 8       The 95% confidence interval for the test MSE (and RMSE)
                 contains the true value about 95% of the time.

Run all:          python src/simulation.py
Or cell by cell:  in VS Code, click "Run Cell" above each "# %%" line.
Output:           printed checks, summary tables, and the figure
                  figures/simulation_objective.png

Owners: Muhammad Saad Malik (ALS: Prop 1, Prop 2, Thm 4)
        Jagadeesh Aravind Kalyan Shetti (SVD: Prop 5, Q5)
        Emmanuel Adjei Danso (bias baseline and error bars: Prop 6, Prop 8)
"""

# %% Setup
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Repo root, whether this runs as a script or cell by cell in VS Code
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Factor model (ALS and SVD): these are the "known truth" of the simulation.
M = 100           # number of users (rows of R)
N = 80            # number of movies (columns of R)
K = 3             # true rank of R, and the rank we fit
LAM = 1.0         # ALS regularization strength lambda
NOISE_SD = 0.1    # standard deviation of the noise added to the true ratings
OBS_FRAC = 0.3    # fraction of entries we observe (the other 70% are hidden)
N_SWEEPS = 50     # number of ALS sweeps (one sweep = update users, then movies)

# Bias baseline (Prop 6): true ratings are mu + b_u + c_i plus noise.
MU_TRUE = 3.5        # true global mean rating
USER_BIAS_SD = 0.3   # spread of the true user biases b_u
MOVIE_BIAS_SD = 0.5  # spread of the true movie biases c_i
BIAS_NOISE_SD = 0.3  # noise on each observed rating
LAM_B = 1.0          # bias regularization strength lambda_b

# Confidence interval (Prop 8)
TRUE_RMSE = 0.9      # true RMSE of the simulated predictions
N_TEST = 20_000      # test ratings per experiment (about our real test size)
N_REPEATS = 2_000    # number of repeated experiments


# %% Functions
def objective(R, mask, U, V, lam):
    """ALS objective J(U, V) (Q1).

    J(U, V) = sum over known (u, i) of (r_ui - u_u^T v_i)^2
              + lam * (||U||_F^2 + ||V||_F^2)

    Multiplying by the mask sets the error to 0 wherever the rating is
    unknown, so only the known ratings count.
    """
    E = (R - U @ V.T) * mask
    return np.sum(E**2) + lam * (np.sum(U**2) + np.sum(V**2))


def normal_equations(Vu, ru, lam):
    """Return G_u = Vu^T Vu + lam*I and the right-hand side Vu^T ru for one
    user (Prop 1).

    Vu holds the rows of V for the movies this user rated (size |I_u| x k)
    and ru holds the user's known ratings (length |I_u|). The user's vector
    u_u is the solution of G_u u_u = Vu^T ru.
    """
    k = Vu.shape[1]
    G = Vu.T @ Vu + lam * np.eye(k)
    rhs = Vu.T @ ru
    return G, rhs


def update_rows(R, mask, V, lam):
    """With V fixed, give every row its best vector by solving its own
    normal equations: u_u = (V_u^T V_u + lam*I)^(-1) V_u^T r_u  (Prop 1).

    The same function updates the movies: pass R.T, mask.T and U instead,
    because swapping the roles of users and movies transposes the problem.
    """
    n_rows, k = R.shape[0], V.shape[1]
    U = np.zeros((n_rows, k))
    for u in range(n_rows):
        Vu = V[mask[u]]          # rows of V for the movies user u rated
        ru = R[u, mask[u]]       # user u's known ratings
        G, rhs = normal_equations(Vu, ru, lam)
        U[u] = np.linalg.solve(G, rhs)   # solve G u_u = rhs without forming G^(-1)
    return U


def truncated_svd(A, k):
    """Rank-k truncated SVD of a full matrix A, built from the eigenvalues
    and eigenvectors of A^T A.

    A^T A is symmetric positive semi-definite, so its eigenvalues are real
    and non-negative, and the singular values are sigma_j = sqrt(eigenvalue).
    Its eigenvectors are the right singular vectors z_j. With Z_k holding the
    top k of them, A Z_k = W_k Sigma_k, so the rank-k approximation is
    A_k = W_k Sigma_k Z_k^T = A Z_k Z_k^T.

    Returns all singular values (largest first), A_k, and Z_k.
    """
    evals, evecs = np.linalg.eigh(A.T @ A)      # eigh returns ascending order
    evals, evecs = evals[::-1], evecs[:, ::-1]  # reorder: largest first
    sig = np.sqrt(np.clip(evals, 0, None))      # clip tiny negatives from rounding
    Zk = evecs[:, :k]
    Ak = A @ Zk @ Zk.T
    return sig, Ak, Zk


def bias_objective(R, mask, mu, b, c, lam_b):
    """Bias objective B(b, c) (Prop 6).

    B(b, c) = sum over known (u, i) of (r_ui - mu - b_u - c_i)^2
              + lam_b * (sum_u b_u^2 + sum_i c_i^2)
    """
    E = (R - mu - b[:, None] - c[None, :]) * mask
    return np.sum(E**2) + lam_b * (np.sum(b**2) + np.sum(c**2))


def fit_biases(R, mask, lam_b, tol=1e-13, max_sweeps=10_000):
    """Fit mu, b and c by alternating the closed-form updates of Prop 6.

    With c fixed:  b_u = sum_{i in I_u} (r_ui - mu - c_i) / (|I_u| + lam_b)
    With b fixed:  c_i = sum_{u in U_i} (r_ui - mu - b_u) / (|U_i| + lam_b)

    Sweeps stop when no bias changes by more than tol. Returns mu, b, c and
    the value of B after every half-step.
    """
    mu = R[mask].mean()
    b = np.zeros(R.shape[0])
    c = np.zeros(R.shape[1])
    history = [bias_objective(R, mask, mu, b, c, lam_b)]
    for _ in range(max_sweeps):
        b_old, c_old = b, c
        # Residuals with the movie biases removed; unknown entries count as 0.
        b = ((R - mu - c[None, :]) * mask).sum(axis=1) / (mask.sum(axis=1) + lam_b)
        history.append(bias_objective(R, mask, mu, b, c, lam_b))
        c = ((R - mu - b[:, None]) * mask).sum(axis=0) / (mask.sum(axis=0) + lam_b)
        history.append(bias_objective(R, mask, mu, b, c, lam_b))
        if max(np.abs(b - b_old).max(), np.abs(c - c_old).max()) < tol:
            break
    return mu, b, c, np.array(history)


def rmse(a, b):
    """Root mean squared difference between two arrays."""
    return np.sqrt(np.mean((a - b) ** 2))


# %% Part A: build the data with known truth
rng = np.random.default_rng(0)   # fixed seed: every team member gets the same numbers

# True ratings: a rank-3 matrix, the product of an (M x 3) and a (3 x N) matrix.
R_true = rng.normal(size=(M, K)) @ rng.normal(size=(K, N))

# Observed ratings: the truth plus small Gaussian noise.
R = R_true + NOISE_SD * rng.normal(size=(M, N))

# Mask of known entries (the set Omega): each entry is observed with
# probability 0.3, so about 70% of the matrix is hidden.
mask = rng.random((M, N)) < OBS_FRAC

print(f"R is {M} x {N}, true rank {K}, noise sd {NOISE_SD}")
print(f"Observed entries: {mask.sum()} of {M * N} ({mask.mean():.1%})")
print(f"Fewest ratings for a user: {mask.sum(axis=1).min()}, "
      f"for a movie: {mask.sum(axis=0).min()}")


# %% Part A: run ALS and record J after every half-step
# Random starting factors: U is M x K (one row per user), V is N x K (one
# row per movie). A predicted rating is r_hat_ui = u_u^T v_i.
U = rng.normal(size=(M, K))
V = rng.normal(size=(N, K))

J_history = [objective(R, mask, U, V, LAM)]
for sweep in range(N_SWEEPS):
    U = update_rows(R, mask, V, LAM)          # update all users, V fixed
    J_history.append(objective(R, mask, U, V, LAM))
    V = update_rows(R.T, mask.T, U, LAM)      # update all movies, U fixed
    J_history.append(objective(R, mask, U, V, LAM))
J_history = np.array(J_history)

print(f"J at the start: {J_history[0]:.4f}")
print(f"J after {N_SWEEPS} sweeps: {J_history[-1]:.4f}")


# %% Check Theorem 4 (Q4): the objective never increases
# Compare each value of J with the one before it. A change larger than a
# tiny rounding tolerance (relative to J) would count as an increase.
changes = np.diff(J_history)
increases = changes > 1e-9 * J_history[:-1]

print("Thm 4   Half-steps where J increased:", increases.sum(), "of", len(changes))
print(f"        Largest change between half-steps: {changes.max():.2e}  (should be <= 0)")
assert not increases.any(), "J increased: Theorem 4 is violated"

# Plot J against the sweep number; half-steps sit at 0.5, 1.5, ...
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(np.arange(len(J_history)) / 2, J_history, marker=".", color="steelblue")
ax.set_yscale("log")
ax.set_xlabel("ALS sweep")
ax.set_ylabel("Objective J(U, V) (log scale)")
ax.set_title("The ALS objective never increases")
fig.tight_layout()
fig.savefig(FIG_DIR / "simulation_objective.png", dpi=150)


# %% Check Proposition 1 (Q2): every user update solves its normal equations
# The last half-step updated the movies, so U is not yet the best choice for
# the current V. One more user update gives the U that Prop 1 describes.
U_check = update_rows(R, mask, V, LAM)

max_formula_gap = 0.0   # solve() result vs. the explicit formula in Prop 1
max_gradient = 0.0      # gradient of f_u at the update
for u in range(M):
    Vu, ru = V[mask[u]], R[u, mask[u]]
    G, rhs = normal_equations(Vu, ru, LAM)

    # The formula exactly as written in Prop 1, with an explicit inverse.
    u_formula = np.linalg.inv(G) @ rhs
    max_formula_gap = max(max_formula_gap, np.abs(U_check[u] - u_formula).max())

    # Gradient of f_u: -2 Vu^T (ru - Vu u_u) + 2 lam u_u.
    # At the minimizer it must be zero.
    grad = -2 * Vu.T @ (ru - Vu @ U_check[u]) + 2 * LAM * U_check[u]
    max_gradient = max(max_gradient, np.abs(grad).max())

print(f"Prop 1  Largest gap between solve() and the formula: {max_formula_gap:.2e}")
print(f"        Largest gradient entry at the update:         {max_gradient:.2e}")
assert max_formula_gap < 1e-8 and max_gradient < 1e-8


# %% Check Proposition 2 (Q3): G_u = V_u^T V_u + lambda*I is positive definite
# Prop 2 gives x^T G_u x = ||V_u x||^2 + lambda ||x||^2 >= lambda ||x||^2, so
# every eigenvalue of G_u is at least lambda. Check every user and movie.
min_eig = np.inf
for u in range(M):
    G, _ = normal_equations(V[mask[u]], R[u, mask[u]], LAM)
    min_eig = min(min_eig, np.linalg.eigvalsh(G).min())
for i in range(N):
    G, _ = normal_equations(U[mask[:, i]], R[mask[:, i], i], LAM)
    min_eig = min(min_eig, np.linalg.eigvalsh(G).min())

print(f"Prop 2  Smallest eigenvalue over all users and movies: {min_eig:.4f} "
      f"(lambda = {LAM})")
assert min_eig >= LAM - 1e-9

# Why lambda matters: a user who rated fewer movies than k. V_u^T V_u then
# has rank at most 2 < k = 3, so it is singular, and adding lambda*I fixes it.
V_few = V[:2]                              # pretend the user rated only 2 movies
G_no_reg = V_few.T @ V_few                 # lambda = 0
G_reg = G_no_reg + LAM * np.eye(K)         # lambda = 1
print(f"        User with 2 ratings: rank of V_u^T V_u = "
      f"{np.linalg.matrix_rank(G_no_reg)} of {K}")
print(f"        Smallest eigenvalue with lambda = 0: "
      f"{np.linalg.eigvalsh(G_no_reg).min():.1e}  (zero up to rounding: singular)")
print(f"        Smallest eigenvalue with lambda = 1: "
      f"{np.linalg.eigvalsh(G_reg).min():.4f}  (invertible)")


# %% Part A: ALS predictions compared with the truth
R_hat = U @ V.T
rmse_hidden_als = rmse(R_hat[~mask], R_true[~mask])   # entries ALS never saw
rmse_train_als = rmse(R_hat[mask], R[mask])           # fit to the known ratings

print(f"ALS error on hidden entries (vs. R_true): {rmse_hidden_als:.4f}")
print(f"ALS error on known entries (vs. R):       {rmse_train_als:.4f}  "
      f"(noise sd = {NOISE_SD})")


# %% Part B: truncated SVD of a filled-in matrix
# SVD needs a full matrix, so the hidden entries must be filled first.
F_zero = np.where(mask, R, 0.0)                 # fill with zeros
F_mean = np.where(mask, R, R[mask].mean())      # fill with the mean known rating
# (Here the mean is close to 0, so the two fills behave alike. On MovieLens,
# with a mean rating of 3.53, the choice matters much more.)

sig_zero, Fk_zero, Zk_zero = truncated_svd(F_zero, K)
sig_mean, Fk_mean, _ = truncated_svd(F_mean, K)

# Our singular values against the library routine (final comparison only).
sig_lib = np.linalg.svd(F_zero, compute_uv=False)
print(f"SVD     Largest gap between our singular values and numpy.linalg.svd: "
      f"{np.abs(sig_zero[:len(sig_lib)] - sig_lib).max():.2e}")


# %% Check Proposition 5: Eckart-Young, the error formula
# The squared error of the rank-k truncation equals the sum of the squared
# singular values that were dropped.
best_err = np.sum((F_zero - Fk_zero) ** 2)
tail_sum = np.sum(sig_zero[K:] ** 2)
print(f"Prop 5  ||A - A_k||_F^2 = {best_err:.4f},  "
      f"sum of dropped sigma^2 = {tail_sum:.4f}")
assert np.isclose(best_err, tail_sum)


# %% Check Proposition 5: Eckart-Young, optimality
# No rank-k matrix should fit A = F_zero better than A_k. We try three kinds
# of rank-k competitor, 1000 of each:
#   (a) random:     B = X Y^T with random X (M x k) and Y (N x k);
#   (b) projection: the best fit of A whose rows lie in a random
#                   k-dimensional subspace, B = A Q Q^T (Q orthonormal);
#   (c) nearby:     A_k with both factors nudged slightly at random.
# (c) is the hardest test, because these matrices sit right next to A_k.
A = F_zero
AZ = A @ Zk_zero                        # = W_k Sigma_k, the left factor of A_k
competitor_best = {"random": np.inf, "projection": np.inf, "nearby": np.inf}
for _ in range(1000):
    B_rand = rng.normal(size=(M, K)) @ rng.normal(size=(K, N))
    Q, _ = np.linalg.qr(rng.normal(size=(N, K)))
    B_proj = A @ Q @ Q.T
    B_near = ((AZ + 0.01 * rng.normal(size=AZ.shape))
              @ (Zk_zero + 0.01 * rng.normal(size=Zk_zero.shape)).T)
    for name, B in [("random", B_rand), ("projection", B_proj), ("nearby", B_near)]:
        competitor_best[name] = min(competitor_best[name], np.sum((A - B) ** 2))

print(f"        Error of A_k: {best_err:.4f}")
for name, err in competitor_best.items():
    print(f"        Best of 1000 {name:<10} competitors: {err:.4f}")
assert min(competitor_best.values()) > best_err


# %% Check Q5: the filled SVD against ALS on the hidden entries
# For contrast: if every entry were known, the SVD would be the right tool.
sig_full, Rk_full, _ = truncated_svd(R, K)
print("Q5      Singular values of the fully observed R (first 6):",
      np.round(sig_full[:6], 2), "-> sharp drop after sigma_3")

rmse_hidden_zero = rmse(Fk_zero[~mask], R_true[~mask])
rmse_hidden_mean = rmse(Fk_mean[~mask], R_true[~mask])
rmse_hidden_full = rmse(Rk_full[~mask], R_true[~mask])
print(f"        Hidden-entry error: ALS {rmse_hidden_als:.4f}, "
      f"zero-filled SVD {rmse_hidden_zero:.4f}")
assert rmse_hidden_als < rmse_hidden_zero   # ALS must beat the filled SVD


# %% Part C: bias baseline data with known truth (Prop 6)
# True ratings are mu + b_u + c_i, observed with noise on the same mask.
b_true = USER_BIAS_SD * rng.normal(size=M)
c_true = MOVIE_BIAS_SD * rng.normal(size=N)
Rb_true = MU_TRUE + b_true[:, None] + c_true[None, :]
Rb = Rb_true + BIAS_NOISE_SD * rng.normal(size=(M, N))

mu_hat, b_hat, c_hat, B_history = fit_biases(Rb, mask, LAM_B)
n_bias_sweeps = (len(B_history) - 1) // 2
print(f"Bias baseline fitted in {n_bias_sweeps} sweeps, mu_hat = {mu_hat:.4f}")


# %% Check Proposition 6: the updates solve the stacked least-squares system
# Stack all biases into x = (b, c), of length M + N. Each known rating gives
# one row of the design matrix D, with a 1 in column u and a 1 in column
# M + i, and the response is y_ui = r_ui - mu. Prop 6 says the updates solve
# (D^T D + lam_b I) x = D^T y. numpy.linalg.lstsq minimizes ||D x - y||^2
# only, so we append the rows sqrt(lam_b) I (with response 0) to include the
# penalty lam_b ||x||^2.
rows, cols = np.nonzero(mask)
D_stack = np.zeros((len(rows), M + N))
D_stack[np.arange(len(rows)), rows] = 1.0
D_stack[np.arange(len(rows)), M + cols] = 1.0
y_stack = Rb[rows, cols] - mu_hat

D_aug = np.vstack([D_stack, np.sqrt(LAM_B) * np.eye(M + N)])
y_aug = np.concatenate([y_stack, np.zeros(M + N)])
x_lstsq = np.linalg.lstsq(D_aug, y_aug, rcond=None)[0]

bias_gap = np.abs(x_lstsq - np.concatenate([b_hat, c_hat])).max()
print(f"Prop 6  Largest gap between our biases and lstsq: {bias_gap:.2e}")
assert bias_gap < 1e-8

# Remark 7: like ALS, the bias objective never increases.
bias_increases = np.diff(B_history) > 1e-9 * B_history[:-1]
print("        Half-steps where B increased:", bias_increases.sum(),
      "of", len(B_history) - 1)
assert not bias_increases.any()

# Individual biases are only defined up to a shared shift (adding d to every
# b_u and subtracting d from every c_i leaves every prediction unchanged), so
# we compare predictions with the truth rather than the biases themselves.
Rb_hat = mu_hat + b_hat[:, None] + c_hat[None, :]
rmse_hidden_bias = rmse(Rb_hat[~mask], Rb_true[~mask])
rmse_hidden_mu = rmse(mu_hat, Rb_true[~mask])
print(f"        Hidden-entry error: mu + b_u + c_i {rmse_hidden_bias:.4f}, "
      f"global mean only {rmse_hidden_mu:.4f}")


# %% Check Proposition 8: coverage of the 95% confidence interval
# Each experiment draws N_TEST prediction errors from N(0, TRUE_RMSE^2), so
# the true MSE is TRUE_RMSE^2 and the true RMSE is TRUE_RMSE. We build the
# interval of Prop 8 from the squared errors and count how often it contains
# the truth. Prop 8 says this should happen about 95% of the time.
true_mse = TRUE_RMSE**2
covered_mse = 0
covered_rmse = 0
for _ in range(N_REPEATS):
    e = rng.normal(scale=TRUE_RMSE, size=N_TEST) ** 2   # squared test errors
    e_bar, s = e.mean(), e.std(ddof=1)
    half_width = 1.96 * s / np.sqrt(N_TEST)
    lo, hi = e_bar - half_width, e_bar + half_width
    covered_mse += lo <= true_mse <= hi
    # RMSE interval: square roots of the endpoints, lower end clipped at 0.
    covered_rmse += np.sqrt(max(lo, 0.0)) <= TRUE_RMSE <= np.sqrt(hi)

coverage = covered_mse / N_REPEATS
coverage_se = np.sqrt(0.95 * 0.05 / N_REPEATS)   # Monte Carlo standard error
print(f"Prop 8  MSE interval contained the truth in {covered_mse} of "
      f"{N_REPEATS} experiments ({coverage:.1%})")
print(f"        RMSE interval contained the truth in {covered_rmse} of "
      f"{N_REPEATS} experiments (same events, as Step 3 says)")
assert abs(coverage - 0.95) < 3 * coverage_se
assert covered_rmse == covered_mse


# %% Summary tables for the "Simulation Check" subsection
# U and V cannot be compared with true factors directly: for any invertible
# Q, U V^T = (U Q)(V Q^(-T))^T, so the factors are only determined up to
# that change. We compare quantities that do not depend on Q instead.
sig_true, _, _ = truncated_svd(R_true, K)
sig_als, _, _ = truncated_svd(R_hat, K)

print("\nTable 1: true values vs. ALS estimates")
print(f"{'Quantity':<38}{'True':>10}{'ALS':>10}")
for j in range(K + 1):
    print(f"{f'Singular value {j + 1} of R_true / U V^T':<38}"
          f"{sig_true[j]:>10.3f}{sig_als[j]:>10.3f}")
print(f"{'Noise sd / error on known entries':<38}"
      f"{NOISE_SD:>10.3f}{rmse_train_als:>10.3f}")

print("\nTable 2: error on the hidden entries (RMSE vs. the truth)")
print(f"{'Method':<38}{'RMSE':>10}")
print(f"{'ALS, rank 3, lambda = 1':<38}{rmse_hidden_als:>10.4f}")
print(f"{'Truncated SVD, zero fill':<38}{rmse_hidden_zero:>10.4f}")
print(f"{'Truncated SVD, mean fill':<38}{rmse_hidden_mean:>10.4f}")
print(f"{'Truncated SVD, all entries known':<38}{rmse_hidden_full:>10.4f}")
print(f"{'Bias baseline (bias data)':<38}{rmse_hidden_bias:>10.4f}")
print(f"{'Global mean only (bias data)':<38}{rmse_hidden_mu:>10.4f}")

print("\nTable 3: bias baseline and confidence interval, true vs. estimate")
print(f"{'Quantity':<38}{'True':>10}{'Estimate':>10}")
# mu_hat is the mean of the observed ratings, so it estimates the mean of
# all true ratings (MU_TRUE plus the average biases), not MU_TRUE itself.
print(f"{'Mean of all true ratings / mu_hat':<38}{Rb_true.mean():>10.3f}{mu_hat:>10.3f}")
print(f"{'Interval coverage':<38}{0.95:>10.3f}{coverage:>10.3f}")

print("\nAll checks passed: Prop 1, Prop 2, Thm 4, Prop 5, Q5, Prop 6, Prop 8.")