"""Simulation with known truth for the matrix-factorization recommender.

We build a rating matrix whose true structure we know (rank 3 plus small
noise), hide 70% of its entries, and fit the known entries with Alternating
Least Squares (ALS). Because the truth is known, we can check each derived
result numerically:

    Q2  Each ALS row update solves its own normal equations.
    Q3  V_u^T V_u + lambda*I is positive definite (every eigenvalue >= lambda).
    Q4  The ALS objective J(U, V) never increases.
    Q5  The truncated SVD is the best rank-k fit of a full matrix
        (Eckart-Young), but after filling in the missing entries it predicts
        the hidden ratings much worse than ALS.

Run all:          python src/simulation.py
Or cell by cell:  in VS Code, click "Run Cell" above each "# %%" line.
Output:           printed checks, two summary tables, and the figure
                  figures/simulation_objective.png

Owners: Muhammad Saad Malik (ALS part, Q2-Q4)
        Jagadeesh Aravind Kalyan Shetti (SVD part, Q5)
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

# Model parameters: these are the "known truth" of the simulation.
M = 100           # number of users (rows of R)
N = 80            # number of movies (columns of R)
K = 3             # true rank of R, and the rank we fit
LAM = 1.0         # regularization strength lambda
NOISE_SD = 0.1    # standard deviation of the noise added to the true ratings
OBS_FRAC = 0.3    # fraction of entries we observe (the other 70% are hidden)
N_SWEEPS = 50     # number of ALS sweeps (one sweep = update users, then movies)


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
    """Return A = Vu^T Vu + lam*I and b = Vu^T ru for one user (Q2).

    Vu holds the rows of V for the movies this user rated (size |Omega_u| x k)
    and ru holds the user's known ratings (length |Omega_u|). The user's
    vector u_u is the solution of A u_u = b.
    """
    k = Vu.shape[1]
    A = Vu.T @ Vu + lam * np.eye(k)
    b = Vu.T @ ru
    return A, b


def update_rows(R, mask, V, lam):
    """With V fixed, give every row its best vector by solving its own
    normal equations: u_u = (V_u^T V_u + lam*I)^(-1) V_u^T r_u  (Q2).

    The same function updates the movies: pass R.T, mask.T and U instead,
    because swapping the roles of users and movies transposes the problem.
    """
    n_rows, k = R.shape[0], V.shape[1]
    U = np.zeros((n_rows, k))
    for u in range(n_rows):
        Vu = V[mask[u]]          # rows of V for the movies user u rated
        ru = R[u, mask[u]]       # user u's known ratings
        A, b = normal_equations(Vu, ru, lam)
        U[u] = np.linalg.solve(A, b)   # solve A u_u = b without forming A^(-1)
    return U


def truncated_svd(F, k):
    """Rank-k truncated SVD of a full matrix F, built from the eigenvalues
    and eigenvectors of F^T F.

    F^T F is symmetric positive semi-definite, so its eigenvalues lambda_j
    are real and non-negative, and the singular values are
    sigma_j = sqrt(lambda_j). With V_k holding the top-k eigenvectors,
    U_k Sigma_k = F V_k, so the rank-k approximation is
    F_k = U_k Sigma_k V_k^T = F V_k V_k^T.

    Returns all singular values (largest first) and F_k.
    """
    evals, evecs = np.linalg.eigh(F.T @ F)    # eigh returns ascending order
    evals, evecs = evals[::-1], evecs[:, ::-1]  # reorder: largest first
    sig = np.sqrt(np.clip(evals, 0, None))    # clip tiny negatives from rounding
    Vk = evecs[:, :k]
    Fk = F @ Vk @ Vk.T
    return sig, Fk


def rmse(a, b):
    """Root mean squared difference between two arrays."""
    return np.sqrt(np.mean((a - b) ** 2))


# %% Build the data with known truth
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


# %% Run ALS and record J after every half-step
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


# %% Check Q4: the objective never increases
# Compare each value of J with the one before it. A change larger than a
# tiny rounding tolerance (relative to J) would count as an increase.
changes = np.diff(J_history)
increases = changes > 1e-9 * J_history[:-1]

print("Q4  Half-steps where J increased:", increases.sum(), "of", len(changes))
print(f"    Largest change between half-steps: {changes.max():.2e}  (should be <= 0)")
assert not increases.any(), "J increased: Q4 is violated"

# Plot J against the sweep number; half-steps sit at 0.5, 1.5, ...
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(np.arange(len(J_history)) / 2, J_history, marker=".", color="steelblue")
ax.set_yscale("log")
ax.set_xlabel("ALS sweep")
ax.set_ylabel("Objective J(U, V) (log scale)")
ax.set_title("The ALS objective never increases")
fig.tight_layout()
fig.savefig(FIG_DIR / "simulation_objective.png", dpi=150)


# %% Check Q2: every user update solves its own normal equations
# The last half-step updated the movies, so U is not yet the best choice for
# the current V. One more user update gives the U that Q2 describes.
U_check = update_rows(R, mask, V, LAM)

max_formula_gap = 0.0   # solve() result vs. the explicit formula in Q2
max_gradient = 0.0      # gradient of J with respect to u_u at the update
for u in range(M):
    Vu, ru = V[mask[u]], R[u, mask[u]]
    A, b = normal_equations(Vu, ru, LAM)

    # The formula exactly as written in Q2, with an explicit inverse.
    u_formula = np.linalg.inv(A) @ b
    max_formula_gap = max(max_formula_gap, np.abs(U_check[u] - u_formula).max())

    # Gradient of J in u_u: -2 Vu^T (ru - Vu u_u) + 2 lam u_u.
    # At the minimizer it must be zero.
    grad = -2 * Vu.T @ (ru - Vu @ U_check[u]) + 2 * LAM * U_check[u]
    max_gradient = max(max_gradient, np.abs(grad).max())

print(f"Q2  Largest gap between solve() and the formula: {max_formula_gap:.2e}")
print(f"    Largest gradient entry at the update:         {max_gradient:.2e}")
assert max_formula_gap < 1e-8 and max_gradient < 1e-8


# %% Check Q3: V_u^T V_u + lambda*I is positive definite
# Q3 gives x^T A x = ||V_u x||^2 + lambda ||x||^2 >= lambda ||x||^2, so every
# eigenvalue of A is at least lambda. Check this for every user and movie.
min_eig = np.inf
for u in range(M):
    A, _ = normal_equations(V[mask[u]], R[u, mask[u]], LAM)
    min_eig = min(min_eig, np.linalg.eigvalsh(A).min())
for i in range(N):
    A, _ = normal_equations(U[mask[:, i]], R[mask[:, i], i], LAM)
    min_eig = min(min_eig, np.linalg.eigvalsh(A).min())

print(f"Q3  Smallest eigenvalue over all users and movies: {min_eig:.4f} "
      f"(lambda = {LAM})")
assert min_eig >= LAM - 1e-9

# Why lambda matters: a user who rated fewer movies than k. V_u^T V_u then
# has rank at most 2 < k = 3, so it is singular, and adding lambda*I fixes it.
V_few = V[:2]                              # pretend the user rated only 2 movies
A_no_reg = V_few.T @ V_few                 # lambda = 0
A_reg = A_no_reg + LAM * np.eye(K)         # lambda = 1
print(f"    User with 2 ratings: rank of V_u^T V_u = "
      f"{np.linalg.matrix_rank(A_no_reg)} of {K}")
print(f"    Smallest eigenvalue with lambda = 0: "
      f"{np.linalg.eigvalsh(A_no_reg).min():.1e}  (zero up to rounding: singular)")
print(f"    Smallest eigenvalue with lambda = 1: "
      f"{np.linalg.eigvalsh(A_reg).min():.4f}  (invertible)")


# %% ALS predictions compared with the truth
R_hat = U @ V.T
rmse_hidden_als = rmse(R_hat[~mask], R_true[~mask])   # entries ALS never saw
rmse_train_als = rmse(R_hat[mask], R[mask])           # fit to the known ratings

print(f"ALS error on hidden entries (vs. R_true): {rmse_hidden_als:.4f}")
print(f"ALS error on known entries (vs. R):       {rmse_train_als:.4f}  "
      f"(noise sd = {NOISE_SD})")


# %% Check Q5: truncated SVD of a filled-in matrix
# SVD needs a full matrix, so the hidden entries must be filled first.
F_zero = np.where(mask, R, 0.0)                 # fill with zeros
F_mean = np.where(mask, R, R[mask].mean())      # fill with the mean known rating
# (Here the mean is close to 0, so the two fills behave alike. On MovieLens,
# with a mean rating of 3.53, the choice matters much more.)

sig_zero, Fk_zero = truncated_svd(F_zero, K)
sig_mean, Fk_mean = truncated_svd(F_mean, K)

# Our singular values against the library routine (final comparison only).
sig_lib = np.linalg.svd(F_zero, compute_uv=False)
print(f"Q5  Largest gap between our singular values and numpy.linalg.svd: "
      f"{np.abs(sig_zero[:len(sig_lib)] - sig_lib).max():.2e}")

# Eckart-Young: the squared error of the best rank-k approximation equals
# the sum of the squared singular values that were dropped.
lhs = np.sum((F_zero - Fk_zero) ** 2)
rhs = np.sum(sig_zero[K:] ** 2)
print(f"    ||F - F_k||_F^2 = {lhs:.4f},  sum of dropped sigma^2 = {rhs:.4f}")
assert np.isclose(lhs, rhs)

# For contrast: if every entry were known, the SVD would be the right tool.
sig_full, Rk_full = truncated_svd(R, K)
print("    Singular values of the fully observed R (first 6):",
      np.round(sig_full[:6], 2), "-> sharp drop after sigma_3")

rmse_hidden_zero = rmse(Fk_zero[~mask], R_true[~mask])
rmse_hidden_mean = rmse(Fk_mean[~mask], R_true[~mask])
rmse_hidden_full = rmse(Rk_full[~mask], R_true[~mask])
assert rmse_hidden_als < rmse_hidden_zero   # ALS must beat the filled SVD


# %% Summary tables for the "Simulation Check" subsection
# U and V cannot be compared with true factors directly: for any invertible
# Q, U V^T = (U Q)(V Q^(-T))^T, so the factors are only determined up to
# that change. We compare quantities that do not depend on Q instead.
sig_true, _ = truncated_svd(R_true, K)
sig_als, _ = truncated_svd(R_hat, K)

print("\nTable 1: true values vs. ALS estimates")
print(f"{'Quantity':<38}{'True':>10}{'ALS':>10}")
for j in range(K + 1):
    print(f"{f'Singular value {j + 1} of R_true / U V^T':<38}"
          f"{sig_true[j]:>10.3f}{sig_als[j]:>10.3f}")
print(f"{'Noise sd / error on known entries':<38}"
      f"{NOISE_SD:>10.3f}{rmse_train_als:>10.3f}")

print("\nTable 2: error on the hidden entries (RMSE vs. R_true)")
print(f"{'Method':<38}{'RMSE':>10}")
print(f"{'ALS, rank 3, lambda = 1':<38}{rmse_hidden_als:>10.4f}")
print(f"{'Truncated SVD, zero fill':<38}{rmse_hidden_zero:>10.4f}")
print(f"{'Truncated SVD, mean fill':<38}{rmse_hidden_mean:>10.4f}")
print(f"{'Truncated SVD, all entries known':<38}{rmse_hidden_full:>10.4f}")

print("\nAll checks passed: Q2, Q3, Q4 and Q5.")