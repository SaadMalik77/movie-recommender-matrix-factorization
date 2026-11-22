"""
From-scratch functions for the matrix-factorization recommender, one function per
step of each method in the report:
  Method 1  the bias baseline  r̂_ui = µ + b_u + c_i          (Proposition 6)
  Method 2  alternating least squares (ALS)                   (Propositions 1, 2, Theorem 4)
  Method 3  truncated SVD of the filled matrix, power iteration (Proposition 5)
plus the 95% confidence interval for the test RMSE (Proposition 8).

Notation (as in the report):
  m, n         number of users and movies
  R, mask      rating matrix (m × n) and the mask M of known ratings, M_ui = 1 iff (u, i) ∈ Ω
  k, λ         number of latent factors, regularization strength
  U, V         user factors (m × k, row u_u) and movie factors (n × k, row v_i)
  I_u          movies rated by user u;   V_u, r_u: rows of V for those movies, u's ratings
  𝒰_i          users who rated movie i;  U_i, r_(i): rows of U for those users, i's ratings
  G_u, h_u     G_u = V_uᵀ V_u + λ I_k  and  h_u = V_uᵀ r_u,  so that  G_u u_u = h_u
  µ            mean of the training ratings

The core method uses only NumPy arrays, matrix products and our own Gaussian
elimination. numpy.linalg appears only in the verification scripts, as a check.

Every function also accepts NumPy arrays of fractions.Fraction (dtype=object),
so the small worked example runs exactly on the same code as the simulation
and the real data.

Owners: Muhammad Saad Malik (Method 2, ALS)
        Jagadeesh Aravind Kalyan Shetti (Method 3, truncated SVD)
        Emmanuel Adjei Danso (Method 1, bias baseline; confidence intervals)
"""
import numpy as np


# ---------------------------------------------------------------------------
# Method 2: alternating least squares
# ---------------------------------------------------------------------------

# Step 1. Centre the ratings.
#   µ = (1 / |Ω|) Σ_{(u,i)∈Ω} r_ui,    and the factors are fitted to r_ui − µ.
# Entries outside Ω are never read, so their values do not matter.
def my_centre(R, mask):
    mu = np.sum(R[mask]) / int(mask.sum())
    return mu, R - mu


# Step 2. Initialize the factors with small random entries.
#   U⁽⁰⁾ ∈ R^{m×k},  V⁽⁰⁾ ∈ R^{n×k},  entries from N(0, scale²),
#   drawn with np.random.default_rng(seed).
def my_init_factors(m, n, k, seed=0, scale=0.1):
    rng = np.random.default_rng(seed)
    U = scale * rng.normal(size=(m, k))
    V = scale * rng.normal(size=(n, k))
    return U, V


# Normal equations of one row (Proposition 1):
#   G = Fᵀ F + λ I_k   (k × k),     h = Fᵀ r   (length k),     G x = h.
#   User u:   F = V_u,  r = r_u      →  G_u u_u = h_u
#   Movie i:  F = U_i,  r = r_(i)    →  (U_iᵀ U_i + λ I_k) v_i = U_iᵀ r_(i)
# A row with no ratings gives G = λ I_k and h = 0, so x = 0.
def my_normal_equations(F, r, lam):
    k = F.shape[1]
    G = F.T @ F + lam * np.eye(k, dtype=F.dtype)
    h = F.T @ r
    return G, h


# Solve G x = h by Gaussian elimination with partial pivoting.
#   Forward:  for each column c, swap up the row with the largest |G[r, c]| (r ≥ c),
#             then subtract (G[r, c] / G[c, c]) × row c from every row r > c
#             (all rows r > c at once, as one outer product).
#   Backward: x_r = (h_r − Σ_{j>r} G[r, j] x_j) / G[r, r],   r = k−1, …, 0.
# About (2/3) k³ operations. G is invertible by Proposition 2 (λ > 0).
def my_gauss_solve(G, h):
    k = G.shape[0]
    A = np.empty((k, k + 1), dtype=np.result_type(G, h, np.float64))   # [G | h]
    A[:, :k] = G
    A[:, k] = h
    for c in range(k):
        p = c + int(np.argmax(np.abs(A[c:, c])))
        if A[p, c] == 0:
            raise ValueError("G is singular")
        if p != c:
            A[[c, p]] = A[[p, c]]
        A[c + 1:, c:] = A[c + 1:, c:] - np.outer(A[c + 1:, c] / A[c, c], A[c, c:])
    x = np.empty(k, dtype=A.dtype)
    for r in range(k - 1, -1, -1):
        x[r] = (A[r, k] - np.dot(A[r, r + 1:k], x[r + 1:])) / A[r, r]
    return x


# Step 3. Update every user with V fixed (Proposition 1):
#   for u = 1, …, m:   G_u u_u = h_u,   G_u = V_uᵀ V_u + λ I_k,   h_u = V_uᵀ r_u.
def my_update_U(R, mask, V, lam):
    m, k = R.shape[0], V.shape[1]
    U = np.empty((m, k), dtype=np.result_type(R, V, np.float64))
    for u in range(m):
        I_u = mask[u]                                   # movies rated by user u
        G_u, h_u = my_normal_equations(V[I_u], R[u, I_u], lam)
        U[u] = my_gauss_solve(G_u, h_u)
    return U


# Step 4. Update every movie with U fixed:
#   for i = 1, …, n:   (U_iᵀ U_i + λ I_k) v_i = U_iᵀ r_(i).
# Exchanging users and movies turns this into Step 3 applied to Rᵀ ≈ V Uᵀ.
def my_update_V(R, mask, U, lam):
    return my_update_U(R.T, mask.T, U, lam)


# Step 5. The objective, in its two parts:
#   J(U, V) = Σ_{(u,i)∈Ω} (r_ui − u_uᵀ v_i)²  +  λ (‖U‖_F² + ‖V‖_F²)
#             └────── squared error ──────┘     └──── penalty ────┘
def my_objective(R, mask, U, V, lam, parts=False):
    E = R - U @ V.T
    sse = np.sum(E[mask] ** 2)
    penalty = lam * (np.sum(U ** 2) + np.sum(V ** 2))
    return (sse + penalty, sse, penalty) if parts else sse + penalty


# Step 6. Run ALS: Step 2, then Steps 3–5 until the relative drop in J over one
# sweep is below tol, or for at most `iters` sweeps.
# Step 5's check (Theorem 4): an increase of J after any half-step is an error.
# Returns U, V and J after every half-step:
#   J_hist[0] at the start, J_hist[2t−1] after the user update of sweep t,
#   J_hist[2t] after its movie update.
def my_als(R, mask, k, lam, iters=50, seed=0, tol=0.0, scale=0.1):
    m, n = R.shape
    U, V = my_init_factors(m, n, k, seed, scale)
    J_hist = [my_objective(R, mask, U, V, lam)]
    for _ in range(iters):
        J_start = J_hist[-1]
        U = my_update_U(R, mask, V, lam)
        J_hist.append(my_objective(R, mask, U, V, lam))
        V = my_update_V(R, mask, U, lam)
        J_hist.append(my_objective(R, mask, U, V, lam))
        for a, b in ((J_hist[-3], J_hist[-2]), (J_hist[-2], J_hist[-1])):
            if b > a + 1e-9 * abs(a):
                raise RuntimeError(f"J increased from {a} to {b}: Theorem 4 violated")
        if tol > 0 and J_start - J_hist[-1] <= tol * J_start:
            break
    return U, V, np.array(J_hist)


# Step 7. Predict:  r̂_ui = µ + u_uᵀ v_i,  clipped to [lo, hi] when clip = (lo, hi).
def my_predict(U, V, mu=0, clip=None):
    P = mu + U @ V.T
    return P if clip is None else np.clip(P, *clip)


# ---------------------------------------------------------------------------
# Method 1: the bias baseline  r̂_ui = µ + b_u + c_i  (Proposition 6)
# ---------------------------------------------------------------------------

# Step 3, users. With the movie biases c fixed:
#   b_u = Σ_{i∈I_u} (r_ui − µ − c_i) / (|I_u| + λ_b)
def my_update_b(R, mask, mu, c, lam_b):
    resid = np.where(mask, R - mu - c[None, :], 0.0)
    return resid.sum(axis=1) / (mask.sum(axis=1) + lam_b)


# Step 3, movies. With the user biases b fixed:
#   c_i = Σ_{u∈𝒰_i} (r_ui − µ − b_u) / (|𝒰_i| + λ_b)
# Exchanging users and movies turns this into the user update on Rᵀ.
def my_update_c(R, mask, mu, b, lam_b):
    return my_update_b(R.T, mask.T, mu, b, lam_b)


# The bias objective:
#   B(b, c) = Σ_{(u,i)∈Ω} (r_ui − µ − b_u − c_i)² + λ_b (Σ_u b_u² + Σ_i c_i²)
def my_bias_objective(R, mask, mu, b, c, lam_b):
    E = np.where(mask, R - mu - b[:, None] - c[None, :], 0.0)
    return np.sum(E ** 2) + lam_b * (b @ b + c @ c)


# Steps 1–3. µ = mean of the known ratings, b = c = 0, then alternate the two
# updates until no bias changes by more than tol. Like ALS, B must never rise
# (Remark 7); an increase is an error. Returns µ, b, c and B after every half-step.
def my_fit_biases(R, mask, lam_b, tol=1e-12, max_sweeps=10_000):
    mu = R[mask].mean()
    b, c = np.zeros(R.shape[0]), np.zeros(R.shape[1])
    hist = [my_bias_objective(R, mask, mu, b, c, lam_b)]
    for _ in range(max_sweeps):
        b_new = my_update_b(R, mask, mu, c, lam_b)
        hist.append(my_bias_objective(R, mask, mu, b_new, c, lam_b))
        c_new = my_update_c(R, mask, mu, b_new, lam_b)
        hist.append(my_bias_objective(R, mask, mu, b_new, c_new, lam_b))
        for a, z in ((hist[-3], hist[-2]), (hist[-2], hist[-1])):
            if z > a + 1e-9 * abs(a):
                raise RuntimeError(f"B increased from {a} to {z}: Remark 7 violated")
        change = max(np.abs(b_new - b).max(), np.abs(c_new - c).max())
        b, c = b_new, c_new
        if change < tol:
            break
    return mu, b, c, np.array(hist)


# Step 4. Predict:  r̂_ui = µ + b_u + c_i,  clipped to [lo, hi] when clip = (lo, hi).
def my_predict_biases(mu, b, c, clip=None):
    P = mu + b[:, None] + c[None, :]
    return P if clip is None else np.clip(P, *clip)


# ---------------------------------------------------------------------------
# Method 3: truncated SVD of the filled matrix, by power iteration (Prop. 5)
# ---------------------------------------------------------------------------

# Step 1. Fill the gaps:  R̃_ui = r_ui on Ω, and elsewhere the mean of movie i's
# known ratings (how="movie"; the global mean for a movie with none), the global
# mean (how="global"), or 0 (how="zero").
def my_fill(R, mask, how="movie"):
    mu = R[mask].mean()
    if how == "zero":
        fill = np.zeros(R.shape[1])
    elif how == "global":
        fill = np.full(R.shape[1], mu)
    else:
        cnt = mask.sum(axis=0)
        fill = np.where(cnt > 0, np.where(mask, R, 0.0).sum(axis=0) / np.maximum(cnt, 1), mu)
    return np.where(mask, R, fill[None, :])


# Step 2. The top k singular triples of A, one at a time. For j = 1, …, k start
# from a random unit vector x ∈ R^n and repeat until x stops changing (‖Δx‖ < tol):
#   y ← Aᵀ(A x),     y ← y − Σ_{l<j} (z_lᵀ y) z_l,     x ← y / ‖y‖.
# Then  z_j = x,  σ_j = ‖A z_j‖,  w_j = A z_j / σ_j.
# AᵀA (n × n) is never formed: each iteration is two matrix–vector products.
# The error shrinks like (σ_{j+1}/σ_j)^{2t} after t iterations (Assumption 7).
# Returns σ (length k), W (m × k), Z (n × k) and the iterations each factor used.
def my_svd_power(A, k, tol=1e-10, max_iter=5_000, seed=0):
    rng = np.random.default_rng(seed)
    m, n = A.shape
    sig, W, Z = np.zeros(k), np.zeros((m, k)), np.zeros((n, k))
    iters = np.zeros(k, dtype=int)
    for j in range(k):
        x = rng.normal(size=n)
        x = x - Z[:, :j] @ (Z[:, :j].T @ x)
        x = x / np.sqrt(x @ x)
        for t in range(1, max_iter + 1):
            y = A.T @ (A @ x)
            y = y - Z[:, :j] @ (Z[:, :j].T @ y)
            x_new = y / np.sqrt(y @ y)
            done = np.sqrt(np.sum((x_new - x) ** 2)) < tol
            x = x_new
            if done:
                break
        iters[j] = t
        Az = A @ x
        sig[j] = np.sqrt(Az @ Az)
        Z[:, j], W[:, j] = x, Az / sig[j]
    return sig, W, Z, iters


# Step 4. Predict with the first k triples:  R̂_k = Σ_{j≤k} σ_j w_j z_jᵀ,
# clipped to [lo, hi] when clip = (lo, hi).
def my_svd_predict(sig, W, Z, k=None, clip=None):
    k = len(sig) if k is None else k
    P = (W[:, :k] * sig[:k]) @ Z[:, :k].T
    return P if clip is None else np.clip(P, *clip)


# ---------------------------------------------------------------------------
# Error bars (Proposition 8)
# ---------------------------------------------------------------------------

# 95% confidence interval for the test error. With squared errors e_j over the N
# test pairs, mean ē and sample sd s:
#   θ ∈ [L, H] = [ē − 1.96 s/√N,  ē + 1.96 s/√N]   (true MSE),
#   √θ ∈ [√max(L, 0), √H]                           (true RMSE).
# Returns (RMSE, lower, upper).
def my_rmse_ci(P, T, mask, z=1.96):
    e = (P - T)[mask] ** 2
    half = z * e.std(ddof=1) / np.sqrt(e.size)
    lo, hi = e.mean() - half, e.mean() + half
    return float(np.sqrt(e.mean())), float(np.sqrt(max(lo, 0.0))), float(np.sqrt(hi))


# 95% interval for the difference in MSE between two predictors on the same test
# pairs. Proposition 8 applied to d_j = e_j(A) − e_j(B):  d̄ ± 1.96 s_d / √N.
# An interval that excludes 0 means the two errors really differ.
# Returns (difference, lower, upper).
def my_mse_diff_ci(P_a, P_b, T, mask, z=1.96):
    d = (P_a - T)[mask] ** 2 - (P_b - T)[mask] ** 2
    half = z * d.std(ddof=1) / np.sqrt(d.size)
    return float(d.mean()), float(d.mean() - half), float(d.mean() + half)


# ---------------------------------------------------------------------------
# Evaluation and comparison helpers
# ---------------------------------------------------------------------------

# RMSE over the cells selected by a mask:
#   RMSE = ( (1/N) Σ_{(u,i) in mask} (r̂_ui − r_ui)² )^{1/2}
def my_rmse(P, T, mask):
    return float(np.sqrt(np.mean((P - T)[mask] ** 2)))


# Print one comparison in the common format:
#   max |mine − ref|   and   relative error ‖mine − ref‖ / ‖ref‖
def report(name, mine, ref):
    mine, ref = np.asarray(mine, float), np.asarray(ref, float)
    abs_err = np.abs(mine - ref).max()
    rel_err = np.linalg.norm(mine - ref) / np.linalg.norm(ref)
    print(f"{name:38s} max|diff| = {abs_err:.2e}   rel = {rel_err:.2e}")
    return abs_err
