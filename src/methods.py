"""
From-scratch functions for the matrix-factorization recommender.

Method 2 of the report: alternating least squares (ALS), one function per step.

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

Owner: Muhammad Saad Malik (ALS)
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
#             then subtract (G[r, c] / G[c, c]) × row c from every row r > c.
#   Backward: x_r = (h_r − Σ_{j>r} G[r, j] x_j) / G[r, r],   r = k−1, …, 0.
# About (2/3) k³ operations. G is invertible by Proposition 2 (λ > 0).
def my_gauss_solve(G, h):
    k = G.shape[0]
    A = np.empty((k, k + 1), dtype=np.result_type(G, h, np.float64))   # [G | h]
    A[:, :k] = G
    A[:, k] = h
    for c in range(k):
        p = c + int(np.argmax([abs(A[r, c]) for r in range(c, k)]))
        if A[p, c] == 0:
            raise ValueError("G is singular")
        if p != c:
            A[[c, p]] = A[[p, c]]
        for r in range(c + 1, k):
            A[r, c:] = A[r, c:] - (A[r, c] / A[c, c]) * A[c, c:]
    x = np.empty(k, dtype=A.dtype)
    for r in range(k - 1, -1, -1):
        s = A[r, k]
        for j in range(r + 1, k):
            s = s - A[r, j] * x[j]
        x[r] = s / A[r, r]
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
