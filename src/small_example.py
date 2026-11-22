"""
Small worked example of ALS (Method 2), computed exactly with fractions.

Data:  a 3 × 3 rating matrix with 2 missing entries, k = 1, λ = 1.
Steps, numbered as in Method 2 of the report:
   Step 2  initialize: v⁽⁰⁾ = (1, 1, 1), chosen instead of random entries
           so that every number stays a short fraction
   Step 3  update every user  (V fixed):   G_u u_u = h_u,
           G_u = V_uᵀ V_u + λ I_k,  h_u = V_uᵀ r_u
   Step 4  update every movie (U fixed):   (U_iᵀ U_i + λ I_k) v_i = U_iᵀ r_(i)
   Step 5  objective J after each update
   Step 7  predict the missing entries,  r̂_ui = u_uᵀ v_i
Step 1 (centring) only shifts every rating by µ; it is skipped here so that
the fractions stay short. Every intermediate result is printed. The functions
come from methods.py, the same ones that run the simulation and the real data.

Second-way checks:
   (a) substitute each new row back into its normal equations: G x − h = 0
       exactly (equivalently ∇f_u = 2(G_u u_u − h_u) = 0, Proposition 1);
   (b) the objective does not increase (Theorem 4).

Run from the repository root:   python src/small_example.py
"""
from fractions import Fraction

import numpy as np

from methods import (my_normal_equations, my_gauss_solve, my_update_U,
                     my_update_V, my_objective, my_predict)

USERS = ["user 1", "user 2", "user 3"]
MOVIES = ["movie 1", "movie 2", "movie 3"]


def num(x):
    """Exact fraction, with a decimal when it is not a whole number."""
    x = Fraction(x)
    return str(x) if x.denominator == 1 else f"{x} (≈ {float(x):.4f})"


def vec(a):
    """Short exact form of a 1-D or 2-D array, e.g. [5, 4] or [[3]]."""
    a = np.asarray(a, dtype=object)
    if a.ndim == 1:
        return "[" + ", ".join(str(Fraction(x)) for x in a) + "]"
    return "[" + ", ".join(vec(row) for row in a) + "]"


def show_table(title, M, mask=None, mark=None):
    """Print a users × movies table; '?' for missing cells, '*' for predicted ones."""
    if title:
        print(title)
    cells = [[("?" if mask is not None and not mask[u, i] and mark is None
               else str(Fraction(M[u, i])) + ("*" if mark is not None and not mark[u, i] else ""))
              for i in range(M.shape[1])] for u in range(M.shape[0])]
    w = max(9, max(len(c) for row in cells for c in row) + 2)
    print(" " * 10 + "".join(f"{m:>{w}}" for m in MOVIES))
    for u, row in enumerate(cells):
        print(f"{USERS[u]:10s}" + "".join(f"{c:>{w}}" for c in row))
    print()


def show_J(label, R, mask, U, V, lam):
    J, sse, pen = my_objective(R, mask, U, V, lam, parts=True)
    print(label)
    print(f"  squared error  Σ_Ω (r_ui − u_uᵀ v_i)² = {num(sse)}")
    print(f"  penalty        λ(‖U‖_F² + ‖V‖_F²)     = {num(pen)}")
    print(f"  J                                      = {num(J)}\n")
    return J


def rule(title):
    print("=" * 66)
    print(title)
    print("=" * 66)


# ---------------------------------------------------------------------------
# Data and Step 2 (initialize)
# ---------------------------------------------------------------------------
ratings = [[5,    4,    None],
           [4,    None, 2   ],
           [1,    2,    1   ]]

mask = np.array([[r is not None for r in row] for row in ratings])        # Ω
R = np.array([[Fraction(r if r is not None else 0) for r in row]
              for row in ratings], dtype=object)                            # 0 where missing (never read)
k, lam = 1, Fraction(1)
V0 = np.array([[Fraction(1)], [Fraction(1)], [Fraction(1)]], dtype=object)

rule("DATA AND STEP 2 (INITIALIZE)")
show_table("Rating matrix R (? = missing):", R, mask)
print(f"k = {k},  λ = {lam},  |Ω| = {mask.sum()} known ratings")
print(f"v⁽⁰⁾ = {vec(V0[:, 0])}\n")

# ---------------------------------------------------------------------------
# Step 3: update every user with V fixed
# ---------------------------------------------------------------------------
rule("STEP 3  Update every user (V fixed):  G_u u_u = h_u")
for u in range(3):
    I_u = mask[u]
    V_u, r_u = V0[I_u], R[u, I_u]
    G_u, h_u = my_normal_equations(V_u, r_u, lam)
    u_u = my_gauss_solve(G_u, h_u)
    rated = ", ".join(m for m, s in zip(MOVIES, I_u) if s)
    print(f"{USERS[u]} rated {rated}")
    print(f"  V_u = {vec(V_u)}   (rows of V for those movies)")
    print(f"  r_u = {vec(r_u)}   (ratings given)")
    print(f"  G_u = V_uᵀ V_u + λI = {vec(G_u)}")
    print(f"  h_u = V_uᵀ r_u      = {vec(h_u)}")
    print(f"  solve G_u u_u = h_u  →  u_u = {num(u_u[0])}\n")

U1 = my_update_U(R, mask, V0, lam)
print(f"U⁽¹⁾ = {vec(U1[:, 0])}\n")
J_after_users = show_J("STEP 5  Objective after Step 3  (U⁽¹⁾, V⁽⁰⁾):", R, mask, U1, V0, lam)

# ---------------------------------------------------------------------------
# Step 4: update every movie with U fixed (using the new U⁽¹⁾)
# ---------------------------------------------------------------------------
rule("STEP 4  Update every movie (U fixed):  (U_iᵀ U_i + λI) v_i = U_iᵀ r_(i)")
for i in range(3):
    in_i = mask[:, i]
    U_i, r_i = U1[in_i], R[in_i, i]
    G_i, h_i = my_normal_equations(U_i, r_i, lam)
    v_i = my_gauss_solve(G_i, h_i)
    raters = ", ".join(u for u, s in zip(USERS, in_i) if s)
    print(f"{MOVIES[i]} rated by {raters}")
    print(f"  U_i   = {vec(U_i)}   (rows of U for those users)")
    print(f"  r_(i) = {vec(r_i)}   (ratings received)")
    print(f"  U_iᵀ U_i + λI = {vec(G_i)}")
    print(f"  U_iᵀ r_(i)    = {vec(h_i)}")
    print(f"  solve  →  v_i = {num(v_i[0])}\n")

V1 = my_update_V(R, mask, U1, lam)
print(f"V⁽¹⁾ = {vec(V1[:, 0])}\n")
J_after_items = show_J("STEP 5  Objective after Step 4  (U⁽¹⁾, V⁽¹⁾):", R, mask, U1, V1, lam)

# ---------------------------------------------------------------------------
# Step 7: predictions
# ---------------------------------------------------------------------------
rule("STEP 7  Predict  r̂_ui = u_uᵀ v_i  after one iteration (* = was missing)")
R_hat = my_predict(U1, V1)
show_table("", R_hat, mark=mask)
for u in range(3):
    for i in range(3):
        if not mask[u, i]:
            print(f"  {USERS[u]}, {MOVIES[i]}:  {num(U1[u, 0])} × {num(V1[i, 0])} = {num(R_hat[u, i])}")
print()

# ---------------------------------------------------------------------------
# Second-way checks
# ---------------------------------------------------------------------------
rule("CHECKS")
res_users = []
for u in range(3):
    G_u, h_u = my_normal_equations(V0[mask[u]], R[u, mask[u]], lam)
    res_users.append(G_u @ U1[u] - h_u)
res_items = []
for i in range(3):
    G_i, h_i = my_normal_equations(U1[mask[:, i]], R[mask[:, i], i], lam)
    res_items.append(G_i @ V1[i] - h_i)
print("(a) Substitute back into the normal equations, G x − h:")
print(f"    users:  {[vec(r) for r in res_users]}")
print(f"    movies: {[vec(r) for r in res_items]}")
ok_a = all(x == 0 for r in res_users + res_items for x in r)
print(f"    all exactly zero: {ok_a}\n")

print("(b) The objective does not increase (Theorem 4):")
print(f"    J after Step 3 = {num(J_after_users)}")
print(f"    J after Step 4 = {num(J_after_items)}")
ok_b = J_after_items <= J_after_users
print(f"    J did not increase: {ok_b}\n")

# Further iterations in floating point (exact fractions grow very long),
# with the same functions, to show J keeps decreasing.
Rf, U, V = R.astype(float), U1.astype(float), V1.astype(float)
J_hist = [float(J_after_items)]
for _ in range(9):
    U = my_update_U(Rf, mask, V, 1.0)
    V = my_update_V(Rf, mask, U, 1.0)
    J_hist.append(my_objective(Rf, mask, U, V, 1.0))
print("(c) J after iterations 1–10 (floating point):")
print("    " + ", ".join(f"{j:.4f}" for j in J_hist))
print(f"    non-increasing: {bool(np.all(np.diff(J_hist) <= 1e-12))}")

assert ok_a and ok_b
