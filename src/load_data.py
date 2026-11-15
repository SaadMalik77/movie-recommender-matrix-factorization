"""Load and explore MovieLens 100K.

Loads the ratings, describes the data, builds the rating matrix R and the mask
of known ratings, makes a seeded 80/20 train/test split, and saves plots to
figures/.

Run it all:       python src/load_data.py
Or cell by cell:  in VS Code, click "Run Cell" above each "# %%" line.

Owner: Muhammad Saad Malik | Checked by: Jagadeesh Aravind Kalyan Shetti
"""

# %% Setup
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Repo root, whether this runs as a script or cell by cell in VS Code
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent

DATA_DIR = ROOT / "data" / "ml-100k"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

SEED = 0          # fixed so every team member gets the same split
TEST_FRAC = 0.2   # hold out 20% of ratings for testing


# %% Functions
def load_ratings(data_dir=DATA_DIR):
    """Read u.data: one row per rating (user, movie, rating, timestamp)."""
    return pd.read_csv(
        data_dir / "u.data",
        sep="\t",
        names=["user", "movie", "rating", "timestamp"],
    )


def load_movies(data_dir=DATA_DIR):
    """Read movie ids and titles from u.item (pipe-separated, latin-1 text)."""
    return pd.read_csv(
        data_dir / "u.item",
        sep="|",
        encoding="latin-1",
        header=None,
        usecols=[0, 1],
        names=["movie", "title"],
    )


def build_matrix(ratings, n_users, n_movies):
    """Return the rating matrix R and the boolean mask of known ratings.

    Ids in the file start at 1, so user u goes to row u-1 and movie i to
    column i-1. R is 0 wherever no rating is known, so always use it together
    with the mask: a 0 in R means "unknown", never "zero stars".
    """
    rows = ratings["user"].to_numpy() - 1
    cols = ratings["movie"].to_numpy() - 1
    R = np.zeros((n_users, n_movies))
    mask = np.zeros((n_users, n_movies), dtype=bool)
    R[rows, cols] = ratings["rating"].to_numpy()
    mask[rows, cols] = True
    return R, mask


def split_ratings(ratings, test_frac, rng):
    """Randomly put a fraction test_frac of the ratings in the test set."""
    idx = rng.permutation(len(ratings))
    n_test = int(round(test_frac * len(ratings)))
    test = ratings.iloc[idx[:n_test]]
    train = ratings.iloc[idx[n_test:]]
    return train, test


# %% Load the data
ratings = load_ratings()
movies = load_movies()
print(ratings.head())

# %% Describe the ratings table
print("Shape of the ratings table:", ratings.shape)
print("\nColumn types:\n", ratings.dtypes, sep="")
print("\nMissing values per column:\n", ratings.isna().sum(), sep="")
print("\nDuplicate (user, movie) pairs:", ratings.duplicated(["user", "movie"]).sum())

dates = pd.to_datetime(ratings["timestamp"], unit="s")
print(f"\nRatings collected from {dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}")

# %% Describe the rating matrix
n_users = ratings["user"].max()
n_movies = ratings["movie"].max()
n_ratings = len(ratings)
density = n_ratings / (n_users * n_movies)

print(f"Users: {n_users}, movies: {n_movies}, ratings: {n_ratings}")
print(f"Matrix entries: {n_users * n_movies}")
print(f"Observed: {density:.2%}, unobserved: {1 - density:.2%}")

print("\nHow many ratings of each value:")
print(ratings["rating"].value_counts().sort_index())
print(f"Mean rating: {ratings['rating'].mean():.3f}, "
      f"standard deviation: {ratings['rating'].std():.3f}")

per_user = ratings.groupby("user").size()
per_movie = ratings.groupby("movie").size()
print(f"\nRatings per user:  min {per_user.min()}, "
      f"median {per_user.median():.0f}, max {per_user.max()}")
print(f"Ratings per movie: min {per_movie.min()}, "
      f"median {per_movie.median():.0f}, max {per_movie.max()}")
print(f"Movies with only 1 rating: {(per_movie == 1).sum()}")

top = (per_movie.sort_values(ascending=False).head(5)
       .rename("n_ratings").reset_index().merge(movies, on="movie"))
print("\nMost-rated movies:")
print(top.to_string(index=False))

# %% Build R and the mask of known ratings
R, mask = build_matrix(ratings, n_users, n_movies)
print("R shape:", R.shape)
print("Known entries:", mask.sum())
assert mask.sum() == n_ratings   # every rating landed in its own cell

# %% Train/test split (80/20)
rng = np.random.default_rng(SEED)
train, test = split_ratings(ratings, TEST_FRAC, rng)
R_train, mask_train = build_matrix(train, n_users, n_movies)
R_test, mask_test = build_matrix(test, n_users, n_movies)

print(f"Train ratings: {len(train)}, test ratings: {len(test)}")
assert not (mask_train & mask_test).any()   # no rating is in both sets

# Test ratings whose user or movie has no training rating ("cold start")
print("Test ratings whose user is not in train:",
      (~test["user"].isin(train["user"])).sum())
print("Test ratings whose movie is not in train:",
      (~test["movie"].isin(train["movie"])).sum())
print("Fewest training ratings for any user:",
      train.groupby("user").size().min())

# %% Plot 1: distribution of ratings
counts = ratings["rating"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.bar(counts.index, counts.values, color="steelblue")
ax.set_xlabel("Rating (stars)")
ax.set_ylabel("Number of ratings")
ax.set_title("Distribution of ratings")
fig.tight_layout()
fig.savefig(FIG_DIR / "rating_distribution.png", dpi=150)

# %% Plot 2: ratings per user and per movie
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
axes[0].hist(per_user, bins=50, color="steelblue")
axes[0].set_xlabel("Ratings per user")
axes[0].set_ylabel("Number of users")
axes[1].hist(per_movie, bins=50, color="darkorange")
axes[1].set_yscale("log")
axes[1].set_xlabel("Ratings per movie")
axes[1].set_ylabel("Number of movies (log scale)")
fig.suptitle("How many ratings each user and movie has")
fig.tight_layout()
fig.savefig(FIG_DIR / "ratings_per_user_and_movie.png", dpi=150)

# %% Plot 3: which entries of R are known
fig, ax = plt.subplots(figsize=(7, 4))
ax.spy(mask, markersize=0.02, aspect="auto", color="black")
ax.set_xlabel("Movie (column)")
ax.set_ylabel("User (row)")
ax.set_title(f"Known ratings: {density:.1%} of R")
fig.tight_layout()
fig.savefig(FIG_DIR / "sparsity_pattern.png", dpi=150)