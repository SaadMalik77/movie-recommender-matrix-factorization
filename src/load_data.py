"""Load MovieLens 100K, build the rating matrix R and the mask of known ratings,
create a train/test split, and describe the data.

Owner: Muhammad Saad Malik | Checked by: Jagadeesh Aravind Kalyan Shetti
"""

import numpy as np
import pandas as pd

DATA_PATH = "data/ml-100k/u.data"


def main():
    rng = np.random.default_rng(0)
    # TODO: load u.data, build R and mask, split 80/20, describe, plot


if __name__ == "__main__":
    main()
