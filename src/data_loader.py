"""Data-loading utilities for Pantheon+ and ancillary datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_pantheon_table(path):
    """Load Pantheon+ `.dat` table into a pandas DataFrame."""
    return pd.read_csv(path, sep=r"\s+", engine="python", comment="#")


def load_covariance(path):
    """Load Pantheon+ covariance matrix from flattened text format."""
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline().strip()
        n = int(first)
        arr = np.fromiter((float(line.strip()) for line in f if line.strip()), dtype=float)
    if arr.size != n * n:
        raise ValueError(f"Covariance size mismatch: expected {n*n}, got {arr.size}")
    return arr.reshape((n, n))


def apply_pantheon_cuts(df, z_min=0.01):
    """Exclude calibrators and low-z points for cosmology fitting."""
    required = {"zHD", "IS_CALIBRATOR"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    mask = (df["IS_CALIBRATOR"] == 0) & (df["zHD"] > z_min)
    return df.loc[mask].reset_index(drop=True)


def select_mu_vectors(df, mu_col="MU_SH0ES"):
    """Return redshift and distance-modulus vectors for fitting."""
    required = {"zHD", mu_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    z_data = df["zHD"].to_numpy(dtype=float)
    mu_data = df[mu_col].to_numpy(dtype=float)
    return z_data, mu_data
