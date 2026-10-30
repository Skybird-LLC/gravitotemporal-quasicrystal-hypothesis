"""Model fitting helpers: chi-squared, marginalization, and information criteria."""

from __future__ import annotations

import numpy as np


def chi2_with_marginalization(model_mu_func, params, z_data, mu_data, cov_inv):
    """Compute chi² analytically marginalizing over absolute magnitude M."""
    mu_theory = np.array([model_mu_func(zi, *params) for zi in z_data], dtype=float)
    delta = np.asarray(mu_data, dtype=float) - mu_theory
    ones = np.ones_like(delta)

    numerator = ones @ cov_inv @ delta
    denominator = ones @ cov_inv @ ones
    if denominator == 0.0:
        raise ZeroDivisionError("Degenerate covariance projection in M marginalization")

    M_best = numerator / denominator
    resid = delta - M_best
    return float(resid @ cov_inv @ resid), float(M_best)


def chi2_cholesky(resid, L):
    """Stable chi² from residual vector using Cholesky factor L of covariance."""
    x = np.linalg.solve(L, resid)
    return float(x @ x)


def chi2_with_marginalization_cholesky(mu_theory, mu_data, L):
    """Compute chi² and best-fit M using Cholesky factor of covariance.

    Inputs are theory and data distance moduli vectors, and L such that
    covariance C = L @ L.T.
    """
    delta = np.asarray(mu_data, dtype=float) - np.asarray(mu_theory, dtype=float)
    ones = np.ones_like(delta)

    delta_w = np.linalg.solve(L, delta)
    ones_w = np.linalg.solve(L, ones)

    denominator = float(ones_w @ ones_w)
    if denominator == 0.0:
        raise ZeroDivisionError("Degenerate covariance projection in M marginalization")

    M_best = float((ones_w @ delta_w) / denominator)
    resid_w = delta_w - M_best * ones_w
    chi2 = float(resid_w @ resid_w)
    return chi2, M_best


def make_fast_chi2_marginalized(L):
    """Return a fast chi2 closure that precomputes solve(L, ones) once."""
    n = L.shape[0]
    ones = np.ones(n, dtype=float)
    ones_w = np.linalg.solve(L, ones)
    ones_w_dot = float(ones_w @ ones_w)
    if ones_w_dot == 0.0:
        raise ZeroDivisionError("Degenerate covariance projection in M marginalization")

    def fast_chi2(mu_theory, mu_data):
        delta = np.asarray(mu_data, dtype=float) - np.asarray(mu_theory, dtype=float)
        delta_w = np.linalg.solve(L, delta)
        M_best = float((ones_w @ delta_w) / ones_w_dot)
        resid_w = delta_w - M_best * ones_w
        return float(resid_w @ resid_w), M_best

    return fast_chi2


def aic(chi2, k):
    """Akaike information criterion."""
    return float(chi2 + 2.0 * k)


def bic(chi2, k, n):
    """Bayesian information criterion."""
    return float(chi2 + k * np.log(n))
