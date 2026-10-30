"""Tests for the fitting module: chi-squared, marginalization, AIC/BIC."""

import numpy as np
import pytest

from src.fitting import (
    aic,
    bic,
    chi2_cholesky,
    chi2_with_marginalization,
    chi2_with_marginalization_cholesky,
    make_fast_chi2_marginalized,
)


# ── Helpers ─────────────────────────────────────────────────────────────


def _identity_cov_inv(n):
    return np.eye(n, dtype=float)


def _simple_model(z, slope):
    """Trivial linear model for testing."""
    return slope * z


# ── chi2_with_marginalization ───────────────────────────────────────────


def test_chi2_marg_zero_residuals():
    """Identical data and model ⇒ chi² = 0."""
    z = np.array([0.1, 0.2, 0.3])
    mu = np.array([38.0, 39.0, 40.0])

    def model(zi, a, b):
        return a + b * zi

    # Find params that exactly reproduce mu (trivially: a=37, b=10)
    cov_inv = _identity_cov_inv(3)
    chi2, M = chi2_with_marginalization(model, (37.0, 10.0), z, mu, cov_inv)
    # After marginalizing M, residual should be ~0
    assert chi2 < 1e-10


def test_chi2_marg_returns_float():
    z = np.array([0.1, 0.5, 1.0])
    mu = np.array([38.0, 42.0, 44.0])
    cov_inv = _identity_cov_inv(3)
    chi2, M = chi2_with_marginalization(_simple_model, (100.0,), z, mu, cov_inv)
    assert isinstance(chi2, float)
    assert isinstance(M, float)


def test_chi2_marg_positive():
    """chi² must be non-negative."""
    rng = np.random.default_rng(0)
    z = np.linspace(0.01, 1.5, 50)
    mu = 40.0 + 5.0 * z + rng.normal(0, 0.1, len(z))
    cov_inv = _identity_cov_inv(50)
    chi2, _ = chi2_with_marginalization(_simple_model, (5.0,), z, mu, cov_inv)
    assert chi2 >= 0.0


# ── chi2_cholesky ───────────────────────────────────────────────────────


def test_chi2_cholesky_matches_direct():
    """Cholesky chi² should match resid @ cov_inv @ resid for identity cov."""
    resid = np.array([0.5, -0.3, 0.1])
    L = np.eye(3, dtype=float)
    chi2_chol = chi2_cholesky(resid, L)
    chi2_direct = float(resid @ resid)
    assert abs(chi2_chol - chi2_direct) < 1e-12


def test_chi2_cholesky_with_real_cov():
    """Test with a non-trivial symmetric positive-definite covariance."""
    rng = np.random.default_rng(1)
    A = rng.normal(size=(5, 5))
    cov = A @ A.T + np.eye(5)
    L = np.linalg.cholesky(cov)
    resid = np.array([1.0, -0.5, 0.3, 0.2, -0.1])
    chi2_chol = chi2_cholesky(resid, L)
    cov_inv = np.linalg.inv(cov)
    chi2_direct = float(resid @ cov_inv @ resid)
    assert abs(chi2_chol - chi2_direct) < 1e-8


# ── chi2_with_marginalization_cholesky ──────────────────────────────────


def test_cholesky_marg_matches_standard():
    """Cholesky marginalized chi² should match the cov_inv version."""
    rng = np.random.default_rng(2)
    n = 20
    mu_theory = rng.normal(40, 2, n)
    mu_data = mu_theory + rng.normal(0, 0.5, n) + 3.0  # offset by M=3

    A = rng.normal(size=(n, n))
    cov = A @ A.T + 5 * np.eye(n)
    L = np.linalg.cholesky(cov)
    cov_inv = np.linalg.inv(cov)

    chi2_L, M_L = chi2_with_marginalization_cholesky(mu_theory, mu_data, L)

    # Replicate via standard marginalization
    delta = mu_data - mu_theory
    ones = np.ones(n)
    M_std = float((ones @ cov_inv @ delta) / (ones @ cov_inv @ ones))
    resid = delta - M_std
    chi2_std = float(resid @ cov_inv @ resid)

    assert abs(chi2_L - chi2_std) < 1e-6
    assert abs(M_L - M_std) < 1e-6


# ── make_fast_chi2_marginalized ─────────────────────────────────────────


def test_fast_closure_matches_full():
    rng = np.random.default_rng(3)
    n = 15
    cov = np.eye(n) * 2.0
    L = np.linalg.cholesky(cov)
    fast = make_fast_chi2_marginalized(L)

    mu_theory = rng.normal(40, 1, n)
    mu_data = mu_theory + rng.normal(0, 0.3, n) + 1.5

    chi2_fast, M_fast = fast(mu_theory, mu_data)
    chi2_full, M_full = chi2_with_marginalization_cholesky(mu_theory, mu_data, L)

    assert abs(chi2_fast - chi2_full) < 1e-10
    assert abs(M_fast - M_full) < 1e-10


# ── Information criteria ────────────────────────────────────────────────


def test_aic_formula():
    assert aic(100.0, 3) == pytest.approx(106.0)


def test_bic_formula():
    # BIC = chi2 + k * ln(n) → 100 + 2*ln(100) ≈ 109.21
    result = bic(100.0, 2, 100)
    assert result == pytest.approx(100.0 + 2.0 * np.log(100))


def test_bic_penalises_more_params():
    """More parameters ⇒ higher BIC for same chi²."""
    b1 = bic(100.0, 2, 1000)
    b2 = bic(100.0, 5, 1000)
    assert b2 > b1
