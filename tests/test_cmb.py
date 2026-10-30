"""Tests for the CMB analysis and φ-periodicity search modules."""

import numpy as np
import os
import pytest

from src.cmb_analysis import (
    bootstrap_significance,
    cosmic_variance,
    load_planck_tt,
    lombscargle_in_lnell,
    residual_spectrum,
)


# ── residual_spectrum ───────────────────────────────────────────────────


def test_residual_spectrum_basic():
    data = np.array([100.0, 200.0, 300.0])
    model = np.array([90.0, 210.0, 295.0])
    resid = residual_spectrum(data, model)
    np.testing.assert_allclose(resid, [10.0, -10.0, 5.0])


def test_residual_spectrum_zero():
    vals = np.array([1.0, 2.0, 3.0])
    resid = residual_spectrum(vals, vals)
    np.testing.assert_allclose(resid, 0.0)


# ── cosmic_variance ─────────────────────────────────────────────────────


def test_cosmic_variance_decreases_with_ell():
    """Cosmic variance ∝ 1/√ℓ, so it decreases for higher multipoles."""
    ell = np.array([10, 100, 1000], dtype=float)
    Dl = np.array([1000.0, 5000.0, 3000.0])
    sigma = cosmic_variance(ell, Dl)
    # σ/|Dl| = sqrt(2/(2ℓ+1)) decreases with ℓ
    relative = sigma / np.abs(Dl)
    assert np.all(np.diff(relative) < 0)


def test_cosmic_variance_floor():
    """Very small Dl should still return sigma >= 1."""
    ell = np.array([100.0])
    Dl = np.array([0.001])
    sigma = cosmic_variance(ell, Dl)
    assert sigma[0] >= 1.0


# ── load_planck_tt ──────────────────────────────────────────────────────


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cmb")
BINNED_FILE = os.path.join(DATA_DIR, "COM_PowerSpect_CMB-TT-binned_R3.01.txt")


@pytest.mark.skipif(not os.path.exists(BINNED_FILE), reason="Planck data not present")
def test_load_planck_tt_shape():
    data = load_planck_tt(BINNED_FILE)
    assert data.ndim == 2
    assert data.shape[1] >= 4  # ell, Dl, err_minus, err_plus
    assert data.shape[0] > 20  # at least ~80 bins


@pytest.mark.skipif(not os.path.exists(BINNED_FILE), reason="Planck data not present")
def test_load_planck_tt_values_positive():
    data = load_planck_tt(BINNED_FILE)
    ell = data[:, 0]
    Dl = data[:, 1]
    assert np.all(ell > 0)
    assert np.all(Dl > 0)


# ── lombscargle_in_lnell ───────────────────────────────────────────────


def test_lombscargle_detects_injected_signal():
    """Inject a known sinusoidal signal at k=5.0 and check it's recovered."""
    rng = np.random.default_rng(99)
    ell = np.arange(30, 2000, dtype=float)
    ln_ell = np.log(ell)
    k_inject = 5.0
    signal = 10.0 * np.cos(2 * np.pi * k_inject * ln_ell)
    noise = rng.normal(0, 1.0, len(ell))
    resid = signal + noise

    freqs, power = lombscargle_in_lnell(ell, resid, n_freqs=5000)
    peak_freq = freqs[np.argmax(power)]
    assert abs(peak_freq - k_inject) < 0.2


def test_lombscargle_returns_correct_shapes():
    ell = np.arange(10, 500, dtype=float)
    resid = np.random.default_rng(0).normal(size=len(ell))
    freqs, power = lombscargle_in_lnell(ell, resid, n_freqs=1000)
    assert freqs.shape == (1000,)
    assert power.shape == (1000,)


def test_lombscargle_power_nonnegative():
    ell = np.arange(10, 500, dtype=float)
    resid = np.random.default_rng(1).normal(size=len(ell))
    _, power = lombscargle_in_lnell(ell, resid)
    assert np.all(power >= 0)


# ── bootstrap_significance ──────────────────────────────────────────────


def test_bootstrap_returns_thresholds():
    """Quick bootstrap (small n_bootstrap) returns expected keys."""
    ell = np.arange(30, 300, dtype=float)
    resid = np.random.default_rng(2).normal(size=len(ell))
    sig, null_max = bootstrap_significance(ell, resid, n_bootstrap=50, seed=0)
    assert "1sigma" in sig
    assert "2sigma" in sig
    assert "3sigma" in sig
    assert len(null_max) == 50


def test_bootstrap_thresholds_ordered():
    """Higher sigma thresholds should be larger values."""
    ell = np.arange(30, 300, dtype=float)
    resid = np.random.default_rng(3).normal(size=len(ell))
    sig, _ = bootstrap_significance(ell, resid, n_bootstrap=200, seed=1)
    assert sig["1sigma"] <= sig["2sigma"] <= sig["3sigma"]
