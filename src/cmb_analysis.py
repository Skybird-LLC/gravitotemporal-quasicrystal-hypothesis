"""CMB residual and φ-periodicity search helpers.

Provides lightweight utilities for loading Planck TT data,
computing residuals, and running Lomb-Scargle periodogram analysis
in ln(ℓ) space.  For the full end-to-end search pipeline see
``cmb_phi_search.py``.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lombscargle

from .constants import K_PHI


def load_planck_tt(path):
    """Load Planck TT spectrum file.

    Handles ``#``-prefixed comment lines and returns a structured array
    with columns: ell, Dl, err_minus, err_plus (and BestFit if present).
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            rows.append(vals)
    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    return np.array(rows)


def residual_spectrum(data_dl, model_dl):
    """Compute residuals data − model."""
    return np.asarray(data_dl, dtype=float) - np.asarray(model_dl, dtype=float)


def cosmic_variance(ell, Dl):
    """Cosmic-variance uncertainty on D_ℓ: σ(D_ℓ) ≈ D_ℓ·√(2/(2ℓ+1))."""
    ell = np.asarray(ell, dtype=float)
    Dl = np.asarray(Dl, dtype=float)
    sigma = np.abs(Dl) * np.sqrt(2.0 / (2.0 * ell + 1.0))
    return np.maximum(sigma, 1.0)  # floor to avoid division by zero


def lombscargle_in_lnell(
    ell,
    resid,
    freq_min=0.1,
    freq_max=20.0,
    n_freqs=10000,
):
    """Lomb-Scargle periodogram of CMB residuals vs ln(ℓ).

    Equivalent to a Mellin transform — detects scale-invariant
    (geometric / multiplicative) periodicities.

    Target: peak at k_φ = 1/ln(φ) ≈ 2.078 cycles per unit ln(ℓ).

    Parameters
    ----------
    ell : array
        Multipole moments (must be > 0).
    resid : array
        Residuals at each multipole.
    freq_min, freq_max : float
        Frequency range in cycles per unit ln(ℓ).
    n_freqs : int
        Number of frequency bins.

    Returns
    -------
    freqs : array
        Cyclic frequencies (cycles per unit ln(ℓ)).
    power : array
        Normalized Lomb-Scargle power at each frequency.
    """
    ell = np.asarray(ell, dtype=float)
    resid = np.asarray(resid, dtype=float)
    mask = ell > 0
    ln_ell = np.log(ell[mask])
    y = resid[mask] - np.mean(resid[mask])

    freqs = np.linspace(freq_min, freq_max, n_freqs)
    angular_freqs = 2.0 * np.pi * freqs
    power = lombscargle(ln_ell, y, angular_freqs, normalize=True)
    return freqs, power


def bootstrap_significance(ell, resid, n_bootstrap=10000, seed=42, **ls_kwargs):
    """Estimate significance thresholds via bootstrap shuffling.

    Shuffles the residuals ``n_bootstrap`` times and records the
    global maximum Lomb-Scargle power each time.

    Returns
    -------
    sig_levels : dict
        Keys '1sigma' … '5sigma' mapping to power thresholds.
    null_max : array
        Distribution of global-max powers under the null.
    """
    ell = np.asarray(ell, dtype=float)
    resid = np.asarray(resid, dtype=float)
    mask = ell > 0
    ln_ell = np.log(ell[mask])
    y = resid[mask] - np.mean(resid[mask])

    freq_min = ls_kwargs.get("freq_min", 0.1)
    freq_max = ls_kwargs.get("freq_max", 20.0)
    n_freqs = ls_kwargs.get("n_freqs", 10000)
    freqs = np.linspace(freq_min, freq_max, n_freqs)
    angular_freqs = 2.0 * np.pi * freqs

    rng = np.random.default_rng(seed)
    null_max = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        y_shuf = rng.permutation(y)
        p_null = lombscargle(ln_ell, y_shuf, angular_freqs, normalize=True)
        null_max[i] = p_null.max()

    sig_levels = {
        "1sigma": float(np.percentile(null_max, 68.27)),
        "2sigma": float(np.percentile(null_max, 95.45)),
        "3sigma": float(np.percentile(null_max, 99.73)),
        "4sigma": float(np.percentile(null_max, 99.9937)),
        "5sigma": float(np.percentile(null_max, 99.99994)),
    }
    return sig_levels, null_max
