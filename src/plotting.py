"""Plotting helpers for Hubble diagram, residuals, and CMB periodograms."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .constants import K_PHI, PHI


# ── Supernova plots ─────────────────────────────────────────────────────


def plot_hubble_diagram(z, mu_data, mu_model=None, yerr=None, ax=None):
    """Plot observed distance modulus and optional model curve."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    z = np.asarray(z)
    mu_data = np.asarray(mu_data)

    ax.errorbar(z, mu_data, yerr=yerr, fmt=".", alpha=0.6, label="Data")
    if mu_model is not None:
        ax.plot(z, mu_model, lw=2, label="Model")

    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Distance modulus μ")
    ax.legend()
    ax.grid(alpha=0.2)
    return ax


def plot_residuals(z, resid, yerr=None, ax=None):
    """Plot residuals versus redshift."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))

    z = np.asarray(z)
    resid = np.asarray(resid)

    ax.axhline(0.0, color="k", lw=1)
    ax.errorbar(z, resid, yerr=yerr, fmt=".", alpha=0.7)
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("μ residual")
    ax.grid(alpha=0.2)
    return ax


# ── CMB plots ───────────────────────────────────────────────────────────


def plot_cmb_residuals(ell, residuals, sigma_cosmic=None, ax=None):
    """Plot CMB TT residuals (data − ΛCDM) vs multipole ℓ."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ell = np.asarray(ell)
    residuals = np.asarray(residuals)

    if sigma_cosmic is not None:
        ax.fill_between(ell, -sigma_cosmic, sigma_cosmic,
                        color="blue", alpha=0.08, label="Cosmic variance (1σ)")
    ax.scatter(ell, residuals, s=2, color="gray", alpha=0.5, label="Residuals")

    ax.set_xlabel("Multipole ℓ", fontsize=12)
    ax.set_ylabel("ΔDℓ (μK²)", fontsize=12)
    ax.set_title("Planck 2018 TT Residuals (Data − Best-fit ΛCDM)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    return ax


def plot_cmb_residuals_lnell(ell, resid_weighted, ax=None):
    """Plot normalized CMB residuals in ln(ℓ) space with φ-interval markers."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ln_ell = np.log(np.asarray(ell, dtype=float))
    resid_weighted = np.asarray(resid_weighted)

    ax.scatter(ln_ell, resid_weighted, s=2, color="steelblue", alpha=0.5)
    ax.axhline(0, color="black", ls="-", lw=0.5)

    # Mark ln(φ) intervals
    ln_phi = np.log(PHI)
    for i in range(int(ln_ell.min() / ln_phi), int(ln_ell.max() / ln_phi) + 1):
        ax.axvline(i * ln_phi, color="gold", ls="--", alpha=0.3, lw=1)

    ax.set_xlabel("ln(ℓ)", fontsize=12)
    ax.set_ylabel("Normalized residual", fontsize=12)
    ax.set_title("Residuals in ln(ℓ) space — golden lines mark ln(φ) intervals",
                 fontsize=13)
    ax.grid(True, alpha=0.2)
    return ax


def plot_cmb_periodogram(freqs, power, sig_levels=None,
                         phi_freq=None, phi_power=None,
                         global_max_freq=None, global_max_power=None,
                         ax=None):
    """Plot Lomb-Scargle periodogram with significance thresholds and k_φ marker."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    freqs = np.asarray(freqs)
    power = np.asarray(power)
    k_phi = K_PHI

    ax.plot(freqs, power, "k-", lw=0.8, alpha=0.7)

    # Significance bands
    if sig_levels is not None:
        colors_sig = {"1sigma": "green", "2sigma": "orange",
                      "3sigma": "red", "4sigma": "darkred", "5sigma": "purple"}
        for level, val in sig_levels.items():
            ax.axhline(val, color=colors_sig.get(level, "gray"), ls="--", lw=1,
                       alpha=0.7, label=f"{level}: {val:.3f}")

    # Golden-ratio target frequency
    ax.axvline(k_phi, color="gold", ls="-", lw=3, alpha=0.8,
               label=f"k_φ = 1/ln(φ) = {k_phi:.3f}")

    if phi_freq is not None and phi_power is not None:
        ax.plot(phi_freq, phi_power, "r*", markersize=15, zorder=5,
                label=f"Peak near k_φ")

    if global_max_freq is not None and global_max_power is not None:
        ax.plot(global_max_freq, global_max_power, "b^", markersize=10, zorder=5,
                label=f"Global max: k={global_max_freq:.3f}")

    ax.set_xlabel("Frequency k (cycles per unit ln ℓ)", fontsize=12)
    ax.set_ylabel("Lomb-Scargle Power", fontsize=12)
    ax.set_title("Periodogram of CMB Residuals — Searching for φ-periodicity",
                 fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(0, 15)
    ax.grid(True, alpha=0.2)
    return ax
