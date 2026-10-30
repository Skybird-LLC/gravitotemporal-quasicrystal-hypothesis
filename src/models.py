"""Cosmological model definitions for H(z), dL(z), and mu(z)."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from .constants import C_KM_S, PHI


def _asarray_z(z):
    return np.asarray(z, dtype=float)


def _scalar_or_array(value, template):
    template_arr = np.asarray(template)
    if template_arr.ndim == 0:
        return float(np.asarray(value))
    return np.asarray(value)


def H_lcdm(z, H0, Om):
    """Flat-ΛCDM Hubble parameter in km/s/Mpc."""
    z_arr = _asarray_z(z)
    OL = 1.0 - Om
    H_z = H0 * np.sqrt(Om * (1.0 + z_arr) ** 3 + OL)
    return _scalar_or_array(H_z, z)


def H_naive_phi(z, H0):
    """Naive φ^n-inspired stiff-fluid scaling H(z)=H0*(1+z)^3."""
    z_arr = _asarray_z(z)
    H_z = H0 * (1.0 + z_arr) ** 3
    return _scalar_or_array(H_z, z)


def w_de_phi(z, eps):
    """Quasiperiodic dark-energy equation of state."""
    z_arr = _asarray_z(z)
    return -1.0 + eps * np.cos(2.0 * np.pi * np.log(1.0 + z_arr) / np.log(PHI))


def _de_density_factor(z, eps):
    """Compute exp(3∫(1+w_de)/(1+z) dz) for Model C."""
    if z <= 0.0:
        return 1.0

    def integrand(zp):
        return (1.0 + w_de_phi(zp, eps)) / (1.0 + zp)

    val, _ = quad(integrand, 0.0, z)
    return np.exp(3.0 * val)


def H_quasiperiodic(z, H0, Om, eps):
    """Model C Hubble parameter with φ-periodic dark-energy modulation."""
    z_arr = _asarray_z(z)

    def one(zi):
        OL = 1.0 - Om
        de_factor = _de_density_factor(float(zi), eps)
        return H0 * np.sqrt(Om * (1.0 + zi) ** 3 + OL * de_factor)

    if z_arr.ndim == 0:
        return float(one(float(z_arr)))
    return np.array([one(zi) for zi in z_arr], dtype=float)


def dL_from_H(z, H_func, *params):
    """Luminosity distance in Mpc for a provided H(z) model."""
    z_arr = _asarray_z(z)

    def dL_one(zi):
        if zi <= 0.0:
            return 0.0
        integral, _ = quad(lambda zp: C_KM_S / H_func(zp, *params), 0.0, zi)
        return (1.0 + zi) * integral

    if z_arr.ndim == 0:
        return float(dL_one(float(z_arr)))
    return np.array([dL_one(zi) for zi in z_arr], dtype=float)


def mu_from_H(z, H_func, *params):
    """Distance modulus in magnitudes for a provided H(z) model."""
    dL = dL_from_H(z, H_func, *params)
    dL_arr = np.asarray(dL, dtype=float)
    safe = np.maximum(dL_arr, 1e-12)
    mu = 5.0 * np.log10(safe) + 25.0
    return _scalar_or_array(mu, z)


def mu_lcdm(z, H0, Om):
    return mu_from_H(z, H_lcdm, H0, Om)


def mu_naive_phi(z, H0):
    return mu_from_H(z, H_naive_phi, H0)


def mu_quasiperiodic(z, H0, Om, eps):
    return mu_from_H(z, H_quasiperiodic, H0, Om, eps)
