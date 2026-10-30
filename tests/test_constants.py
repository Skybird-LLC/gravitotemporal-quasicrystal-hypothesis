"""Tests for physical constants: verify values match known references."""

import numpy as np
import pytest

from src.constants import (
    ALPHA_EM,
    A_PREFACTOR,
    C_KM_S,
    C_M_S,
    E_EULER,
    G_NEWTON,
    HBAR,
    H0_PLANCK,
    K_B,
    K_PHI,
    LCDM_PARAMS,
    L_PLANCK,
    M_ETACC_GEV,
    M_ETACC_KG,
    M_PLANCK,
    OL_PLANCK,
    OM_PLANCK,
    PHI,
    T_AGE,
    T_PLANCK,
)


# ── Golden ratio ────────────────────────────────────────────────────────


def test_phi_value():
    assert PHI == pytest.approx(1.6180339887, rel=1e-9)


def test_phi_identity():
    """φ² = φ + 1  (defining property)."""
    assert PHI**2 == pytest.approx(PHI + 1, rel=1e-12)


# ── Speed of light ──────────────────────────────────────────────────────


def test_c_km_s():
    assert C_KM_S == pytest.approx(299792.458, rel=1e-9)


def test_c_m_s():
    assert C_M_S == pytest.approx(2.99792458e8, rel=1e-9)


def test_c_units_consistent():
    assert C_M_S == pytest.approx(C_KM_S * 1e3, rel=1e-9)


# ── Fundamental constants ───────────────────────────────────────────────


def test_hbar():
    assert HBAR == pytest.approx(1.054571817e-34, rel=1e-6)


def test_g_newton():
    assert G_NEWTON == pytest.approx(6.67430e-11, rel=1e-4)


def test_k_boltzmann():
    assert K_B == pytest.approx(1.380649e-23, rel=1e-6)


# ── Planck units ────────────────────────────────────────────────────────


def test_planck_time():
    assert T_PLANCK == pytest.approx(5.391247e-44, rel=1e-5)


def test_planck_length():
    assert L_PLANCK == pytest.approx(1.616255e-35, rel=1e-5)


def test_planck_mass():
    assert M_PLANCK == pytest.approx(2.176434e-8, rel=1e-5)


def test_planck_time_from_fundamentals():
    """t_P ≈ √(ℏG/c⁵) — cross-check against fundamental constants."""
    t_p_calc = np.sqrt(HBAR * G_NEWTON / C_M_S**5)
    assert T_PLANCK == pytest.approx(t_p_calc, rel=1e-3)


# ── Master equation prefactor ──────────────────────────────────────────


def test_a_prefactor():
    expected = T_PLANCK * 12 * np.e / np.pi
    assert A_PREFACTOR == pytest.approx(expected, rel=1e-12)


def test_a_prefactor_order_of_magnitude():
    assert 1e-43 < A_PREFACTOR < 1e-42


# ── Cosmological parameters ────────────────────────────────────────────


def test_h0_planck():
    assert H0_PLANCK == pytest.approx(67.4, rel=1e-3)


def test_omega_sum():
    """Flat universe: Ωm + ΩΛ = 1."""
    assert OM_PLANCK + OL_PLANCK == pytest.approx(1.0, rel=1e-3)


def test_age_of_universe():
    """~13.8 Gyr in seconds."""
    gyr_in_s = 13.8e9 * 365.25 * 24 * 3600
    assert T_AGE == pytest.approx(gyr_in_s, rel=0.01)


# ── Fine structure constant ─────────────────────────────────────────────


def test_alpha_em():
    assert ALPHA_EM == pytest.approx(1.0 / 137.036, rel=1e-5)


# ── Charmonium mass ────────────────────────────────────────────────────


def test_etacc_gev():
    assert M_ETACC_GEV == pytest.approx(2.9839, rel=1e-4)


def test_etacc_kg_positive():
    assert M_ETACC_KG > 0


# ── CMB analysis constant ──────────────────────────────────────────────


def test_k_phi():
    expected = 1.0 / np.log(PHI)
    assert K_PHI == pytest.approx(expected, rel=1e-10)
    assert K_PHI == pytest.approx(2.078, rel=1e-2)


# ── LCDM_PARAMS dict ───────────────────────────────────────────────────


def test_lcdm_params_h0():
    assert LCDM_PARAMS["H0"] == pytest.approx(67.36, rel=1e-3)


def test_lcdm_params_omega_flat():
    assert LCDM_PARAMS["Omega_m"] + LCDM_PARAMS["Omega_L"] == pytest.approx(1.0, rel=1e-3)


def test_lcdm_params_as_consistent():
    """As ≈ exp(ln10As) * 1e-10."""
    expected = np.exp(LCDM_PARAMS["ln10As"]) * 1e-10
    assert LCDM_PARAMS["As"] == pytest.approx(expected, rel=1e-6)
