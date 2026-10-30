import numpy as np

from src.models import H_lcdm, H_naive_phi, mu_lcdm


def test_h_lcdm_positive():
    z = np.array([0.01, 0.5, 1.0])
    values = H_lcdm(z, 70.0, 0.3)
    assert np.all(values > 0)


def test_h_naive_phi_positive():
    z = np.array([0.01, 0.5, 1.0])
    values = H_naive_phi(z, 70.0)
    assert np.all(values > 0)


def test_mu_lcdm_monotonic_in_z():
    z = np.array([0.05, 0.2, 0.8])
    mu = mu_lcdm(z, 70.0, 0.3)
    assert np.all(np.diff(mu) > 0)
