"""Physical and mathematical constants for Fibonacci Cosmology project."""
import numpy as np

# ── Fundamental mathematical constants ──────────────────────────────────
PHI = (1 + np.sqrt(5)) / 2            # golden ratio ≈ 1.6180339887
E_EULER = np.e                          # Euler's number ≈ 2.7182818284

# ── Speed of light ──────────────────────────────────────────────────────
C_KM_S = 299792.458                     # km/s
C_M_S  = 2.99792458e8                   # m/s

# ── Fundamental physical constants ──────────────────────────────────────
HBAR      = 1.054571817e-34             # reduced Planck constant (J·s)
G_NEWTON  = 6.67430e-11                 # gravitational constant (m³/kg/s²)
K_B       = 1.380649e-23                # Boltzmann constant (J/K)

# ── Planck units ────────────────────────────────────────────────────────
T_PLANCK = 5.391247e-44                 # Planck time (s)
L_PLANCK = 1.616255e-35                 # Planck length (m)
M_PLANCK = 2.176434e-8                  # Planck mass (kg)

# ── Master equation prefactor (lambda0 = 1) ────────────────────────────
A_PREFACTOR = T_PLANCK * 12 * E_EULER / np.pi  # ≈ 5.59e-43 s

# ── Cosmological parameters (Planck 2018 best-fit) ─────────────────────
H0_PLANCK = 67.4                        # km/s/Mpc
OM_PLANCK = 0.315                       # matter density parameter
OL_PLANCK = 0.685                       # dark energy density parameter
T_AGE     = 4.35e17                     # age of universe in seconds (13.8 Gyr)

# ── Fine structure constant ─────────────────────────────────────────────
ALPHA_EM = 1.0 / 137.035999084

# ── Charmonium meson mass (Brandenburg G derivation) ────────────────────
M_ETACC_GEV = 2.9839                    # GeV/c²
M_ETACC_KG  = M_ETACC_GEV * 1.602176634e-19 * 1e9 / C_M_S**2

# ── CMB analysis constants ──────────────────────────────────────────────
K_PHI = 1.0 / np.log(PHI)              # target Fourier frequency ≈ 2.078 cycles/ln(ℓ)

# ── Planck 2018 best-fit ΛCDM parameters (TT,TE,EE+lowE+lensing) ──────
LCDM_PARAMS = {
    'H0':      67.36,
    'ombh2':   0.02237,
    'omch2':   0.1200,
    'tau':     0.0544,
    'ns':      0.9649,
    'ln10As':  3.044,
    'As':      np.exp(3.044) * 1e-10,
    'Omega_m': 0.3153,
    'Omega_L': 0.6847,
}
