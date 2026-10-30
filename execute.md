## GitHub Copilot Primer

This is a context file you place at the root of your repository. Copilot reads it automatically and uses it to understand your project when suggesting code. Save it as `.github/copilot-instructions.md` — that's the official location Copilot checks first.

---

```markdown
# Copilot Instructions — Gravitotemporal Quasicrystal Cosmology

## Project Overview

This project tests a novel cosmological framework called the **Gravitotemporal
Quasicrystal Hypothesis** against empirical data. The core claim is that cosmic
expansion follows discrete golden-ratio ($\phi = 1.618...$) time steps rather
than smooth continuous evolution, and that this geometric rhythm eliminates the
need for dark energy as a separate substance — reframing it as an emergent
property of the expansion clock.

We are fitting three competing cosmological models to Type Ia supernova data
(Pantheon+ dataset) and searching for $\phi$-periodic signatures in CMB and BAO
data.

---

## Physics Context (for code generation)

### Constants Used Throughout

```python
import numpy as np

# Fundamental constants
c_km_s   = 299792.458          # speed of light in km/s
c_m_s    = 2.99792458e8        # speed of light in m/s
hbar      = 1.054571817e-34    # reduced Planck constant (J·s)
G_newton  = 6.67430e-11        # gravitational constant (m³/kg/s²)
k_B       = 1.380649e-23       # Boltzmann constant (J/K)

# Planck units
t_P = 5.391247e-44             # Planck time (s)
l_P = 1.616255e-35             # Planck length (m)
m_P = 2.176434e-8              # Planck mass (kg)

# Key mathematical constants
phi = (1 + np.sqrt(5)) / 2     # golden ratio ≈ 1.6180339887
e_euler = np.e                  # Euler's number ≈ 2.7182818284

# Cosmological parameters (Planck 2018 best-fit for comparison)
H0_planck = 67.4               # km/s/Mpc
Om_planck = 0.315              # matter density parameter
OL_planck = 0.685              # dark energy density parameter
T_age     = 4.35e17            # age of universe in seconds (13.8 Gyr)

# Fine structure constant
alpha_em = 1 / 137.035999084

# Charmonium meson mass (for Brandenburg G derivation)
M_etacc_GeV = 2.9839           # GeV/c²
M_etacc_kg  = M_etacc_GeV * 1.602176634e-19 * 1e9 / c_m_s**2
```

### The Three Models We Test

**Model A — ΛCDM (the standard model, our benchmark):**
- Hubble parameter: `H(z) = H0 * sqrt(Om*(1+z)^3 + OL)`
- Free parameters: `H0`, `Om` (with `OL = 1 - Om` for flat universe)
- This is what we must match or beat.

**Model B — Naive φⁿ (the original master equation, expected to fail):**
- Hubble parameter: `H(z) = H0 * (1+z)^3`
- This corresponds to a stiff-fluid universe (equation of state w = 1)
- Scale factor: `a(t) ∝ t^(1/3)`
- Free parameters: `H0` only
- We include this to show the raw equation fails and motivate Model C.

**Model C — Quasiperiodic φ-potential (the real test):**
- Dark energy equation of state oscillates with golden-ratio period:
  `w_de(z) = -1 + epsilon * cos(2*pi*ln(1+z) / ln(phi))`
- Free parameters: `H0`, `Om`, `epsilon`
- If `epsilon = 0`, this reduces to ΛCDM.
- If `epsilon > 0` at >2σ significance, we have evidence for φ-periodicity.

### Master Time Equation

The framework's foundational equation:

```
T(n) = t_P * (12 * lambda0^3 * e / pi) * phi^n
```

Where:
- `n` = integer tick number (discrete time step)
- `lambda0` = initial lattice parameter (dimensionless, = L / l_P)
- With `lambda0 = 1`: prefactor `A = t_P * 12*e/pi ≈ 5.59e-43 s`
- Today's tick: `N ≈ 287` (derived from T_age / A)

### Scale Factor

```
a(n) = phi^((n - N) / 3)
```

Where N ≈ 287 is today's tick number.

### Quasiperiodic Potential

```
V(lam) = V0 * [1 - cos(2*pi*lam/phi) - (1/phi)*cos(2*pi*lam/phi^2)]
```

This has incommensurate frequencies (ratio = φ), creating quasiperiodic minima.

---

## Data Sources

### Primary: Pantheon+ Type Ia Supernovae

- **Repository:** `github.com/PantheonPlusSH0ES/DataRelease`
- **Key file:** `Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat`
- **Covariance:** `Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov`
- **Size:** 1,701 light curves, 1,550 unique SNe Ia, z = 0.001 to 2.26
- **Important columns in .dat file:**
  - `CID` — supernova identifier
  - `zHD` — Hubble-diagram redshift (CMB frame, peculiar velocity corrected)
  - `m_b_corr` — corrected apparent magnitude (Tripp standardization applied)
  - `m_b_corr_err_DIAG` — diagonal error (FOR PLOTTING ONLY, not fitting)
  - `MU_SH0ES` — distance modulus with SH0ES Cepheid calibration
  - `MU_SH0ES_ERR_DIAG` — diagonal error on distance modulus
  - `IS_CALIBRATOR` — 1 if Cepheid-host SN (exclude from cosmology fits)
  - `IDSURVEY` — survey identifier

- **Covariance matrix format:**
  - First line: integer N (= 1701)
  - Following N*N lines: matrix elements read sequentially row by row
  - MUST be used for any parameter fitting (diagonal errors are insufficient)
  - Matrix is symmetric and positive semi-definite

- **Filtering rules:**
  - Exclude `IS_CALIBRATOR == 1` for cosmological fits
  - Apply `zHD > 0.01` cut to avoid peculiar velocity dominated regime
  - When using `m_b_corr`, analytically marginalize over absolute magnitude M

### Secondary: CMB Power Spectrum (Planck 2018)

- **Source:** Planck Legacy Archive or LAMBDA (NASA)
- **Key file:** `COM_PowerSpect_CMB-TT-full_R3.01.txt`
- **Columns:** multipole l, D_l (μK²), error_minus, error_plus
- **Usage:** Fourier-transform residuals (data minus best-fit ΛCDM) and search
  for peak at frequency 1/ln(φ) ≈ 2.078 in ln(l) space

### Tertiary: BAO (DESI DR1, SDSS)

- **Source:** DESI collaboration public data releases
- **Key observable:** Galaxy correlation function ξ(r)
- **Search for:** Secondary peaks at r ≈ 93 Mpc (150/φ) and r ≈ 243 Mpc (150×φ)

---

## Project Structure

```
fibonacci-cosmology/
├── .github/
│   └── copilot-instructions.md     ← THIS FILE
├── data/
│   ├── pantheon/
│   │   ├── Pantheon+SH0ES.dat      ← downloaded from GitHub
│   │   └── Pantheon+SH0ES_STAT+SYS.cov
│   ├── cmb/
│   │   └── planck_tt_spectrum.txt
│   └── bao/
│       └── (DESI correlation data)
├── src/
│   ├── constants.py                ← all physical constants
│   ├── models.py                   ← H(z), dL(z), mu(z) for all 3 models
│   ├── data_loader.py              ← parse Pantheon+ .dat and .cov files
│   ├── fitting.py                  ← chi-squared, MCMC, marginalization
│   ├── cmb_analysis.py             ← CMB residual Fourier analysis
│   └── plotting.py                 ← Hubble diagram, residuals, corner plots
├── notebooks/
│   ├── 01_pantheon_fit.ipynb       ← main supernova analysis
│   ├── 02_cmb_phi_search.ipynb     ← CMB φ-periodicity search
│   └── 03_bao_harmonics.ipynb      ← BAO secondary peak search
├── results/
│   ├── figures/
│   └── tables/
├── tests/
│   ├── test_models.py              ← unit tests for H(z) implementations
│   └── test_data_loader.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Coding Conventions

### General

- Python 3.10+
- Use numpy for all numerical work (no symbolic math in production code)
- Use scipy.integrate.quad for single integrals, scipy.integrate.quad_vec for arrays
- Use scipy.optimize.minimize (Nelder-Mead) for quick fits
- Use emcee for MCMC when doing proper parameter estimation
- All distances in Mpc, all speeds in km/s, all times in seconds unless noted
- All angles in radians unless explicitly stated
- Use natural log (`np.log`) unless base-10 is specified (`np.log10`)

### Variable Naming

```python
# Redshift
z       # single redshift value
z_arr   # array of redshifts
z_hd    # Hubble-diagram redshift from data file

# Distances
dL      # luminosity distance (Mpc)
dA      # angular diameter distance (Mpc)
dC      # comoving distance (Mpc)
mu      # distance modulus (mag) = 5*log10(dL/Mpc) + 25

# Hubble parameter
H0      # Hubble constant today (km/s/Mpc)
H_z     # Hubble parameter at redshift z (km/s/Mpc)

# Density parameters (dimensionless, at z=0)
Om      # total matter (baryonic + dark matter)
Ob      # baryonic matter
OL      # dark energy / cosmological constant
Or      # radiation
Ok      # curvature (= 0 for flat universe)

# Model-specific
eps     # epsilon: amplitude of phi-periodic modulation in w_de(z)
n_tick  # integer tick number in master time equation
lam     # lambda: lattice parameter (dimensionless)
lam0    # initial lattice parameter at n=0
w_de    # dark energy equation of state parameter
N_today # tick number corresponding to present epoch (≈ 287)

# Fitting
chi2    # chi-squared statistic
chi2_nu # reduced chi-squared (chi2 / degrees of freedom)
ndof    # number of degrees of freedom
cov     # covariance matrix (1701 x 1701 for Pantheon+)
cov_inv # inverse covariance matrix
```

### Key Functions to Implement

```python
# In models.py — each model needs these three functions:

def H_model(z, *params):
    """Hubble parameter H(z) in km/s/Mpc."""
    ...

def dL_model(z, *params):
    """Luminosity distance in Mpc. Integrates c/H(z') from 0 to z."""
    from scipy.integrate import quad
    integral, _ = quad(lambda zp: c_km_s / H_model(zp, *params), 0, z)
    return (1 + z) * integral

def mu_model(z, *params):
    """Distance modulus in magnitudes."""
    d = dL_model(z, *params)
    return 5.0 * np.log10(d) + 25.0
```

```python
# In fitting.py — chi-squared with analytic M marginalization:

def chi2_with_marginalization(model_mu_func, params, z_data, mu_data, cov_inv):
    """
    Compute chi² analytically marginalizing over the absolute magnitude M.

    For diagonal approximation (quick checks):
        cov_inv = np.diag(1.0 / sigma**2)

    For full covariance (publication quality):
        cov_inv = np.linalg.inv(cov_matrix)

    The marginalization over M is:
        M_best = (1ᵀ C⁻¹ Δμ) / (1ᵀ C⁻¹ 1)
    where Δμ = mu_data - mu_model and 1 is a vector of ones.
    """
    mu_theory = np.array([model_mu_func(zi, *params) for zi in z_data])
    delta = mu_data - mu_theory
    ones = np.ones_like(delta)

    # Analytic marginalization
    M_best = (ones @ cov_inv @ delta) / (ones @ cov_inv @ ones)
    resid = delta - M_best

    return resid @ cov_inv @ resid
```

### Performance Notes

- The `quad` integration is the bottleneck. For Model C (quasiperiodic),
  each mu(z) call requires two nested integrations.
- For MCMC with emcee (~100k samples), vectorize where possible:
  - Pre-compute a spline interpolation of mu(z) on a fine grid
  - Use `scipy.interpolate.CubicSpline` to evaluate at data redshifts
- The full covariance matrix inversion (1701×1701) should be done ONCE
  and cached. Use `np.linalg.cholesky` for numerical stability:

```python
L = np.linalg.cholesky(cov_matrix)
# Solve L @ x = delta, then L.T @ y = x → y = cov_inv @ delta
# This avoids explicitly computing cov_inv
x = np.linalg.solve(L, resid)
chi2 = x @ x  # equivalent to resid @ cov_inv @ resid
```

---

## Success Criteria

| Test | Pass condition | Fail condition |
|------|---------------|----------------|
| Model B vs ΛCDM | (expected to fail) | chi2/dof >> 1 confirms stiff fluid is ruled out |
| Model C chi2/dof | ≤ 1.05 | > 1.10 (poor fit) |
| Model C vs ΛCDM | Δchi2 < 4 (comparable) | Δchi2 > 10 (significantly worse) |
| epsilon significance | > 0 at 2σ+ → φ-signal detected | consistent with 0 → no signal at current precision |
| CMB φ-periodicity | Peak in Fourier spectrum at 1/ln(φ) at > 3σ | No peak → lattice imprint below CMB sensitivity |

---

## References

- Pantheon+ data paper: arXiv:2112.03863
- Pantheon+ cosmological constraints: arXiv:2202.04077
- Planck 2018 cosmological parameters: arXiv:1807.06209
- Fibonacci time crystal (qubit stabilization): Nature 2022
- Brandenburg GEM theory: IEEE Trans. Plasma Sci. Vol. 35, No. 4, p845 (2007)

---

## What NOT to do

- Do NOT use diagonal errors for final parameter constraints (plotting only)
- Do NOT fit calibrator SNe (IS_CALIBRATOR == 1) in cosmological analysis
- Do NOT hardcode H0 = 70 — it must be a free parameter in every fit
- Do NOT ignore the absolute magnitude degeneracy — always marginalize over M
- Do NOT use scipy.optimize for final results — use MCMC (emcee) for proper posteriors
- Do NOT compare models by chi2 alone — use AIC or BIC for model selection:
  - AIC = chi2 + 2*k (k = number of free parameters)
  - BIC = chi2 + k*ln(N) (N = number of data points)
  - Lower AIC/BIC = preferred model
```

---

## How to Use This

1. Create the directory `.github/` at the root of your repo
2. Save the content above as `.github/copilot-instructions.md`
3. Copilot will automatically read this file whenever you work in the repository
4. When you open a new `.py` file and start typing, Copilot will know:
    - What `phi` means in your context
    - How to structure `H(z)` functions
    - Which data columns to use
    - How to properly compute $\chi^2$
    - What variable names to use

Additionally, create a `requirements.txt`:

```text
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
emcee>=3.1
corner>=2.2
astropy>=5.3
pandas>=2.0
```

And a minimal `src/constants.py` that Copilot can import from:

```python
"""Physical and mathematical constants for Fibonacci Cosmology project."""
import numpy as np

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2

# Speed of light
C_KM_S = 299792.458       # km/s
C_M_S  = 2.99792458e8     # m/s

# Planck units
T_PLANCK = 5.391247e-44   # s
L_PLANCK = 1.616255e-35   # m

# Master equation prefactor (lambda0 = 1)
A_PREFACTOR = T_PLANCK * 12 * np.e / np.pi  # ≈ 5.59e-43 s

# Today's tick number
N_TODAY = np.log(4.35e17 / A_PREFACTOR) / np.log(PHI)  # ≈ 287

# Fine structure constant
ALPHA_EM = 1 / 137.035999084
```

Once these files are in place, open VS Code in your project folder and Copilot will have full context. When you type something like `def H_quasiphi(z,` it will autocomplete with the correct physics, variable names, and integration structure.