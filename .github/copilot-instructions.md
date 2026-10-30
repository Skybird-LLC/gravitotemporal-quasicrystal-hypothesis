# Updated Copilot Instructions

Here's the revised file. I've marked every changed section with comments so you can see what moved and why. The biggest shifts are:

1. **Planck data paths and formats** — corrected to match your actual files
2. **Quasicrystal framework** — the theoretical reframing from periodic → quasiperiodic
3. **LIGO results** — documented as context
4. **New scripts** — three analysis scripts, not one
5. **"Never fall back to mock data silently"** — added as a hard rule

## Repository Structure Reference (Source of Truth)

Use the auto-generated directory tree in `README.md` as the canonical project
layout reference when locating files.

- Tree block markers: `<!-- TREE:START -->` … `<!-- TREE:END -->`
- Refresh command: `bash scripts/update_tree.sh`
- If path assumptions conflict with the README tree, trust the README tree.

---

```markdown
# Copilot Instructions — Gravitotemporal Quasicrystal Cosmology

## Repository Structure Reference (Source of Truth)

Use the auto-generated directory tree in `README.md` as the canonical project
layout reference when locating files.

- Tree block markers: `<!-- TREE:START -->` … `<!-- TREE:END -->`
- Refresh command: `bash scripts/update_tree.sh`
- If path assumptions conflict with the README tree, trust the README tree.

## Project Overview

This project tests a novel cosmological framework called the **Gravitotemporal
Quasicrystal Hypothesis** against empirical data. The core claim is that cosmic
expansion follows discrete golden-ratio (φ = 1.618...) time steps rather than
smooth continuous evolution, and that this geometric rhythm eliminates the need
for dark energy as a separate substance — reframing it as an emergent property
of the expansion clock.

The framework predicts a **quasicrystalline** (not periodic) structure to
spacetime, which produces **log-periodic** modulations in cosmological
observables — oscillations that are periodic in ln(k), ln(ℓ), or ln(r),
with a characteristic period of ln(φ) ≈ 0.4812.

We are fitting competing cosmological models to Type Ia supernova data
(Pantheon+), searching for φ-log-periodic signatures in CMB power spectrum
residuals (Planck 2018), and planning BAO correlation function analysis
(DESI DR1).

---

## Previous Results (Context for New Work)

### Pantheon+ Supernova Analysis — COMPLETED

| Model | χ²/dof | Parameters | Verdict |
|-------|--------|------------|---------|
| Model A (ΛCDM) | 0.880 | H0, Ωm | Benchmark |
| Model B (φⁿ naive) | 4.04 | H0 | ❌ Ruled out |
| Model C (φ-periodic w_de) | 0.880 | H0, Ωm, ε | ✅ Survived |

**Model C key result:**
- ε = 0.052 ± 0.138 (consistent with zero at 0.4σ)
- Ωm = 0.335 ± 0.019 (matches Planck and Pantheon+ ΛCDM)
- H0 unconstrained (flat posterior, expected when marginalizing over M)
- P(ε > 0) = 64.7% (mild preference for positive oscillation)
- Upper bound: |ε| < 0.33 at 95% confidence

**Interpretation:** Model C is not ruled out, not detected. The supernova data
integrates over H(z), washing out oscillatory signals.

### LIGO Ringdown Analysis — COMPLETED (mixed)

| Test | Method | Result | Status |
|------|--------|--------|--------|
| Test A: QNM frequency ratios | Ratio-based, mass-independent | m=1 above 2σ | ⚠️ Marginal |
| Test B: Stacked ringdown PSD | Harmonic comb search | above 3σ initially | ❌ Systematic artifact |

**Test B failure explanation:** The "signal" appeared in an off-source null
test, meaning it was caused by instrumental/systematic structure in LIGO's
noise floor, not astrophysical signal. This revealed that searching for a
**linear-periodic** harmonic comb was the wrong test — the quasicrystal
hypothesis predicts **log-periodic** signatures, not linear harmonics.

This failure motivated the theoretical reframing below.

---

## Theoretical Framework — Quasicrystalline Spacetime

### Key Insight: Quasiperiodic, Not Periodic

The hypothesis implies a 3D icosahedral quasicrystal (Penrose-tiling-like
structure), NOT a regular crystal lattice. The critical difference:

| Property | Periodic crystal | Quasicrystal |
|----------|-----------------|--------------|
| Real-space structure | Repeating unit cell | Aperiodic, self-similar |
| Diffraction pattern | Discrete Bragg peaks at integer spacings | Dense set of peaks at φ-related spacings |
| Observable signature | Harmonics at k, 2k, 3k, ... | Log-periodic: period ln(φ) in ln(k) |
| What to search for | Comb in linear space | Oscillation in log space |

### Structure Factor

The quasicrystal structure factor S(k) has peaks at wavevectors:

    k_G = G_perp · (projection from 6D → 3D)

These peaks are dense but hierarchically ordered, with intensities falling
off as a power law. The dominant spacing ratio between strong peaks is φ.

### Testable Predictions (Log-Periodic Modulations)

All predictions take the form of small fractional modulations on top of ΛCDM:

**Power Spectrum (matter or CMB):**
```
δP(k)/P_ΛCDM(k) = ε · cos(2π · ln(k/k₀) / ln(φ) + φ₀)
```

**Correlation Function (BAO):**
```
δξ(r)/ξ_ΛCDM(r) = ε' · cos(2π · ln(r/r₀) / ln(φ) + ψ₀)
```

**CMB Angular Power Spectrum:**
```
δCℓ/Cℓ^ΛCDM = ε'' · cos(2π · ln(ℓ/ℓ₀) / ln(φ) + χ₀)
```

The Lomb-Scargle target frequency in all cases is:
```
f_target = 1 / ln(φ) ≈ 2.078 cycles per unit of ln(variable)
```

### Theoretical Architecture (Three Layers)

```
LAYER 3 (Foundation): Fibonacci Anyon Fusion Algebra
    τ × τ = 1 + τ  →  quantum dimension d = φ
    Total quantum dimension D = √(1 + φ²) ≈ 1.902
    Topological entanglement entropy S_topo = ln(D) ≈ 0.643
    WHY φ appears: it is the quantum dimension of the simplest
    non-Abelian anyon model capable of universal error correction.
         │
         ▼
LAYER 2 (Dynamics): Quasiperiodic Hamiltonian
    H(t) driven by Fibonacci sequence → time quasicrystal
    w_de(z) = -1 + ε·cos(2π ln(1+z) / ln φ)
    G = (ℏc/M²)·exp(-1/φα)  [Brandenburg GEM formula]
    The exponential is a tunneling amplitude through topological
    protection — gravity is weak because the vacuum error-corrects.
         │
         ▼
LAYER 1 (Observables): Log-periodic Scaling
    T(n) = A·φⁿ  [master time equation]
    a(n) = φ^((n-N)/3)  [scale factor]
    V(λ) = quasiperiodic potential → era transitions
         │
         ▼
DATA: Pantheon+ (✅), LIGO (⚠️/❌), CMB (← CURRENT), BAO (pending)
```

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
ln_phi = np.log(phi)            # ≈ 0.48121182506
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

# CMB / DSI analysis constants
K_PHI = 1.0 / np.log(phi)     # target Fourier frequency ≈ 2.078 cycles/ln(ℓ)
LN_PHI = np.log(phi)           # ≈ 0.4812, the characteristic log-period
```

### The Three Cosmological Models

**Model A — ΛCDM (benchmark):**
- `H(z) = H0 * sqrt(Om*(1+z)^3 + OL)`
- Free parameters: `H0`, `Om` (with `OL = 1 - Om` for flat universe)

**Model B — Naive φⁿ (expected to fail):**
- `H(z) = H0 * (1+z)^3` (stiff-fluid, w = 1)
- Free parameters: `H0` only

**Model C — Quasiperiodic φ-potential (the real test):**
- `w_de(z) = -1 + epsilon * cos(2*pi*ln(1+z) / ln(phi))`
- Free parameters: `H0`, `Om`, `epsilon`
- If `epsilon = 0`, reduces to ΛCDM.

### Master Time Equation

```
T(n) = t_P * (12 * lambda0^3 * e / pi) * phi^n
```

Where n = integer tick number, lambda0 = initial lattice parameter.
With lambda0 = 1: prefactor A = t_P * 12*e/pi ≈ 5.59e-43 s.
Today's tick: N ≈ 287.

### Scale Factor

```
a(n) = phi^((n - N) / 3)
```

### Quasiperiodic Potential

```
V(lam) = V0 * [1 - cos(2*pi*lam/phi) - (1/phi)*cos(2*pi*lam/phi^2)]
```

Incommensurate frequencies (ratio = φ) → quasiperiodic minima.

---

## Data Sources

### Primary: Pantheon+ Type Ia Supernovae — COMPLETED

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
  - MUST be used for any parameter fitting
- **Filtering rules:**
  - Exclude `IS_CALIBRATOR == 1` for cosmological fits
  - Apply `zHD > 0.01` cut to avoid peculiar velocity dominated regime
  - When using `m_b_corr`, analytically marginalize over absolute magnitude M

### Secondary: CMB Power Spectrum (Planck 2018) — CURRENT ANALYSIS

- **Source:** Planck Legacy Archive (ESA)
- **Reference papers:**
  - Planck 2018 V — Power spectra and likelihoods: arXiv:1907.12875
  - Planck 2018 VI — Cosmological parameters: arXiv:1807.06209

- **Data files (already downloaded to `data/planck/`):**

  | File | Columns | Description |
  |------|---------|-------------|
  | `COM_PowerSpect_CMB-TT-full_R3.01.txt` | ℓ, Dℓ, −dDℓ, +dDℓ | TT spectrum, ℓ = 2–2508 |
  | `COM_PowerSpect_CMB-TE-full_R3.01.txt` | ℓ, Dℓ, −dDℓ, +dDℓ | TE spectrum, ℓ = 2–1996 |
  | `COM_PowerSpect_CMB-EE-full_R3.01.txt` | ℓ, Dℓ, −dDℓ, +dDℓ | EE spectrum, ℓ = 2–1996 |
  | `COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt` | ℓ, DℓTT, DℓTE, DℓEE, DℓBB, DℓPP | Best-fit ΛCDM theory |
  | `COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum_R3.01.txt` | (key-value pairs) | Contains `calPlanck` calibration value |

- **CRITICAL: calPlanck correction**
  The theory spectra MUST be divided by `calPlanck**2` before comparison to
  the coadded CMB spectra. The calPlanck value is in the parameters file.
  See: https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/CMB_spectrum_%26_Likelihood_Code

- **Column format for `-full` data files (4 columns):**
  ```
  # ℓ    Dℓ(μK²)    -dDℓ    +dDℓ
  ```
  where Dℓ = ℓ(ℓ+1)Cℓ/2π in μK². Errors are asymmetric; symmetrize as:
  `sigma = 0.5 * (abs(col2) + abs(col3))`

- **Column format for theory file (6 columns):**
  ```
  # ℓ    DℓTT    DℓTE    DℓEE    DℓBB    DℓPP
  ```

- **Data loader (use this exact pattern):**
  ```python
  def load_planck_spectrum(spec_type="TT"):
      """Load Planck -full spectrum files from data/planck/."""
      fname = Path("data/planck") / f"COM_PowerSpect_CMB-{spec_type}-full_R3.01.txt"
      if not fname.exists():
          raise FileNotFoundError(f"Planck data not found: {fname}")
      data  = np.loadtxt(fname, comments="#")
      ell   = data[:, 0]
      Dl    = data[:, 1]
      sigma = 0.5 * (np.abs(data[:, 2]) + np.abs(data[:, 3]))
      return ell, Dl, sigma

  def load_planck_theory():
      """Load Planck best-fit ΛCDM theory, corrected by calPlanck²."""
      # Get calPlanck
      param_file = Path("data/planck") / \
          "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum_R3.01.txt"
      cal_planck = 1.0
      with open(param_file) as f:
          for line in f:
              if "calPlanck" in line and "=" in line:
                  cal_planck = float(line.split("=")[-1].strip())
                  break

      theory_file = Path("data/planck") / \
          "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt"
      data = np.loadtxt(theory_file, comments="#")
      ell_th = data[:, 0]
      # Columns: ℓ, TT, TE, EE, BB, PP — divide by calPlanck²
      theory = {
          "TT": data[:, 1] / cal_planck**2,
          "TE": data[:, 2] / cal_planck**2,
          "EE": data[:, 3] / cal_planck**2,
      }
      return ell_th, theory
  ```

- **What we search for in the CMB:**

  The residuals (observed Dℓ minus best-fit ΛCDM theory) are analyzed with
  a Lomb-Scargle periodogram in **ln(ℓ)** space (NOT linear ℓ space).

  **Target signal:**
  ```
  ΔCℓ/Cℓ = ε_CMB · cos(2π · ln(ℓ/ℓ₀) / ln(φ) + δ)
  ```

  **Target Lomb-Scargle frequency:**
  ```
  f_φ = 1 / ln(φ) ≈ 2.078 cycles per unit ln(ℓ)
  ```

  Also check harmonics: m=1 at f_φ, m=2 at 2·f_φ, m=3 at 3·f_φ.

  **Significance:** Bootstrap with 10,000 shuffles of residuals.

  **Why ln(ℓ) and not ℓ:** A quasicrystal produces self-similar (scale-invariant)
  modulations. These are periodic in log-space, not linear space. Searching
  in linear ℓ-space (as done previously) would miss the signal — like trying
  to hear a melody by sampling at the wrong rate.

### Tertiary: BAO Correlation Function (DESI DR1) — PENDING

- **Documentation:** https://data.desi.lbl.gov/doc/releases/dr1/vac/full-shape-bao-clustering/
- **Data files (actual DR1 VAC path):**
  - Base: `https://data.desi.lbl.gov/public/dr1/vac/dr1/full-shape-bao-clustering/v1.0/`
  - Correlation poles (recsym):
    `.../data/recsym/correlation/correlation-recsym-poles_LRG_GCcomb_z0.6-0.8.h5`
  - Covariance (RascalC):
    `.../data/covariance/RascalC/covariance_correlation-recsym-poles_LRG_GCcomb_z0.6-0.8.h5`
- **Alternative:** Zenodo deposit https://zenodo.org/records/14276816
- **Alternative:** CosmoSIS compressed BAO: https://cosmosis.readthedocs.io/en/latest/reference/standard_library/desi_dr1.html
- **Key observable:** Galaxy correlation function ξ₀(s) for LRG z=0.6–0.8
- **Search for:** Log-periodic modulation: `δξ(r)/ξ_ΛCDM(r) = ε' · cos(2π·ln(r/r₀)/ln(φ) + ψ₀)`
- **NOT searching for:** Linear-periodic secondary peaks (that was the old, incorrect prediction)
- **File format:** Columns are typically `s [Mpc/h]`, `ξ₀(s)`, possibly `ξ₂`, `ξ₄`
- **Covariance:** In a separate `cov/` subdirectory with matching filename

---

## Coding Conventions

### General

- Python 3.10+
- Use numpy for all numerical work
- Use scipy.integrate.quad for single integrals
- Use scipy.signal.lombscargle for periodogram analysis in ln-space
- Use scipy.interpolate.interp1d to align theory with data ℓ values
- All distances in Mpc, all speeds in km/s, all times in seconds unless noted
- Use natural log (`np.log`) unless base-10 is specified (`np.log10`)

### Output Paths (Repository Standard)

- Save figures to `results/figures/`
- Save machine-readable outputs (`.json`, `.csv`, `.txt`) to `results/tables/`
- Save MCMC chains to `results/chains/`
- Do **NOT** save outputs to `output/` or repo root unless explicitly requested
- Create output directories with `mkdir(parents=True, exist_ok=True)` before writing

### Variable Naming

```python
# Redshift
z, z_arr, z_hd

# Distances
dL, dA, dC       # luminosity, angular diameter, comoving (Mpc)
mu               # distance modulus (mag)

# Hubble parameter
H0, H_z          # km/s/Mpc

# Density parameters
Om, Ob, OL, Or, Ok

# Model-specific
eps              # epsilon: amplitude of modulation
n_tick           # integer tick number
lam, lam0        # lattice parameter
w_de             # dark energy equation of state
N_today          # present-epoch tick (≈ 287)

# CMB analysis
ell              # multipole moment
Dl, Dl_err       # Dℓ in μK², symmetrized error
ln_ell           # = np.log(ell)
f_phi            # target frequency = 1/ln(φ) ≈ 2.078
resid            # data - theory residuals
resid_norm       # resid / sigma

# DSI modulation parameters
eps_dsi          # fractional modulation amplitude
k0, r0, ell0     # reference scale for log-periodic phase
phase0           # phase offset

# Fitting
chi2, chi2_nu, ndof
cov, cov_inv
```

### Key Data Loading Pattern

```python
# ALWAYS load Planck data this way:
def load_planck_spectrum(spec_type="TT"):
    """
    Load from data/planck/COM_PowerSpect_CMB-{spec_type}-full_R3.01.txt
    4 columns: ℓ, Dℓ, -dDℓ, +dDℓ
    """
    fname = PLANCK_DIR / f"COM_PowerSpect_CMB-{spec_type}-full_R3.01.txt"
    if not fname.exists():
        raise FileNotFoundError(
            f"Missing: {fname}\n"
            f"Download from Planck Legacy Archive and place in data/planck/"
        )
    data  = np.loadtxt(fname, comments="#")
    ell   = data[:, 0]
    Dl    = data[:, 1]
    sigma = 0.5 * (np.abs(data[:, 2]) + np.abs(data[:, 3]))
    return ell, Dl, sigma
```

### Performance Notes

- Pantheon+ covariance inversion: do ONCE, cache result
- Use Cholesky decomposition for numerical stability
- CMB bootstrap (10,000 iterations): ~2–5 minutes, no optimization needed
- Lomb-Scargle is O(N·M), fast enough as-is

---

## Success Criteria

| Test | Pass condition | Fail condition |
|------|---------------|----------------|
| Model B vs ΛCDM | (expected to fail) | χ²/dof >> 1 ✓ confirmed |
| Model C χ²/dof | ≤ 1.05 | > 1.10 |
| Model C vs ΛCDM | Δχ² < 4 | Δχ² > 10 |
| ε significance | > 0 at 2σ+ → detected | consistent with 0 → not detected |
| **CMB ln(ℓ) periodicity** | **Peak at f≈2.078 at >3σ** | **No peak → below CMB sensitivity** |
| **CMB cross-check** | **Same peak in TT, TE, and EE** | **TT-only → likely systematic** |
| **BAO ln(r) modulation** | **Δχ² > 6 vs null (ε=0)** | **Δχ² < 2 → no signal** |

### Model selection:
- AIC = χ² + 2k (k = free parameters)
- BIC = χ² + k·ln(N) (N = data points)
- Lower AIC/BIC = preferred model

---

## Updated Scorecard

| Prediction | Test | Result | Status |
|---|---|---|---|
| φⁿ expansion (Model B) | Pantheon+ χ² | χ²/ν = 4.04 | ❌ Ruled out |
| φ-modulated dark energy (Model C) | Pantheon+ MCMC | ε = 0.05 ± 0.14 | ⚠️ Survived, not detected |
| φ-harmonic QNM ratios | LIGO ringdown ratios | m=1 above 2σ | ⚠️ Marginal |
| φ-harmonic ringdown PSD | LIGO stacked PSD | Systematic artifact | ❌ Failed null test |
| φ-log-periodic CMB modulations | Planck TT/TE/EE Lomb-Scargle in ln(ℓ) | — | ← CURRENT |
| φ-log-periodic BAO modulations | DESI ξ(r) fit in ln(r) | — | 🔲 Pending |
| Lattice scale ≈ proton radius | Analytical | L ≈ 0.6 fm vs rₚ ≈ 0.88 fm | ⚠️ Suggestive |
| G = (ℏc/M²)exp(-1/φα) | Brandenburg formula | Matches measured G | ✅ Correct order of magnitude |

---

## Dependencies

```
# requirements.txt
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
emcee>=3.1
corner>=2.2
# Optional:
astropy>=5.3
```

Note: CAMB is NOT required. We use the official Planck best-fit theory file
(`*-theory_R3.01.txt`) as the ΛCDM template, corrected by calPlanck².

---

## References

- Pantheon+ data paper: arXiv:2112.03863
- Pantheon+ cosmological constraints: arXiv:2202.04077
- Planck 2018 overview: arXiv:1807.06205
- Planck 2018 cosmological parameters: arXiv:1807.06209
- Planck 2018 power spectra and likelihoods: arXiv:1907.12875
- Planck Legacy Archive: https://pla.esac.esa.int/pla/#cosmology
- Planck Wiki (calPlanck): https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/CMB_spectrum_%26_Likelihood_Code
- DESI DR1: arXiv:2503.14745
- DESI 2024 II (clustering): arXiv:2411.12020
- DESI 2024 VI (cosmological constraints): arXiv:2404.03002
- DESI DR1 clustering products: https://data.desi.lbl.gov/doc/releases/dr1/vac/full-shape-bao-clustering/
- Fibonacci time crystal: Nature 2022
- Brandenburg GEM theory: IEEE Trans. Plasma Sci. Vol. 35, No. 4, p845 (2007)

---

## What NOT to Do

- Do NOT use diagonal errors for final parameter constraints (plotting only)
- Do NOT fit calibrator SNe (IS_CALIBRATOR == 1) in cosmological analysis
- Do NOT hardcode H0 = 70 — always a free parameter
- Do NOT ignore absolute magnitude degeneracy — always marginalize over M
- Do NOT use scipy.optimize for final results — use MCMC for posteriors
- Do NOT compare models by χ² alone — use AIC or BIC
- Do NOT perform FFT on event display images — use actual data products
- Do NOT assume CMB data format — always check header/comment lines
- Do NOT search in linear ℓ/k/r space — the quasicrystal predicts LOG-periodic signals
- Do NOT use the parametric ΛCDM approximation for amplitude claims
  (use the official Planck theory file instead)
- Do NOT look for Planck data in `data/cmb/` — it is in `data/planck/`
- Do NOT look for `-binned` files — we use the `-full` (unbinned) files
- Do NOT write figures or results to `output/` or project root; use `results/figures` and `results/tables`
- **Do NOT silently fall back to mock/synthetic data.** If real data files
  are missing, raise an error with the expected path. Mock data may only
  be used when explicitly requested for pipeline validation, and MUST be
  clearly labeled in all output and figure titles.
```

---

## What Changed (Quick Diff)

| Section | Old | New | Why |
|---------|-----|-----|-----|
| Data directory | `data/cmb/` | `data/planck/` | Matches your actual folder |
| Filenames | `-binned_R3.01.txt` | `-full_R3.01.txt` | Matches what you downloaded |
| Column format | 6 columns (binned) | 4 columns (ℓ, Dℓ, −dDℓ, +dDℓ) | Different file format |
| Theory template | CAMB or downloaded | Local file + calPlanck² correction | Already on disk, no dependency |
| Search method | Linear ℓ periodicity | **Log-periodic in ln(ℓ)** | Quasicrystal ≠ crystal |
| LIGO results | Not mentioned | Documented with lessons learned | Informs why we switched approach |
| Mock fallback | Silent fallback | **Hard error** | Prevents false results |
| Scripts | 1 (`cmb_phi_search.py`) | 3 (`quasicrystal_structure_factor.py`, `cmb_dsi_lnell.py`, `bao_dsi_fit.py`) | Matches current codebase |
| DESI source | Generic | Specific navigation path + alternatives | Actionable |

Save this as `.github/copilot-instructions.md`, then tell Copilot to patch `cmb_dsi_lnell.py` using the instructions from my previous message. The key fix is: **read from `data/planck/`, use `-full` filenames, parse 4 columns, and never fall back to mock data.**