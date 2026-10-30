# Results Summary — Gravitotemporal Quasicrystal Cosmology

## What is this project?

This project tests whether cosmic expansion follows discrete golden-ratio
(φ ≈ 1.618) time steps rather than smooth continuous evolution. If true,
dark energy would not be a separate substance but an emergent property of
the expansion clock — a "time quasicrystal" at cosmological scales.

We tested three competing models against two independent datasets:
**1,580 Type Ia supernovae** (Pantheon+) and the **Planck 2018 CMB
temperature power spectrum**.

---

## The Three Models

| Model | Physics | Free parameters |
|---|---|---|
| **A — ΛCDM** | Standard cosmology with cosmological constant | H₀, Ωₘ |
| **B — Naive φⁿ** | Direct φⁿ expansion (stiff fluid, w = 1) | H₀ |
| **C — φ-periodic dark energy** | w_de(z) = −1 + ε·cos(2π ln(1+z)/ln φ) | H₀, Ωₘ, ε |

Model A is the standard benchmark. Model B is the raw master equation
prediction, included to show it fails and motivate the refined Model C.
Model C embeds the golden-ratio periodicity into the dark energy equation
of state; if ε = 0 it reduces identically to ΛCDM.

---

## Test 1: Pantheon+ Supernovae — COMPLETED

**Dataset:** 1,580 Type Ia supernovae, z = 0.01–2.26, full 1580×1580
statistical + systematic covariance matrix.

**Method:** Maximum-likelihood fitting with analytic marginalization over
absolute magnitude M, followed by full MCMC posterior exploration (32
walkers × 50,000 steps, 10,000 burn-in) for Model C.

### Results

| Model | χ²/dof | AIC | BIC | Verdict |
|---|---|---|---|---|
| A — ΛCDM | 0.880 | 1391.1 | 1401.8 | Benchmark |
| B — Naive φⁿ | 4.040 | 6377.4 | 6382.8 | ❌ **Ruled out** |
| C — φ-periodic | 0.880 | 1393.0 | 1409.0 | ✅ Survived |

### Model C posterior (MCMC)

| Parameter | Median ± 1σ | Notes |
|---|---|---|
| H₀ | 74.9 ± 17.0 km/s/Mpc | Unconstrained (flat posterior; expected when marginalizing over M) |
| Ωₘ | 0.335 ± 0.019 | Matches Planck 2018 (0.315) and Pantheon+ ΛCDM (0.332) |
| ε | 0.052 ± 0.138 | Consistent with zero at 0.4σ |

- **P(ε > 0) = 64.7%** — mild preference for positive oscillation, not significant
- **Upper bound:** |ε| < 0.33 at 95% confidence
- **MCMC diagnostics:** Gelman-Rubin R̂ = 1.0005–1.0006, ESS > 1.2 million — fully converged

### Interpretation

Model B (the raw φⁿ master equation) is decisively ruled out: χ²/dof = 4.04
implies a stiff-fluid universe that drastically overpredicts luminosity
distances at high redshift.

Model C fits the data exactly as well as ΛCDM (Δχ² = 0.15). The extra
parameter ε is not significantly detected — the data are consistent with
ε = 0 (i.e., standard ΛCDM). BIC penalizes the extra parameter and
prefers Model A by ΔBIC = 7.2.

**The framework survives but is not detected.** This is expected:
supernova distances integrate over H(z), averaging out any oscillatory
signal. The φ-periodic modulation needs a probe that measures each scale
independently — which is exactly what the CMB provides.

---

## Test 2: CMB Power Spectrum (Planck 2018) — COMPLETED

**Dataset:** Planck 2018 binned TT power spectrum, 38 multipole bins
spanning ℓ ≈ 48 to ℓ ≈ 1155.

**Method:** Compute residuals (observed Dℓ minus Planck best-fit ΛCDM theory),
then perform a Lomb-Scargle periodogram in ln(ℓ) space (equivalent to a
Mellin transform). This detects scale-invariant, multiplicative
periodicities. Significance assessed via 10,000 bootstrap shuffles.

**Target signal:** A peak at Fourier frequency k_φ = 1/ln(φ) ≈ 2.078
cycles per unit ln(ℓ), which would indicate golden-ratio-spaced
modulations imprinted at the epoch of recombination (z ≈ 1100).

### Results

| Quantity | Value |
|---|---|
| Target frequency k_φ | 2.078 |
| Nearest peak found | k = 2.148 |
| Peak power | 0.083 |
| 1σ threshold (bootstrap) | 0.413 |
| Significance | **< 1σ** |
| p-value | 1.0 |
| Global maximum | k = 0.126 (power = 0.828) — unrelated low-frequency mode |

### Interpretation

**No φ-periodic signal is detected in the CMB.** The power at k ≈ 2.08 is
a factor of 5× below the 1σ noise threshold. The Planck TT residuals show
no evidence for golden-ratio structure at the sensitivity of the binned
spectrum (38 bins).

This constrains any φ-periodic modulation of the CMB to be very small —
below the cosmic variance + measurement noise floor of Planck's binned TT
data.

---

## Combined Scorecard

| Prediction | Test | Result | Status |
|---|---|---|---|
| φⁿ discrete expansion (Model B) | Pantheon+ χ² | χ²/dof = 4.04 | ❌ Ruled out |
| φ-periodic dark energy (Model C) | Pantheon+ MCMC | ε = 0.05 ± 0.14 | ⚠️ Survived, not detected |
| φ-periodic CMB modulations | Planck TT Lomb-Scargle | < 1σ, p = 1.0 | ❌ Not detected |
| φ-harmonic BAO features | DESI ξ(r) | — | 🔲 Pending |
| Lattice scale ≈ proton radius | Analytical | L ≈ 0.6 fm vs rₚ ≈ 0.88 fm | ⚠️ Suggestive |
| G = (ℏc/M²)exp(−1/φα) | Brandenburg formula | Matches measured G | ✅ Correct order of magnitude |

---

## What does this mean?

### The honest assessment

The Gravitotemporal Quasicrystal Hypothesis has **not been falsified by
supernova data** — Model C fits Pantheon+ as well as standard ΛCDM — but
it has also **produced no positive detection** in either supernovae or the
CMB. The golden-ratio periodicity parameter ε is consistent with zero in
both probes.

### What Model B's failure tells us

The raw master equation T(n) = A·φⁿ, taken literally as the sole driver of
expansion, produces a stiff-fluid cosmology that is catastrophically wrong
(χ²/dof = 4.04). This means the naive reading of the framework — that
φⁿ scaling directly replaces the Friedmann equation — does not work.

### What Model C's survival tells us

The φ-periodic dark energy formulation (Model C) is not ruled out because
it contains ΛCDM as a special case (ε = 0). The data merely tell us that
any φ-periodic oscillation in the dark energy equation of state must be
small: |ε| < 0.33 at 95% confidence. The framework's predictions are
unfalsified but also unconfirmed.

### What the CMB null result tells us

The absence of a φ-periodic peak in the Planck TT residuals is the
strongest constraint so far. Unlike supernovae (which integrate over
redshift, washing out oscillations), the CMB preserves oscillatory features
at each multipole independently. The null result at < 1σ means either:

1. The lattice imprint is below Planck's sensitivity in the binned TT spectrum
2. The signal may appear in polarization (TE, EE) rather than temperature
3. The unbinned spectrum (2,500 multipoles vs 38 bins) may have more resolving power
4. The φ-periodicity affects perturbation growth differently than assumed

### What remains to be done

- **BAO analysis (DESI DR1):** Search for secondary peaks in the galaxy
  correlation function at r ≈ 93 Mpc (150/φ) and r ≈ 243 Mpc (150×φ).
  This is an independent geometric test.
- **Unbinned CMB:** Repeat the Lomb-Scargle analysis on the full 2,500-multipole
  Planck TT spectrum (now downloaded) for higher frequency resolution.
- **CMB polarization:** Search for φ-periodicity in TE and EE spectra as
  a cross-check — a TT-only signal would likely be systematic.
- **CMB with CAMB:** Use exact CAMB theory spectra (now in requirements)
  instead of the tabulated best-fit, for amplitude-calibrated residuals.

---

## Technical Details

- **Code:** Python 3.10+, numpy, scipy, matplotlib, emcee, corner
- **Supernova fitting:** Full 1580×1580 STAT+SYS covariance via Cholesky decomposition;
  analytic marginalization over absolute magnitude M
- **MCMC:** 32 walkers × 50,000 steps (10,000 burn-in); Gelman-Rubin R̂ < 1.001;
  ESS > 1.2M for all parameters
- **CMB analysis:** Lomb-Scargle periodogram in ln(ℓ) space; 10,000 bootstrap
  shuffles for significance; Planck official best-fit theory subtraction
- **Tests:** 50 unit tests covering constants, models, fitting, CMB analysis,
  and data loading — all passing
