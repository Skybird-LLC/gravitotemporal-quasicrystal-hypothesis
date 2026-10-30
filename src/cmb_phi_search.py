"""
CMB φ-Periodicity Search
=========================
Searches for golden-ratio periodic modulations in the Planck 2018
CMB TT power spectrum residuals using a Fourier analysis in ln(ℓ) space.

Downloads all required data automatically.
No dependency on Pantheon+ data — this is an independent test.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import lombscargle
import matplotlib.pyplot as plt
import urllib.request
import os
import json
from pathlib import Path

# ══════════════════════════════════════════════════════════════
# 0. CONSTANTS
# ══════════════════════════════════════════════════════════════

PHI = (1 + np.sqrt(5)) / 2          # golden ratio
K_PHI = 1.0 / np.log(PHI)           # target frequency in ln(ℓ) space ≈ 2.078
BASE_DIR = Path(__file__).resolve().parent.parent


def resolve_cmb_data_dir() -> str:
    data_planck = BASE_DIR / 'data' / 'planck'
    data_cmb = BASE_DIR / 'data' / 'cmb'
    marker = 'COM_PowerSpect_CMB-TT-full_R3.01.txt'

    if (data_planck / marker).exists():
        return str(data_planck)
    if (data_cmb / marker).exists():
        return str(data_cmb)
    if data_planck.exists():
        return str(data_planck)
    return str(data_cmb)


DATA_DIR = resolve_cmb_data_dir()
RESULTS_DIR = str(BASE_DIR / 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')

for d in [DATA_DIR, RESULTS_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 1. DOWNLOAD PLANCK 2018 DATA
# ══════════════════════════════════════════════════════════════

# Planck 2018 binned TT power spectrum (from Plik likelihood)
# Source: Planck Legacy Archive / ESA
# Paper: Planck 2018 results V (arXiv:1907.12875)
PLANCK_TT_URL = (
    "https://pla.esac.esa.int/pla/aio/product-action"
    "?COSMOLOGY.FILE_ID=COM_PowerSpect_CMB-TT-binned_R3.01.txt"
)

# Fallback: NASA LAMBDA hosts the same data
LAMBDA_TT_URL = (
    "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/"
    "actpol_2016/auxiliary_data/bf_base_plikHM_TTTEEE_lowl_lowE_lensing.minimum.theory_cl"
)

# Best-fit ΛCDM parameters (Planck 2018 TT,TE,EE+lowE+lensing)
# From Table 2 of arXiv:1807.06209
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


def download_file(url, filepath, description=""):
    """Download a file if not already present."""
    if os.path.exists(filepath):
        print(f"  Found {os.path.basename(filepath)} locally.")
        return True
    print(f"  Downloading {description or os.path.basename(filepath)} ...")
    try:
        urllib.request.urlretrieve(url, filepath)
        size = os.path.getsize(filepath) / 1024
        print(f"    Done ({size:.1f} KB)")
        return True
    except Exception as e:
        print(f"    FAILED: {e}")
        return False


def load_two_numeric_columns(filepath):
    """Load first two numeric columns from mixed-content text files."""
    col0 = []
    col1 = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            col0.append(x)
            col1.append(y)

    if not col0:
        raise ValueError(f"No numeric rows found in {filepath}")

    return np.asarray(col0), np.asarray(col1)


def generate_lcdm_theory(ell_max=2600):
    """
    Generate a best-fit ΛCDM TT power spectrum.

    Uses the analytic approximation for the CMB TT spectrum.
    For production work, use CAMB or CLASS. For this search,
    we need the SHAPE of the residuals, not absolute calibration.

    The Planck team provides their best-fit theory curve in the
    data release. We try to download it; if unavailable, we use
    a simplified parametric model.
    """
    try:
        # Try to use CAMB if installed
        import camb
        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=LCDM_PARAMS['H0'],
            ombh2=LCDM_PARAMS['ombh2'],
            omch2=LCDM_PARAMS['omch2'],
            tau=LCDM_PARAMS['tau']
        )
        pars.InitPower.set_params(
            As=LCDM_PARAMS['As'],
            ns=LCDM_PARAMS['ns']
        )
        pars.set_for_lmax(ell_max + 100)
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
        totCL = powers['total']
        ells = np.arange(totCL.shape[0])
        # CAMB returns C_ℓ; we want D_ℓ = ℓ(ℓ+1)C_ℓ/2π
        Dl = totCL[:, 0]  # TT is column 0, already in D_ℓ convention
        mask = (ells >= 2) & (ells <= ell_max)
        print("  Using CAMB for ΛCDM theory spectrum.")
        return ells[mask], Dl[mask], 'CAMB'
    except ImportError:
        pass

    # Fallback: download tabulated best-fit theory spectra
    theory_candidates = [
        (
            "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/"
            "actpol_2016/auxiliary_data/"
            "bf_base_plikHM_TTTEEE_lowl_lowE_lensing.minimum.theory_cl",
            os.path.join(DATA_DIR, 'lambda_bestfit_theory_cl.dat'),
            'LAMBDA_bestfit',
            "LAMBDA best-fit theory Cl",
        ),
        (
            "https://pla.esac.esa.int/pla/aio/product-action"
            "?COSMOLOGY.FILE_ID="
            "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum_R3.01.txt",
            os.path.join(DATA_DIR, 'planck_bestfit_theory_cl.dat'),
            'Planck_official',
            "Planck best-fit theory Cl",
        ),
    ]

    for theory_url, theory_file, source_name, description in theory_candidates:
        if not download_file(theory_url, theory_file, description):
            continue

        try:
            ell_raw, dl_raw = load_two_numeric_columns(theory_file)
            ells = np.rint(ell_raw).astype(int)
            valid = np.isfinite(dl_raw) & (ells >= 2)
            ells = ells[valid]
            Dl = dl_raw[valid]

            if len(ells) == 0:
                raise ValueError("No valid spectrum rows after filtering")

            order = np.argsort(ells)
            ells = ells[order]
            Dl = Dl[order]

            uniq_ells, first_idx = np.unique(ells, return_index=True)
            ells = uniq_ells
            Dl = Dl[first_idx]

            if len(ells) < 100 or ells.max() < min(1000, ell_max):
                raise ValueError(
                    "Parsed table does not look like a full theory spectrum "
                    f"(rows={len(ells)}, ell_max={ells.max()})"
                )

            mask = (ells >= 2) & (ells <= ell_max)
            if mask.sum() < 50:
                raise ValueError("Insufficient overlap with requested ell range")

            print(f"  Using {source_name} best-fit theory spectrum.")
            return ells[mask], Dl[mask], source_name
        except Exception as e:
            print(f"  Could not parse {source_name} theory file: {e}")

    # Last resort: simplified parametric approximation
    print("  WARNING: Using simplified parametric ΛCDM approximation.")
    print("  Install CAMB (`pip install camb`) for exact theory spectrum.")
    ells = np.arange(2, ell_max + 1)
    # This is a rough envelope — NOT suitable for precision cosmology
    # but adequate for frequency-domain analysis of residual SHAPE
    x = ells / 300.0  # scaled multipole
    # Approximate the acoustic peaks
    Dl_approx = (
        5500 * (x / 1.0)**2 / (1 + (x / 1.0)**4) *  # envelope
        (1 + 0.15 * np.cos(np.pi * x / 1.03))**2      # acoustic oscillations
    )
    return ells, Dl_approx, 'parametric_approximation'


def load_planck_tt():
    """
    Load Planck 2018 TT power spectrum data.

    Tries multiple sources in order of preference.
    Returns (ell, Dl, Dl_err) where Dl = ℓ(ℓ+1)Cℓ/2π in μK².
    """

    # --- Attempt 1: Planck binned TT spectrum ---
    binned_file = os.path.join(DATA_DIR, 'COM_PowerSpect_CMB-TT-binned_R3.01.txt')
    if download_file(PLANCK_TT_URL, binned_file, "Planck 2018 TT binned spectrum"):
        try:
            data = np.loadtxt(binned_file)
            ell = data[:, 0]
            Dl = data[:, 1]
            err_minus = data[:, 2] if data.shape[1] > 2 else np.zeros_like(Dl)
            err_plus = data[:, 3] if data.shape[1] > 3 else err_minus
            Dl_err = 0.5 * (np.abs(err_minus) + np.abs(err_plus))
            print(f"  Loaded binned TT: {len(ell)} multipole bins, "
                  f"ℓ = {ell.min():.0f} to {ell.max():.0f}")
            return ell, Dl, Dl_err, 'planck_binned'
        except Exception as e:
            print(f"  Could not parse binned file: {e}")

    # --- Attempt 2: Planck full (unbinned) TT spectrum ---
    full_file = os.path.join(DATA_DIR, 'COM_PowerSpect_CMB-TT-full_R3.01.txt')
    full_url = (
        "https://pla.esac.esa.int/pla/aio/product-action"
        "?COSMOLOGY.FILE_ID=COM_PowerSpect_CMB-TT-full_R3.01.txt"
    )
    if download_file(full_url, full_file, "Planck 2018 TT full spectrum"):
        try:
            data = np.loadtxt(full_file)
            ell = data[:, 0]
            Dl = data[:, 1]
            err_minus = data[:, 2] if data.shape[1] > 2 else np.zeros_like(Dl)
            err_plus = data[:, 3] if data.shape[1] > 3 else err_minus
            Dl_err = 0.5 * (np.abs(err_minus) + np.abs(err_plus))
            print(f"  Loaded full TT: {len(ell)} multipoles, "
                  f"ℓ = {ell.min():.0f} to {ell.max():.0f}")
            return ell, Dl, Dl_err, 'planck_full'
        except Exception as e:
            print(f"  Could not parse full file: {e}")

    # --- Attempt 3: NASA LAMBDA compilation ---
    lambda_file = os.path.join(DATA_DIR, 'lambda_tt_spectrum.txt')
    lambda_url = (
        "https://lambda.gsfc.nasa.gov/data/mapping/wmap/data_products/"
        "spectra/planck_coadded_TT.txt"
    )
    if download_file(lambda_url, lambda_file, "LAMBDA TT compilation"):
        try:
            data = np.loadtxt(lambda_file, comments='#')
            ell = data[:, 0]
            Dl = data[:, 1]
            Dl_err = data[:, 2] if data.shape[1] > 2 else np.ones_like(Dl) * 50
            print(f"  Loaded LAMBDA TT: {len(ell)} points")
            return ell, Dl, Dl_err, 'lambda'
        except Exception as e:
            print(f"  Could not parse LAMBDA file: {e}")

    raise RuntimeError(
        "Could not download Planck TT data from any source.\n"
        "Please download manually from:\n"
        "  https://pla.esac.esa.int/pla/#cosmology\n"
        "  Select: Power Spectra > CMB > TT\n"
        f"  Save to: {DATA_DIR}/\n"
        "Or install CAMB and we'll compute a theory spectrum."
    )


# ══════════════════════════════════════════════════════════════
# 2. THE ANALYSIS
# ══════════════════════════════════════════════════════════════

def compute_residuals(ell_data, Dl_data, ell_theory, Dl_theory):
    """
    Compute residuals: data - theory.

    Interpolates theory onto data multipoles.
    Returns normalized residuals (Δ/σ_cosmic) and raw residuals.
    """
    # Interpolate theory onto data multipoles
    theory_interp = CubicSpline(ell_theory, Dl_theory)

    # Only use multipoles in range of both data and theory
    ell_min = max(ell_data.min(), ell_theory.min())
    ell_max = min(ell_data.max(), ell_theory.max())
    mask = (ell_data >= ell_min) & (ell_data <= ell_max)

    ell = ell_data[mask]
    Dl = Dl_data[mask]
    Dl_th = theory_interp(ell)

    residuals = Dl - Dl_th

    # Cosmic variance: σ²(Cℓ) = 2/(2ℓ+1) * Cℓ²
    # In Dℓ units: σ(Dℓ) ≈ Dℓ * sqrt(2/(2ℓ+1))
    sigma_cosmic = np.abs(Dl_th) * np.sqrt(2.0 / (2.0 * ell + 1.0))
    sigma_cosmic = np.maximum(sigma_cosmic, 1.0)  # floor to avoid division by zero

    residuals_normalized = residuals / sigma_cosmic

    return ell, residuals, residuals_normalized, sigma_cosmic


def fourier_in_lnell(ell, residuals, weights=None):
    """
    Fourier transform of residuals as a function of ln(ℓ).

    This is equivalent to a Mellin transform — it detects
    scale-invariant (geometric/multiplicative) periodicities.

    Uses Lomb-Scargle periodogram because the data may not be
    uniformly spaced in ln(ℓ).

    Returns (frequencies, power, significance_levels).
    """
    ln_ell = np.log(ell)

    # Subtract mean
    if weights is not None:
        mean_resid = np.average(residuals, weights=weights)
    else:
        mean_resid = np.mean(residuals)
    y = residuals - mean_resid

    # Define frequency grid to search
    # The golden ratio frequency is k_φ = 1/ln(φ) ≈ 2.078
    # Search from 0.1 to 20 cycles per unit ln(ℓ)
    n_freqs = 10000
    freq_min = 0.1
    freq_max = 20.0
    freqs = np.linspace(freq_min, freq_max, n_freqs)

    # Angular frequencies for scipy's lombscargle
    angular_freqs = 2 * np.pi * freqs

    # Lomb-Scargle periodogram
    power = lombscargle(ln_ell, y, angular_freqs, normalize=True)

    # Significance levels via bootstrap
    n_bootstrap = 10000
    power_max_null = np.zeros(n_bootstrap)
    rng = np.random.default_rng(42)

    print(f"  Running {n_bootstrap} bootstrap iterations for significance...")
    for i in range(n_bootstrap):
        y_shuffled = rng.permutation(y)
        power_null = lombscargle(ln_ell, y_shuffled, angular_freqs, normalize=True)
        power_max_null[i] = power_null.max()

    # Significance thresholds
    sig_levels = {
        '1sigma': np.percentile(power_max_null, 68.27),
        '2sigma': np.percentile(power_max_null, 95.45),
        '3sigma': np.percentile(power_max_null, 99.73),
        '4sigma': np.percentile(power_max_null, 99.9937),
        '5sigma': np.percentile(power_max_null, 99.99994),
    }

    return freqs, power, sig_levels, power_max_null


def find_peak_near_phi(freqs, power, tolerance=0.3):
    """
    Find the highest peak within ±tolerance of k_φ = 1/ln(φ).
    """
    k_phi = 1.0 / np.log(PHI)
    mask = np.abs(freqs - k_phi) < tolerance
    if not np.any(mask):
        return None, None, None

    idx_local = np.argmax(power[mask])
    idx_global = np.where(mask)[0][idx_local]

    return freqs[idx_global], power[idx_global], idx_global


# ══════════════════════════════════════════════════════════════
# 3. MAIN EXECUTION
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  CMB φ-PERIODICITY SEARCH")
    print("  Looking for golden-ratio oscillations in Planck 2018 residuals")
    print("=" * 70)

    # --- Load data ---
    print("\n[1/5] Loading Planck 2018 TT power spectrum...")
    ell_data, Dl_data, Dl_err, data_source = load_planck_tt()

    # --- Generate theory ---
    print("\n[2/5] Generating ΛCDM theory spectrum...")
    ell_theory, Dl_theory, theory_source = generate_lcdm_theory(
        ell_max=int(ell_data.max()) + 100
    )

    # --- Compute residuals ---
    print("\n[3/5] Computing residuals...")
    ell, residuals, resid_norm, sigma_cosmic = compute_residuals(
        ell_data, Dl_data, ell_theory, Dl_theory
    )
    print(f"  {len(ell)} multipoles in analysis range")
    print(f"  ℓ range: {ell.min():.0f} to {ell.max():.0f}")
    print(f"  Mean residual: {np.mean(residuals):.2f} μK²")
    print(f"  RMS residual: {np.std(residuals):.2f} μK²")

    # Use data errors if available, otherwise cosmic variance
    if Dl_err is not None and np.all(Dl_err > 0):
        # Need to match the mask used in compute_residuals
        ell_min = max(ell_data.min(), ell_theory.min())
        ell_max_val = min(ell_data.max(), ell_theory.max())
        err_mask = (ell_data >= ell_min) & (ell_data <= ell_max_val)
        analysis_err = Dl_err[err_mask]
        if len(analysis_err) == len(ell):
            weights = 1.0 / analysis_err**2
            resid_weighted = residuals / analysis_err
            print(f"  Using data errors for weighting")
        else:
            weights = 1.0 / sigma_cosmic**2
            resid_weighted = resid_norm
            print(f"  Using cosmic variance for weighting")
    else:
        weights = 1.0 / sigma_cosmic**2
        resid_weighted = resid_norm
        print(f"  Using cosmic variance for weighting")

    # --- Fourier analysis ---
    print("\n[4/5] Performing Lomb-Scargle periodogram in ln(ℓ) space...")
    freqs, power, sig_levels, null_dist = fourier_in_lnell(
        ell, resid_weighted, weights=weights
    )

    # --- Find φ signal ---
    k_phi = 1.0 / np.log(PHI)
    phi_freq, phi_power, phi_idx = find_peak_near_phi(freqs, power)

    # Find global maximum
    global_max_idx = np.argmax(power)
    global_max_freq = freqs[global_max_idx]
    global_max_power = power[global_max_idx]

    # Determine significance of φ peak
    if phi_power is not None:
        p_value = np.mean(null_dist >= phi_power)
        if phi_power > sig_levels['5sigma']:
            phi_significance = '>5σ'
        elif phi_power > sig_levels['4sigma']:
            phi_significance = '>4σ'
        elif phi_power > sig_levels['3sigma']:
            phi_significance = '>3σ'
        elif phi_power > sig_levels['2sigma']:
            phi_significance = '>2σ'
        elif phi_power > sig_levels['1sigma']:
            phi_significance = '>1σ'
        else:
            phi_significance = '<1σ'
    else:
        p_value = 1.0
        phi_significance = 'not found'

    # --- Report ---
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"\n  Target frequency:     k_φ = 1/ln(φ) = {k_phi:.4f}")
    print(f"  Nearest peak:         k = {phi_freq:.4f}" if phi_freq else "  No peak found near k_φ")
    print(f"  Peak power:           {phi_power:.4f}" if phi_power else "")
    print(f"  Significance:         {phi_significance}")
    print(f"  p-value (bootstrap):  {p_value:.6f}")
    print(f"\n  Global maximum:       k = {global_max_freq:.4f} (power = {global_max_power:.4f})")
    print(f"\n  Significance thresholds:")
    for level, val in sig_levels.items():
        marker = " ← φ peak" if (phi_power and phi_power > val and 
                                   (level == '1sigma' or phi_power <= list(sig_levels.values())[
                                       list(sig_levels.keys()).index(level) - 1] 
                                    if list(sig_levels.keys()).index(level) > 0 else True)) else ""
        print(f"    {level}: {val:.4f}{marker}")

    # --- Save results ---
    results = {
        'target_frequency_k_phi': k_phi,
        'phi_peak_frequency': float(phi_freq) if phi_freq else None,
        'phi_peak_power': float(phi_power) if phi_power else None,
        'phi_significance': phi_significance,
        'phi_p_value': float(p_value),
        'global_max_frequency': float(global_max_freq),
        'global_max_power': float(global_max_power),
        'significance_thresholds': {k: float(v) for k, v in sig_levels.items()},
        'n_multipoles': int(len(ell)),
        'ell_range': [float(ell.min()), float(ell.max())],
        'data_source': data_source,
        'theory_source': theory_source,
        'n_bootstrap': len(null_dist),
        'lcdm_params': {k: float(v) for k, v in LCDM_PARAMS.items()},
    }

    results_file = os.path.join(RESULTS_DIR, 'cmb_phi_search_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_file}")

    # ══════════════════════════════════════════════════════════
    # 5. PLOTTING
    # ══════════════════════════════════════════════════════════

    print("\n[5/5] Generating plots...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 14))

    # --- Panel 1: Residuals in ℓ-space ---
    ax1 = axes[0]
    ax1.fill_between(ell, -sigma_cosmic, sigma_cosmic, color='blue', alpha=0.08,
                     label='Cosmic variance (1σ)')
    ax1.scatter(ell, residuals, s=1, color='gray', alpha=0.5, label='Residuals')

    # Overlay the best-fit φ-oscillation if detected
    if phi_power and phi_power > sig_levels['1sigma']:
        # Reconstruct the sinusoidal fit at the detected frequency
        ln_ell = np.log(ell)
        # Estimate amplitude from Lomb-Scargle power
        amp = np.sqrt(2 * phi_power * np.var(resid_weighted)) * np.mean(sigma_cosmic)
        # Find best phase by fitting
        phases = np.linspace(0, 2*np.pi, 100)
        best_phase = 0
        best_corr = -np.inf
        for phase in phases:
            model = amp * np.cos(2*np.pi * phi_freq * ln_ell + phase)
            corr = np.sum(residuals * model)
            if corr > best_corr:
                best_corr = corr
                best_phase = phase

        oscillation = amp * np.cos(2*np.pi * phi_freq * np.log(ell) + best_phase)
        ax1.plot(ell, oscillation, 'r-', lw=1.5, alpha=0.8,
                 label=f'φ-oscillation (k={phi_freq:.3f})')

    ax1.set_xlabel('Multipole ℓ', fontsize=12)
    ax1.set_ylabel('ΔDℓ (μK²)', fontsize=12)
    ax1.set_title('Planck 2018 TT Residuals (Data − Best-fit ΛCDM)', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2)

    # --- Panel 2: Residuals in ln(ℓ)-space ---
    ax2 = axes[1]
    ln_ell = np.log(ell)
    ax2.scatter(ln_ell, resid_weighted, s=2, color='steelblue', alpha=0.5)
    ax2.axhline(0, color='black', ls='-', lw=0.5)

    # Mark φ-period intervals
    for i in range(int(ln_ell.min() / np.log(PHI)), 
                    int(ln_ell.max() / np.log(PHI)) + 1):
        ax2.axvline(i * np.log(PHI), color='gold', ls='--', alpha=0.3, lw=1)

    ax2.set_xlabel('ln(ℓ)', fontsize=12)
    ax2.set_ylabel('Normalized residual', fontsize=12)
    ax2.set_title('Residuals in ln(ℓ) space — golden lines mark ln(φ) intervals',
                  fontsize=13)
    ax2.grid(True, alpha=0.2)

    # --- Panel 3: Lomb-Scargle periodogram ---
    ax3 = axes[2]
    ax3.plot(freqs, power, 'k-', lw=0.8, alpha=0.7)

    # Significance bands
    colors_sig = {'1sigma': 'green', '2sigma': 'orange', 
                  '3sigma': 'red', '4sigma': 'darkred', '5sigma': 'purple'}
    for level, val in sig_levels.items():
        ax3.axhline(val, color=colors_sig.get(level, 'gray'), ls='--', lw=1,
                     alpha=0.7, label=f'{level}: {val:.3f}')

    # Mark the golden ratio frequency
    ax3.axvline(k_phi, color='gold', ls='-', lw=3, alpha=0.8,
                label=f'k_φ = 1/ln(φ) = {k_phi:.3f}')

    # Mark any peak near φ
    if phi_power:
        ax3.plot(phi_freq, phi_power, 'r*', markersize=15, zorder=5,
                 label=f'Peak near k_φ: {phi_significance}')

    # Mark global maximum
    ax3.plot(global_max_freq, global_max_power, 'b^', markersize=10, zorder=5,
             label=f'Global max: k={global_max_freq:.3f}')

    ax3.set_xlabel('Frequency k (cycles per unit ln ℓ)', fontsize=12)
    ax3.set_ylabel('Lomb-Scargle Power', fontsize=12)
    ax3.set_title('Periodogram of CMB Residuals — Searching for φ-periodicity',
                  fontsize=13)
    ax3.legend(fontsize=9, loc='upper right')
    ax3.set_xlim(0, 15)
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'cmb_phi_periodogram.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved to {fig_path}")
    # Only show interactively when a display is available and not piped
    import sys
    if sys.stdout.isatty() and os.environ.get('DISPLAY') or os.environ.get('TERM_PROGRAM'):
        try:
            plt.show(block=False)
            plt.pause(0.5)
            plt.close('all')
        except Exception:
            plt.close('all')

    # ══════════════════════════════════════════════════════════
    # 6. VERDICT
    # ══════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)

    if phi_power and phi_power > sig_levels['3sigma']:
        print("\n  ██████████████████████████████████████████████████")
        print("  ██  φ-PERIODIC SIGNAL DETECTED AT >3σ          ██")
        print("  ██  This is evidence for golden-ratio           ██")
        print("  ██  structure in the CMB power spectrum.         ██")
        print("  ██████████████████████████████████████████████████")
        print(f"\n  Frequency: {phi_freq:.4f} (expected: {k_phi:.4f})")
        print(f"  Deviation from k_φ: {abs(phi_freq - k_phi)/k_phi*100:.2f}%")
        print(f"\n  NEXT STEP: Verify with TE and EE spectra.")

    elif phi_power and phi_power > sig_levels['2sigma']:
        print("\n  ▶ TENTATIVE HINT (2-3σ) of φ-periodicity.")
        print(f"    Peak at k = {phi_freq:.4f}, expected k_φ = {k_phi:.4f}")
        print(f"    This is suggestive but not conclusive.")
        print(f"\n  NEXT STEP: Cross-check with TE/EE spectra and BAO data.")

    elif phi_power and phi_power > sig_levels['1sigma']:
        print("\n  ▶ WEAK EXCESS (1-2σ) near φ frequency.")
        print(f"    Consistent with noise fluctuation.")
        print(f"    Upper limit on modulation amplitude can be derived.")
        print(f"\n  NEXT STEP: Proceed to BAO analysis for independent test.")

    else:
        print("\n  ▶ NO SIGNIFICANT SIGNAL at the golden-ratio frequency.")
        print(f"    The CMB residuals do not show φ-periodicity above noise.")
        print(f"    This constrains the lattice imprint to be very small.")
        print(f"\n  This does NOT rule out the framework — it means:")
        print(f"    1. The lattice signal may be below Planck sensitivity")
        print(f"    2. The signal may appear in polarization (TE/EE) not TT")
        print(f"    3. The φ-structure may affect perturbations differently")

    return results


if __name__ == '__main__':
    results = main()
