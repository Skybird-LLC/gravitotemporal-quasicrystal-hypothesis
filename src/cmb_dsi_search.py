"""
CMB Discrete Scale Invariance (DSI) Search
============================================
Searches for log-periodic (golden-ratio) oscillations in Planck 2018
TT, TE, and EE power spectrum residuals.

Tests the prediction:
    f(φ) ≈ φ^{-n} · cos(2π ln(r) / ln(φ))

which should produce peaks at k_m = m/ln(φ) for m = 1, 2, 3, ...
in the Lomb-Scargle periodogram of residuals vs ln(ℓ).

Downloads all required data automatically.
"""

import numpy as np
from scipy.signal import lombscargle
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import urllib.request
import os
import json
from pathlib import Path

# ══════════════════════════════════════════════════════════════
# 0. CONSTANTS AND PATHS
# ══════════════════════════════════════════════════════════════

PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)
K_PHI = 1.0 / LN_PHI              # fundamental: ≈ 2.078
K_PHI_HARMONICS = [K_PHI * m for m in range(1, 5)]  # m=1,2,3,4

BASE_DIR = Path(__file__).resolve().parent.parent


def resolve_cmb_data_dir() -> Path:
    data_planck = BASE_DIR / 'data' / 'planck'
    data_cmb = BASE_DIR / 'data' / 'cmb'
    marker = 'COM_PowerSpect_CMB-TT-full_R3.01.txt'

    if (data_planck / marker).exists():
        return data_planck
    if (data_cmb / marker).exists():
        return data_cmb
    if data_planck.exists():
        return data_planck
    return data_cmb


DATA_DIR = resolve_cmb_data_dir()
RESULTS_DIR = BASE_DIR / 'results'
FIG_DIR = RESULTS_DIR / 'figures'

for d in [DATA_DIR, RESULTS_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Planck 2018 best-fit ΛCDM (TT,TE,EE+lowE+lensing, Table 2 of 1807.06209)
LCDM_PARAMS = {
    'H0': 67.36, 'ombh2': 0.02237, 'omch2': 0.1200,
    'tau': 0.0544, 'ns': 0.9649, 'ln10As': 3.044,
}

# ══════════════════════════════════════════════════════════════
# 1. DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════

# Planck Legacy Archive URLs
PLA_BASE = "https://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID="

DATA_PRODUCTS = {
    'TT_full': {
        'filename': 'COM_PowerSpect_CMB-TT-full_R3.01.txt',
        'url': PLA_BASE + 'COM_PowerSpect_CMB-TT-full_R3.01.txt',
        'description': 'Planck 2018 TT full (unbinned) power spectrum',
    },
    'TT_binned': {
        'filename': 'COM_PowerSpect_CMB-TT-binned_R3.01.txt',
        'url': PLA_BASE + 'COM_PowerSpect_CMB-TT-binned_R3.01.txt',
        'description': 'Planck 2018 TT binned power spectrum',
    },
    'TE_full': {
        'filename': 'COM_PowerSpect_CMB-TE-full_R3.01.txt',
        'url': PLA_BASE + 'COM_PowerSpect_CMB-TE-full_R3.01.txt',
        'description': 'Planck 2018 TE full power spectrum',
    },
    'TE_binned': {
        'filename': 'COM_PowerSpect_CMB-TE-binned_R3.01.txt',
        'url': PLA_BASE + 'COM_PowerSpect_CMB-TE-binned_R3.01.txt',
        'description': 'Planck 2018 TE binned power spectrum',
    },
    'EE_full': {
        'filename': 'COM_PowerSpect_CMB-EE-full_R3.01.txt',
        'url': PLA_BASE + 'COM_PowerSpect_CMB-EE-full_R3.01.txt',
        'description': 'Planck 2018 EE full power spectrum',
    },
    'EE_binned': {
        'filename': 'COM_PowerSpect_CMB-EE-binned_R3.01.txt',
        'url': PLA_BASE + 'COM_PowerSpect_CMB-EE-binned_R3.01.txt',
        'description': 'Planck 2018 EE binned power spectrum',
    },
    'theory': {
        'filename': 'planck_bestfit_theory_cl.dat',
        'url': PLA_BASE + 'COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum_R3.01.txt',
        'description': 'Planck best-fit ΛCDM theory spectrum',
    },
}


def download_file(url, filepath, description=""):
    """Download a file if not already cached."""
    filepath = Path(filepath)
    if filepath.exists():
        print(f"    ✓ {filepath.name} (cached)")
        return True
    print(f"    ↓ Downloading {description or filepath.name}...")
    try:
        urllib.request.urlretrieve(url, str(filepath))
        size_kb = filepath.stat().st_size / 1024
        print(f"      Done ({size_kb:.1f} KB)")
        return True
    except Exception as e:
        print(f"      FAILED: {e}")
        return False


def download_all():
    """Download all required Planck data products."""
    print("\n[1/5] Downloading Planck 2018 data products...")
    results = {}
    for key, info in DATA_PRODUCTS.items():
        path = DATA_DIR / info['filename']
        success = download_file(info['url'], path, info['description'])
        results[key] = success
    return results


# ══════════════════════════════════════════════════════════════
# 2. DATA LOADING
# ══════════════════════════════════════════════════════════════

def load_spectrum(spectrum_type='TT', prefer_full=True):
    """
    Load a Planck 2018 CMB power spectrum.

    Parameters
    ----------
    spectrum_type : str
        'TT', 'TE', or 'EE'
    prefer_full : bool
        If True, try unbinned first; if False, try binned first.

    Returns
    -------
    ell, Dl, Dl_err, source_label
    """
    order = ['full', 'binned'] if prefer_full else ['binned', 'full']

    for variant in order:
        key = f'{spectrum_type}_{variant}'
        if key not in DATA_PRODUCTS:
            continue
        filepath = DATA_DIR / DATA_PRODUCTS[key]['filename']
        if not filepath.exists():
            continue
        try:
            data = np.loadtxt(str(filepath))
            ell = data[:, 0]
            Dl = data[:, 1]

            # Handle asymmetric errors
            if data.shape[1] >= 4:
                err_minus = np.abs(data[:, 2])
                err_plus = np.abs(data[:, 3])
                Dl_err = 0.5 * (err_minus + err_plus)
            elif data.shape[1] >= 3:
                Dl_err = np.abs(data[:, 2])
            else:
                # Cosmic variance estimate
                Dl_err = np.abs(Dl) * np.sqrt(2.0 / (2.0 * ell + 1.0))

            # Sanity check
            print(f"    Loaded {spectrum_type} ({variant}): {len(ell)} points, "
                  f"ℓ = {ell.min():.0f}–{ell.max():.0f}")
            print(f"    Dℓ range: {Dl.min():.1f} to {Dl.max():.1f} μK²")

            return ell, Dl, Dl_err, f'planck_{variant}'

        except Exception as e:
            print(f"    Could not parse {filepath.name}: {e}")
            continue

    raise FileNotFoundError(
        f"No {spectrum_type} data found. Run download_all() first."
    )


def load_theory():
    """
    Load the Planck best-fit ΛCDM theory spectrum.
    Falls back to CAMB if the file isn't available.

    Returns dict with keys 'TT', 'TE', 'EE', each containing (ell, Dl).
    """
    theory_file = DATA_DIR / DATA_PRODUCTS['theory']['filename']

    if theory_file.exists():
        try:
            data = np.loadtxt(str(theory_file))
            # Planck theory files typically have columns:
            # ell, TT, TE, EE, BB, PP (or subset)
            ell = data[:, 0]
            result = {'ell': ell}

            if data.shape[1] >= 2:
                result['TT'] = data[:, 1]
            if data.shape[1] >= 3:
                result['TE'] = data[:, 2]
            if data.shape[1] >= 4:
                result['EE'] = data[:, 3]

            print(f"    Loaded theory: {len(ell)} multipoles, "
                  f"ℓ = {ell.min():.0f}–{ell.max():.0f}")
            print(f"    Columns available: {[k for k in result if k != 'ell']}")
            return result, 'planck_official'

        except Exception as e:
            print(f"    Could not parse theory file: {e}")

    # Fallback: CAMB
    try:
        import camb
        print("    Generating theory spectrum with CAMB...")
        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=LCDM_PARAMS['H0'],
            ombh2=LCDM_PARAMS['ombh2'],
            omch2=LCDM_PARAMS['omch2'],
            tau=LCDM_PARAMS['tau']
        )
        pars.InitPower.set_params(
            As=np.exp(LCDM_PARAMS['ln10As']) * 1e-10,
            ns=LCDM_PARAMS['ns']
        )
        pars.set_for_lmax(3000)
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK2')
        totCL = powers['total']
        ell = np.arange(totCL.shape[0])

        result = {
            'ell': ell[2:],
            'TT': totCL[2:, 0],
            'TE': totCL[2:, 3],
            'EE': totCL[2:, 1],
        }
        print(f"    CAMB theory: {len(result['ell'])} multipoles")
        return result, 'CAMB'

    except ImportError:
        raise RuntimeError(
            "No theory spectrum available.\n"
            "Either download from Planck Legacy Archive or install CAMB:\n"
            "  pip install camb"
        )


# ══════════════════════════════════════════════════════════════
# 3. RESIDUAL COMPUTATION
# ══════════════════════════════════════════════════════════════

def compute_residuals(ell_data, Dl_data, Dl_err, ell_theory, Dl_theory):
    """
    Compute residuals and verify unit consistency.

    Returns (ell, residuals, errors, diagnostics_dict).
    """
    # Interpolate theory onto data multipoles
    ell_min = max(ell_data.min(), ell_theory.min())
    ell_max = min(ell_data.max(), ell_theory.max())
    mask = (ell_data >= ell_min) & (ell_data <= ell_max)

    ell = ell_data[mask]
    Dl = Dl_data[mask]
    err = Dl_err[mask]

    theory_interp = CubicSpline(ell_theory, Dl_theory)
    Dl_th = theory_interp(ell)

    residuals = Dl - Dl_th

    # Diagnostic: check unit consistency
    signal_max = np.max(np.abs(Dl_th))
    resid_rms = np.std(residuals)
    ratio = resid_rms / signal_max if signal_max > 0 else np.inf

    diagnostics = {
        'n_points': len(ell),
        'ell_range': (float(ell.min()), float(ell.max())),
        'theory_Dl_range': (float(Dl_th.min()), float(Dl_th.max())),
        'data_Dl_range': (float(Dl.min()), float(Dl.max())),
        'residual_rms': float(resid_rms),
        'signal_max': float(signal_max),
        'resid_to_signal_ratio': float(ratio),
    }

    # Warning if residuals are suspiciously large
    if ratio > 0.5:
        print(f"    ⚠ WARNING: RMS(residual)/max(signal) = {ratio:.2f}")
        print(f"      This suggests a unit or convention mismatch!")
        print(f"      Theory range: {Dl_th.min():.1f} to {Dl_th.max():.1f}")
        print(f"      Data range:   {Dl.min():.1f} to {Dl.max():.1f}")
    else:
        print(f"    Residual/Signal ratio: {ratio:.4f} (OK)")

    return ell, residuals, err, diagnostics


# ══════════════════════════════════════════════════════════════
# 4. DSI SEARCH (LOMB-SCARGLE + HARMONICS)
# ══════════════════════════════════════════════════════════════

def dsi_search(ell, residuals, errors=None, n_bootstrap=10000, seed=42):
    """
    Search for Discrete Scale Invariance in CMB residuals.

    Performs Lomb-Scargle periodogram in ln(ℓ) space and checks
    for peaks at k_φ = m/ln(φ) for m = 1, 2, 3, 4.

    Returns dict with frequencies, power, significance, and harmonic analysis.
    """
    ln_ell = np.log(ell)

    # Weight by inverse errors if available
    if errors is not None and np.all(errors > 0):
        y = residuals / errors
    else:
        y = residuals / np.std(residuals)

    y = y - np.mean(y)

    # Frequency grid
    n_freqs = 10000
    freq_min = 0.05
    freq_max = min(len(ell) / (2 * (ln_ell.max() - ln_ell.min())), 25.0)
    freqs = np.linspace(freq_min, freq_max, n_freqs)
    angular_freqs = 2 * np.pi * freqs

    # Periodogram
    power = lombscargle(ln_ell, y, angular_freqs, normalize=True)

    # Bootstrap significance
    rng = np.random.default_rng(seed)
    null_max = np.zeros(n_bootstrap)
    print(f"    Bootstrap ({n_bootstrap} iterations)...", end='', flush=True)
    for i in range(n_bootstrap):
        y_shuf = rng.permutation(y)
        p_null = lombscargle(ln_ell, y_shuf, angular_freqs, normalize=True)
        null_max[i] = p_null.max()
        if (i + 1) % 2500 == 0:
            print(f" {i+1}", end='', flush=True)
    print(" done.")

    sig_levels = {}
    for name, pct in [('1sigma', 68.27), ('2sigma', 95.45), 
                       ('3sigma', 99.73), ('4sigma', 99.9937),
                       ('5sigma', 99.99994)]:
        sig_levels[name] = float(np.percentile(null_max, pct))

    # Analyze each harmonic
    harmonic_results = []
    tolerance = 0.3  # search window ± this value around each harmonic

    for m in range(1, 5):
        k_target = m * K_PHI
        mask = np.abs(freqs - k_target) < tolerance
        if not np.any(mask):
            harmonic_results.append({
                'm': m, 'k_target': k_target,
                'k_found': None, 'power': None, 'sigma': '<1σ'
            })
            continue

        idx_local = np.argmax(power[mask])
        idx_global = np.where(mask)[0][idx_local]
        k_found = freqs[idx_global]
        p_found = power[idx_global]

        # Determine significance
        p_value = float(np.mean(null_max >= p_found))
        if p_found > sig_levels.get('5sigma', np.inf):
            sigma = '>5σ'
        elif p_found > sig_levels.get('4sigma', np.inf):
            sigma = '>4σ'
        elif p_found > sig_levels.get('3sigma', np.inf):
            sigma = '>3σ'
        elif p_found > sig_levels.get('2sigma', np.inf):
            sigma = '>2σ'
        elif p_found > sig_levels.get('1sigma', np.inf):
            sigma = '>1σ'
        else:
            sigma = '<1σ'

        harmonic_results.append({
            'm': m,
            'k_target': float(k_target),
            'k_found': float(k_found),
            'power': float(p_found),
            'p_value': p_value,
            'sigma': sigma,
            'deviation_pct': float(abs(k_found - k_target) / k_target * 100),
        })

    # Global maximum
    idx_max = np.argmax(power)

    return {
        'freqs': freqs,
        'power': power,
        'sig_levels': sig_levels,
        'null_distribution': null_max,
        'harmonics': harmonic_results,
        'global_max_freq': float(freqs[idx_max]),
        'global_max_power': float(power[idx_max]),
        'ln_ell_range': float(ln_ell.max() - ln_ell.min()),
        'n_points': len(ell),
        'nyquist_freq': float(freq_max),
    }


# ══════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  CMB DISCRETE SCALE INVARIANCE (DSI) SEARCH")
    print("  f(φ) ≈ φ^{-n} · cos(2π ln(r) / ln(φ))")
    print("  Target harmonics: k_m = m/ln(φ) for m = 1,2,3,4")
    print("=" * 70)

    # Download
    download_status = download_all()

    # Load theory
    print("\n[2/5] Loading ΛCDM theory spectrum...")
    theory, theory_source = load_theory()

    # Analyze each spectrum type
    all_results = {}
    spectra_to_analyze = []

    for stype in ['TT', 'TE', 'EE']:
        if stype in theory:
            spectra_to_analyze.append(stype)
        else:
            print(f"\n  Skipping {stype} — no theory curve available")

    for stype in spectra_to_analyze:
        print(f"\n{'─' * 70}")
        print(f"  Analyzing {stype} spectrum")
        print(f"{'─' * 70}")

        # Load data
        print(f"\n[3/5] Loading Planck {stype} data...")
        try:
            ell_d, Dl_d, Dl_err, data_src = load_spectrum(stype, prefer_full=True)
        except FileNotFoundError as e:
            print(f"  {e}")
            continue

        # Compute residuals
        print(f"\n[4/5] Computing {stype} residuals...")
        ell, resid, err, diag = compute_residuals(
            ell_d, Dl_d, Dl_err,
            theory['ell'], theory[stype]
        )

        # DSI search
        print(f"\n[5/5] Running DSI search on {stype}...")
        dsi = dsi_search(ell, resid, err, n_bootstrap=10000)

        # Report
        print(f"\n  {stype} RESULTS:")
        print(f"  {'─' * 50}")
        print(f"  Data points:     {dsi['n_points']}")
        print(f"  ln(ℓ) range:     {dsi['ln_ell_range']:.2f}")
        print(f"  Nyquist freq:    {dsi['nyquist_freq']:.1f}")
        print(f"  Global max:      k = {dsi['global_max_freq']:.3f} "
              f"(power = {dsi['global_max_power']:.4f})")

        print(f"\n  DSI Harmonics:")
        print(f"  {'m':>3} {'k_target':>10} {'k_found':>10} "
              f"{'power':>8} {'σ-level':>10} {'Δk/k':>8}")
        print(f"  {'─' * 55}")
        for h in dsi['harmonics']:
            if h['k_found'] is not None:
                print(f"  {h['m']:>3} {h['k_target']:>10.3f} "
                      f"{h['k_found']:>10.3f} {h['power']:>8.4f} "
                      f"{h['sigma']:>10} {h['deviation_pct']:>7.1f}%")
            else:
                print(f"  {h['m']:>3} {h['k_target']:>10.3f} "
                      f"{'—':>10} {'—':>8} {'—':>10} {'—':>8}")

        # Store
        all_results[stype] = {
            'data_source': data_src,
            'diagnostics': diag,
            'dsi': {k: v for k, v in dsi.items() 
                    if k not in ('freqs', 'power', 'null_distribution')},
            'dsi_harmonics': dsi['harmonics'],
        }

        # Keep arrays for plotting
        all_results[f'{stype}_arrays'] = {
            'ell': ell, 'resid': resid,
            'freqs': dsi['freqs'], 'power': dsi['power'],
            'sig_levels': dsi['sig_levels'],
        }

    # ══════════════════════════════════════════════════════════
    # CROSS-SPECTRUM CONSISTENCY CHECK
    # ══════════════════════════════════════════════════════════

    print(f"\n{'=' * 70}")
    print("  CROSS-SPECTRUM CONSISTENCY")
    print(f"{'=' * 70}")

    detected_in = []
    for stype in spectra_to_analyze:
        if stype not in all_results:
            continue
        fundamental = all_results[stype]['dsi_harmonics'][0]  # m=1
        if fundamental['sigma'] in ['>2σ', '>3σ', '>4σ', '>5σ']:
            detected_in.append(stype)

    if len(detected_in) == 0:
        print("\n  No significant φ-periodic signal in any spectrum.")
        print("  The DSI hypothesis is not supported by Planck data")
        print("  at the current sensitivity level.")
    elif len(detected_in) == 1:
        print(f"\n  Signal detected in {detected_in[0]} only.")
        print("  ⚠ Single-spectrum detection — likely systematic or noise.")
    elif len(detected_in) >= 2:
        print(f"\n  ██ Signal detected in: {', '.join(detected_in)} ██")
        print("  Multi-spectrum detection strengthens the case for DSI.")

    # ══════════════════════════════════════════════════════════
    # PLOTTING
    # ══════════════════════════════════════════════════════════

    n_spectra = len([s for s in spectra_to_analyze if s in all_results])
    if n_spectra == 0:
        print("\n  No spectra to plot.")
        return all_results

    fig, axes = plt.subplots(n_spectra, 2, figsize=(16, 5 * n_spectra))
    if n_spectra == 1:
        axes = axes.reshape(1, -1)

    colors = {'TT': 'steelblue', 'TE': 'seagreen', 'EE': 'coral'}

    for i, stype in enumerate(spectra_to_analyze):
        if f'{stype}_arrays' not in all_results:
            continue

        arr = all_results[f'{stype}_arrays']

        # Left panel: residuals in ln(ℓ) space
        ax_left = axes[i, 0]
        ln_ell = np.log(arr['ell'])
        ax_left.scatter(ln_ell, arr['resid'], s=1, color=colors.get(stype, 'gray'),
                        alpha=0.3)
        ax_left.axhline(0, color='black', lw=0.5)

        # Mark φ intervals
        for j in range(20):
            x = j * LN_PHI
            if ln_ell.min() <= x <= ln_ell.max():
                ax_left.axvline(x, color='gold', ls='--', alpha=0.3)

        ax_left.set_xlabel('ln(ℓ)', fontsize=11)
        ax_left.set_ylabel(f'{stype} Residual (μK²)', fontsize=11)
        ax_left.set_title(f'{stype} Residuals in ln(ℓ) space', fontsize=12)
        ax_left.grid(True, alpha=0.15)

        # Right panel: periodogram
        ax_right = axes[i, 1]
        ax_right.plot(arr['freqs'], arr['power'], '-', color='gray',
                      lw=0.6, alpha=0.7)

        # Significance levels
        sig_colors = {'1sigma': 'green', '2sigma': 'orange', '3sigma': 'red'}
        for level in ['1sigma', '2sigma', '3sigma']:
            val = arr['sig_levels'][level]
            ax_right.axhline(val, color=sig_colors[level], ls='--', lw=1,
                             alpha=0.7, label=f'{level}: {val:.3f}')

        # Mark DSI harmonics
        harmonic_colors = ['gold', 'darkorange', 'tomato', 'darkred']
        for m in range(1, 5):
            k_m = m * K_PHI
            ax_right.axvline(k_m, color=harmonic_colors[m-1], ls='-',
                             lw=2 if m == 1 else 1.2, alpha=0.7,
                             label=f'm={m}: k={k_m:.2f}')

        # Mark detected peaks
        harmonics = all_results[stype]['dsi_harmonics']
        for h in harmonics:
            if h['k_found'] is not None:
                marker = '★' if h['sigma'] not in ['<1σ'] else '○'
                color = 'red' if h['sigma'] not in ['<1σ'] else 'gray'
                ax_right.plot(h['k_found'], h['power'], '*',
                              color=color, markersize=12, zorder=5)

        ax_right.set_xlabel('Frequency k (cycles per unit ln ℓ)', fontsize=11)
        ax_right.set_ylabel('Lomb-Scargle Power', fontsize=11)
        ax_right.set_title(f'{stype} Periodogram — DSI harmonic search',
                           fontsize=12)
        ax_right.legend(fontsize=8, loc='upper right')
        ax_right.set_xlim(0, 15)
        ax_right.grid(True, alpha=0.15)

    plt.tight_layout()
    fig_path = FIG_DIR / 'cmb_dsi_periodogram.png'
    plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved to {fig_path}")
    plt.show()

    # Save results
    serializable = {k: v for k, v in all_results.items() 
                    if not k.endswith('_arrays')}
    serializable['theory_source'] = theory_source
    serializable['phi'] = float(PHI)
    serializable['k_phi'] = float(K_PHI)
    serializable['harmonics_tested'] = [float(k) for k in K_PHI_HARMONICS]
    serializable['detected_in_spectra'] = detected_in

    results_file = RESULTS_DIR / 'cmb_dsi_search_results.json'
    with open(str(results_file), 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"  Results saved to {results_file}")

    return all_results


if __name__ == '__main__':
    results = main()
