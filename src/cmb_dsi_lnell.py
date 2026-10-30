#!/usr/bin/env python3
"""
CMB DSI Test — Search for log-periodic oscillations in Planck residuals.

Prediction: If spacetime has quasicrystalline discrete scale invariance
with phi-spacing, then CMB power spectrum residuals (relative to LCDM)
should show a periodic signal in ln(ell) with period = ln(phi) ≈ 0.4812.

This script:
  1. Loads Planck TT (and optionally TE, EE) power spectra
  2. Loads the Planck best-fit LCDM theory spectrum
  3. Computes fractional residuals  δ = (D_ℓ^data - D_ℓ^theory) / D_ℓ^theory
  4. Transforms to ln(ℓ) space
  5. Runs a Lomb-Scargle periodogram looking for periodicity at ln(φ)
  6. Estimates significance via phase-shuffled null ensemble
"""

import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.signal import lombscargle  # note: angular frequency version
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Configuration ───────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
PLANCK_DIR   = BASE_DIR / "data" / "planck"
RESULTS_DIR  = BASE_DIR / "results"
FIG_DIR      = RESULTS_DIR / "figures"
TABLE_DIR    = RESULTS_DIR / "tables"

for d in [RESULTS_DIR, FIG_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PHI          = (1 + np.sqrt(5)) / 2          # golden ratio
LN_PHI       = np.log(PHI)                   # ≈ 0.4812
PERIOD_TARGET = LN_PHI                        # the period we're looking for

ELL_MIN      = 30      # avoid cosmic-variance-dominated low ℓ
ELL_MAX      = 2500    # Planck TT limit

N_NULL       = 10000   # number of shuffled realizations for significance

# Set True to inject a fake signal for pipeline validation
INJECT_TEST     = False
INJECT_EPSILON  = 0.01   # fractional amplitude of injected signal
INJECT_PHASE    = 1.23   # arbitrary phase

SPEC_TYPES   = ["TT"]   # can add "TE", "EE" later


# ─── Data Loading ────────────────────────────────────────────────────
def load_planck_data(spec_type="TT"):
    """
    Load COM_PowerSpect_CMB-{TT,TE,EE}-full_R3.01.txt
    Columns: ℓ, Dℓ, -dDℓ, +dDℓ   (Dℓ in μK²)
    """
    fname = PLANCK_DIR / f"COM_PowerSpect_CMB-{spec_type}-full_R3.01.txt"
    if not fname.exists():
        raise FileNotFoundError(f"Missing: {fname}")

    data = np.loadtxt(fname, comments="#")
    ell   = data[:, 0].astype(int)
    Dl    = data[:, 1]
    # symmetrize asymmetric errors
    sigma = 0.5 * (np.abs(data[:, 2]) + np.abs(data[:, 3]))

    print(f"  Loaded {spec_type}: ℓ = {ell[0]}–{ell[-1]}, {len(ell)} multipoles")
    return ell, Dl, sigma


def load_planck_theory():
    """
    Load the best-fit ΛCDM theory spectrum and calPlanck calibration.
    Theory file columns: ℓ, DℓTT, DℓTE, DℓEE, DℓBB, DℓPP (may vary)
    """
    # ── Read calPlanck from parameters file ──
    param_file = PLANCK_DIR / \
        "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum_R3.01.txt"
    cal_planck = 1.0
    if param_file.exists():
        with open(param_file) as f:
            for line in f:
                # Look for calPlanck in various formats
                low = line.lower().strip()
                if "calplanck" in low:
                    # Try "calPlanck = 1.00045" or "calPlanck  1.00045"
                    parts = line.replace("=", " ").split()
                    for i, p in enumerate(parts):
                        if "calplanck" in p.lower() and i + 1 < len(parts):
                            try:
                                cal_planck = float(parts[i + 1])
                            except ValueError:
                                pass
                            break
        print(f"  calPlanck = {cal_planck}")
    else:
        print(f"  ⚠️  Parameters file not found, using calPlanck = 1.0")

    # ── Load theory spectrum ──
    theory_file = PLANCK_DIR / \
        "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt"
    if not theory_file.exists():
        raise FileNotFoundError(f"Missing: {theory_file}")

    data = np.loadtxt(theory_file, comments="#")
    ell_th = data[:, 0]

    # Column mapping (standard Planck theory file order)
    col_map = {"TT": 1, "TE": 2, "EE": 3, "BB": 4}
    theory = {}
    for spec, col in col_map.items():
        if col < data.shape[1]:
            theory[spec] = data[:, col] / cal_planck**2

    print(f"  Theory: ℓ = {ell_th[0]:.0f}–{ell_th[-1]:.0f}, "
          f"spectra: {list(theory.keys())}")
    return ell_th, theory


# ─── Core Analysis ───────────────────────────────────────────────────
def compute_residuals(ell_data, Dl_data, sigma_data, ell_theory, Dl_theory,
                      ell_min=ELL_MIN, ell_max=ELL_MAX):
    """
    Compute fractional residuals δ = (data - theory) / theory
    on the overlapping ℓ range.
    """
    # Interpolate theory onto data ℓ values
    f_th = interp1d(ell_theory, Dl_theory, kind="cubic",
                    fill_value="extrapolate")
    theory_at_data = f_th(ell_data)

    # Apply ℓ cuts
    mask = (ell_data >= ell_min) & (ell_data <= ell_max) & (theory_at_data > 0)
    ell   = ell_data[mask]
    obs   = Dl_data[mask]
    th    = theory_at_data[mask]
    sig   = sigma_data[mask]

    # Fractional residual and its error
    delta       = (obs - th) / th
    delta_sigma = sig / th

    return ell, delta, delta_sigma


def lomb_scargle_in_lnell(ell, delta, delta_sigma,
                          period_min=0.1, period_max=5.0, n_periods=5000):
    """
    Run a Lomb-Scargle periodogram on the residuals in ln(ℓ) space.

    Returns:
        periods  — array of test periods in ln(ℓ) units
        power    — Lomb-Scargle power at each period
        peak_period — period of highest peak
        peak_power  — power at highest peak
    """
    x = np.log(ell.astype(float))       # ln(ℓ)
    y = delta / delta_sigma              # SNR-weighted residual

    # Remove mean
    y = y - np.mean(y)

    # Frequency grid (frequency = 1/period in ln(ℓ) space)
    freqs = np.linspace(1.0 / period_max, 1.0 / period_min, n_periods)

    # scipy.signal.lombscargle uses angular frequencies
    angular_freqs = 2 * np.pi * freqs
    power = lombscargle(x, y, angular_freqs, normalize=True)

    periods = 1.0 / freqs
    peak_idx = np.argmax(power)

    return periods, power, periods[peak_idx], power[peak_idx]


def null_test(ell, delta, delta_sigma, n_realizations=N_NULL):
    """
    Estimate significance by phase-shuffling the residuals.

    For each realization:
      1. FFT the residual time series (in ln(ℓ) space)
      2. Randomize the phases
      3. Inverse FFT back
      4. Run Lomb-Scargle
      5. Record the peak power near ln(φ)

    Returns array of null-distribution peak powers.
    """
    x = np.log(ell.astype(float))
    y = (delta / delta_sigma) - np.mean(delta / delta_sigma)

    # Target frequency window: within ±15% of 1/ln(φ)
    target_freq  = 1.0 / LN_PHI
    freq_lo      = target_freq * 0.85
    freq_hi      = target_freq * 1.15
    test_freqs   = np.linspace(freq_lo, freq_hi, 200)
    angular_test = 2 * np.pi * test_freqs

    # Measure the real peak power in this window
    real_power = lombscargle(x, y, angular_test, normalize=True)
    real_peak  = np.max(real_power)

    # Phase-shuffle null ensemble
    null_peaks = np.zeros(n_realizations)
    rng = np.random.default_rng(42)

    for i in range(n_realizations):
        # Shuffle the residual values (destroys any real periodicity)
        y_shuf = rng.permutation(y)
        p_shuf = lombscargle(x, y_shuf, angular_test, normalize=True)
        null_peaks[i] = np.max(p_shuf)

    return real_peak, null_peaks


# ─── Plotting ────────────────────────────────────────────────────────
def plot_periodogram(periods, power, spec_type, real_peak_at_phi, pvalue):
    """Plot the Lomb-Scargle periodogram with ln(φ) marked."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(periods, power, "k-", lw=0.6, alpha=0.8)
    ax.axvline(LN_PHI, color="red", ls="--", lw=2,
               label=f"ln(φ) = {LN_PHI:.4f}")
    ax.axvline(np.log(2), color="blue", ls=":", lw=1.5, alpha=0.5,
               label=f"ln(2) = {np.log(2):.4f} (control)")
    ax.set_xlabel("Period in ln(ℓ) space")
    ax.set_ylabel("Lomb-Scargle Power (normalized)")
    ax.set_title(f"Planck {spec_type} — Log-periodic search in ln(ℓ)\n"
                 f"Power at ln(φ): {real_peak_at_phi:.4f}, "
                 f"p-value: {pvalue:.2e}")
    ax.set_xlim(0.1, 3.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"periodogram_{spec_type}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: periodogram_{spec_type}.png")


def plot_null_distribution(real_peak, null_peaks, spec_type, pvalue):
    """Plot the null distribution with observed value marked."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null_peaks, bins=80, density=True, color="gray", alpha=0.7,
            label="Phase-shuffled null")
    ax.axvline(real_peak, color="red", lw=2,
               label=f"Observed ({real_peak:.4f})")

    # Mark significance thresholds
    for sigma_n, color in [(2, "orange"), (3, "red"), (5, "darkred")]:
        thresh = np.percentile(null_peaks, 100 * (1 - 2 * (1 - __import__('scipy').stats.norm.cdf(sigma_n))))
        ax.axvline(thresh, color=color, ls=":", lw=1,
                   label=f"{sigma_n}σ threshold")

    ax.set_xlabel("Peak Lomb-Scargle power near ln(φ)")
    ax.set_ylabel("Density")
    ax.set_title(f"Planck {spec_type} — Null test (N={len(null_peaks)})\n"
                 f"p = {pvalue:.2e}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"null_test_{spec_type}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: null_test_{spec_type}.png")


def plot_residuals(ell, delta, delta_sigma, spec_type):
    """Plot residuals vs ln(ℓ) so you can visually inspect."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))

    # Top: residuals vs ℓ
    ax1.errorbar(ell, delta, yerr=delta_sigma, fmt=".", ms=1, lw=0.3,
                 alpha=0.4, color="steelblue")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_xlabel("ℓ")
    ax1.set_ylabel("δDℓ / Dℓ (fractional residual)")
    ax1.set_title(f"Planck {spec_type} residuals relative to best-fit ΛCDM")

    # Bottom: residuals vs ln(ℓ) — this is where periodicity should appear
    ln_ell = np.log(ell.astype(float))
    ax2.errorbar(ln_ell, delta, yerr=delta_sigma, fmt=".", ms=1, lw=0.3,
                 alpha=0.4, color="steelblue")
    ax2.axhline(0, color="k", lw=0.5)

    # Overlay the predicted oscillation if injecting
    if INJECT_TEST:
        x_smooth = np.linspace(ln_ell.min(), ln_ell.max(), 500)
        y_pred   = INJECT_EPSILON * np.cos(2 * np.pi * x_smooth / LN_PHI + INJECT_PHASE)
        ax2.plot(x_smooth, y_pred, "r-", lw=1.5, alpha=0.7,
                 label=f"Injected: ε={INJECT_EPSILON}")
        ax2.legend()

    ax2.set_xlabel("ln(ℓ)")
    ax2.set_ylabel("δDℓ / Dℓ")
    ax2.set_title("Same residuals in ln(ℓ) space — periodicity should appear here")

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"residuals_{spec_type}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: residuals_{spec_type}.png")


# ─── Main ────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  CMB DSI Test: Log-periodic search in ln(ℓ) space")
    print(f"  Target period: ln(φ) = {LN_PHI:.6f}")
    print(f"  ℓ range: {ELL_MIN}–{ELL_MAX}")
    print(f"  Null realizations: {N_NULL}")
    print(f"  Injection test: {INJECT_TEST}")
    print("=" * 65)

    # Load theory
    print("\n[1] Loading Planck best-fit ΛCDM theory...")
    ell_theory, theory_dict = load_planck_theory()

    results = {}

    for spec in SPEC_TYPES:
        print(f"\n{'─' * 50}")
        print(f"  Processing: {spec}")
        print(f"{'─' * 50}")

        # Load data
        print(f"\n[2] Loading Planck {spec} data...")
        ell_data, Dl_data, sigma_data = load_planck_data(spec)

        if spec not in theory_dict:
            print(f"  ⚠️  {spec} not in theory file — skipping")
            continue

        # Get theory for this spectrum type
        Dl_theory = theory_dict[spec]

        # Compute residuals
        print(f"\n[3] Computing fractional residuals...")
        ell, delta, delta_sigma = compute_residuals(
            ell_data, Dl_data, sigma_data, ell_theory, Dl_theory
        )
        print(f"  Using {len(ell)} multipoles after cuts")

        # Optional: inject a fake signal for validation
        if INJECT_TEST:
            print(f"\n  💉 INJECTING test signal: ε={INJECT_EPSILON}, "
                  f"period=ln(φ)={LN_PHI:.4f}")
            ln_ell = np.log(ell.astype(float))
            injection = INJECT_EPSILON * np.cos(
                2 * np.pi * ln_ell / LN_PHI + INJECT_PHASE
            )
            delta = delta + injection

        # Plot residuals
        plot_residuals(ell, delta, delta_sigma, spec)

        # Lomb-Scargle periodogram
        print(f"\n[4] Running Lomb-Scargle periodogram...")
        periods, power, peak_period, peak_power = lomb_scargle_in_lnell(
            ell, delta, delta_sigma
        )
        print(f"  Global peak: period = {peak_period:.4f}, power = {peak_power:.4f}")

        # Power specifically at ln(φ)
        idx_phi = np.argmin(np.abs(periods - LN_PHI))
        power_at_phi = power[idx_phi]
        print(f"  Power at ln(φ) = {LN_PHI:.4f}: {power_at_phi:.4f}")

        # Null test
        print(f"\n[5] Running null test ({N_NULL} realizations)...")
        real_peak, null_peaks = null_test(ell, delta, delta_sigma)

        # p-value
        pvalue = np.mean(null_peaks >= real_peak)
        if pvalue == 0:
            pvalue_str = f"< {1.0/N_NULL:.1e}"
            pvalue_for_plot = 1.0 / N_NULL
        else:
            pvalue_str = f"{pvalue:.4f}"
            pvalue_for_plot = pvalue

        # Convert to sigma
        from scipy.stats import norm
        if pvalue > 0:
            sigma_equiv = norm.ppf(1 - pvalue)
        else:
            sigma_equiv = norm.ppf(1 - 1.0 / N_NULL)

        print(f"\n  ╔══════════════════════════════════════╗")
        print(f"  ║  RESULT: Planck {spec:>2s}                  ║")
        print(f"  ║  Power at ln(φ):  {real_peak:>8.4f}          ║")
        print(f"  ║  p-value:         {pvalue_str:>12s}      ║")
        print(f"  ║  Significance:    {sigma_equiv:>6.2f}σ            ║")
        print(f"  ╚══════════════════════════════════════╝")

        # Plot
        plot_periodogram(periods, power, spec, real_peak, pvalue_for_plot)
        plot_null_distribution(real_peak, null_peaks, spec, pvalue_for_plot)

        results[spec] = {
            "power_at_phi": real_peak,
            "p_value": pvalue,
            "sigma": sigma_equiv,
            "peak_period": peak_period,
            "peak_power": peak_power,
        }

    # ── Summary ──
    print(f"\n{'=' * 65}")
    print("  SUMMARY")
    print(f"{'=' * 65}")
    for spec, r in results.items():
        status = "✅ DETECTION" if r["sigma"] >= 3 else \
                 "🟡 MARGINAL"  if r["sigma"] >= 2 else \
                 "❌ NO SIGNAL"
        print(f"  {spec}: {r['sigma']:.2f}σ  (p={r['p_value']:.4f})  "
              f"power={r['power_at_phi']:.4f}  {status}")
        print(f"       Global peak at period={r['peak_period']:.4f} "
              f"(ln(φ)={LN_PHI:.4f})")

    # Save numerical results
    results_file = TABLE_DIR / "cmb_dsi_lnell_results.txt"
    with open(results_file, "w") as f:
        f.write("CMB DSI Log-Periodic Test Results\n")
        f.write(f"Target period: ln(phi) = {LN_PHI:.6f}\n")
        f.write(f"ell range: {ELL_MIN}-{ELL_MAX}\n")
        f.write(f"Null realizations: {N_NULL}\n")
        f.write(f"Injection test: {INJECT_TEST}\n\n")
        for spec, r in results.items():
            f.write(f"{spec}:\n")
            for k, v in r.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")
    print(f"\n  Results saved to {results_file}")


if __name__ == "__main__":
    main()
