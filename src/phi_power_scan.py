#!/usr/bin/env python3
"""
φ^m Power Scan — Systematic test of DSI at all physically motivated periods.

For each m = 1..6, fits a multi-harmonic template with fundamental period
m·ln(φ) to CMB, DESI BAO, and LIGO QNM data.

Reports Δχ², AIC, BIC for each m and each channel.
Identifies which (if any) m is preferred.
"""

import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2_dist, norm
from scipy.integrate import quad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Configuration ───────────────────────────────────────────────────
PHI        = (1 + np.sqrt(5)) / 2
LN_PHI     = np.log(PHI)

M_VALUES   = [1, 2, 3, 4, 5, 6]          # powers of φ to test
N_TEMPLATE = 2                             # harmonics per template (keep small
                                           # to avoid overfitting with few points)
N_STARTS   = 80                            # random restarts for optimizer
N_NULL     = 5000                          # Monte Carlo shuffles

PLANCK_DIR = Path("data/planck")
DESI_DIR   = Path("data/desi")
FIG_DIR    = Path("results/figures")
TAB_DIR    = Path("results/tables")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# ΛCDM fiducial (Planck 2018 best-fit)
OM   = 0.3153
H0   = 67.36
RS   = 147.09   # sound horizon in Mpc
C_KM = 299792.458


# ─── Template Fitting Engine ────────────────────────────────────────
def template(x, period, params):
    """
    Multi-harmonic cosine template.

    params = [ε₁, ψ₁, ε₂, ψ₂, ...]
    """
    result = np.zeros_like(x)
    n_harm = len(params) // 2
    for n in range(n_harm):
        eps = params[2 * n]
        psi = params[2 * n + 1]
        freq = 2 * np.pi * (n + 1) / period
        result += eps * np.cos(freq * x + psi)
    return result


def fit_template(x, y, sigma, period, n_harmonics=N_TEMPLATE, n_starts=None):
    """
    Fit template at a given period. Returns dict with fit results.
    """
    if n_starts is None:
        n_starts = N_STARTS
    n_params = 2 * n_harmonics
    chi2_null = np.sum((y / sigma) ** 2)

    def objective(params):
        model = template(x, period, params)
        return 0.5 * np.sum(((y - model) / sigma) ** 2)

    best_fun = np.inf
    best_par = None
    rng = np.random.default_rng(42)

    for _ in range(n_starts):
        p0 = np.concatenate([
            rng.uniform(-0.05, 0.05, n_harmonics),   # amplitudes
            rng.uniform(0, 2 * np.pi, n_harmonics),  # phases
        ])
        # Interleave: [ε₁, ψ₁, ε₂, ψ₂, ...]
        p0_interleaved = np.empty(n_params)
        p0_interleaved[0::2] = p0[:n_harmonics]
        p0_interleaved[1::2] = p0[n_harmonics:]

        res = minimize(objective, p0_interleaved, method="L-BFGS-B")
        if res.fun < best_fun:
            best_fun = res.fun
            best_par = res.x

    chi2_fit = 2 * best_fun
    dchi2 = chi2_null - chi2_fit
    n_data = len(x)

    # Analytic p-value (χ² distribution)
    pval = 1 - chi2_dist.cdf(max(dchi2, 0), df=n_params)
    sig = norm.ppf(1 - pval) if 0 < pval < 1 else (8.0 if pval <= 0 else -8.0)

    # Information criteria
    aic = chi2_fit + 2 * n_params
    bic = chi2_fit + n_params * np.log(n_data)

    # Per-harmonic amplitudes
    amplitudes = [np.abs(best_par[2 * n]) for n in range(n_harmonics)]

    return {
        "period": period,
        "chi2_null": chi2_null,
        "chi2_fit": chi2_fit,
        "delta_chi2": dchi2,
        "n_params": n_params,
        "p_value": pval,
        "sigma": sig,
        "aic": aic,
        "bic": bic,
        "params": best_par,
        "amplitudes": amplitudes,
        "aic_null": chi2_null,  # null has 0 params
        "bic_null": chi2_null,
    }


def null_test_mc(x, y, sigma, period, n_realizations=N_NULL, mc_starts=10):
    """Monte Carlo null distribution for a given period."""
    rng = np.random.default_rng(99)
    null_dchi2 = np.zeros(n_realizations)
    for i in range(n_realizations):
        y_shuf = rng.permutation(y)
        res = fit_template(x, y_shuf, sigma, period, n_starts=mc_starts)
        null_dchi2[i] = max(res["delta_chi2"], 0)
    return null_dchi2


# ─── ΛCDM Distance Functions ────────────────────────────────────────
def _E(z):
    return np.sqrt(OM * (1 + z)**3 + (1 - OM))

def DM_over_rs(z):
    integral, _ = quad(lambda zp: 1.0 / _E(zp), 0, z)
    return (C_KM / H0) * integral / RS

def DH_over_rs(z):
    return (C_KM / H0) / (_E(z) * RS)

def DV_over_rs(z):
    dm = DM_over_rs(z) * RS
    dh = DH_over_rs(z) * RS
    return (z * dm**2 * dh)**(1./3.) / RS

_LCDM_FUNCS = {
    "DM_over_rs": DM_over_rs,
    "DH_over_rs": DH_over_rs,
    "DV_over_rs": DV_over_rs,
}


# ─── Data Loaders ───────────────────────────────────────────────────
def load_cmb(spec_type="TT"):
    """Load Planck spectrum and theory, return ln(ℓ), δDℓ/Dℓ, σ."""
    # Data
    data = np.loadtxt(
        PLANCK_DIR / f"COM_PowerSpect_CMB-{spec_type}-full_R3.01.txt",
        comments="#")
    ell_d, Dl_d = data[:, 0], data[:, 1]
    sig_d = 0.5 * (np.abs(data[:, 2]) + np.abs(data[:, 3]))

    # Theory
    theory = np.loadtxt(
        PLANCK_DIR / "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt",
        comments="#")
    col = {"TT": 1, "TE": 2, "EE": 3}[spec_type]

    # calPlanck
    cal = 1.0
    pf = PLANCK_DIR / "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum_R3.01.txt"
    if pf.exists():
        with open(pf) as f:
            for line in f:
                if "calplanck" in line.lower():
                    parts = line.replace("=", " ").split()
                    for i, p in enumerate(parts):
                        if "calplanck" in p.lower() and i + 1 < len(parts):
                            try: cal = float(parts[i + 1])
                            except: pass

    f_th = interp1d(theory[:, 0], theory[:, col] / cal**2,
                    kind="cubic", fill_value="extrapolate")

    mask = (ell_d >= 30) & (f_th(ell_d) > 0)
    ell = ell_d[mask]
    delta = (Dl_d[mask] - f_th(ell)) / f_th(ell)
    sigma = sig_d[mask] / f_th(ell)

    return np.log(ell.astype(float)), delta, sigma, f"CMB {spec_type}"


def load_desi():
    """Load DESI compressed BAO, return ln(z), fractional residuals, σ."""
    data_rows = []

    # Try combined file first
    for f in sorted(DESI_DIR.glob("desi_*_mean.txt")):
        if "ALL" in f.name:
            continue
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("#") or len(line) == 0:
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        data_rows.append(
                            (float(parts[0]), float(parts[1]), parts[2]))
                    except ValueError:
                        continue

    if not data_rows:
        # Try ALL file
        for f in sorted(DESI_DIR.glob("desi_*ALL*_mean.txt")):
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith("#") or len(line) == 0:
                        continue
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            data_rows.append(
                                (float(parts[0]), float(parts[1]), parts[2]))
                        except ValueError:
                            continue

    if not data_rows:
        raise FileNotFoundError("No DESI data found")

    # Load covariance if available
    cov_matrix = None
    cov_files = list(DESI_DIR.glob("desi_*ALL*_cov.txt"))
    if cov_files:
        cov_matrix = np.loadtxt(cov_files[0])

    z_arr, resid, sigma = [], [], []
    preds = []
    for z_val, meas, obs_type in data_rows:
        if obs_type not in _LCDM_FUNCS:
            continue
        pred = _LCDM_FUNCS[obs_type](z_val)
        z_arr.append(z_val)
        resid.append((meas - pred) / pred)
        preds.append(pred)
        sigma.append(0.02)

    z_arr = np.array(z_arr)
    resid = np.array(resid)
    preds = np.array(preds)
    sigma = np.array(sigma)

    if cov_matrix is not None and len(cov_matrix) == len(resid):
        frac_cov = cov_matrix / np.outer(preds, preds)
        sigma = np.sqrt(np.diag(frac_cov))

    return np.log(z_arr), resid, sigma, "DESI BAO"


def load_ligo_qnm():
    """
    LIGO QNM frequencies → compute ln(ratio), assign uniform errors.
    """
    qnm_freqs = np.array([
        251.2, 390.6, 297.5, 253.5, 326.7, 171.4,
        275.3, 373.8, 262.1, 350.2, 287.4, 310.5,
        245.8, 298.3,
    ])
    # All pairwise ratios > 1
    ratios = []
    for i in range(len(qnm_freqs)):
        for j in range(i + 1, len(qnm_freqs)):
            r = max(qnm_freqs[i], qnm_freqs[j]) / \
                min(qnm_freqs[i], qnm_freqs[j])
            if r > 1.01:  # skip near-unity
                ratios.append(r)
    ratios = np.sort(ratios)
    ln_ratios = np.log(ratios)

    # Build a "spectrum" from the distribution of ln(ratios)
    # Use kernel density estimation, then treat as a function to fit
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(ln_ratios, bw_method=0.05)
    x_grid = np.linspace(0.01, ln_ratios.max() + 0.1, 200)
    y_grid = kde(x_grid)
    # Normalize to zero mean
    y_grid = y_grid - np.mean(y_grid)
    sigma_grid = np.ones_like(y_grid) * np.std(y_grid)

    return x_grid, y_grid, sigma_grid, "LIGO QNM"


# ─── Main Scan ──────────────────────────────────────────────────────
def scan_channel(x, y, sigma, label, run_mc=False):
    """Scan all φ^m periods for one data channel."""
    baseline = x.max() - x.min()
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"  N = {len(x)}, baseline = {baseline:.3f} in ln-space")
    print(f"{'─' * 60}")

    results = []
    for m in M_VALUES:
        period = m * LN_PHI
        cycles = baseline / period
        res = fit_template(x, y, sigma, period)
        res["m"] = m
        res["cycles"] = cycles
        results.append(res)

        status = "✅" if res["sigma"] >= 3 else \
                 "🟡" if res["sigma"] >= 2 else "·"
        print(f"  φ^{m}: period={period:.3f}  cycles={cycles:.1f}  "
              f"Δχ²={res['delta_chi2']:>6.1f}  "
              f"AIC={res['aic']:>8.1f}  "
              f"p={res['p_value']:.4f}  "
              f"{res['sigma']:>5.1f}σ  {status}")

    # Optional: MC null test for the best-performing m
    if run_mc:
        best = max(results, key=lambda r: r["delta_chi2"])
        print(f"\n  Running MC null test for best m={best['m']}...")
        null_dchi2 = null_test_mc(x, y, sigma, best["period"])
        mc_p = np.mean(null_dchi2 >= best["delta_chi2"])
        mc_sig = norm.ppf(1 - mc_p) if 0 < mc_p < 1 else \
                 (8.0 if mc_p == 0 else -8.0)
        print(f"  MC p-value: {mc_p:.4f} ({mc_sig:.1f}σ)")
        best["mc_p"] = mc_p
        best["mc_sigma"] = mc_sig

    return results


def plot_scan(all_results, channel_label):
    """Create comparison plot for one channel across all m values."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"φ^m Power Scan — {channel_label}", fontsize=14, y=1.02)

    ms = [r["m"] for r in all_results]
    labels = [f"φ^{m}\n({m*LN_PHI:.2f})" for m in ms]

    # Panel 1: Δχ²
    ax = axes[0, 0]
    dchi2s = [r["delta_chi2"] for r in all_results]
    colors = ["#e74c3c" if r["sigma"] >= 2 else
              "#f39c12" if r["sigma"] >= 1 else
              "#95a5a6" for r in all_results]
    ax.bar(ms, dchi2s, color=colors, edgecolor="k", alpha=0.8)
    # Expected Δχ² from noise (= n_params)
    ax.axhline(all_results[0]["n_params"], color="blue", ls="--", lw=1,
               label=f"Expected from noise ({all_results[0]['n_params']} params)")
    ax.set_xlabel("Power m (period = m·ln φ)")
    ax.set_ylabel("Δχ²")
    ax.set_title("Fit improvement over null")
    ax.set_xticks(ms)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=8)

    # Panel 2: AIC
    ax = axes[0, 1]
    aics = [r["aic"] for r in all_results]
    aic_null = all_results[0]["aic_null"]
    ax.bar(ms, aics, color=colors, edgecolor="k", alpha=0.8)
    ax.axhline(aic_null, color="blue", ls="--", lw=1, label="ΛCDM (AIC)")
    ax.set_xlabel("Power m")
    ax.set_ylabel("AIC")
    ax.set_title("Akaike Information Criterion (lower = better)")
    ax.set_xticks(ms)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=8)

    # Panel 3: Significance (σ)
    ax = axes[1, 0]
    sigs = [r["sigma"] for r in all_results]
    ax.bar(ms, sigs, color=colors, edgecolor="k", alpha=0.8)
    ax.axhline(2, color="orange", ls=":", label="2σ")
    ax.axhline(3, color="red", ls=":", label="3σ")
    ax.set_xlabel("Power m")
    ax.set_ylabel("Significance (σ)")
    ax.set_title("Detection significance")
    ax.set_xticks(ms)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=8)

    # Panel 4: Best-fit template overlaid on data
    best = max(all_results, key=lambda r: r["delta_chi2"])
    ax = axes[1, 1]

    # We need x, y, sigma from the channel — store them
    # (passed via closure or we re-derive)
    ax.text(0.5, 0.5,
            f"Best: φ^{best['m']}\n"
            f"Δχ² = {best['delta_chi2']:.1f}\n"
            f"p = {best['p_value']:.4f}\n"
            f"{best['sigma']:.1f}σ\n"
            f"AIC = {best['aic']:.1f}\n"
            f"|ε₁| = {best['amplitudes'][0]:.4f}",
            transform=ax.transAxes, fontsize=14,
            ha="center", va="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_title("Best-fit summary")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    safe_label = channel_label.replace(" ", "_").lower()
    fig.savefig(FIG_DIR / f"phi_scan_{safe_label}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: phi_scan_{safe_label}.png")


def plot_best_overlay(x, y, sigma, results, channel_label):
    """Plot data with the top-3 best-fit templates overlaid."""
    ranked = sorted(results, key=lambda r: r["delta_chi2"], reverse=True)[:3]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.errorbar(x, y, yerr=sigma, fmt=".", ms=2, lw=0.3, alpha=0.4,
                color="steelblue", label="Data")

    x_smooth = np.linspace(x.min(), x.max(), 500)
    line_styles = ["-", "--", ":"]
    line_colors = ["#e74c3c", "#f39c12", "#27ae60"]

    for i, res in enumerate(ranked):
        tmpl = template(x_smooth, res["period"], res["params"])
        ax.plot(x_smooth, tmpl, color=line_colors[i], ls=line_styles[i],
                lw=2, alpha=0.8,
                label=f"φ^{res['m']} ({res['sigma']:.1f}σ, Δχ²={res['delta_chi2']:.1f})")

    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("ln-space coordinate")
    ax.set_ylabel("Fractional residual")
    ax.set_title(f"{channel_label} — Top 3 φ^m template fits")
    ax.legend(fontsize=9)

    fig.tight_layout()
    safe_label = channel_label.replace(" ", "_").lower()
    fig.savefig(FIG_DIR / f"phi_scan_overlay_{safe_label}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: phi_scan_overlay_{safe_label}.png")


# ─── Main ────────────────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   φ^m POWER SCAN — Systematic DSI Period Search            ║")
    print("║   Testing m = 1, 2, 3, 4, 5, 6                            ║")
    print("║   Periods: 0.48, 0.96, 1.44, 1.93, 2.41, 2.89            ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    all_channel_results = {}

    # ── CMB ──
    for spec in ["TT", "TE", "EE"]:
        try:
            x, y, sigma, label = load_cmb(spec)
            results = scan_channel(x, y, sigma, label)
            plot_scan(results, label)
            plot_best_overlay(x, y, sigma, results, label)
            all_channel_results[label] = results
        except Exception as e:
            print(f"  ❌ {spec}: {e}")

    # ── DESI ──
    try:
        x, y, sigma, label = load_desi()
        results = scan_channel(x, y, sigma, label, run_mc=True)
        plot_scan(results, label)
        plot_best_overlay(x, y, sigma, results, label)
        all_channel_results[label] = results
    except Exception as e:
        print(f"  ❌ DESI: {e}")

    # ── LIGO ──
    try:
        x, y, sigma, label = load_ligo_qnm()
        results = scan_channel(x, y, sigma, label)
        plot_scan(results, label)
        plot_best_overlay(x, y, sigma, results, label)
        all_channel_results[label] = results
    except Exception as e:
        print(f"  ❌ LIGO: {e}")

    # ═══ Grand Summary Table ═══
    print("\n" + "=" * 80)
    print("  GRAND SUMMARY — φ^m Power Scan")
    print("=" * 80)

    header = f"{'Channel':<12} | " + " | ".join(
        [f"  φ^{m} (Δχ²/σ)  " for m in M_VALUES])
    print(f"  {header}")
    print(f"  {'─' * len(header)}")

    for label, results in all_channel_results.items():
        row = f"{label:<12} | "
        for res in results:
            sig_str = f"{res['sigma']:+.1f}σ"
            row += f" {res['delta_chi2']:>5.1f}/{sig_str:<6s} | "
        # Mark the best
        best = max(results, key=lambda r: r["delta_chi2"])
        row += f" ← best: φ^{best['m']}"
        print(f"  {row}")

    # ── Save summary table ──
    with open(TAB_DIR / "phi_power_scan.txt", "w") as f:
        f.write("φ^m Power Scan Results\n")
        f.write(f"Tested: m = {M_VALUES}\n")
        f.write(f"Harmonics per template: {N_TEMPLATE}\n\n")

        for label, results in all_channel_results.items():
            f.write(f"\n{label}:\n")
            f.write(f"  {'m':>2}  {'period':>7}  {'cycles':>6}  "
                    f"{'Δχ²':>7}  {'p-val':>8}  {'σ':>6}  "
                    f"{'AIC':>8}  {'|ε₁|':>7}\n")
            for r in results:
                f.write(f"  {r['m']:>2}  {r['period']:>7.3f}  "
                        f"{r['cycles']:>6.1f}  {r['delta_chi2']:>7.1f}  "
                        f"{r['p_value']:>8.4f}  {r['sigma']:>6.1f}  "
                        f"{r['aic']:>8.1f}  {r['amplitudes'][0]:>7.4f}\n")

            best = max(results, key=lambda r: r["delta_chi2"])
            f.write(f"  Best: φ^{best['m']} "
                    f"(Δχ²={best['delta_chi2']:.1f}, {best['sigma']:.1f}σ)\n")

    print(f"\n  Saved: phi_power_scan.txt")

    # ── Final verdict ──
    print("\n" + "=" * 80)
    print("  INTERPRETATION GUIDE")
    print("=" * 80)
    print("""
  • If ONE specific m is consistently preferred across channels → 
    strong evidence for that DSI period

  • If DIFFERENT m values win in different channels → 
    likely fitting noise (each channel picks a different random peak)

  • If NO m exceeds 2σ in any channel → 
    DSI at any φ^m period is not detected at current sensitivity

  • If Δχ² ≈ n_params for all m →
    all models are fitting noise equally well (expected null)

  • CRITICAL: if m=3 (icosahedral inflation) wins across CMB + BAO →
    that's the most physically motivated result
    """)


if __name__ == "__main__":
    main()
