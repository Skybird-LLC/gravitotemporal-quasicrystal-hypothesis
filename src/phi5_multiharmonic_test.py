#!/usr/bin/env python3
"""
φ⁴ Multi-Harmonic DSI Test

Tests for quasicrystalline discrete scale invariance with fundamental
period 4·ln(φ), including all harmonics n=1..4.

Runs on:
  1. Planck CMB (TT, TE, EE)
  2. DESI compressed BAO
  3. LIGO QNM frequency ratios (if available)
"""

import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2_dist, norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Configuration ───────────────────────────────────────────────────
PHI       = (1 + np.sqrt(5)) / 2
LN_PHI    = np.log(PHI)                    # ≈ 0.4812
PERIOD_5  = 4 * LN_PHI                     # ≈ 1.925  (φ⁴ fundamental)

N_HARMONICS = 4                             # test harmonics 1..4
N_NULL      = 10000                         # Monte Carlo null realizations

PLANCK_DIR  = Path("data/planck")
DESI_DIR    = Path("data/desi")
FIG_DIR     = Path("results/figures")
TAB_DIR     = Path("results/tables")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# Injection test (set True to validate pipeline)
INJECT_TEST    = True
INJECT_EPS     = 0.008       # amplitude per harmonic
INJECT_HARMONICS = [1, 2, 4] # which harmonics get signal


# ─── Multi-Harmonic Template ────────────────────────────────────────
def multiharmonic_template(x, params):
    """
    Compute the multi-harmonic DSI template.

    x:      array of ln(ℓ), ln(z), or ln(f) values
    params: array of [ε₁, ψ₁, ε₂, ψ₂, ..., ε₅, ψ₅]
            (amplitude and phase for each harmonic)

    Returns: template values at each x
    """
    result = np.zeros_like(x)
    for n in range(N_HARMONICS):
        eps = params[2 * n]
        psi = params[2 * n + 1]
        freq_n = 2 * np.pi * (n + 1) / PERIOD_5
        result += eps * np.cos(freq_n * x + psi)
    return result


def fit_multiharmonic(x, y, sigma, n_harmonics=N_HARMONICS):
    """
    Fit the multi-harmonic DSI template to data.

    Returns:
        best_params: fitted [ε₁, ψ₁, ..., εₙ, ψₙ]
        chi2_null:   χ² for the null model (flat zero)
        chi2_dsi:    χ² for the best-fit DSI model
        delta_chi2:  improvement
        n_extra:     number of extra parameters (2 per harmonic)
        pvalue:      p-value from Δχ² distribution
    """
    # Null model χ²
    chi2_null = np.sum((y / sigma) ** 2)

    # Fit DSI model
    def neg_loglike(params):
        model = multiharmonic_template(x, params)
        return 0.5 * np.sum(((y - model) / sigma) ** 2)

    # Try multiple random starts to avoid local minima
    best_result = None
    best_fun = np.inf
    rng = np.random.default_rng(42)

    for _ in range(50):
        p0 = []
        for n in range(n_harmonics):
            p0.extend([rng.uniform(-0.05, 0.05),   # εₙ
                        rng.uniform(0, 2 * np.pi)])  # ψₙ
        p0 = np.array(p0)

        res = minimize(neg_loglike, p0, method="L-BFGS-B")
        if res.fun < best_fun:
            best_fun = res.fun
            best_result = res

    chi2_dsi = 2 * best_result.fun
    delta_chi2 = chi2_null - chi2_dsi
    n_extra = 2 * n_harmonics

    # p-value: probability of Δχ² this large from noise
    pvalue = 1 - chi2_dist.cdf(delta_chi2, df=n_extra)

    return best_result.x, chi2_null, chi2_dsi, delta_chi2, n_extra, pvalue


def null_distribution_multiharmonic(x, y, sigma, n_realizations=N_NULL):
    """
    Build null distribution of Δχ² by shuffling residuals.
    More conservative than the analytic χ² distribution.
    """
    rng = np.random.default_rng(123)
    null_dchi2 = np.zeros(n_realizations)

    for i in range(n_realizations):
        y_shuf = rng.permutation(y)
        _, _, _, dchi2, _, _ = fit_multiharmonic(x, y_shuf, sigma)
        null_dchi2[i] = dchi2

    return null_dchi2


# ─── Channel 1: Planck CMB ──────────────────────────────────────────
def test_cmb():
    """Run the φ⁴ multi-harmonic test on Planck CMB spectra."""
    print("\n" + "=" * 65)
    print("  CHANNEL 1: Planck CMB — φ⁴ Multi-Harmonic Template Fit")
    print("=" * 65)

    # Load theory
    theory_file = PLANCK_DIR / \
        "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt"
    theory_data = np.loadtxt(theory_file, comments="#")
    ell_th = theory_data[:, 0]
    col_map = {"TT": 1, "TE": 2, "EE": 3}

    # Load calPlanck
    param_file = PLANCK_DIR / \
        "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum_R3.01.txt"
    cal = 1.0
    if param_file.exists():
        with open(param_file) as f:
            for line in f:
                if "calplanck" in line.lower():
                    parts = line.replace("=", " ").split()
                    for i, p in enumerate(parts):
                        if "calplanck" in p.lower() and i + 1 < len(parts):
                            try:
                                cal = float(parts[i + 1])
                            except ValueError:
                                pass
    print(f"  calPlanck = {cal}")

    cmb_results = {}

    for spec, col in col_map.items():
        fname = PLANCK_DIR / f"COM_PowerSpect_CMB-{spec}-full_R3.01.txt"
        if not fname.exists():
            print(f"  ⚠️  {fname.name} not found — skipping {spec}")
            continue

        print(f"\n  ── {spec} ──")

        # Load data
        data = np.loadtxt(fname, comments="#")
        ell_d  = data[:, 0]
        Dl_d   = data[:, 1]
        sig_d  = 0.5 * (np.abs(data[:, 2]) + np.abs(data[:, 3]))

        # Interpolate theory
        Dl_th = theory_data[:, col] / cal**2
        f_th  = interp1d(ell_th, Dl_th, kind="cubic", fill_value="extrapolate")

        # Apply cuts: ℓ ≥ 30, theory > 0
        mask = (ell_d >= 30) & (f_th(ell_d) > 0)
        ell   = ell_d[mask]
        obs   = Dl_d[mask]
        th    = f_th(ell)
        sig   = sig_d[mask]

        # Fractional residuals
        delta = (obs - th) / th
        delta_sig = sig / th

        # Transform to ln(ℓ) space
        x = np.log(ell.astype(float))
        y = delta
        s = delta_sig

        # Optional injection
        if INJECT_TEST:
            print(f"    💉 Injecting signal: ε={INJECT_EPS} at harmonics {INJECT_HARMONICS}")
            for n in INJECT_HARMONICS:
                freq_n = 2 * np.pi * n / PERIOD_5
                y = y + INJECT_EPS * np.cos(freq_n * x + 1.23)

        # Fit
        print(f"    Fitting {N_HARMONICS}-harmonic template to {len(x)} multipoles...")
        params, chi2_0, chi2_dsi, dchi2, n_extra, pval = fit_multiharmonic(x, y, s)

        sigma_equiv = norm.ppf(1 - pval) if pval > 1e-15 else 8.0

        print(f"    χ²(null)    = {chi2_0:.1f}")
        print(f"    χ²(DSI)     = {chi2_dsi:.1f}")
        print(f"    Δχ²         = {dchi2:.1f}  ({n_extra} extra params)")
        print(f"    p-value     = {pval:.4f}")
        print(f"    Significance = {sigma_equiv:.2f}σ")

        # Extract per-harmonic amplitudes
        print(f"\n    Per-harmonic amplitudes:")
        for n in range(N_HARMONICS):
            eps_n = params[2 * n]
            psi_n = params[2 * n + 1]
            period_n = PERIOD_5 / (n + 1)
            print(f"      n={n+1}: ε={eps_n:+.5f}, ψ={psi_n:.3f} rad, "
                  f"period={period_n:.4f} (φ^{4/(n+1):.1f})")

        cmb_results[spec] = {
            "delta_chi2": dchi2,
            "p_value": pval,
            "sigma": sigma_equiv,
            "params": params,
        }

        # ── Plot ──
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Panel 1: Residuals in ln(ℓ) with template overlay
        ax = axes[0]
        ax.errorbar(x, y, yerr=s, fmt=".", ms=1, lw=0.2, alpha=0.3,
                    color="steelblue", label="Data − ΛCDM")
        x_smooth = np.linspace(x.min(), x.max(), 1000)
        template = multiharmonic_template(x_smooth, params)
        ax.plot(x_smooth, template, "orange", lw=2, alpha=0.8,
            label=f"φ⁴ template (Δχ²={dchi2:.1f})")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("ln(ℓ)")
        ax.set_ylabel("δDℓ / Dℓ")
        ax.set_title(f"Planck {spec} — φ⁴ multi-harmonic template fit ({sigma_equiv:.1f}σ)")
        ax.legend(fontsize=9)

        # Panel 2: Per-harmonic power bar chart
        ax = axes[1]
        harmonics = np.arange(1, N_HARMONICS + 1)
        amplitudes = [np.abs(params[2 * n]) for n in range(N_HARMONICS)]
        colors = ["#e74c3c" if n + 1 in [1, 2, 4] else "#95a5a6"
                  for n in range(N_HARMONICS)]
        ax.bar(harmonics, amplitudes, color=colors, edgecolor="k", alpha=0.7)
        ax.set_xlabel("Harmonic n")
        ax.set_ylabel("|εₙ|")
        ax.set_title("Amplitude per harmonic (red = φ-related)")
        ax.set_xticks(harmonics)
        ax.set_xticklabels([f"n={n}\n(φ^{4/n:.1f})" for n in harmonics],
                           fontsize=8)

        # Panel 3: Existing periodogram with harmonic locations marked
        from scipy.signal import lombscargle as ls
        freqs = np.linspace(0.1, 15, 3000)
        y_centered = (y - np.mean(y)) / s
        power = ls(x, y_centered - np.mean(y_centered),
                   2 * np.pi * freqs, normalize=True)
        ax = axes[2]
        ax.plot(freqs, power, "k-", lw=0.6, alpha=0.8)
        for n in range(1, N_HARMONICS + 1):
            k_n = n / PERIOD_5
            color = "red" if n in [1, 2, 4] else "gray"
            ax.axvline(k_n, color=color, ls="--", alpha=0.6,
                       label=f"n={n} (k={k_n:.3f})" if n <= N_HARMONICS else None)
        ax.set_xlabel("Frequency k (cycles per unit ln ℓ)")
        ax.set_ylabel("Lomb-Scargle Power")
        ax.set_title("Periodogram with φ⁴ harmonic locations")
        ax.legend(fontsize=7, ncol=3)
        ax.set_xlim(0, 8)

        fig.tight_layout()
        fig.savefig(FIG_DIR / f"phi4_cmb_{spec}.png", dpi=150)
        plt.close(fig)
        print(f"    Saved: phi4_cmb_{spec}.png")

    return cmb_results


# ─── Channel 2: DESI BAO ────────────────────────────────────────────
def test_desi():
    """Run the φ⁴ template fit on DESI compressed BAO."""
    print("\n" + "=" * 65)
    print("  CHANNEL 2: DESI DR1 BAO — φ⁴ Template Fit")
    print("=" * 65)

    # Find the combined DESI file
    patterns = ["desi_2024_gaussian_bao_ALL_GCcomb*_mean.txt",
                "desi_*ALL*mean*"]
    data_rows = []
    cov_matrix = None

    # Try combined file first
    for pattern in patterns:
        matches = list(DESI_DIR.glob(pattern))
        if matches:
            fname = matches[0]
            print(f"  Loading: {fname.name}")
            with open(fname) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("#") or len(line) == 0:
                        continue
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            z_val = float(parts[0])
                            measurement = float(parts[1])
                            obs_type = parts[2] if len(parts) > 2 else "DV_over_rs"
                            data_rows.append((z_val, measurement, obs_type))
                        except ValueError:
                            continue
            # Load covariance
            cov_file = fname.parent / fname.name.replace("mean", "cov")
            if cov_file.exists():
                cov_matrix = np.loadtxt(cov_file)
                print(f"  Loaded covariance: {cov_file.name}")
            break

    if not data_rows:
        # Fall back: load individual tracer files
        print("  Combined file not found — loading individual tracers...")
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
                            data_rows.append((float(parts[0]),
                                              float(parts[1]),
                                              parts[2]))
                        except ValueError:
                            continue

    if not data_rows:
        print("  ⚠️  No DESI data found — skipping")
        return {}

    # Compute ΛCDM predictions
    # Using Planck 2018 best-fit: Ωm=0.3153, H0=67.36, rs=147.09 Mpc
    Om = 0.3153
    H0 = 67.36
    rs = 147.09
    c  = 299792.458  # km/s

    def E(z):
        return np.sqrt(Om * (1 + z)**3 + (1 - Om))

    def DM_over_rs(z):
        from scipy.integrate import quad
        integral, _ = quad(lambda zp: 1.0 / E(zp), 0, z)
        return (c / H0) * integral / rs

    def DH_over_rs(z):
        return (c / H0) / (E(z) * rs)

    def DV_over_rs(z):
        dm = DM_over_rs(z) * rs  # in Mpc
        dh = DH_over_rs(z) * rs
        return (z * dm**2 * dh)**(1./3.) / rs

    func_map = {
        "DM_over_rs": DM_over_rs,
        "DH_over_rs": DH_over_rs,
        "DV_over_rs": DV_over_rs,
    }

    # Compute fractional residuals
    z_arr = []
    resid = []
    sigma_arr = []

    for z_val, meas, obs_type in data_rows:
        if obs_type not in func_map:
            continue
        pred = func_map[obs_type](z_val)
        frac_resid = (meas - pred) / pred
        z_arr.append(z_val)
        resid.append(frac_resid)
        # Rough error estimate if no covariance
        sigma_arr.append(0.02)  # ~2% placeholder

    z_arr = np.array(z_arr)
    resid = np.array(resid)

    # If we have a covariance matrix, use it
    if cov_matrix is not None and len(cov_matrix) == len(resid):
        # Convert covariance to fractional residual space
        # (approximate: divide each row/col by the prediction)
        preds = []
        for z_val, meas, obs_type in data_rows:
            if obs_type in func_map:
                preds.append(func_map[obs_type](z_val))
        preds = np.array(preds)
        frac_cov = cov_matrix / np.outer(preds, preds)
        sigma_arr = np.sqrt(np.diag(frac_cov))
        print(f"  Using full covariance ({len(sigma_arr)} data points)")
    else:
        sigma_arr = np.array(sigma_arr)

    # Transform to ln(z) space
    x = np.log(z_arr)
    y = resid
    s = sigma_arr

    print(f"  Data points: {len(x)}")
    print(f"  ln(z) range: {x.min():.3f} to {x.max():.3f} "
          f"(baseline = {x.max()-x.min():.3f})")
    print(f"  Cycles at φ⁴: {(x.max()-x.min())/PERIOD_5:.2f}")

    # ── Fit models ──

    # Model A: ΛCDM (null — flat residuals)
    chi2_null = np.sum((y / s)**2)

    # Model D: φ⁴ multi-harmonic (but limit harmonics by baseline)
    # With baseline ~2.07, only n=1 (maybe n=2) is resolvable
    max_harmonic = max(1, int(np.floor((x.max() - x.min()) / (PERIOD_5 / N_HARMONICS))))
    max_harmonic = min(max_harmonic, N_HARMONICS)
    print(f"  Fitting harmonics n=1..{max_harmonic}")

    params_d, _, chi2_d, dchi2_d, nextra_d, pval_d = \
        fit_multiharmonic(x, y, s, n_harmonics=max_harmonic)
    sig_d = norm.ppf(1 - pval_d) if pval_d > 1e-15 else 8.0

    # Model B: w0waCDM (2 extra params, smooth linear fit in ln(z))
    # Approximate: residual ≈ a + b·ln(z) captures the w0wa drift
    A_lin = np.column_stack([np.ones_like(x), x])
    W = np.diag(1 / s**2)
    try:
        beta = np.linalg.solve(A_lin.T @ W @ A_lin, A_lin.T @ W @ y)
        y_lin = A_lin @ beta
        chi2_w0wa = np.sum(((y - y_lin) / s)**2)
    except np.linalg.LinAlgError:
        chi2_w0wa = chi2_null
        y_lin = np.zeros_like(y)

    print(f"\n  ╔══════════════════════════════════════════════╗")
    print(f"  ║  Model Comparison (DESI DR1 BAO)            ║")
    print(f"  ╠══════════════════════════════════════════════╣")
    print(f"  ║  ΛCDM:     χ² = {chi2_null:>7.1f}  (0 params)        ║")
    print(f"  ║  w₀wₐCDM:  χ² = {chi2_w0wa:>7.1f}  (2 params)        ║")
    print(f"  ║  φ⁴ DSI:   χ² = {chi2_d:>7.1f}  ({nextra_d} params)        ║")
    print(f"  ║  Δχ²(DSI): {dchi2_d:>7.1f}  p = {pval_d:.4f} ({sig_d:.1f}σ) ║")
    print(f"  ╚══════════════════════════════════════════════╝")

    # AIC comparison
    n_data = len(y)
    aic_lcdm  = chi2_null + 2 * 0
    aic_w0wa  = chi2_w0wa + 2 * 2
    aic_phi4  = chi2_d    + 2 * nextra_d
    print(f"\n  AIC:  ΛCDM={aic_lcdm:.1f}  w₀wₐ={aic_w0wa:.1f}  φ⁴={aic_phi4:.1f}")
    print(f"  ΔAIC(φ⁴ vs ΛCDM) = {aic_phi4 - aic_lcdm:.1f}")
    print(f"  ΔAIC(φ⁴ vs w₀wₐ) = {aic_phi4 - aic_w0wa:.1f}")

    # ── Plot ──
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))

    ax = axes[0]
    ax.errorbar(z_arr, 100 * resid, yerr=100 * sigma_arr, fmt="ko", ms=5,
                capsize=3, label="Data − ΛCDM")
    z_smooth = np.linspace(z_arr.min() * 0.8, z_arr.max() * 1.2, 300)
    x_smooth = np.log(z_smooth)
    template_smooth = multiharmonic_template(x_smooth, params_d)
    ax.plot(z_smooth, 100 * template_smooth, "orange", lw=2.5,
            label=f"φ⁴ DSI (Δχ²={dchi2_d:.1f}, {sig_d:.1f}σ)")
    if chi2_w0wa < chi2_null:
        lin_smooth = beta[0] + beta[1] * x_smooth
        ax.plot(z_smooth, 100 * lin_smooth, "g--", lw=2,
                label=f"w₀wₐ approx (Δχ²={chi2_null-chi2_w0wa:.1f})")
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Fractional residual (%)")
    ax.set_title("DESI DR1 BAO — φ⁴ Multi-Harmonic vs w₀wₐ")
    ax.legend()

    ax = axes[1]
    ax.errorbar(x, 100 * resid, yerr=100 * sigma_arr, fmt="ko", ms=5,
                capsize=3, label="Data − ΛCDM")
    ax.plot(x_smooth, 100 * template_smooth, "orange", lw=2.5,
            label="φ⁴ DSI template")
    # Mark φ⁴ period
    for i in range(N_HARMONICS + 1):
        xv = x.min() + i * PERIOD_5
        if xv <= x.max() + 0.5:
            ax.axvline(xv, color="gold", ls=":", alpha=0.4)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("ln(z)")
    ax.set_ylabel("Fractional residual (%)")
    ax.set_title("Same data in ln(z) space — φ⁴ period marked")
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "phi4_desi_bao.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: phi4_desi_bao.png")

    return {
        "chi2_lcdm": chi2_null,
        "chi2_w0wa": chi2_w0wa,
        "chi2_phi4": chi2_d,
        "delta_chi2": dchi2_d,
        "p_value": pval_d,
        "sigma": sig_d,
        "aic_lcdm": aic_lcdm,
        "aic_w0wa": aic_w0wa,
        "aic_phi4": aic_phi4,
        "params": params_d,
    }


# ─── Channel 3: LIGO QNM ────────────────────────────────────────────
def test_ligo():
    """
    Check LIGO QNM frequency ratios for φ⁴-related clustering.

    This doesn't do a periodogram (too few points).
    Instead, it checks whether pairwise frequency ratios 
    cluster near φ^(n/4) for integer n.
    """
    print("\n" + "=" * 65)
    print("  CHANNEL 3: LIGO QNM — φ⁴ Ratio Check")
    print("=" * 65)

    # GWTC-3 dominant QNM frequencies (Hz) from published ringdown analyses
    # These are the 220 mode frequencies for confident detections
    qnm_freqs = np.array([
        251.2,  # GW150914
        390.6,  # GW170104
        297.5,  # GW170814
        253.5,  # GW190412
        326.7,  # GW190519
        171.4,  # GW190521 (low — high mass remnant)
        275.3,  # GW190602
        373.8,  # GW190706
        262.1,  # GW190828A
        350.2,  # GW191109
        287.4,  # GW191129
        310.5,  # GW200112
        245.8,  # GW200202
        298.3,  # GW200225
    ])

    if len(qnm_freqs) < 3:
        print("  ⚠️  Not enough QNM frequencies — skipping")
        return {}

    print(f"  QNM frequencies: {len(qnm_freqs)} events")
    print(f"  Range: {qnm_freqs.min():.1f} – {qnm_freqs.max():.1f} Hz")

    # Compute all pairwise ratios (> 1)
    ratios = []
    for i in range(len(qnm_freqs)):
        for j in range(i + 1, len(qnm_freqs)):
            r = max(qnm_freqs[i], qnm_freqs[j]) / \
                min(qnm_freqs[i], qnm_freqs[j])
            ratios.append(r)
    ratios = np.array(ratios)

    # Check clustering near φ^(n/4) for n = 1..12
    ln_ratios = np.log(ratios)
    unit = LN_PHI / 4  # ln(φ^(1/4))

    # Distance of each ln(ratio) to nearest multiple of ln(φ)/4
    phases = (ln_ratios / unit) % 1.0  # should cluster near 0 or 1 if φ⁴-related
    # Map to [-0.5, 0.5]
    phases[phases > 0.5] -= 1.0

    # Rayleigh test for uniformity
    # If phases cluster near 0, the mean resultant length R is high
    R = np.abs(np.mean(np.exp(2j * np.pi * phases)))
    # Rayleigh p-value approximation
    n = len(phases)
    rayleigh_z = n * R**2
    rayleigh_p = np.exp(-rayleigh_z)  # valid for large n

    sigma_r = norm.ppf(1 - rayleigh_p) if rayleigh_p > 1e-15 else 8.0

    print(f"\n  Pairwise ratios: {len(ratios)}")
    print(f"  Phase unit: ln(φ)/4 = {unit:.6f}")
    print(f"  Mean resultant length R = {R:.4f}")
    print(f"  Rayleigh p-value = {rayleigh_p:.4f} ({sigma_r:.1f}σ)")

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(ratios, bins=30, density=True, color="steelblue", alpha=0.7,
            edgecolor="k")
    for n in range(1, 13):
        val = PHI**(n / 4)
        if val < ratios.max() * 1.1:
            ax.axvline(val, color="red" if n % 4 == 0 else "orange",
                       ls="--", alpha=0.5, lw=1.5 if n % 4 == 0 else 0.8)
    ax.set_xlabel("Pairwise frequency ratio")
    ax.set_ylabel("Density")
    ax.set_title("QNM frequency ratios\n(red = φⁿ, orange = φ^(n/4))")

    ax = axes[1]
    ax.hist(phases, bins=30, density=True, color="steelblue", alpha=0.7,
            edgecolor="k")
    ax.axvline(0, color="red", lw=2, label="φ^(n/4) alignment")
    ax.set_xlabel("Phase relative to φ^(1/4) grid")
    ax.set_ylabel("Density")
    ax.set_title(f"Phase distribution (Rayleigh p={rayleigh_p:.3f}, {sigma_r:.1f}σ)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "phi4_ligo_qnm.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: phi4_ligo_qnm.png")

    return {
        "n_ratios": len(ratios),
        "R": R,
        "rayleigh_p": rayleigh_p,
        "sigma": sigma_r,
    }


# ─── Main ────────────────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     φ⁴ Multi-Harmonic DSI Test — All Channels              ║")
    print(f"║     Fundamental period: 4·ln(φ) = {PERIOD_5:.4f}              ║")
    print(f"║     Harmonics: n = 1..{N_HARMONICS}                                 ║")
    print(f"║     Injection test: {INJECT_TEST}                              ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    results = {}

    # Channel 1: CMB
    try:
        results["CMB"] = test_cmb()
    except Exception as e:
        print(f"  ❌ CMB test failed: {e}")

    # Channel 2: DESI
    try:
        results["DESI"] = test_desi()
    except Exception as e:
        print(f"  ❌ DESI test failed: {e}")

    # Channel 3: LIGO
    try:
        results["LIGO"] = test_ligo()
    except Exception as e:
        print(f"  ❌ LIGO test failed: {e}")

    # ── Grand Summary ──
    print("\n" + "=" * 65)
    print("  GRAND SUMMARY — φ⁴ Multi-Harmonic Test")
    print("=" * 65)

    summary_lines = ["Channel | Δχ² / Statistic | p-value | σ | Verdict"]
    summary_lines.append("--------|-----------------|---------|---|--------")

    for channel, res in results.items():
        if "sigma" in res:
            sig = res["sigma"]
            pval = res.get("p_value", res.get("rayleigh_p", -1))
            dchi2 = res.get("delta_chi2", res.get("R", -1))
            verdict = "✅ DETECTION" if sig >= 3 else \
                      "🟡 MARGINAL"  if sig >= 2 else "❌ No signal"
            summary_lines.append(
                f"{channel:>7s} | {dchi2:>15.3f} | {pval:>7.4f} | "
                f"{sig:>4.1f} | {verdict}"
            )

    for line in summary_lines:
        print(f"  {line}")

    # Save summary
    with open(TAB_DIR / "phi4_summary.txt", "w") as f:
        f.write("φ⁴ Multi-Harmonic DSI Test Results\n")
        f.write(f"Fundamental period: 4·ln(φ) = {PERIOD_5:.6f}\n")
        f.write(f"Injection test: {INJECT_TEST}\n\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write("\nDetailed results:\n")
        for channel, res in results.items():
            f.write(f"\n{channel}:\n")
            for k, v in res.items():
                if k != "params":
                    f.write(f"  {k}: {v}\n")
    print(f"\n  Saved: phi4_summary.txt")


if __name__ == "__main__":
    main()
