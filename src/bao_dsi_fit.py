"""
bao_dsi_fit.py
==============
Fit a log-periodic DSI modulation to the DESI DR1 BAO correlation function ξ₀(s).

Data: DESI DR1 full-shape BAO clustering VAC — LRG z=0.6–0.8
  Correlation: correlation-recsym-poles_LRG_GCcomb_z0.6-0.8.h5
  Covariance:  covariance_correlation-recsym-poles_LRG_GCcomb_z0.6-0.8.h5
  Source: https://data.desi.lbl.gov/public/dr1/vac/dr1/full-shape-bao-clustering/v1.0/

Model:
    ξ_obs(s) = ξ_ΛCDM(s) · [1 + ε·cos(2π·ln(s/s₀)/ln(φ) + ψ₀)]

Null:
    ξ_obs(s) = ξ_ΛCDM(s)     (ε = 0)

Requirements: numpy, scipy, matplotlib, h5py, emcee (optional)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
from pathlib import Path
import json
import h5py

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════
PHI    = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / 'results'
FIG_DIR = RESULTS_DIR / 'figures'
TABLE_DIR = RESULTS_DIR / 'tables'

for d in [RESULTS_DIR, FIG_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# 1. LOAD BAO DATA — DESI DR1 HDF5
# ═══════════════════════════════════════════════════════════════
DATA_DIR = BASE_DIR / "data" / "desi"

# Range limits for the fit (Mpc/h)
S_MIN = 30   # avoid small-scale nonlinearities
S_MAX = 198  # upper limit of data


def load_desi_bao():
    """
    Load DESI DR1 LRG z=0.6–0.8 monopole ξ₀(s) and RascalC covariance.

    The correlation file has 50 bins (s=3–198 Mpc/h).
    The covariance file has 45 bins per multipole (s=22–198), 3 multipoles
    (ℓ=0,2,4) stacked → 135×135 matrix.  We use only the monopole block
    (top-left 45×45) and restrict the data to matching s values.

    Returns: s, xi0, sigma, cov_mono
    """
    corr_file = DATA_DIR / "corr_LRG_z06_08.h5"
    cov_file  = DATA_DIR / "cov_LRG_z06_08.h5"

    if not corr_file.exists():
        raise FileNotFoundError(
            f"DESI correlation file not found: {corr_file}\n"
            f"Download from: https://data.desi.lbl.gov/public/dr1/vac/dr1/"
            f"full-shape-bao-clustering/v1.0/data/recsym/correlation/"
            f"correlation-recsym-poles_LRG_GCcomb_z0.6-0.8.h5"
        )
    if not cov_file.exists():
        raise FileNotFoundError(
            f"DESI covariance file not found: {cov_file}\n"
            f"Download from: https://data.desi.lbl.gov/public/dr1/vac/dr1/"
            f"full-shape-bao-clustering/v1.0/data/covariance/RascalC/"
            f"covariance_correlation-recsym-poles_LRG_GCcomb_z0.6-0.8.h5"
        )

    # Load monopole from correlation file
    with h5py.File(corr_file, 'r') as f:
        s_corr  = f['0/s'][:]       # 50 bins, 3–198 Mpc/h
        xi_corr = f['0/value'][:]

    # Load covariance — monopole block is top-left 45×45
    with h5py.File(cov_file, 'r') as f:
        cov_full = f['value'][:]          # 135×135
        s_cov    = f['observable/0/s'][:] # 45 bins, 22–198 Mpc/h

    n_cov = len(s_cov)   # 45
    cov_mono = cov_full[:n_cov, :n_cov]  # monopole block

    # Match: restrict correlation data to the s values in the covariance
    # (finding nearest match for each covariance s-bin)
    idx_match = []
    for sc in s_cov:
        i = np.argmin(np.abs(s_corr - sc))
        if np.abs(s_corr[i] - sc) < 1.0:  # tolerance 1 Mpc/h
            idx_match.append(i)
    idx_match = np.array(idx_match)

    s_matched  = s_corr[idx_match]
    xi_matched = xi_corr[idx_match]

    # Apply s range cut
    mask = (s_matched >= S_MIN) & (s_matched <= S_MAX)
    s_out  = s_matched[mask]
    xi_out = xi_matched[mask]

    # Corresponding covariance sub-block
    # idx_match aligned with s_cov, so mask indices into that
    cov_cut = cov_mono[np.ix_(mask, mask)]
    sigma   = np.sqrt(np.diag(cov_cut))

    print(f"  Loaded DESI DR1 LRG z=0.6–0.8 monopole ξ₀(s)")
    print(f"  s range: {s_out[0]:.0f}–{s_out[-1]:.0f} Mpc/h  ({len(s_out)} bins)")
    print(f"  Covariance: {cov_cut.shape[0]}×{cov_cut.shape[1]} (RascalC)")

    return s_out, xi_out, sigma, cov_cut


# ═══════════════════════════════════════════════════════════════
# 2. ΛCDM TEMPLATE
# ═══════════════════════════════════════════════════════════════
def get_lcdm_xi(s, s_data=None, xi_data=None):
    """
    Return a smooth ΛCDM template ξ(s).

    Uses a Savitzky-Golay filter to smooth the measured ξ₀(s),
    removing oscillations shorter than ~30 Mpc/h while preserving
    the BAO peak.  This is the null hypothesis against which we
    test for log-periodic modulation.
    """
    if s_data is not None and xi_data is not None:
        from scipy.signal import savgol_filter
        window = min(15, len(xi_data) // 3)
        if window % 2 == 0:
            window += 1
        if window < 5:
            window = 5
        xi_smooth = savgol_filter(xi_data, window_length=window, polyorder=3)
        f_interp  = interp1d(s_data, xi_smooth, kind='cubic',
                             fill_value='extrapolate')
        return f_interp(s)

    # Analytic fallback (approximate)
    xi_smooth = 1200 / s**2
    bao_bump  = 0.015 * np.exp(-0.5 * ((s - 105) / 8)**2)
    return xi_smooth + bao_bump


# ═══════════════════════════════════════════════════════════════
# 3. DSI MODEL
# ═══════════════════════════════════════════════════════════════
def dsi_modulation(s, s0, epsilon, psi0):
    """Log-periodic modulation factor: 1 + ε·cos(2π·ln(s/s₀)/ln(φ) + ψ₀)"""
    return 1.0 + epsilon * np.cos(
        2.0 * np.pi * np.log(s / s0) / LN_PHI + psi0
    )

def xi_model(s, xi_lcdm, s0, epsilon, psi0):
    """Full DSI-modulated correlation function."""
    return xi_lcdm * dsi_modulation(s, s0, epsilon, psi0)


# ═══════════════════════════════════════════════════════════════
# 4. CHI-SQUARED FITTING
# ═══════════════════════════════════════════════════════════════
def chi2(params, s, xi_obs, xi_lcdm, sigma, cov_inv=None):
    """
    χ² for DSI model. Supports diagonal errors or full covariance.
    params = [s0, epsilon, psi0]
    """
    s0, epsilon, psi0 = params

    # Physical priors
    if s0 < 5 or s0 > 500:
        return 1e12
    if abs(epsilon) > 1.0:
        return 1e12

    xi_th   = xi_model(s, xi_lcdm, s0, epsilon, psi0)
    residual = xi_obs - xi_th

    if cov_inv is not None:
        return float(residual @ cov_inv @ residual)
    else:
        return float(np.sum((residual / sigma)**2))


def chi2_null(s, xi_obs, xi_lcdm, sigma, cov_inv=None):
    """χ² for the null model (ε = 0)."""
    residual = xi_obs - xi_lcdm
    if cov_inv is not None:
        return float(residual @ cov_inv @ residual)
    else:
        return float(np.sum((residual / sigma)**2))


def fit_dsi(s, xi_obs, sigma, cov=None):
    """
    Fit the DSI model and compare to null.

    Returns: results dictionary with best-fit, χ², Δχ², p-value, etc.
    """
    # ── ΛCDM template ──
    xi_lcdm = get_lcdm_xi(s, s, xi_obs)

    # ── Covariance handling ──
    cov_inv = None
    if cov is not None:
        cov_inv = np.linalg.inv(cov)

    # ── Null χ² ──
    chi2_0 = chi2_null(s, xi_obs, xi_lcdm, sigma, cov_inv)
    nu_null = len(s)    # dof for null (no free params beyond ΛCDM)

    # ── DSI fit: global search then refine ──
    bounds = [
        (10, 400),       # s0 [Mpc/h]
        (-0.5, 0.5),    # epsilon
        (0, 2 * np.pi), # psi0
    ]

    print("  Running global optimization (differential evolution)...")
    result_de = differential_evolution(
        chi2, bounds=bounds,
        args=(s, xi_obs, xi_lcdm, sigma, cov_inv),
        seed=42, maxiter=1000, tol=1e-8, polish=True
    )

    # Refine with Nelder-Mead
    result = minimize(
        chi2, x0=result_de.x,
        args=(s, xi_obs, xi_lcdm, sigma, cov_inv),
        method='Nelder-Mead',
        options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 50000}
    )

    s0_fit, eps_fit, psi_fit = result.x
    chi2_dsi = result.fun
    nu_dsi   = len(s) - 3   # 3 free parameters

    # ── Δχ² and significance ──
    delta_chi2 = chi2_0 - chi2_dsi   # positive = DSI is better
    delta_dof  = 3                    # DSI has 3 extra params

    from scipy.stats import chi2 as chi2_dist
    # p-value: probability that Δχ² this large arises by chance with 3 extra dof
    p_value = 1 - chi2_dist.cdf(max(delta_chi2, 0), df=delta_dof)

    # Convert to sigma
    from scipy.stats import norm
    if p_value > 1e-15:
        n_sigma = norm.ppf(1 - p_value / 2)
    else:
        n_sigma = 8.0  # cap display

    # ── AIC / BIC comparison ──
    n = len(s)
    aic_null = chi2_0 + 2 * 0
    aic_dsi  = chi2_dsi + 2 * 3
    bic_null = chi2_0 + 0 * np.log(n)
    bic_dsi  = chi2_dsi + 3 * np.log(n)

    results = {
        "best_fit": {
            "s0":      float(s0_fit),
            "epsilon": float(eps_fit),
            "psi0":    float(psi_fit),
        },
        "chi2_null":   float(chi2_0),
        "chi2_dsi":    float(chi2_dsi),
        "nu_null":     int(nu_null),
        "nu_dsi":      int(nu_dsi),
        "delta_chi2":  float(delta_chi2),
        "p_value":     float(p_value),
        "n_sigma":     float(n_sigma),
        "aic_null":    float(aic_null),
        "aic_dsi":     float(aic_dsi),
        "bic_null":    float(bic_null),
        "bic_dsi":     float(bic_dsi),
        "xi_lcdm":     xi_lcdm,
    }

    return results


# ═══════════════════════════════════════════════════════════════
# 5. OPTIONAL: MCMC FOR PROPER POSTERIORS
# ═══════════════════════════════════════════════════════════════
def run_mcmc(s, xi_obs, sigma, cov=None, best_fit=None, nwalkers=32, nsteps=5000):
    """
    Run emcee MCMC to get proper posteriors on (s₀, ε, ψ₀).
    """
    try:
        import emcee
    except ImportError:
        print("  ⚠️  emcee not installed — skipping MCMC")
        return None

    xi_lcdm = get_lcdm_xi(s, s, xi_obs)
    cov_inv = np.linalg.inv(cov) if cov is not None else None

    def log_prior(theta):
        s0, eps, psi = theta
        if 10 < s0 < 400 and -0.5 < eps < 0.5 and 0 < psi < 2 * np.pi:
            return 0.0
        return -np.inf

    def log_likelihood(theta):
        return -0.5 * chi2(theta, s, xi_obs, xi_lcdm, sigma, cov_inv)

    def log_prob(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    ndim = 3
    if best_fit is not None:
        p0 = best_fit + 1e-3 * np.random.randn(nwalkers, ndim)
    else:
        p0 = np.column_stack([
            np.random.uniform(50, 200, nwalkers),
            np.random.uniform(-0.1, 0.1, nwalkers),
            np.random.uniform(0, 2 * np.pi, nwalkers),
        ])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)

    print(f"  Running MCMC: {nwalkers} walkers × {nsteps} steps ...")
    sampler.run_mcmc(p0, nsteps, progress=True)

    # Discard burn-in
    burnin  = nsteps // 2
    samples = sampler.get_chain(discard=burnin, flat=True)

    labels = [r"$s_0$", r"$\varepsilon$", r"$\psi_0$"]
    medians = np.median(samples, axis=0)
    stds    = np.std(samples, axis=0)

    print("\n  MCMC Results:")
    for lbl, med, std in zip(labels, medians, stds):
        print(f"    {lbl}: {med:.4f} ± {std:.4f}")

    return samples


# ═══════════════════════════════════════════════════════════════
# 6. PLOTTING
# ═══════════════════════════════════════════════════════════════
def plot_bao_results(s, xi_obs, sigma, results, save=True):
    """Three-panel figure: data+fits, residuals, modulation pattern."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
    fig.suptitle('DESI DR1 LRG z=0.6–0.8 — DSI Log-Periodic Modulation Test',
                 fontsize=14, fontweight='bold')

    bf      = results["best_fit"]
    xi_lcdm = results["xi_lcdm"]
    xi_dsi  = xi_model(s, xi_lcdm, bf["s0"], bf["epsilon"], bf["psi0"])

    # ── Panel 1: Data and models (s² weighting) ──
    ax = axes[0]
    ax.errorbar(s, s**2 * xi_obs, yerr=s**2 * sigma,
                fmt='o', ms=3, color='k', alpha=0.6, label='DESI DR1 data')
    ax.plot(s, s**2 * xi_lcdm, 'b-', lw=2, label=r'$\Lambda$CDM (smoothed)')
    ax.plot(s, s**2 * xi_dsi, 'r--', lw=2,
            label=rf'DSI ($\varepsilon$={bf["epsilon"]:.4f})')
    ax.set_ylabel(r'$s^2 \cdot \xi_0(s)$  [(Mpc/$h$)$^2$]')
    ax.set_xlabel(r'$s$  [Mpc/$h$]')
    ax.legend()
    ax.set_title(
        rf'$\chi^2_{{\mathrm{{null}}}} = {results["chi2_null"]:.1f}$,  '
        rf'$\chi^2_{{\mathrm{{DSI}}}} = {results["chi2_dsi"]:.1f}$,  '
        rf'$\Delta\chi^2 = {results["delta_chi2"]:.1f}$  '
        rf'({results["n_sigma"]:.1f}$\sigma$,  $p = {results["p_value"]:.4f}$)'
    )

    # ── Panel 2: Fractional residuals in s ──
    ax = axes[1]
    # Avoid division by near-zero xi_lcdm values
    safe_mask = np.abs(xi_lcdm) > 1e-6
    frac_res = np.zeros_like(xi_obs)
    frac_err = np.zeros_like(sigma)
    frac_res[safe_mask] = (xi_obs[safe_mask] - xi_lcdm[safe_mask]) / xi_lcdm[safe_mask]
    frac_err[safe_mask] = sigma[safe_mask] / np.abs(xi_lcdm[safe_mask])
    ax.errorbar(s, frac_res, yerr=frac_err,
                fmt='o', ms=3, color='k', alpha=0.6)

    s_fine = np.linspace(s.min(), s.max(), 500)
    dsi_curve = bf["epsilon"] * np.cos(
        2 * np.pi * np.log(s_fine / bf["s0"]) / LN_PHI + bf["psi0"]
    )
    ax.plot(s_fine, dsi_curve, 'r-', lw=2,
            label=rf'DSI: $\varepsilon\cos(2\pi\ln(s/s_0)/\ln\phi + \psi_0)$')
    ax.axhline(0, color='gray', ls='-', lw=0.5)
    ax.set_xlabel(r'$s$  [Mpc/$h$]')
    ax.set_ylabel(r'$(\xi - \xi_{\Lambda\mathrm{CDM}}) / \xi_{\Lambda\mathrm{CDM}}$')
    ax.legend(fontsize=9)

    # ── Panel 3: Same residuals in ln(s) ──
    ax = axes[2]
    ln_s = np.log(s)
    ax.errorbar(ln_s, frac_res, yerr=frac_err,
                fmt='o', ms=3, color='k', alpha=0.6)
    ax.plot(np.log(s_fine), dsi_curve, 'r-', lw=2)
    ax.axhline(0, color='gray', ls='-', lw=0.5)

    # Mark φ-periodic grid
    for n in range(-5, 10):
        x_grid = np.log(bf["s0"]) + n * LN_PHI
        if ln_s.min() - 0.1 < x_grid < ln_s.max() + 0.1:
            ax.axvline(x_grid, color='goldenrod', ls=':', lw=0.8, alpha=0.5)

    ax.set_xlabel(r'$\ln(s)$  [$s$ in Mpc/$h$]')
    ax.set_ylabel(r'Fractional residual')
    ax.set_title(r'Residuals in $\ln s$ space — vertical lines at $\ln s_0 + n\ln\phi$')

    plt.tight_layout()
    if save:
        fig_path = FIG_DIR / 'bao_dsi_fit.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {fig_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# 7. MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("BAO DSI Fit — Log-periodic modulation search")
    print(f"Data: DESI DR1 LRG z=0.6–0.8 ξ₀(s)")
    print(f"Model: ξ(s) = ξ_ΛCDM(s) · [1 + ε·cos(2π·ln(s/s₀)/ln(φ) + ψ₀)]")
    print(f"ln(φ) = {LN_PHI:.4f}")
    print("=" * 60)

    # Load data
    s, xi_obs, sigma, cov = load_desi_bao()

    # Fit
    results = fit_dsi(s, xi_obs, sigma, cov)

    # Report
    bf = results["best_fit"]
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Best-fit parameters:")
    print(f"    s₀      = {bf['s0']:.2f} Mpc/h")
    print(f"    ε       = {bf['epsilon']:.5f}")
    print(f"    ψ₀      = {bf['psi0']:.4f} rad")
    print(f"")
    print(f"  Null model:  χ² = {results['chi2_null']:.2f}  (ν = {results['nu_null']})")
    print(f"  DSI model:   χ² = {results['chi2_dsi']:.2f}   (ν = {results['nu_dsi']})")
    print(f"  Δχ²        = {results['delta_chi2']:.2f}")
    print(f"  p-value    = {results['p_value']:.6f}")
    print(f"  Significance: {results['n_sigma']:.1f}σ")
    print(f"")
    print(f"  AIC null = {results['aic_null']:.1f},  AIC DSI = {results['aic_dsi']:.1f}")
    print(f"  BIC null = {results['bic_null']:.1f},  BIC DSI = {results['bic_dsi']:.1f}")
    print(f"  ΔBIC = {results['bic_null'] - results['bic_dsi']:.1f}  "
          f"({'DSI preferred' if results['bic_dsi'] < results['bic_null'] else 'null preferred'})")

    # Plot
    plot_bao_results(s, xi_obs, sigma, results)

    # Save results
    output = {k: v for k, v in results.items() if k != "xi_lcdm"}
    results_file = TABLE_DIR / "bao_dsi_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {results_file}")

    # Optional MCMC
    print("\n── Optional MCMC ──")
    p0 = np.array([bf["s0"], bf["epsilon"], bf["psi0"]])
    samples = run_mcmc(s, xi_obs, sigma, cov, best_fit=p0)
