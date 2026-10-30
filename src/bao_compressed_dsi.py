#!/usr/bin/env python3
"""
bao_compressed_dsi.py
Test for φ-scaling in DESI DR1 compressed BAO distances.

Uses Cobaya-format files:
  desi_2024_gaussian_bao_ALL_GCcomb_mean.txt
  desi_2024_gaussian_bao_ALL_GCcomb_cov.txt
"""

import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import json

# ── Configuration ──────────────────────────────────────────
DATA_DIR  = Path("data/desi")
BASE_DIR  = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIG_DIR   = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"

for d in [RESULTS_DIR, FIG_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)
PHI       = (1 + np.sqrt(5)) / 2
LN_PHI    = np.log(PHI)

# ── Fiducial ΛCDM (Planck 2018 best-fit) ──────────────────
# Used to compute theory predictions for DM/rs, DH/rs, DV/rs
from scipy.integrate import quad

# Planck 2018 best-fit parameters
Om   = 0.3153
Ode  = 1 - Om
H0   = 67.36   # km/s/Mpc
rd   = 147.09  # Mpc, sound horizon at drag epoch
c_km = 299792.458

def E(z):
    """Hubble parameter E(z) = H(z)/H0 for flat ΛCDM."""
    return np.sqrt(Om * (1+z)**3 + Ode)

def DM_over_rs(z):
    """Comoving angular diameter distance / rs."""
    integral, _ = quad(lambda zp: 1.0/E(zp), 0, z)
    DM = (c_km / H0) * integral  # Mpc
    return DM / rd

def DH_over_rs(z):
    """Hubble distance / rs."""
    return (c_km / H0) / (E(z) * rd)

def DV_over_rs(z):
    """Volume-averaged distance / rs."""
    dm = DM_over_rs(z) * rd
    dh = DH_over_rs(z) * rd
    DV = (z * dm**2 * dh) ** (1./3.)
    return DV / rd


# ── Load DESI data ─────────────────────────────────────────
def load_compressed_bao():
    """Load the ALL_GCcomb combined data vector and covariance."""
    mean_file = DATA_DIR / "desi_2024_gaussian_bao_ALL_GCcomb_mean.txt"
    cov_file  = DATA_DIR / "desi_2024_gaussian_bao_ALL_GCcomb_cov.txt"

    # Parse mean file: columns are z, value, quantity_name
    redshifts = []
    values    = []
    quantities = []

    with open(mean_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            redshifts.append(float(parts[0]))
            values.append(float(parts[1]))
            quantities.append(parts[2])

    data = {
        'z':     np.array(redshifts),
        'value': np.array(values),
        'qty':   quantities,
    }

    # Load covariance
    cov = np.loadtxt(cov_file)
    data['cov']   = cov
    data['sigma']  = np.sqrt(np.diag(cov))
    data['icov']   = np.linalg.inv(cov)

    return data


# ── ΛCDM predictions ──────────────────────────────────────
def lcdm_predictions(data):
    """Compute flat ΛCDM predictions for each data point."""
    pred = np.zeros(len(data['z']))
    for i, (z, qty) in enumerate(zip(data['z'], data['qty'])):
        if 'DM' in qty:
            pred[i] = DM_over_rs(z)
        elif 'DH' in qty:
            pred[i] = DH_over_rs(z)
        elif 'DV' in qty:
            pred[i] = DV_over_rs(z)
    return pred


# ── DSI modulation model ──────────────────────────────────
def dsi_modulation(z, eps, z0, psi0):
    """
    Log-periodic fractional modulation:
      δD/D_ΛCDM = ε · cos(2π ln(z/z0) / ln(φ) + ψ0)
    """
    return eps * np.cos(2 * np.pi * np.log(z / z0) / LN_PHI + psi0)


def chi2_null(data, pred):
    """χ² for pure ΛCDM (no modulation)."""
    residual = data['value'] - pred
    return residual @ data['icov'] @ residual


def chi2_dsi(params, data, pred):
    """χ² for ΛCDM + DSI modulation."""
    eps, z0, psi0 = params
    mod = 1.0 + dsi_modulation(data['z'], eps, z0, psi0)
    residual = data['value'] - pred * mod
    return residual @ data['icov'] @ residual


# ── Main analysis ─────────────────────────────────────────
def main():
    print("=" * 60)
    print("  DESI DR1 COMPRESSED BAO — φ-PERIODICITY TEST")
    print("=" * 60)

    # Load data
    data = load_compressed_bao()
    n_data = len(data['z'])
    print(f"\n  Loaded {n_data} data points across {len(set(data['z']))} redshift bins")
    print(f"  Quantities: {set(data['qty'])}")
    print(f"  Redshift range: {data['z'].min():.3f} – {data['z'].max():.3f}")

    # ΛCDM predictions
    pred = lcdm_predictions(data)

    # ── Null model (pure ΛCDM) ──
    chi2_0 = chi2_null(data, pred)
    dof_null = n_data
    print(f"\n  ΛCDM χ² = {chi2_0:.2f}  (dof = {dof_null})")
    print(f"  Reduced χ²/dof = {chi2_0/dof_null:.3f}")

    # ── DSI model: grid search + optimize ──
    print("\n  Fitting DSI modulation (ε, z₀, ψ₀)...")

    best_chi2 = chi2_0
    best_params = None

    # Grid over z0 and psi0, then refine
    for z0_try in np.linspace(0.1, 2.0, 20):
        for psi0_try in np.linspace(0, 2*np.pi, 12):
            for eps_try in [0.005, 0.01, 0.02, 0.05]:
                x0 = [eps_try, z0_try, psi0_try]
                try:
                    res = minimize(chi2_dsi, x0, args=(data, pred),
                                   method='Nelder-Mead',
                                   options={'maxiter': 5000, 'xatol': 1e-8})
                    if res.fun < best_chi2:
                        best_chi2 = res.fun
                        best_params = res.x
                except:
                    pass

    if best_params is not None:
        eps_fit, z0_fit, psi0_fit = best_params
        dof_dsi = n_data - 3
        delta_chi2 = chi2_0 - best_chi2

        # Significance: Δχ² follows χ²(3) for 3 extra parameters
        from scipy.stats import chi2 as chi2_dist
        p_value = 1 - chi2_dist.cdf(delta_chi2, df=3)
        n_sigma = chi2_dist.isf(p_value, df=1)**0.5 if p_value < 0.5 else 0.0

        print(f"\n  ── DSI Fit Results ──")
        print(f"  ε       = {eps_fit:.4f}")
        print(f"  z₀      = {z0_fit:.4f}")
        print(f"  ψ₀      = {psi0_fit:.4f} rad")
        print(f"  χ²(DSI) = {best_chi2:.2f}  (dof = {dof_dsi})")
        print(f"  Δχ²     = {delta_chi2:.2f}")
        print(f"  p-value = {p_value:.4f}")
        print(f"  ~{n_sigma:.1f}σ")
    else:
        print("  No improvement over ΛCDM found.")
        eps_fit = z0_fit = psi0_fit = 0.0
        delta_chi2 = 0.0
        p_value = 1.0
        n_sigma = 0.0
        best_chi2 = chi2_0

    # ── Plot ──
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])

    z_unique = sorted(set(data['z']))

    # Top: data vs ΛCDM vs DSI
    ax = axes[0]
    # Plot DV/rs points
    for qty_type, marker, color in [('DV_over_rs', 'o', 'royalblue'),
                                      ('DM_over_rs', 's', 'forestgreen'),
                                      ('DH_over_rs', '^', 'firebrick')]:
        mask = [q == qty_type for q in data['qty']]
        if any(mask):
            mask = np.array(mask)
            ax.errorbar(data['z'][mask], data['value'][mask], 
                       yerr=data['sigma'][mask],
                       fmt=marker, color=color, label=qty_type.replace('_', '/'),
                       capsize=3, markersize=6)

    # Theory curves
    z_fine = np.linspace(0.05, 2.5, 200)
    ax.plot(z_fine, [DV_over_rs(z) for z in z_fine], 'b--', alpha=0.5, label='ΛCDM DV/rs')
    ax.plot(z_fine, [DM_over_rs(z) for z in z_fine], 'g--', alpha=0.5, label='ΛCDM DM/rs')
    ax.plot(z_fine, [DH_over_rs(z) for z in z_fine], 'r--', alpha=0.5, label='ΛCDM DH/rs')

    ax.set_xlabel('Redshift z')
    ax.set_ylabel('Distance / $r_s$')
    ax.set_title('DESI DR1 BAO Distances vs ΛCDM')
    ax.legend(fontsize=8)

    # Bottom: fractional residuals
    ax = axes[1]
    residuals = (data['value'] - pred) / pred
    res_err   = data['sigma'] / pred

    ax.errorbar(data['z'], residuals * 100, yerr=res_err * 100,
               fmt='ko', capsize=3, markersize=5, label='Data − ΛCDM')
    ax.axhline(0, color='gray', ls='--')

    if best_params is not None and abs(eps_fit) > 1e-6:
        z_fine = np.linspace(data['z'].min() * 0.8, data['z'].max() * 1.2, 300)
        mod_fine = dsi_modulation(z_fine, eps_fit, z0_fit, psi0_fit) * 100
        ax.plot(z_fine, mod_fine, 'goldenrod', lw=2, alpha=0.7,
                label=f'DSI fit (ε={eps_fit:.3f}, Δχ²={delta_chi2:.1f})')

    ax.set_xlabel('Redshift z')
    ax.set_ylabel('Fractional residual (%)')
    ax.set_title(f'Residuals — Δχ² = {delta_chi2:.1f}, p = {p_value:.3f} (~{n_sigma:.1f}σ)')
    ax.legend(fontsize=8)
    ax.set_ylim(-5, 5)

    plt.tight_layout()
    fig_path = FIG_DIR / 'bao_compressed_dsi.png'
    plt.savefig(fig_path, dpi=150)
    print(f"\n  Plot saved: {fig_path}")

    # ── Save results ──
    results = {
        'test': 'DESI_DR1_compressed_BAO_DSI',
        'n_data': int(n_data),
        'chi2_lcdm': float(chi2_0),
        'chi2_dsi': float(best_chi2),
        'delta_chi2': float(delta_chi2),
        'p_value': float(p_value),
        'n_sigma': float(n_sigma),
        'epsilon': float(eps_fit),
        'z0': float(z0_fit),
        'psi0': float(psi0_fit),
        'caveat': 'Only 7 redshift bins — limited frequency resolution'
    }

    results_file = TABLE_DIR / 'bao_compressed_dsi_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved: {results_file}")

    # ── Verdict ──
    print(f"\n{'='*60}")
    if delta_chi2 < 3.53:  # p > 0.317 for 3 dof
        print("  VERDICT: No significant improvement over ΛCDM")
    elif delta_chi2 < 8.02:  # ~1-2σ
        print("  VERDICT: Marginal hint — needs full ξ(s) for confirmation")
    else:
        print("  VERDICT: Significant — proceed to full ξ(s) analysis")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
