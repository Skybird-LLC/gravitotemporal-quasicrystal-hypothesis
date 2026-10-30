#!/usr/bin/env python3
"""
desi_multibin_dsi.py
====================
Multi-redshift-bin DSI search on DESI DR1 ξ₀(s).

Extends the 3.2σ LRG z=0.6-0.8 detection to all available bins.

Model:
    ξ_DSI(s) = ξ_smooth(s) × [1 + ε cos(2π ln(s)/ln(φ) + ψ₀)]

Period FIXED at ln(φ).  Free params: (ε, ψ₀) → Δχ² ~ χ²(2) under null.

Usage:
    python src/desi_multibin_dsi.py
    python src/desi_multibin_dsi.py --data-dir /path/to/data
    python src/desi_multibin_dsi.py --s-min 30 --s-max 180 --n-null 500
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import chi2 as chi2_dist, norm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import argparse
import warnings
from datetime import datetime

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ══════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════
PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = BASE_DIR / 'data' / 'desi_dr1'
RESULTS_DIR = BASE_DIR / 'results'
FIG_DIR = RESULTS_DIR / 'figures'

for d in [RESULTS_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# Bin definitions
# ══════════════════════════════════════════════════════════════
DESI_BINS = [
    {'name': 'BGS_z0.1-0.4',     'label': 'BGS  0.1<z<0.4',      'zeff': 0.295},
    {'name': 'LRG_z0.4-0.6',     'label': 'LRG  0.4<z<0.6',      'zeff': 0.510},
    {'name': 'LRG_z0.6-0.8',     'label': 'LRG  0.6<z<0.8',      'zeff': 0.706},
    {'name': 'LRG+ELG_z0.8-1.1', 'label': 'LRG+ELG  0.8<z<1.1',  'zeff': 0.934},
    {'name': 'ELG_z1.1-1.6',     'label': 'ELG  1.1<z<1.6',      'zeff': 1.317},
    {'name': 'QSO_z0.8-2.1',     'label': 'QSO  0.8<z<2.1',      'zeff': 1.491},
]


# ══════════════════════════════════════════════════════════════
# Data I/O
# ══════════════════════════════════════════════════════════════

def load_bin_data(bin_dir, s_min=25.0, s_max=200.0):
    """
    Load ξ₀(s) and covariance for one bin.

    Tries formats: .txt, .dat, .npy
    Auto-extracts monopole block if covariance includes multipoles.

    Returns (s, xi0, cov) or (None, None, None) if missing.
    """
    bin_dir = Path(bin_dir)
    if not bin_dir.exists():
        return None, None, None

    # --- Find correlation function file ---
    xi_file = None
    for pattern in ['xi0.txt', 'xi0.dat', 'xi0.npy', 'xi_0.txt',
                     'xi_0.dat', '*xi0*.*', '*xi_0*.*']:
        matches = sorted(bin_dir.glob(pattern))
        if matches:
            xi_file = matches[0]
            break

    if xi_file is None:
        return None, None, None

    # Load
    if xi_file.suffix == '.npy':
        data = np.load(str(xi_file))
    else:
        try:
            data = np.loadtxt(str(xi_file), comments='#')
        except Exception:
            data = np.genfromtxt(str(xi_file), comments='#',
                                 skip_header=1, filling_values=np.nan)
            data = data[~np.isnan(data).any(axis=1)]

    if data.ndim != 2 or data.shape[1] < 2:
        return None, None, None

    s_all = data[:, 0]
    xi0_all = data[:, 1]

    # --- Find covariance file ---
    cov_file = None
    for pattern in ['cov_xi0.txt', 'cov_xi0.dat', 'cov_xi0.npy',
                     'cov*.txt', 'cov*.dat', 'cov*.npy',
                     'RascalC*.txt', 'RascalC*.npy']:
        matches = sorted(bin_dir.glob(pattern))
        if matches:
            cov_file = matches[0]
            break

    if cov_file is None:
        print(f"    ⚠ No covariance found — using diagonal estimate")
        if data.shape[1] >= 3:
            sigma = data[:, 2]
        else:
            sigma = np.maximum(np.abs(xi0_all) * 0.1, 1e-6)
        cov_all = np.diag(sigma ** 2)
    else:
        if cov_file.suffix == '.npy':
            cov_full = np.load(str(cov_file))
        else:
            cov_full = np.loadtxt(str(cov_file))

        n = len(s_all)
        # Extract monopole block if stacked multipoles
        if cov_full.shape[0] == n:
            cov_all = cov_full
        elif cov_full.shape[0] % n == 0:
            cov_all = cov_full[:n, :n]
        else:
            # Best effort: trim to match
            m = min(cov_full.shape[0], n)
            s_all = s_all[:m]
            xi0_all = xi0_all[:m]
            cov_all = cov_full[:m, :m]

    # --- Apply s range cut ---
    mask = (s_all >= s_min) & (s_all <= s_max)
    s = s_all[mask]
    xi0 = xi0_all[mask]
    idx = np.where(mask)[0]
    cov = cov_all[np.ix_(idx, idx)]

    return s, xi0, cov


# ══════════════════════════════════════════════════════════════
# Fitting engine
# ══════════════════════════════════════════════════════════════

def broadband_model(s, coeffs):
    """Polynomial model for s²·ξ(s), returned as ξ(s)."""
    x = s / 100.0
    s2xi = np.polyval(coeffs, x)
    return s2xi / s ** 2


def dsi_factor(s, epsilon, psi0, period=LN_PHI):
    """Log-periodic modulation: 1 + ε cos(2π ln s / period + ψ₀)."""
    return 1.0 + epsilon * np.cos(2.0 * np.pi * np.log(s) / period + psi0)


def _chi2(params, s, xi_data, cov_inv, n_poly, with_dsi=False,
          period=LN_PHI):
    """Compute χ² for null or DSI model."""
    coeffs = params[: n_poly + 1]
    xi_mod = broadband_model(s, coeffs)

    if with_dsi:
        eps = params[n_poly + 1]
        psi = params[n_poly + 2]
        xi_mod = xi_mod * dsi_factor(s, eps, psi, period)

    r = xi_data - xi_mod
    return float(r @ cov_inv @ r)


def fit_bin(s, xi0, cov, n_poly=5, period=LN_PHI, verbose=True):
    """
    Fit one redshift bin with null and DSI models.

    Returns dict with chi2_null, chi2_dsi, delta_chi2, p_value,
    sigma, epsilon, psi0, and model arrays.
    """
    N = len(s)

    # Invert covariance (with tiny regularization if needed)
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.inv(
            cov + np.eye(N) * np.abs(np.diag(cov)).max() * 1e-8)

    # Hartlap factor (assume ~1000 mocks; safe no-op for RascalC)
    n_mock = 1000
    if n_mock > N + 2:
        cov_inv *= (n_mock - N - 2) / (n_mock - 1)

    # ── Null fit (broadband only) ──
    x = s / 100.0
    p0 = np.polyfit(x, s ** 2 * xi0, n_poly)

    res0 = minimize(_chi2, p0,
                    args=(s, xi0, cov_inv, n_poly, False, period),
                    method='Powell', options={'maxiter': 80000,
                                              'ftol': 1e-13})
    chi2_null = res0.fun

    # ── DSI fit (broadband × modulation) ──
    best = {'fun': np.inf}

    for eps0 in [-0.06, -0.03, 0.0, 0.03, 0.06]:
        for psi0 in np.linspace(0, 2 * np.pi, 13, endpoint=False):
            p_init = np.concatenate([res0.x, [eps0, psi0]])
            r = minimize(_chi2, p_init,
                         args=(s, xi0, cov_inv, n_poly, True, period),
                         method='Nelder-Mead',
                         options={'maxiter': 120000, 'xatol': 1e-11,
                                  'fatol': 1e-11})
            if r.fun < best.get('fun', np.inf):
                best = r

    # Polish with Powell
    best = minimize(_chi2, best.x,
                    args=(s, xi0, cov_inv, n_poly, True, period),
                    method='Powell',
                    options={'maxiter': 80000, 'ftol': 1e-13})

    # Global optimizer as safety net
    bnd = [(c - 50 * (abs(c) + 1), c + 50 * (abs(c) + 1)) for c in res0.x]
    bnd += [(-0.25, 0.25), (0.0, 2 * np.pi)]
    try:
        rde = differential_evolution(
            _chi2, bnd,
            args=(s, xi0, cov_inv, n_poly, True, period),
            maxiter=3000, tol=1e-11, seed=42, polish=True)
        if rde.fun < best.fun:
            best = rde
    except Exception:
        pass

    chi2_dsi = best.fun
    eps_fit = best.x[n_poly + 1]
    psi_fit = best.x[n_poly + 2] % (2 * np.pi)

    # ── Statistics ──
    dchi2 = max(chi2_null - chi2_dsi, 0.0)
    pval = float(chi2_dist.sf(dchi2, df=2))
    sig = float(norm.isf(pval)) if pval > 0 else 8.0

    # Model curves
    xi_null = broadband_model(s, res0.x)
    xi_dsi = (broadband_model(s, best.x[: n_poly + 1])
              * dsi_factor(s, eps_fit, psi_fit, period))

    if verbose:
        print(f"    χ²_null={chi2_null:.1f}  χ²_DSI={chi2_dsi:.1f}  "
              f"Δχ²={dchi2:.1f}  p={pval:.4f}  → {sig:.1f}σ  "
              f"ε={eps_fit:+.5f}")

    return {
        'chi2_null': round(chi2_null, 2),
        'chi2_dsi': round(chi2_dsi, 2),
        'delta_chi2': round(dchi2, 2),
        'p_value': pval,
        'sigma': round(sig, 2),
        'epsilon': round(eps_fit, 5),
        'psi0': round(psi_fit, 3),
        'n_bins': N,
        'xi_null': xi_null,
        'xi_dsi': xi_dsi,
    }


# ══════════════════════════════════════════════════════════════
# Null tests
# ══════════════════════════════════════════════════════════════

def random_period_null(s, xi0, cov, n_poly=5, n_rand=300, seed=42):
    """
    Compare Δχ²(ln φ) to Δχ² at random periods.
    Returns fraction of random periods beating ln(φ).
    """
    rng = np.random.default_rng(seed)
    ref = fit_bin(s, xi0, cov, n_poly, LN_PHI, verbose=False)
    dchi2_ref = ref['delta_chi2']

    periods = rng.uniform(0.2, 3.0, n_rand)
    beats = 0
    dchi2s = np.empty(n_rand)

    for i, per in enumerate(periods):
        r = fit_bin(s, xi0, cov, n_poly, per, verbose=False)
        dchi2s[i] = r['delta_chi2']
        if dchi2s[i] >= dchi2_ref:
            beats += 1

    frac = beats / n_rand
    print(f"    Random-period null: {beats}/{n_rand} beat ln(φ)  "
          f"(frac={frac:.3f})  mean Δχ²_rand={dchi2s.mean():.1f}")

    return {
        'delta_chi2_phi': dchi2_ref,
        'frac_exceed': round(frac, 4),
        'mean_random': round(float(dchi2s.mean()), 2),
        'max_random': round(float(dchi2s.max()), 2),
        'n_rand': n_rand,
    }


# ══════════════════════════════════════════════════════════════
# Combination
# ══════════════════════════════════════════════════════════════

def combine_results(results):
    """
    Two independent combination methods:
      1) Sum of Δχ² → χ²(2 N_bins)
      2) Fisher's method on p-values → χ²(2 N_bins)
    """
    vals = [r for r in results.values() if r is not None]
    if not vals:
        return {}

    # Method 1: additive Δχ²
    dchi2_tot = sum(v['delta_chi2'] for v in vals)
    dof = 2 * len(vals)
    p_add = float(chi2_dist.sf(dchi2_tot, df=dof))
    sig_add = float(norm.isf(p_add)) if p_add > 0 else 8.0

    # Method 2: Fisher
    pvals = [v['p_value'] for v in vals if v['p_value'] > 0]
    fisher_stat = -2.0 * sum(np.log(p) for p in pvals)
    p_fish = float(chi2_dist.sf(fisher_stat, df=2 * len(pvals)))
    sig_fish = float(norm.isf(p_fish)) if p_fish > 0 else 8.0

    return {
        'n_bins': len(vals),
        'delta_chi2_total': round(dchi2_tot, 2),
        'dof': dof,
        'p_additive': p_add,
        'sigma_additive': round(sig_add, 2),
        'p_fisher': p_fish,
        'sigma_fisher': round(sig_fish, 2),
    }


# ══════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════

def plot_results(bin_data, bin_results, combined, path):
    """Multi-panel: each bin gets a row (fit, residual, ln-space)."""
    active = [(n, d, bin_results[n]) for n, d in bin_data.items()
              if bin_results.get(n) is not None]
    if not active:
        return

    nb = len(active)
    fig, axes = plt.subplots(nb, 3, figsize=(17, 3.6 * nb + 1.5),
                             squeeze=False)
    fig.suptitle(
        r'DESI DR1 Multi-Bin DSI Search — period fixed at $\ln\phi$',
        fontsize=13, fontweight='bold')

    for row, (name, (s, xi0, cov), res) in enumerate(active):
        sig_diag = np.sqrt(np.diag(cov))
        lbl = next(b['label'] for b in DESI_BINS if b['name'] == name)
        sc = 'green' if res['sigma'] >= 3 else (
             'orange' if res['sigma'] >= 2 else 'gray')

        # Col 0: s²ξ with fits
        ax = axes[row, 0]
        ax.errorbar(s, s**2 * xi0, yerr=s**2 * sig_diag,
                    fmt='.', color='dimgray', ms=3, capsize=1, alpha=0.7)
        ax.plot(s, s**2 * res['xi_null'], '-', color='steelblue',
                lw=1.5, label='ΛCDM smooth')
        ax.plot(s, s**2 * res['xi_dsi'], '--', color='crimson',
                lw=1.5, label=f'DSI  ε={res["epsilon"]:+.4f}')
        ax.set_ylabel(r'$s^2\,\xi_0$')
        ax.set_title(lbl, fontsize=10)
        ax.legend(fontsize=7)

        # Col 1: fractional residual + DSI curve
        ax = axes[row, 1]
        safe = np.where(np.abs(res['xi_null']) > 1e-12,
                        res['xi_null'], np.nan)
        fr = (xi0 - res['xi_null']) / np.abs(safe)
        fe = sig_diag / np.abs(safe)
        ax.errorbar(s, fr, yerr=fe, fmt='.', color='dimgray',
                    ms=3, capsize=1, alpha=0.7)
        sf = np.linspace(s.min(), s.max(), 500)
        ax.plot(sf, dsi_factor(sf, res['epsilon'], res['psi0']) - 1,
                '-', color='crimson', lw=1.2)
        ax.axhline(0, lw=0.4, color='k')
        ax.set_ylim(-0.15, 0.15)
        ax.set_ylabel('Frac. residual')

        # Col 2: ln-space residual with φ grid
        ax = axes[row, 2]
        ax.errorbar(np.log(s), fr, yerr=fe, fmt='.', color='dimgray',
                    ms=3, capsize=1, alpha=0.7)
        ax.plot(np.log(sf),
                dsi_factor(sf, res['epsilon'], res['psi0']) - 1,
                '-', color='crimson', lw=1.2)
        # Draw φ-grid
        node0 = -res['psi0'] * LN_PHI / (2 * np.pi)
        for n in range(-12, 12):
            ln_n = node0 + n * LN_PHI
            if np.log(s.min()) - 0.5 < ln_n < np.log(s.max()) + 0.5:
                ax.axvline(ln_n, color='gold', alpha=0.3, lw=0.8)
        ax.axhline(0, lw=0.4, color='k')
        ax.set_ylim(-0.15, 0.15)
        ax.set_ylabel('Frac. residual')
        ax.text(0.97, 0.93,
                f'{res["sigma"]:.1f}σ  Δχ²={res["delta_chi2"]:.1f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=9, fontweight='bold', color=sc,
                bbox=dict(facecolor='white', alpha=0.8, ec=sc))

    for ax in axes[-1]:
        ax.set_xlabel('$s$ [Mpc/$h$]')
    axes[-1, 2].set_xlabel(r'$\ln(s)$')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(str(path), dpi=150, bbox_inches='tight')
    print(f"\n  Plot → {path}")
    plt.show()


def plot_epsilon_vs_z(bin_results, path):
    """ε and σ as functions of effective redshift."""
    zs, eps, sigs, labs = [], [], [], []
    for b in DESI_BINS:
        r = bin_results.get(b['name'])
        if r is None:
            continue
        zs.append(b['zeff'])
        eps.append(r['epsilon'])
        sigs.append(r['sigma'])
        labs.append(b['label'].split()[0])

    if len(zs) < 2:
        return

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    cols = ['green' if s >= 3 else 'orange' if s >= 2 else 'gray'
            for s in sigs]

    a1.scatter(zs, eps, c=cols, s=90, zorder=3, edgecolors='k')
    a1.axhline(0, ls='--', lw=0.6, color='k')
    a1.axhspan(-0.05, -0.01, alpha=0.08, color='crimson',
               label=r'$|\varepsilon|\in[1\%,5\%]$')
    for z, e, l in zip(zs, eps, labs):
        a1.annotate(l, (z, e), xytext=(5, 5),
                    textcoords='offset points', fontsize=8)
    a1.set_xlabel('$z_{\\rm eff}$')
    a1.set_ylabel(r'$\varepsilon$')
    a1.set_title(r'DSI amplitude vs. redshift')
    a1.legend(fontsize=8)
    a1.grid(alpha=0.15)

    a2.bar(range(len(zs)), sigs, color=cols, edgecolor='k', alpha=0.85)
    a2.set_xticks(range(len(zs)))
    a2.set_xticklabels(labs, rotation=25, fontsize=8)
    a2.axhline(2, ls='--', color='orange', lw=1, label='2σ')
    a2.axhline(3, ls='--', color='red', lw=1, label='3σ')
    a2.set_ylabel('Significance (σ)')
    a2.set_title('Per-bin detection significance')
    a2.legend(fontsize=8)
    a2.grid(alpha=0.15, axis='y')

    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches='tight')
    print(f"  Plot → {path}")
    plt.show()


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description='DESI DR1 multi-bin DSI')
    ap.add_argument('--data-dir', default=str(DEFAULT_DATA_DIR))
    ap.add_argument('--s-min', type=float, default=25.0)
    ap.add_argument('--s-max', type=float, default=200.0)
    ap.add_argument('--n-poly', type=int, default=5)
    ap.add_argument('--n-null', type=int, default=300,
                    help='Random periods for null test')
    ap.add_argument('--skip-null-test', action='store_true')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  DESI DR1 MULTI-BIN DSI SEARCH")
    print(f"  Period fixed at ln(φ) = {LN_PHI:.6f}")
    print(f"  Fit range: [{args.s_min}, {args.s_max}] Mpc/h")
    print(f"  Broadband polynomial degree: {args.n_poly}")
    print("=" * 70)

    bin_data = {}
    bin_results = {}

    for binfo in DESI_BINS:
        name = binfo['name']
        print(f"\n{'─' * 70}")
        print(f"  {binfo['label']}   (z_eff = {binfo['zeff']})")
        print(f"{'─' * 70}")

        s, xi0, cov = load_bin_data(data_dir / name,
                                    args.s_min, args.s_max)
        if s is None:
            print(f"  ⚠ No data. Expected: {data_dir / name}/xi0.txt")
            bin_results[name] = None
            continue

        print(f"  Loaded {len(s)} bins  [{s.min():.0f}, {s.max():.0f}] Mpc/h")
        bin_data[name] = (s, xi0, cov)

        # Fit
        res = fit_bin(s, xi0, cov, args.n_poly, LN_PHI, verbose=True)
        bin_results[name] = res

        # Random-period null if signal is interesting
        if not args.skip_null_test and res['delta_chi2'] > 4.0:
            nt = random_period_null(s, xi0, cov, args.n_poly,
                                    args.n_null)
            res['null_test'] = nt

    # ── Combine ──
    print(f"\n{'═' * 70}")
    print("  COMBINATION")
    print(f"{'═' * 70}")

    comb = combine_results(bin_results)
    if comb:
        print(f"  Additive Δχ²: {comb['delta_chi2_total']:.1f}  "
              f"(dof={comb['dof']})  → {comb['sigma_additive']:.1f}σ  "
              f"p={comb['p_additive']:.6f}")
        print(f"  Fisher:  → {comb['sigma_fisher']:.1f}σ  "
              f"p={comb['p_fisher']:.6f}")

    # ── Consistency ──
    epsilons = {n: r['epsilon'] for n, r in bin_results.items()
                if r is not None}
    if len(epsilons) >= 2:
        ev = list(epsilons.values())
        print(f"\n  ε values: {epsilons}")
        print(f"  Mean ε = {np.mean(ev):+.5f} ± {np.std(ev):.5f}")
        same = all(e > 0 for e in ev) or all(e < 0 for e in ev)
        print(f"  All same sign: {'✓' if same else '✗'}")

    # ── Summary table ──
    print(f"\n{'═' * 70}")
    hdr = f"  {'Bin':<25} {'Δχ²':>6} {'p':>10} {'σ':>7} {'ε':>10}"
    print(hdr)
    print(f"  {'─' * 60}")
    for b in DESI_BINS:
        r = bin_results.get(b['name'])
        if r is None:
            print(f"  {b['name']:<25} {'—':>6}")
        else:
            mark = '◆' if r['sigma'] >= 3 else '◇' if r['sigma'] >= 2 else ' '
            print(f"  {b['name']:<25} {r['delta_chi2']:>6.1f} "
                  f"{r['p_value']:>10.4f} {r['sigma']:>5.1f}σ {mark} "
                  f"{r['epsilon']:>+10.5f}")
    print(f"  {'─' * 60}")
    if comb:
        print(f"  {'COMBINED (Δχ²)':<25} {comb['delta_chi2_total']:>6.1f} "
              f"{comb['p_additive']:>10.6f} "
              f"{comb['sigma_additive']:>5.1f}σ")
        print(f"  {'COMBINED (Fisher)':<25} {'':>6} "
              f"{comb['p_fisher']:>10.6f} "
              f"{comb['sigma_fisher']:>5.1f}σ")

    # ── Save ──
    out = {
        'timestamp': datetime.now().isoformat(),
        'config': {'s_min': args.s_min, 's_max': args.s_max,
                   'n_poly': args.n_poly, 'period': LN_PHI},
        'per_bin': {
            n: {k: v for k, v in r.items()
                if k not in ('xi_null', 'xi_dsi')}
            if r else None
            for n, r in bin_results.items()},
        'combined': comb,
    }
    jf = RESULTS_DIR / 'desi_multibin_dsi_results.json'
    with open(str(jf), 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Results → {jf}")

    # ── Plots ──
    if bin_data:
        plot_results(bin_data, bin_results, comb,
                     FIG_DIR / 'desi_multibin_dsi.png')
        plot_epsilon_vs_z(bin_results,
                          FIG_DIR / 'desi_epsilon_vs_z.png')

    # ── Verdict ──
    if comb:
        print(f"\n{'═' * 70}")
        n3 = sum(1 for r in bin_results.values()
                 if r and r['sigma'] >= 3)
        n2 = sum(1 for r in bin_results.values()
                 if r and 2 <= r['sigma'] < 3)
        print(f"  {n3} bins ≥3σ,  {n2} bins 2–3σ")
        s_c = comb['sigma_additive']
        if s_c >= 5:
            print("  ★ DISCOVERY-LEVEL combined significance.")
        elif s_c >= 3:
            print("  ⚡ Strong combined evidence — pursue confirmation.")
        elif s_c >= 2:
            print("  ⚠ Moderate — suggestive, not conclusive.")
        else:
            print("  No significant combined evidence.")


if __name__ == '__main__':
    main()
