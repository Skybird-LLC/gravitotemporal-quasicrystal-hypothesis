#!/usr/bin/env python3
"""
desi_bao_dsi.py
===============
DSI search on DESI compressed Gaussian BAO distance measurements.

Compares three models on the same footing:
  1. ΛCDM(Ωm)            — 1 free parameter (baseline)
  2. DSI(Ωm, ε, ψ₀)      — 3 free parameters (log-periodic DE)
  3. w₀wₐ(Ωm, w₀, wₐ)    — 3 free parameters (generic evolving DE)

DSI dark energy equation of state:
    w(z) = -1 + ε cos(2π ln(1+z)/ln φ + ψ₀)

Integrated exactly into the dark energy density:
    ρ_DE(z)/ρ_DE(0) = exp{(3ε ln φ/2π)[sin(θ(z)) - sin(ψ₀)]}
    where θ(z) = 2π ln(1+z)/ln φ + ψ₀

Period FIXED at ln(φ). The DSI model is NOT a blind search.

Data: CobayaSampler/bao_data format (Gaussian mean + covariance).

Usage:
    python src/desi_bao_dsi.py
    python src/desi_bao_dsi.py --data-dir data/desi_dr2 --n-random 500
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
from scipy.stats import chi2 as chi2_dist, norm
from pathlib import Path
import matplotlib.pyplot as plt
import json
import argparse
from datetime import datetime
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ══════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════
PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)
C_KMS = 299792.458  # speed of light [km/s]

# Planck 2018 fiducial cosmology
H0_FID = 67.36       # km/s/Mpc
H_FID = H0_FID / 100
OM_FID = 0.3153
OBH2_FID = 0.02237

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / 'results'
FIG_DIR = RESULTS_DIR / 'figures'
TABLE_DIR = RESULTS_DIR / 'tables'
for d in [RESULTS_DIR, FIG_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# Data vector layouts
# ══════════════════════════════════════════════════════════════
# Each entry: (tracer_tag, z_eff, observable_type)
# Order matches the ALL_GCcomb files

LAYOUT_13 = [  # DR2 (13 entries)
    ('BGS',       0.295, 'DV'),
    ('LRG1',      0.510, 'DM'),
    ('LRG1',      0.510, 'DH'),
    ('LRG2',      0.706, 'DM'),
    ('LRG2',      0.706, 'DH'),
    ('LRG3+ELG1', 0.934, 'DM'),
    ('LRG3+ELG1', 0.934, 'DH'),
    ('ELG2',      1.317, 'DM'),
    ('ELG2',      1.317, 'DH'),
    ('QSO',       1.491, 'DM'),
    ('QSO',       1.491, 'DH'),
    ('Lya',       2.330, 'DM'),
    ('Lya',       2.330, 'DH'),
]

LAYOUT_12 = [  # DR1 (12 entries — QSO was DV-only)
    ('BGS',       0.295, 'DV'),
    ('LRG1',      0.510, 'DM'),
    ('LRG1',      0.510, 'DH'),
    ('LRG2',      0.706, 'DM'),
    ('LRG2',      0.706, 'DH'),
    ('LRG3+ELG1', 0.930, 'DM'),
    ('LRG3+ELG1', 0.930, 'DH'),
    ('ELG2',      1.317, 'DM'),
    ('ELG2',      1.317, 'DH'),
    ('QSO',       1.491, 'DV'),
    ('Lya',       2.330, 'DM'),
    ('Lya',       2.330, 'DH'),
]


# ══════════════════════════════════════════════════════════════
# Data I/O
# ══════════════════════════════════════════════════════════════

def load_bao_data(data_dir):
    """
    Load DESI BAO data from CobayaSampler/bao_data format.

    Tries ALL_GCcomb first, falls back to individual tracer files.
    Auto-detects DR1 (12 entries) vs DR2 (13 entries).

    Returns: (data_vec, cov_matrix, layout, version_string)
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # ── Try ALL combined file ──
    all_mean = sorted(data_dir.glob('*ALL*mean*'))
    all_cov = sorted(data_dir.glob('*ALL*cov*'))

    if all_mean and all_cov:
        mean_file = all_mean[0]
        cov_file = all_cov[0]

        # Mean file can be either:
        #  - 1 numeric column (values only)
        #  - 3 columns: z, value, quantity_label (e.g., DM_over_rs)
        try:
            mean = np.loadtxt(str(mean_file))
            layout = None
        except ValueError:
            rows = []
            with open(mean_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    z_eff = float(parts[0])
                    value = float(parts[1])
                    qty = parts[2] if len(parts) > 2 else 'DV_over_rs'

                    qty_low = qty.lower()
                    if 'dm' in qty_low:
                        obs = 'DM'
                    elif 'dh' in qty_low:
                        obs = 'DH'
                    else:
                        obs = 'DV'

                    rows.append((z_eff, value, obs))

            mean = np.array([r[1] for r in rows], dtype=float)
            layout = [(f'z{r[0]:.3f}', r[0], r[2]) for r in rows]

        cov = np.loadtxt(str(cov_file))
        n = len(mean)

        if cov.shape != (n, n):
            # Try reading as flattened and reshaping
            if cov.size == n * n:
                cov = cov.reshape(n, n)
            else:
                raise ValueError(
                    f"Cov shape {cov.shape} incompatible with {n} measurements")

        if layout is None:
            if n == 13:
                layout = LAYOUT_13
                version = 'DR2'
            elif n == 12:
                layout = LAYOUT_12
                version = 'DR1'
            else:
                print(f"  ⚠ Unexpected vector size {n}. Trying individual files.")
                return _load_individual(data_dir)
        else:
            if n == 13:
                version = 'DR2'
            elif n == 12:
                version = 'DR1'
            else:
                version = f'N={n}'

        if n not in (12, 13):
            print(f"  ⚠ Unexpected vector size {n}. Trying individual files.")
            return _load_individual(data_dir)

        print(f"  Loaded ALL_GCcomb: {n} measurements ({version})")
        print(f"    Mean file: {mean_file.name}")
        print(f"    Cov file:  {cov_file.name}")
        return mean, cov, layout, version

    # ── Fallback: individual tracer files ──
    return _load_individual(data_dir)


def _load_individual(data_dir):
    """Assemble data vector from individual tracer files."""
    data_dir = Path(data_dir)

    # Tracer file patterns (order matters!)
    tracers = [
        ('BGS',  '*BGS*mean*',       0.295, ['DV']),
        ('LRG1', '*LRG*z0.4*mean*',  0.510, ['DM', 'DH']),
        ('LRG2', '*LRG*z0.6*mean*',  0.706, ['DM', 'DH']),
        ('LRG+ELG', '*LRG+ELG*mean*', 0.934, ['DM', 'DH']),
        ('ELG',  '*ELG*z1.1*mean*',  1.317, ['DM', 'DH']),
        ('QSO',  '*QSO*mean*',       1.491, ['DM', 'DH']),
        ('Lya',  '*Lya*mean*',       2.330, ['DM', 'DH']),
    ]

    all_vals = []
    all_layout = []

    for tag, pattern, zeff, obs_types in tracers:
        files = sorted(data_dir.glob(pattern))
        if not files:
            print(f"    ⚠ Missing tracer: {tag}")
            continue

        vals = np.loadtxt(str(files[0]))
        vals = np.atleast_1d(vals)

        if len(vals) == len(obs_types):
            for v, ot in zip(vals, obs_types):
                all_vals.append(v)
                all_layout.append((tag, zeff, ot))
        elif len(vals) == 1 and len(obs_types) == 2:
            # Probably DV-only (DR1 QSO)
            all_vals.append(vals[0])
            all_layout.append((tag, zeff, 'DV'))
        else:
            # Best effort
            for i, v in enumerate(vals):
                ot = obs_types[i] if i < len(obs_types) else f'X{i}'
                all_vals.append(v)
                all_layout.append((tag, zeff, ot))

    mean = np.array(all_vals)
    n = len(mean)

    # Assemble block-diagonal covariance
    cov_blocks = []
    idx = 0
    for tag, pattern, zeff, obs_types in tracers:
        cov_pattern = pattern.replace('mean', 'cov')
        cov_files = sorted(data_dir.glob(cov_pattern))
        nobs = sum(1 for _, z, _ in all_layout if z == zeff
                   and _ == tag)
        # Count entries for this tracer
        n_tracer = 0
        for t, z, o in all_layout:
            if t == tag:
                n_tracer += 1

        if cov_files and n_tracer > 0:
            c = np.loadtxt(str(cov_files[0]))
            if c.shape == (n_tracer, n_tracer):
                cov_blocks.append(c)
            else:
                cov_blocks.append(np.eye(n_tracer) * 0.01)
        elif n_tracer > 0:
            cov_blocks.append(np.eye(n_tracer) * 0.01)
        idx += n_tracer

    from scipy.linalg import block_diag
    cov = block_diag(*cov_blocks) if cov_blocks else np.eye(n) * 0.01

    version = 'DR2' if n == 13 else ('DR1' if n == 12 else f'N={n}')
    print(f"  Assembled from individual files: {n} measurements ({version})")
    print(f"  ⚠ Block-diagonal covariance (no cross-tracer correlations)")
    return mean, cov, all_layout, version


# ══════════════════════════════════════════════════════════════
# Cosmology engine
# ══════════════════════════════════════════════════════════════

def sound_horizon(Om, h=H_FID, obh2=OBH2_FID):
    """
    Approximate rd from Aubourg+ (2015) fitting formula.
    Good to ~0.5% for Planck-like parameters.
    """
    omh2 = Om * h ** 2
    rd = 147.49 - 26.302 * (omh2 - 0.1426) + 12.096 * (obh2 - 0.02237)
    return rd


def E_squared(z, Om, model='LCDM', epsilon=0.0, psi0=0.0,
              w0=-1.0, wa=0.0):
    """
    E²(z) = H²(z)/H₀² for different dark energy models.

    LCDM:  Ωm(1+z)³ + (1-Ωm)
    DSI:   Ωm(1+z)³ + (1-Ωm) × f_DE(z)
    w0wa:  Ωm(1+z)³ + (1-Ωm) × f_DE(z)
    """
    matter = Om * (1 + z) ** 3
    OL = 1.0 - Om

    if model == 'LCDM':
        return matter + OL

    elif model == 'DSI':
        # Exact integration of w(z) = -1 + ε cos(2π ln(1+z)/ln φ + ψ₀)
        theta_z = 2 * np.pi * np.log(1 + z) / LN_PHI + psi0
        theta_0 = psi0
        exponent = (3 * epsilon * LN_PHI / (2 * np.pi)
                     * (np.sin(theta_z) - np.sin(theta_0)))
        f_de = np.exp(exponent)
        val = matter + OL * f_de
        return max(val, 1e-12)

    elif model == 'w0wa':
        # Standard CPL parameterization
        lna = -np.log(1 + z)
        f_de = ((1 + z) ** (3 * (1 + w0 + wa))
                * np.exp(-3 * wa * z / (1 + z)))
        val = matter + OL * f_de
        return max(val, 1e-12)

    raise ValueError(f"Unknown model: {model}")


def compute_theory_vector(layout, Om, model='LCDM', h=H_FID,
                          epsilon=0.0, psi0=0.0, w0=-1.0, wa=0.0):
    """
    Compute theory predictions D_X/rd for each entry in layout.

    Returns array matching the data vector.
    """
    rd = sound_horizon(Om, h)
    DH0 = C_KMS / (100 * h)  # c/H₀ in Mpc

    theory = np.empty(len(layout))

    for i, (tag, zeff, obs) in enumerate(layout):
        # E(z)
        E2 = E_squared(zeff, Om, model, epsilon, psi0, w0, wa)
        Ez = np.sqrt(E2)

        # DH(z) / rd
        DH = DH0 / Ez
        DH_over_rd = DH / rd

        # DM(z) / rd  via numerical integration
        def integrand(zp):
            E2p = E_squared(zp, Om, model, epsilon, psi0, w0, wa)
            return 1.0 / np.sqrt(max(E2p, 1e-12))

        integral, _ = quad(integrand, 0, zeff, limit=200)
        DM = DH0 * integral
        DM_over_rd = DM / rd

        # DV(z) / rd
        DV_over_rd = (zeff * DM_over_rd ** 2 * DH_over_rd) ** (1.0 / 3.0)

        if obs == 'DM':
            theory[i] = DM_over_rd
        elif obs == 'DH':
            theory[i] = DH_over_rd
        elif obs == 'DV':
            theory[i] = DV_over_rd
        else:
            theory[i] = DV_over_rd  # fallback

    return theory


# ══════════════════════════════════════════════════════════════
# Chi-squared and fitting
# ══════════════════════════════════════════════════════════════

def chi2_func(params, data, cov_inv, layout, model):
    """Compute χ² for a given parameter set and model."""
    if model == 'LCDM':
        Om = params[0]
        theory = compute_theory_vector(layout, Om, 'LCDM')

    elif model == 'DSI':
        Om, eps, psi = params
        theory = compute_theory_vector(layout, Om, 'DSI',
                                       epsilon=eps, psi0=psi)

    elif model == 'w0wa':
        Om, w0, wa = params
        theory = compute_theory_vector(layout, Om, 'w0wa',
                                       w0=w0, wa=wa)
    else:
        raise ValueError(model)

    r = data - theory
    return float(r @ cov_inv @ r)


def fit_model(data, cov, layout, model='LCDM', verbose=True):
    """
    Fit a cosmological model to the BAO data.

    Uses grid + Nelder-Mead + differential evolution for robustness.
    """
    cov_inv = np.linalg.inv(cov)

    if model == 'LCDM':
        # 1D: scan Ωm
        best = {'fun': np.inf}
        for Om0 in np.linspace(0.20, 0.45, 30):
            r = minimize(chi2_func, [Om0],
                         args=(data, cov_inv, layout, model),
                         method='Nelder-Mead',
                         options={'xatol': 1e-8, 'fatol': 1e-10,
                                  'maxiter': 5000})
            if r.fun < best.get('fun', np.inf):
                best = r

        Om_best = best.x[0]
        result = {
            'model': 'LCDM',
            'chi2': round(best.fun, 3),
            'n_params': 1,
            'Om': round(Om_best, 5),
        }

    elif model == 'DSI':
        # 3D: Ωm, ε, ψ₀
        # First fit ΛCDM for starting point
        lcdm = fit_model(data, cov, layout, 'LCDM', verbose=False)
        Om0 = lcdm['Om']

        best = {'fun': np.inf}

        # Grid search
        for eps0 in np.linspace(-0.15, 0.15, 15):
            for psi0 in np.linspace(0, 2 * np.pi, 13, endpoint=False):
                r = minimize(chi2_func, [Om0, eps0, psi0],
                             args=(data, cov_inv, layout, model),
                             method='Nelder-Mead',
                             options={'xatol': 1e-9, 'fatol': 1e-11,
                                      'maxiter': 30000})
                if r.fun < best.get('fun', np.inf):
                    best = r

        # Polish with differential evolution
        bounds = [(0.15, 0.50), (-0.30, 0.30), (0, 2 * np.pi)]
        try:
            rde = differential_evolution(
                chi2_func, bounds,
                args=(data, cov_inv, layout, model),
                maxiter=3000, tol=1e-12, seed=42, polish=True)
            if rde.fun < best.fun:
                best = rde
        except Exception:
            pass

        Om_best, eps_best, psi_best = best.x
        psi_best = psi_best % (2 * np.pi)

        result = {
            'model': 'DSI',
            'chi2': round(best.fun, 3),
            'n_params': 3,
            'Om': round(Om_best, 5),
            'epsilon': round(eps_best, 5),
            'psi0': round(psi_best, 4),
        }

    elif model == 'w0wa':
        # 3D: Ωm, w₀, wₐ
        lcdm = fit_model(data, cov, layout, 'LCDM', verbose=False)
        Om0 = lcdm['Om']

        best = {'fun': np.inf}

        for w0_init in np.linspace(-1.5, -0.5, 8):
            for wa_init in np.linspace(-2.0, 1.0, 8):
                r = minimize(chi2_func, [Om0, w0_init, wa_init],
                             args=(data, cov_inv, layout, model),
                             method='Nelder-Mead',
                             options={'xatol': 1e-9, 'fatol': 1e-11,
                                      'maxiter': 30000})
                if r.fun < best.get('fun', np.inf):
                    best = r

        bounds = [(0.15, 0.50), (-3.0, 0.0), (-5.0, 3.0)]
        try:
            rde = differential_evolution(
                chi2_func, bounds,
                args=(data, cov_inv, layout, model),
                maxiter=3000, tol=1e-12, seed=42, polish=True)
            if rde.fun < best.fun:
                best = rde
        except Exception:
            pass

        Om_best, w0_best, wa_best = best.x

        result = {
            'model': 'w0wa',
            'chi2': round(best.fun, 3),
            'n_params': 3,
            'Om': round(Om_best, 5),
            'w0': round(w0_best, 4),
            'wa': round(wa_best, 4),
        }
    else:
        raise ValueError(f"Unknown model: {model}")

    # Degrees of freedom and reduced chi2
    result['n_data'] = len(data)
    result['dof'] = len(data) - result['n_params']
    result['chi2_red'] = round(result['chi2'] / result['dof'], 4)

    if verbose:
        tag = model.upper()
        print(f"  {tag:8s}  χ²={result['chi2']:8.3f}  "
              f"χ²/ν={result['chi2_red']:.3f}  "
              f"(ν={result['dof']})")
        if model == 'DSI':
            print(f"           Ωm={result['Om']:.4f}  "
                  f"ε={result['epsilon']:+.5f}  "
                  f"ψ₀={result['psi0']:.3f}")
        elif model == 'w0wa':
            print(f"           Ωm={result['Om']:.4f}  "
                  f"w₀={result['w0']:.3f}  "
                  f"wₐ={result['wa']:.3f}")
        else:
            print(f"           Ωm={result['Om']:.4f}")

    return result


def compare_models(lcdm_result, test_result):
    """Compute Δχ², p-value, and significance for test vs ΛCDM."""
    dchi2 = max(lcdm_result['chi2'] - test_result['chi2'], 0.0)
    dof = test_result['n_params'] - lcdm_result['n_params']

    pval = float(chi2_dist.sf(dchi2, df=dof))
    sig = float(norm.isf(pval)) if pval > 1e-15 else 8.0

    return {
        'delta_chi2': round(dchi2, 3),
        'delta_dof': dof,
        'p_value': pval,
        'sigma': round(sig, 2),
    }


# ══════════════════════════════════════════════════════════════
# Random period null test
# ══════════════════════════════════════════════════════════════

def random_period_null(data, cov, layout, dsi_result, n_rand=300,
                       seed=42):
    """
    Compare Δχ² at ln(φ) to Δχ² at random periods.

    If ln(φ) is special, it should beat most random periods.
    """
    print(f"\n  Random-period null test ({n_rand} periods)...")
    rng = np.random.default_rng(seed)
    cov_inv = np.linalg.inv(cov)

    lcdm = fit_model(data, cov, layout, 'LCDM', verbose=False)
    dchi2_phi = lcdm['chi2'] - dsi_result['chi2']

    periods = rng.uniform(0.15, 3.5, n_rand)
    dchi2_rand = np.empty(n_rand)

    for i, per in enumerate(periods):
        # Fit DSI with this period instead of ln(φ)
        best_chi2 = np.inf
        for eps0 in np.linspace(-0.12, 0.12, 7):
            for psi0 in np.linspace(0, 2 * np.pi, 7, endpoint=False):
                def chi2_custom_period(params):
                    Om, eps, psi = params
                    # Recompute with custom period
                    theory = np.empty(len(layout))
                    rd = sound_horizon(Om)
                    DH0 = C_KMS / (100 * H_FID)
                    for j, (tag, zeff, obs) in enumerate(layout):
                        theta = 2 * np.pi * np.log(1 + zeff) / per + psi
                        exp_arg = (3 * eps * per / (2 * np.pi)
                                   * (np.sin(theta) - np.sin(psi)))
                        f_de = np.exp(exp_arg)
                        E2 = Om * (1 + zeff) ** 3 + (1 - Om) * f_de
                        Ez = np.sqrt(max(E2, 1e-12))
                        DH = DH0 / Ez / rd
                        integ = lambda zp: 1.0 / np.sqrt(max(
                            Om * (1 + zp) ** 3 + (1 - Om) * np.exp(
                                3 * eps * per / (2 * np.pi) *
                                (np.sin(2 * np.pi * np.log(1 + zp) / per + psi)
                                 - np.sin(psi))), 1e-12))
                        I, _ = quad(integ, 0, zeff, limit=100)
                        DM = DH0 * I / rd
                        DV = (zeff * DM ** 2 * DH) ** (1.0 / 3.0)
                        if obs == 'DM':
                            theory[j] = DM
                        elif obs == 'DH':
                            theory[j] = DH
                        else:
                            theory[j] = DV
                    r = data - theory
                    return float(r @ cov_inv @ r)

                try:
                    r = minimize(chi2_custom_period,
                                 [lcdm['Om'], eps0, psi0],
                                 method='Nelder-Mead',
                                 options={'maxiter': 8000,
                                          'xatol': 1e-7,
                                          'fatol': 1e-8})
                    if r.fun < best_chi2:
                        best_chi2 = r.fun
                except Exception:
                    pass

        dchi2_rand[i] = lcdm['chi2'] - best_chi2

        if (i + 1) % 50 == 0:
            print(f"    ... {i + 1}/{n_rand}")

    n_beat = int(np.sum(dchi2_rand >= dchi2_phi))
    frac = n_beat / n_rand

    print(f"  ln(φ) Δχ² = {dchi2_phi:.2f}")
    print(f"  Random: mean Δχ²={dchi2_rand.mean():.2f}, "
          f"max={dchi2_rand.max():.2f}")
    print(f"  {n_beat}/{n_rand} random periods beat ln(φ)  "
          f"→ frac={frac:.3f}")

    return {
        'delta_chi2_phi': round(dchi2_phi, 3),
        'frac_exceed': round(frac, 4),
        'mean_random': round(float(dchi2_rand.mean()), 3),
        'max_random': round(float(dchi2_rand.max()), 3),
        'n_random': n_rand,
        'periods': periods.tolist(),
        'dchi2_random': dchi2_rand.tolist(),
    }


# ══════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════

def plot_results(data, cov, layout, results, null_test, version, path):
    """Four-panel summary figure."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.30)
    fig.suptitle(
        f'DESI {version} BAO — DSI Log-Periodic Dark Energy Test',
        fontsize=14, fontweight='bold')

    # Extract unique redshifts and data for plotting
    zeffs = sorted(set(z for _, z, _ in layout))
    sigma_diag = np.sqrt(np.diag(cov))

    # Compute theory vectors for each model
    r_lcdm = results['LCDM']
    r_dsi = results['DSI']
    r_w0wa = results['w0wa']

    t_lcdm = compute_theory_vector(layout, r_lcdm['Om'], 'LCDM')
    t_dsi = compute_theory_vector(
        layout, r_dsi['Om'], 'DSI',
        epsilon=r_dsi['epsilon'], psi0=r_dsi['psi0'])
    t_w0wa = compute_theory_vector(
        layout, r_w0wa['Om'], 'w0wa',
        w0=r_w0wa['w0'], wa=r_w0wa['wa'])

    # ── Panel 1: D/rd vs z ──
    ax = fig.add_subplot(gs[0, 0])
    markers = {'DM': 'o', 'DH': 's', 'DV': 'D'}
    colors = {'DM': 'steelblue', 'DH': 'coral', 'DV': 'forestgreen'}

    for i, (tag, zeff, obs) in enumerate(layout):
        ax.errorbar(zeff, data[i], yerr=sigma_diag[i],
                    fmt=markers.get(obs, 'o'), color=colors.get(obs, 'gray'),
                    ms=5, capsize=2, alpha=0.8,
                    label=f'${obs.replace("over_rd", "/r_d")}$'
                    if i < 3 else None)

    # Dense theory curves
    z_dense = np.linspace(0.05, 2.8, 200)
    for obs_type, color in [('DM', 'steelblue'), ('DH', 'coral')]:
        t_dense = [compute_theory_vector(
            [(f'x', z, obs_type)], r_lcdm['Om'], 'LCDM')[0]
            for z in z_dense]
        ax.plot(z_dense, t_dense, '-', color=color, alpha=0.3, lw=1)

        t_dense_dsi = [compute_theory_vector(
            [(f'x', z, obs_type)], r_dsi['Om'], 'DSI',
            epsilon=r_dsi['epsilon'], psi0=r_dsi['psi0'])[0]
            for z in z_dense]
        ax.plot(z_dense, t_dense_dsi, '--', color=color, alpha=0.6, lw=1.5)

    ax.set_xlabel('$z$')
    ax.set_ylabel('$D/r_d$')
    ax.set_title('BAO distance measurements')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)

    # ── Panel 2: Fractional residuals from ΛCDM ──
    ax = fig.add_subplot(gs[0, 1])
    frac_res = (data - t_lcdm) / t_lcdm
    frac_err = sigma_diag / t_lcdm
    frac_dsi = (t_dsi - t_lcdm) / t_lcdm
    frac_w0wa = (t_w0wa - t_lcdm) / t_lcdm

    for i, (tag, zeff, obs) in enumerate(layout):
        ax.errorbar(zeff + {'DM': -0.01, 'DH': 0.01, 'DV': 0}.get(obs, 0),
                    frac_res[i], yerr=frac_err[i],
                    fmt=markers.get(obs, 'o'), color=colors.get(obs, 'gray'),
                    ms=5, capsize=2, alpha=0.8)

    # DSI and w0wa residual curves
    z_curve = np.linspace(0.1, 2.6, 300)
    for obs_t, col in [('DM', 'steelblue'), ('DH', 'coral')]:
        t0 = np.array([compute_theory_vector(
            [('x', z, obs_t)], r_lcdm['Om'], 'LCDM')[0] for z in z_curve])
        td = np.array([compute_theory_vector(
            [('x', z, obs_t)], r_dsi['Om'], 'DSI',
            epsilon=r_dsi['epsilon'], psi0=r_dsi['psi0'])[0]
            for z in z_curve])
        tw = np.array([compute_theory_vector(
            [('x', z, obs_t)], r_w0wa['Om'], 'w0wa',
            w0=r_w0wa['w0'], wa=r_w0wa['wa'])[0]
            for z in z_curve])
        ax.plot(z_curve, (td - t0) / t0, '--', color='crimson', lw=1.3,
                alpha=0.7, label='DSI' if obs_t == 'DM' else None)
        ax.plot(z_curve, (tw - t0) / t0, ':', color='purple', lw=1.3,
                alpha=0.7, label='$w_0 w_a$' if obs_t == 'DM' else None)

    ax.axhline(0, ls='-', lw=0.5, color='k')
    ax.set_xlabel('$z$')
    ax.set_ylabel(r'$(D - D_{\Lambda CDM})\,/\,D_{\Lambda CDM}$')
    ax.set_title('Fractional residuals from ΛCDM')
    ax.set_ylim(-0.06, 0.06)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)

    # ── Panel 3: w(z) comparison ──
    ax = fig.add_subplot(gs[1, 0])
    z_w = np.linspace(0.01, 3.0, 300)

    # DSI w(z)
    w_dsi = -1 + r_dsi['epsilon'] * np.cos(
        2 * np.pi * np.log(1 + z_w) / LN_PHI + r_dsi['psi0'])
    ax.plot(z_w, w_dsi, '-', color='crimson', lw=2,
            label=f'DSI (ε={r_dsi["epsilon"]:+.4f})')

    # w0wa w(z)
    w_w0wa = r_w0wa['w0'] + r_w0wa['wa'] * z_w / (1 + z_w)
    ax.plot(z_w, w_w0wa, '--', color='purple', lw=2,
            label=f'$w_0 w_a$ ({r_w0wa["w0"]:.2f}, {r_w0wa["wa"]:.2f})')

    ax.axhline(-1, ls=':', lw=1, color='gray', label='ΛCDM')
    ax.set_xlabel('$z$')
    ax.set_ylabel('$w(z)$')
    ax.set_title('Dark energy equation of state')
    ax.set_ylim(-1.4, -0.5)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.15)

    # ── Panel 4: Model comparison / random period test ──
    ax = fig.add_subplot(gs[1, 1])

    if null_test and null_test.get('dchi2_random'):
        dchi2_rand = np.array(null_test['dchi2_random'])
        ax.hist(dchi2_rand, bins=30, density=True, alpha=0.5,
                color='gray', label=f'Random periods (n={null_test["n_random"]})')
        ax.axvline(null_test['delta_chi2_phi'], color='crimson', lw=2.5,
                   label=f'ln(φ): Δχ²={null_test["delta_chi2_phi"]:.1f}')
        ax.set_xlabel('Δχ²')
        ax.set_ylabel('Density')
        ax.set_title('Random-period null test')
        frac = null_test['frac_exceed']
        ax.text(0.97, 0.93,
                f'{int(frac * null_test["n_random"])}/{null_test["n_random"]} '
                f'beat ln(φ)\nfrac = {frac:.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))
        ax.legend(fontsize=8)
    else:
        # Fallback: bar chart of Δχ² per model
        models = ['DSI', 'w0wa']
        dchi2s = [results[m].get('delta_chi2', 0) for m in models]
        sigs = [results[m].get('sigma', 0) for m in models]
        colors_bar = ['crimson' if s >= 3 else 'orange' if s >= 2
                      else 'gray' for s in sigs]
        ax.bar(models, dchi2s, color=colors_bar, alpha=0.8, edgecolor='k')
        ax.axhline(4, ls='--', lw=1, color='gray',
                   label='Expected from noise (2 dof)')
        for i, (m, d, s) in enumerate(zip(models, dchi2s, sigs)):
            ax.text(i, d + 0.3, f'{s:.1f}σ', ha='center', fontsize=11,
                    fontweight='bold')
        ax.set_ylabel('Δχ² vs ΛCDM')
        ax.set_title('Model comparison')
        ax.legend(fontsize=8)

    ax.grid(alpha=0.15)

    plt.savefig(str(path), dpi=150, bbox_inches='tight')
    print(f"\n  Figure → {path}")
    plt.show()


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='DSI search on DESI BAO compressed distances')
    ap.add_argument('--data-dir',
                    default=str(BASE_DIR / 'data' / 'desi_dr2'),
                    help='Path to DESI BAO directory (default: data/desi_dr2)')
    ap.add_argument('--n-random', type=int, default=300,
                    help='Number of random periods for null test')
    ap.add_argument('--skip-random', action='store_true',
                    help='Skip random-period null (saves ~30 min)')
    args = ap.parse_args()

    print("=" * 70)
    print("  DESI BAO — DSI LOG-PERIODIC DARK ENERGY TEST")
    print(f"  Fixed period: ln(φ) = {LN_PHI:.6f}")
    print(f"  Data directory: {args.data_dir}")
    print("=" * 70)

    # ── Load data ──
    print("\n[1/5] Loading BAO data...")
    data, cov, layout, version = load_bao_data(args.data_dir)

    print(f"\n  Data vector ({len(data)} entries):")
    sigma = np.sqrt(np.diag(cov))
    print(f"  {'#':>3} {'Tracer':<12} {'z_eff':>6} {'Obs':>4} "
          f"{'Value':>10} {'σ':>8}")
    for i, (tag, z, obs) in enumerate(layout):
        print(f"  {i:>3} {tag:<12} {z:>6.3f} {obs:>4} "
              f"{data[i]:>10.4f} {sigma[i]:>8.4f}")

    # ── Fit models ──
    print("\n[2/5] Fitting models...")
    results = {}

    print("\n  ┌─ ΛCDM ─┐")
    results['LCDM'] = fit_model(data, cov, layout, 'LCDM')

    print("\n  ┌─ DSI (log-periodic DE) ─┐")
    results['DSI'] = fit_model(data, cov, layout, 'DSI')

    print("\n  ┌─ w₀wₐ (CPL) ─┐")
    results['w0wa'] = fit_model(data, cov, layout, 'w0wa')

    # ── Compare ──
    print("\n[3/5] Model comparison vs ΛCDM...")
    for model_name in ['DSI', 'w0wa']:
        comp = compare_models(results['LCDM'], results[model_name])
        results[model_name].update(comp)
        mark = '⚡' if comp['sigma'] >= 3 else (
               '⚠' if comp['sigma'] >= 2 else ' ')
        print(f"  {model_name:8s}  Δχ²={comp['delta_chi2']:7.2f}  "
              f"p={comp['p_value']:.5f}  → {comp['sigma']:.1f}σ {mark}")

    # Head-to-head: DSI vs w0wa (same number of params)
    if results['DSI']['chi2'] < results['w0wa']['chi2']:
        better = 'DSI'
    elif results['w0wa']['chi2'] < results['DSI']['chi2']:
        better = 'w0wa'
    else:
        better = 'tie'
    dchi2_head = abs(results['DSI']['chi2'] - results['w0wa']['chi2'])
    print(f"\n  Head-to-head (same dof):  DSI χ²={results['DSI']['chi2']:.2f}"
          f"  vs  w₀wₐ χ²={results['w0wa']['chi2']:.2f}")
    print(f"  → {better} preferred by Δχ² = {dchi2_head:.2f}")
    results['head_to_head'] = {
        'preferred': better,
        'delta_chi2': round(dchi2_head, 3),
    }

    # ── Random-period null ──
    null_test = None
    if not args.skip_random and results['DSI'].get('delta_chi2', 0) > 2:
        print(f"\n[4/5] Random-period null test...")
        null_test = random_period_null(
            data, cov, layout, results['DSI'], args.n_random)
        results['null_test'] = null_test
    else:
        print(f"\n[4/5] Random-period test "
              f"{'skipped (--skip-random)' if args.skip_random else 'not needed (Δχ² < 2)'}")

    # ── Plot ──
    print(f"\n[5/5] Generating plots...")
    plot_results(data, cov, layout, results, null_test, version,
                 FIG_DIR / f'desi_bao_dsi_{version.lower()}.png')

    # ── Summary ──
    print(f"\n{'═' * 70}")
    print("  SUMMARY")
    print(f"{'═' * 70}")
    print(f"\n  DESI {version} BAO — {len(data)} measurements, "
          f"7 redshift bins")
    print(f"\n  {'Model':<12} {'χ²':>8} {'χ²/ν':>8} {'Δχ²':>8} "
          f"{'σ':>6} {'Key params'}")
    print(f"  {'─' * 65}")
    for m in ['LCDM', 'DSI', 'w0wa']:
        r = results[m]
        dchi2 = r.get('delta_chi2', 0)
        sig = r.get('sigma', 0)
        if m == 'LCDM':
            params = f"Ωm={r['Om']:.4f}"
        elif m == 'DSI':
            params = (f"Ωm={r['Om']:.4f}  ε={r['epsilon']:+.5f}  "
                      f"ψ₀={r['psi0']:.2f}")
        else:
            params = (f"Ωm={r['Om']:.4f}  w₀={r['w0']:.3f}  "
                      f"wₐ={r['wa']:.3f}")
        mark = '◆' if sig >= 3 else '◇' if sig >= 2 else ' '
        print(f"  {m:<12} {r['chi2']:>8.2f} {r['chi2_red']:>8.3f} "
              f"{dchi2:>8.2f} {sig:>5.1f}σ{mark}  {params}")

    if null_test:
        print(f"\n  Random-period null: "
              f"{null_test['frac_exceed'] * 100:.1f}% of random periods "
              f"beat ln(φ)")

    # ── Save ──
    out = {
        'timestamp': datetime.now().isoformat(),
        'version': version,
        'n_data': len(data),
        'layout': [(t, z, o) for t, z, o in layout],
        'models': {k: {kk: vv for kk, vv in v.items()
                       if not isinstance(vv, np.ndarray)}
                   for k, v in results.items()
                   if isinstance(v, dict)},
    }
    jf = TABLE_DIR / f'desi_bao_dsi_{version.lower()}_results.json'
    with open(str(jf), 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Results → {jf}")


if __name__ == '__main__':
    main()
