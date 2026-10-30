"""Run full-covariance MCMC for Model C (quasiperiodic) on Pantheon+.

Optimized version:
  1. cos(phase) precomputed once
  2. DE base integral precomputed; per-sample = eps * base (no cumtrapz)
  3. cumsum replaces cumulative_trapezoid for uniform grid
  4. solve(L, ones) precomputed via make_fast_chi2_marginalized

Recommended defaults:
- n_walkers = 32
- n_steps   = 50000
- n_burn    = 10000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import emcee
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.constants import C_KM_S, PHI
from src.fitting import make_fast_chi2_marginalized, chi2_with_marginalization_cholesky


# ── helpers ──────────────────────────────────────────────────────────────────

def _cumtrapz_uniform(y: np.ndarray, dx: float) -> np.ndarray:
    """Cumulative trapezoidal integration on a uniform grid (opt #3)."""
    out = np.empty(y.shape[0], dtype=float)
    out[0] = 0.0
    half_dx = 0.5 * dx
    np.cumsum(half_dx * (y[:-1] + y[1:]), out=out[1:])
    return out


# ── fast evaluator (opts #1-3) ───────────────────────────────────────────────

class QuasiPhiEvaluator:
    """Fast evaluator for Model C μ(z) with maximal precomputation."""

    def __init__(self, z_data: np.ndarray, n_grid: int) -> None:
        self.z_data = np.asarray(z_data, dtype=float)
        self.n_grid = int(n_grid)

        self.z_max = float(np.max(self.z_data))
        self.z_grid = np.linspace(0.0, self.z_max, self.n_grid)
        self.dz = float(self.z_grid[1] - self.z_grid[0])
        self.one_plus_z = 1.0 + self.z_grid
        self.one_plus_z_cubed = self.one_plus_z ** 3

        # Opt #1: precompute cos(phase) once
        phase = 2.0 * np.pi * np.log1p(self.z_grid) / np.log(PHI)
        self.cos_phase = np.cos(phase)

        # Opt #2: precompute DE base integral (the part that just scales with eps)
        self.de_base = _cumtrapz_uniform(
            self.cos_phase / self.one_plus_z, self.dz
        )

        # Precompute interpolation indices/weights for z_data onto z_grid
        idx = np.floor(self.z_data / self.dz).astype(int)
        idx = np.clip(idx, 0, self.n_grid - 2)
        frac = (self.z_data - self.z_grid[idx]) / self.dz
        self.idx = idx
        self.frac = frac
        self.one_minus_frac = 1.0 - frac

    def mu(self, H0: float, Om: float, eps: float) -> np.ndarray:
        # Opt #2: DE integral is just eps * precomputed base
        de_factor = np.exp(3.0 * eps * self.de_base)

        OL = 1.0 - Om
        e_z = np.sqrt(Om * self.one_plus_z_cubed + OL * de_factor)

        # Opt #3: fast uniform cumtrapz
        dC = (C_KM_S / H0) * _cumtrapz_uniform(1.0 / e_z, self.dz)

        dL_grid = self.one_plus_z * dC
        dL_data = self.one_minus_frac * dL_grid[self.idx] + self.frac * dL_grid[self.idx + 1]
        return 5.0 * np.log10(np.maximum(dL_data, 1e-12)) + 25.0


# ── priors ───────────────────────────────────────────────────────────────────

def log_prior(theta: np.ndarray) -> float:
    H0, Om, eps = theta
    if not (50.0 <= H0 <= 100.0):
        return -np.inf
    if not (0.01 <= Om <= 0.99):
        return -np.inf
    if not (-1.0 <= eps <= 1.0):
        return -np.inf
    return 0.0


def build_log_prob(mu_data: np.ndarray, fast_chi2, evaluator: QuasiPhiEvaluator):
    """Build log-probability closure using opt #4 fast_chi2."""
    def log_prob(theta: np.ndarray) -> float:
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        H0, Om, eps = map(float, theta)
        mu_theory = evaluator.mu(H0, Om, eps)
        chi2, _ = fast_chi2(mu_theory, mu_data)
        return lp - 0.5 * chi2

    return log_prob


# ── diagnostics ──────────────────────────────────────────────────────────────

def gelman_rubin(chain_3d: np.ndarray) -> np.ndarray:
    """Compute R-hat for each parameter from shape (n_steps, n_walkers, n_dim)."""
    n_steps, n_walkers, n_dim = chain_3d.shape
    r_hat = np.empty(n_dim, dtype=float)
    for d in range(n_dim):
        chains = chain_3d[:, :, d]
        chain_means = np.mean(chains, axis=0)
        grand_mean = np.mean(chain_means)
        B = n_steps * np.var(chain_means, ddof=1)
        W = np.mean(np.var(chains, axis=0, ddof=1))
        var_hat = ((n_steps - 1) / n_steps) * W + (1.0 / n_steps) * B
        r_hat[d] = np.sqrt(var_hat / W) if W > 0 else float("nan")
    return r_hat


def effective_sample_size(flat_samples: np.ndarray) -> np.ndarray:
    """Simple ESS estimate using initial positive-sequence autocorrelation."""
    n, n_dim = flat_samples.shape
    ess = np.empty(n_dim, dtype=float)
    for d in range(n_dim):
        x = flat_samples[:, d] - np.mean(flat_samples[:, d])
        fft_x = np.fft.rfft(x, n=2 * n)
        acf = np.fft.irfft(fft_x * np.conj(fft_x))[:n]
        acf /= acf[0]
        tau = 1.0
        for k in range(1, n // 2):
            pair_sum = acf[2 * k - 1] + acf[2 * k]
            if pair_sum < 0:
                break
            tau += 2.0 * pair_sum
        ess[d] = float(n / tau)
    return ess


def summarize_chain(samples: np.ndarray) -> dict:
    labels = ["H0", "Om", "eps"]
    summary = {}
    for idx, label in enumerate(labels):
        arr = samples[:, idx]
        q16, q50, q84 = np.percentile(arr, [16, 50, 84])
        summary[label] = {
            "p16": float(q16),
            "median": float(q50),
            "p84": float(q84),
            "minus_1sigma": float(q50 - q16),
            "plus_1sigma": float(q84 - q50),
        }

    eps_samples = samples[:, 2]
    eps_mean = float(np.mean(eps_samples))
    eps_std = float(np.std(eps_samples, ddof=1))
    prob_eps_gt_zero = float(np.mean(eps_samples > 0.0))

    summary["eps_significance"] = {
        "mean": eps_mean,
        "std": eps_std,
        "mean_over_std": float(eps_mean / eps_std) if eps_std > 0 else float("nan"),
        "p_eps_gt_0": prob_eps_gt_zero,
        "eps_zero_excluded_2sigma": bool(abs(eps_mean) > 2.0 * eps_std),
    }
    return summary


def make_corner_plot(flat_samples: np.ndarray, out_path: Path) -> None:
    """Simple 3x3 corner plot without the corner package."""
    labels = [r"$H_0$", r"$\Omega_m$", r"$\varepsilon$"]
    n_dim = flat_samples.shape[1]
    fig, axes = plt.subplots(n_dim, n_dim, figsize=(9, 9))

    for i in range(n_dim):
        for j in range(n_dim):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            if i == j:
                ax.hist(flat_samples[:, i], bins=60, color="steelblue",
                        density=True, histtype="stepfilled", alpha=0.7)
                q16, q50, q84 = np.percentile(flat_samples[:, i], [16, 50, 84])
                ax.axvline(q50, color="k", lw=1.2)
                ax.axvline(q16, color="k", ls="--", lw=0.8)
                ax.axvline(q84, color="k", ls="--", lw=0.8)
            else:
                ax.hist2d(flat_samples[:, j], flat_samples[:, i],
                          bins=50, cmap="Blues", density=True)
            if i == n_dim - 1:
                ax.set_xlabel(labels[j], fontsize=12)
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(labels[i], fontsize=12)
            else:
                ax.set_yticklabels([])

    fig.suptitle("Model C posterior", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Corner plot: {out_path}")


# ── grid convergence check ───────────────────────────────────────────────────

def grid_convergence_check(
    z_data: np.ndarray, mu_data: np.ndarray, L: np.ndarray,
    theta: tuple[float, float, float],
) -> dict:
    results = {}
    for ng in (1200, 2400):
        ev = QuasiPhiEvaluator(z_data, n_grid=ng)
        mu_th = ev.mu(*theta)
        chi2, _ = chi2_with_marginalization_cholesky(mu_th, mu_data, L)
        loglike = -0.5 * chi2
        results[ng] = {"chi2": float(chi2), "loglike": float(loglike)}

    delta_chi2 = results[2400]["chi2"] - results[1200]["chi2"]
    delta_loglike = results[2400]["loglike"] - results[1200]["loglike"]
    return {
        "grid_1200": results[1200],
        "grid_2400": results[2400],
        "delta_chi2": float(delta_chi2),
        "delta_loglike": float(delta_loglike),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-walkers", type=int, default=32)
    parser.add_argument("--n-steps", type=int, default=50000)
    parser.add_argument("--n-burn", type=int, default=10000)
    parser.add_argument("--n-grid", type=int, default=1200)
    args = parser.parse_args()

    if args.n_burn >= args.n_steps:
        raise ValueError("n_burn must be smaller than n_steps")

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "pantheon"
    out_tables = root / "results" / "tables"
    out_chains = root / "results" / "chains"
    out_figs = root / "results" / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_chains.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    table_path = data_dir / "Pantheon+SH0ES_hubbleflow_zgt0p01.csv"
    cov_path = data_dir / "Pantheon+SH0ES_STAT+SYS_hubbleflow_zgt0p01.npy"

    df = pd.read_csv(table_path)
    cov = np.load(cov_path)

    z_data = df["zHD"].to_numpy(dtype=float)
    mu_data = df["MU_SH0ES"].to_numpy(dtype=float)

    if cov.shape != (len(z_data), len(z_data)):
        raise ValueError(
            f"Covariance shape {cov.shape} incompatible with N={len(z_data)}"
        )

    L = np.linalg.cholesky(cov)

    # Opt #4: precompute solve(L, ones) once
    fast_chi2 = make_fast_chi2_marginalized(L)

    # Opts #1-3: precompute grids, cos, DE base, interp indices
    evaluator = QuasiPhiEvaluator(z_data, n_grid=args.n_grid)
    log_prob = build_log_prob(mu_data, fast_chi2, evaluator)

    # Start near quick-fit point from comparison run
    center = np.array([58.3, 0.334, 0.052], dtype=float)
    scale = np.array([0.8, 0.015, 0.02], dtype=float)
    rng = np.random.default_rng(42)
    p0 = center + scale * rng.normal(size=(args.n_walkers, 3))
    p0[:, 0] = np.clip(p0[:, 0], 50.1, 99.9)
    p0[:, 1] = np.clip(p0[:, 1], 0.011, 0.989)
    p0[:, 2] = np.clip(p0[:, 2], -0.99, 0.99)

    sampler = emcee.EnsembleSampler(args.n_walkers, 3, log_prob)

    print("Starting MCMC for Model C (optimized)")
    print(f"  n_walkers = {args.n_walkers}")
    print(f"  n_steps   = {args.n_steps}")
    print(f"  n_burn    = {args.n_burn}")
    print(f"  n_grid    = {args.n_grid}")

    sampler.run_mcmc(p0, args.n_steps, progress=True)

    # ── post-processing ──────────────────────────────────────────────────────

    chain = sampler.get_chain()
    flat_samples = sampler.get_chain(discard=args.n_burn, flat=True)
    flat_log_prob = sampler.get_log_prob(discard=args.n_burn, flat=True)

    best_idx = int(np.argmax(flat_log_prob))
    best_theta = flat_samples[best_idx]
    best_log_prob = float(flat_log_prob[best_idx])

    # Summary statistics
    summary = summarize_chain(flat_samples)

    # R-hat on post-burn chain
    chain_post_burn = sampler.get_chain(discard=args.n_burn)
    r_hat = gelman_rubin(chain_post_burn)
    ess = effective_sample_size(flat_samples)

    summary["diagnostics"] = {
        "r_hat": {"H0": float(r_hat[0]), "Om": float(r_hat[1]), "eps": float(r_hat[2])},
        "ess": {"H0": float(ess[0]), "Om": float(ess[1]), "eps": float(ess[2])},
    }

    summary["meta"] = {
        "model": "ModelC_Quasiperiodic",
        "dataset": "Pantheon+ Hubble-flow (IS_CALIBRATOR=0, zHD>0.01)",
        "n_data": int(len(z_data)),
        "n_walkers": int(args.n_walkers),
        "n_steps": int(args.n_steps),
        "n_burn": int(args.n_burn),
        "n_grid": int(args.n_grid),
        "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
    }
    summary["best_fit"] = {
        "H0": float(best_theta[0]),
        "Om": float(best_theta[1]),
        "eps": float(best_theta[2]),
        "max_log_prob": best_log_prob,
    }

    # Grid convergence check
    theta_best = (float(best_theta[0]), float(best_theta[1]), float(best_theta[2]))
    conv = grid_convergence_check(z_data, mu_data, L, theta_best)
    summary["grid_convergence"] = conv

    # ── save outputs ─────────────────────────────────────────────────────────

    chain_path = out_chains / "modelc_quasiperiodic_chain.npy"
    flat_path = out_chains / "modelc_quasiperiodic_flat_samples.npy"
    flat_log_prob_path = out_chains / "modelc_quasiperiodic_flat_log_prob.npy"
    summary_path = out_tables / "modelc_quasiperiodic_mcmc_summary.json"
    corner_path = out_figs / "modelc_corner_plot.png"

    np.save(chain_path, chain)
    np.save(flat_path, flat_samples)
    np.save(flat_log_prob_path, flat_log_prob)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    make_corner_plot(flat_samples, corner_path)

    # ── print report ─────────────────────────────────────────────────────────

    print("\n" + "=" * 65)
    print("MCMC RESULTS — Model C (Quasiperiodic φ-potential)")
    print("=" * 65)

    for label in ("H0", "Om", "eps"):
        s = summary[label]
        print(f"  {label:>4s} = {s['median']:.6f}  "
              f"(-{s['minus_1sigma']:.6f} / +{s['plus_1sigma']:.6f})")

    sig = summary["eps_significance"]
    print(f"\n  ε mean/std = {sig['mean_over_std']:.3f}σ")
    print(f"  P(ε > 0)  = {sig['p_eps_gt_0']:.4f}")
    print(f"  ε=0 excluded at >2σ? {sig['eps_zero_excluded_2sigma']}")

    diag = summary["diagnostics"]
    print(f"\n  R-hat:  H0={diag['r_hat']['H0']:.4f}  "
          f"Om={diag['r_hat']['Om']:.4f}  eps={diag['r_hat']['eps']:.4f}")
    print(f"  ESS:    H0={diag['ess']['H0']:.0f}  "
          f"Om={diag['ess']['Om']:.0f}  eps={diag['ess']['eps']:.0f}")

    gc = summary["grid_convergence"]
    print(f"\n  Grid convergence (best-fit):")
    print(f"    χ²(n_grid=1200) = {gc['grid_1200']['chi2']:.6f}")
    print(f"    χ²(n_grid=2400) = {gc['grid_2400']['chi2']:.6f}")
    print(f"    Δχ² = {gc['delta_chi2']:.6f}")

    print(f"\n  acceptance fraction = {summary['meta']['acceptance_fraction_mean']:.4f}")
    print("=" * 65)

    print(f"\n  Wrote: {summary_path}")
    print(f"  Wrote: {chain_path}")
    print(f"  Wrote: {flat_path}")
    print(f"  Wrote: {flat_log_prob_path}")


if __name__ == "__main__":
    main()
