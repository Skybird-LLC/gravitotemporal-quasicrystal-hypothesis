"""Check Model C log-likelihood convergence between n_grid=1200 and n_grid=2400.

By default reads best-fit parameters from:
results/tables/modelc_quasiperiodic_mcmc_summary.json

Fallback order if missing:
1) --H0 --Om --eps CLI values
2) Model C params from results/tables/pantheon_model_comparison_fullcov.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.fitting import chi2_with_marginalization_cholesky
from src.run_modelc_mcmc import QuasiPhiEvaluator


def infer_theta(root: Path, h0_arg, om_arg, eps_arg) -> tuple[float, float, float, str]:
    if h0_arg is not None and om_arg is not None and eps_arg is not None:
        return float(h0_arg), float(om_arg), float(eps_arg), "cli"

    mcmc_summary = root / "results" / "tables" / "modelc_quasiperiodic_mcmc_summary.json"
    if mcmc_summary.exists():
        with open(mcmc_summary, "r", encoding="utf-8") as f:
            data = json.load(f)
        best = data.get("best_fit")
        if best and {"H0", "Om", "eps"}.issubset(best.keys()):
            return float(best["H0"]), float(best["Om"]), float(best["eps"]), "mcmc_summary"

    comp_json = root / "results" / "tables" / "pantheon_model_comparison_fullcov.json"
    if comp_json.exists():
        with open(comp_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        for rec in data.get("records", []):
            if rec.get("model") == "ModelC_Quasiperiodic":
                params = rec.get("params_best", [])
                if len(params) == 3:
                    return float(params[0]), float(params[1]), float(params[2]), "model_comparison"

    raise RuntimeError(
        "Could not infer best-fit parameters. Provide --H0 --Om --eps or run MCMC first."
    )


def loglike_for_grid(z_data, mu_data, L, theta, n_grid):
    evaluator = QuasiPhiEvaluator(z_data, n_grid=n_grid)
    mu_theory = evaluator.mu(*theta)
    chi2, _ = chi2_with_marginalization_cholesky(mu_theory, mu_data, L)
    return -0.5 * chi2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--H0", type=float, default=None)
    parser.add_argument("--Om", type=float, default=None)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--grid-low", type=int, default=1200)
    parser.add_argument("--grid-high", type=int, default=2400)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "pantheon"
    out_path = root / "results" / "tables" / "modelc_grid_convergence_check.json"

    df = pd.read_csv(data_dir / "Pantheon+SH0ES_hubbleflow_zgt0p01.csv")
    cov = np.load(data_dir / "Pantheon+SH0ES_STAT+SYS_hubbleflow_zgt0p01.npy")

    z_data = df["zHD"].to_numpy(dtype=float)
    mu_data = df["MU_SH0ES"].to_numpy(dtype=float)
    L = np.linalg.cholesky(cov)

    H0, Om, eps, source = infer_theta(root, args.H0, args.Om, args.eps)
    theta = (H0, Om, eps)

    loglike_low = float(loglike_for_grid(z_data, mu_data, L, theta, args.grid_low))
    loglike_high = float(loglike_for_grid(z_data, mu_data, L, theta, args.grid_high))
    delta = float(loglike_high - loglike_low)
    delta_abs = abs(delta)

    payload = {
        "theta_source": source,
        "theta": {"H0": H0, "Om": Om, "eps": eps},
        "loglike_grid_low": {"n_grid": int(args.grid_low), "value": loglike_low},
        "loglike_grid_high": {"n_grid": int(args.grid_high), "value": loglike_high},
        "delta_loglike_high_minus_low": delta,
        "abs_delta_loglike": delta_abs,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Model C grid convergence check complete")
    print(f"  theta source: {source}")
    print(f"  theta: H0={H0:.8f}, Om={Om:.8f}, eps={eps:.8f}")
    print(f"  logL(n_grid={args.grid_low})  = {loglike_low:.8f}")
    print(f"  logL(n_grid={args.grid_high}) = {loglike_high:.8f}")
    print(f"  delta logL (high-low) = {delta:.8f}")
    print(f"  |delta logL| = {delta_abs:.8f}")
    print(f"  Wrote: {out_path}")


if __name__ == "__main__":
    main()
