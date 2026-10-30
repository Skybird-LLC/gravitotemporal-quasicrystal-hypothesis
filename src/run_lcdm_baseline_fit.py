"""Run first-pass ΛCDM fit on filtered Pantheon+ dataset with full covariance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize

from src.fitting import aic, bic, chi2_with_marginalization_cholesky


def mu_lcdm_fast(z_data: np.ndarray, H0: float, Om: float, n_grid: int = 4000) -> np.ndarray:
    """Fast μ(z) for flat ΛCDM using grid integration + interpolation."""
    c_km_s = 299792.458
    z_max = float(np.max(z_data))
    z_grid = np.linspace(0.0, z_max, n_grid)

    OL = 1.0 - Om
    e_z = np.sqrt(Om * (1.0 + z_grid) ** 3 + OL)
    integrand = 1.0 / e_z

    integral = cumulative_trapezoid(integrand, z_grid, initial=0.0)
    dC = (c_km_s / H0) * integral

    dL_grid = (1.0 + z_grid) * dC
    dL_data = np.interp(z_data, z_grid, dL_grid)
    dL_data = np.maximum(dL_data, 1e-12)
    return 5.0 * np.log10(dL_data) + 25.0


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "pantheon"
    out_dir = root / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

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

    def objective(params: np.ndarray) -> float:
        H0, Om = float(params[0]), float(params[1])
        if not (50.0 <= H0 <= 100.0 and 0.01 <= Om <= 0.99):
            return 1e12
        mu_theory = mu_lcdm_fast(z_data, H0, Om)
        chi2, _ = chi2_with_marginalization_cholesky(mu_theory, mu_data, L)
        return chi2

    result = minimize(
        objective,
        x0=np.array([70.0, 0.30], dtype=float),
        method="Nelder-Mead",
        options={"maxiter": 800, "xatol": 1e-4, "fatol": 1e-2},
    )

    H0_best, Om_best = map(float, result.x)
    chi2_best = float(result.fun)

    mu_best = mu_lcdm_fast(z_data, H0_best, Om_best)
    _, M_best = chi2_with_marginalization_cholesky(mu_best, mu_data, L)

    n_data = int(len(z_data))
    k = 2
    ndof = n_data - (k + 1)

    summary = {
        "model": "LambdaCDM",
        "dataset": "Pantheon+ Hubble-flow (IS_CALIBRATOR=0, zHD>0.01)",
        "n_data": n_data,
        "k_params": k,
        "ndof": ndof,
        "H0_best": H0_best,
        "Om_best": Om_best,
        "M_best_marginalized": float(M_best),
        "chi2": chi2_best,
        "chi2_per_dof": float(chi2_best / ndof),
        "AIC": float(aic(chi2_best, k)),
        "BIC": float(bic(chi2_best, k, n_data)),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }

    out_json = out_dir / "pantheon_lcdm_baseline_fit.json"
    out_csv = out_dir / "pantheon_lcdm_baseline_fit.csv"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame([summary]).to_csv(out_csv, index=False)

    print("ΛCDM baseline fit complete")
    print(f"  H0 = {H0_best:.4f}")
    print(f"  Om = {Om_best:.6f}")
    print(f"  chi2 = {chi2_best:.3f}")
    print(f"  chi2/dof = {chi2_best/ndof:.6f}")
    print(f"  AIC = {summary['AIC']:.3f}")
    print(f"  BIC = {summary['BIC']:.3f}")
    print(f"  Wrote: {out_json}")
    print(f"  Wrote: {out_csv}")


if __name__ == "__main__":
    main()
