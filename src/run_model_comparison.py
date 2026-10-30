"""Fit Model A/B/C on filtered Pantheon+ data with full covariance and compare metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize

from src.constants import C_KM_S, PHI
from src.fitting import aic, bic, chi2_with_marginalization_cholesky


def mu_lcdm_fast(z_data: np.ndarray, H0: float, Om: float, n_grid: int = 4000) -> np.ndarray:
    z_max = float(np.max(z_data))
    z_grid = np.linspace(0.0, z_max, n_grid)

    OL = 1.0 - Om
    e_z = np.sqrt(Om * (1.0 + z_grid) ** 3 + OL)
    dC = (C_KM_S / H0) * cumulative_trapezoid(1.0 / e_z, z_grid, initial=0.0)

    dL_grid = (1.0 + z_grid) * dC
    dL_data = np.interp(z_data, z_grid, dL_grid)
    return 5.0 * np.log10(np.maximum(dL_data, 1e-12)) + 25.0


def mu_naive_fast(z_data: np.ndarray, H0: float) -> np.ndarray:
    # H(z) = H0*(1+z)^3 => integral dz/(1+z)^3 = 0.5*(1 - (1+z)^-2)
    dC = (C_KM_S / H0) * 0.5 * (1.0 - (1.0 + z_data) ** -2)
    dL = (1.0 + z_data) * dC
    return 5.0 * np.log10(np.maximum(dL, 1e-12)) + 25.0


def mu_quasiperiodic_fast(
    z_data: np.ndarray, H0: float, Om: float, eps: float, n_grid: int = 4000
) -> np.ndarray:
    z_max = float(np.max(z_data))
    z_grid = np.linspace(0.0, z_max, n_grid)

    w = -1.0 + eps * np.cos(2.0 * np.pi * np.log1p(z_grid) / np.log(PHI))
    integrand_de = (1.0 + w) / (1.0 + z_grid)
    de_integral = cumulative_trapezoid(integrand_de, z_grid, initial=0.0)
    de_factor = np.exp(3.0 * de_integral)

    OL = 1.0 - Om
    e_z = np.sqrt(Om * (1.0 + z_grid) ** 3 + OL * de_factor)
    dC = (C_KM_S / H0) * cumulative_trapezoid(1.0 / e_z, z_grid, initial=0.0)

    dL_grid = (1.0 + z_grid) * dC
    dL_data = np.interp(z_data, z_grid, dL_grid)
    return 5.0 * np.log10(np.maximum(dL_data, 1e-12)) + 25.0


def fit_model(label, k_params, mu_builder, x0, bounds_check, z_data, mu_data, L):
    def objective(params: np.ndarray) -> float:
        if not bounds_check(params):
            return 1e12
        mu_theory = mu_builder(z_data, *params)
        chi2, _ = chi2_with_marginalization_cholesky(mu_theory, mu_data, L)
        return chi2

    result = minimize(
        objective,
        x0=np.array(x0, dtype=float),
        method="Nelder-Mead",
        options={"maxiter": 3000, "xatol": 1e-4, "fatol": 1e-2},
    )

    params_best = [float(x) for x in result.x]
    chi2_best = float(result.fun)

    mu_best = mu_builder(z_data, *params_best)
    _, M_best = chi2_with_marginalization_cholesky(mu_best, mu_data, L)

    n_data = int(len(z_data))
    ndof = n_data - (k_params + 1)

    return {
        "model": label,
        "n_data": n_data,
        "k_params": k_params,
        "ndof": ndof,
        "params_best": params_best,
        "M_best_marginalized": float(M_best),
        "chi2": chi2_best,
        "chi2_per_dof": float(chi2_best / ndof),
        "AIC": float(aic(chi2_best, k_params)),
        "BIC": float(bic(chi2_best, k_params, n_data)),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }


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

    lcdm = fit_model(
        label="ModelA_LambdaCDM",
        k_params=2,
        mu_builder=mu_lcdm_fast,
        x0=[70.0, 0.30],
        bounds_check=lambda p: 50.0 <= p[0] <= 100.0 and 0.01 <= p[1] <= 0.99,
        z_data=z_data,
        mu_data=mu_data,
        L=L,
    )

    naive = fit_model(
        label="ModelB_NaivePhiN",
        k_params=1,
        mu_builder=mu_naive_fast,
        x0=[70.0],
        bounds_check=lambda p: 50.0 <= p[0] <= 100.0,
        z_data=z_data,
        mu_data=mu_data,
        L=L,
    )

    quasi = fit_model(
        label="ModelC_Quasiperiodic",
        k_params=3,
        mu_builder=mu_quasiperiodic_fast,
        x0=[70.0, 0.30, 0.05],
        bounds_check=lambda p: (
            50.0 <= p[0] <= 100.0 and 0.01 <= p[1] <= 0.99 and -1.0 <= p[2] <= 1.0
        ),
        z_data=z_data,
        mu_data=mu_data,
        L=L,
    )

    records = [lcdm, naive, quasi]
    best_chi2 = min(rec["chi2"] for rec in records)
    for rec in records:
        rec["delta_chi2_vs_best"] = float(rec["chi2"] - best_chi2)

    table_rows = []
    for rec in records:
        row = {
            "model": rec["model"],
            "chi2": rec["chi2"],
            "chi2_per_dof": rec["chi2_per_dof"],
            "delta_chi2_vs_best": rec["delta_chi2_vs_best"],
            "AIC": rec["AIC"],
            "BIC": rec["BIC"],
            "params_best": json.dumps(rec["params_best"]),
            "M_best_marginalized": rec["M_best_marginalized"],
            "optimizer_success": rec["optimizer_success"],
        }
        table_rows.append(row)

    df_out = pd.DataFrame(table_rows).sort_values("chi2", ascending=True)

    out_json = out_dir / "pantheon_model_comparison_fullcov.json"
    out_csv = out_dir / "pantheon_model_comparison_fullcov.csv"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"records": records}, f, indent=2)

    df_out.to_csv(out_csv, index=False)

    print("Model comparison complete")
    print(df_out.to_string(index=False))
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
