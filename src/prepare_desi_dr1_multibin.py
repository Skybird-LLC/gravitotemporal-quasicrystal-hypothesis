#!/usr/bin/env python3
"""
prepare_desi_dr1_multibin.py
============================
Download DESI DR1 GCcomb recsym correlation + RascalC covariance files
and convert them to the folder layout expected by src/desi_multibin_dsi.py:

data/desi_dr1/<bin>/
  xi0.txt       # columns: s [Mpc/h], xi0
  cov_xi0.txt   # covariance matrix for xi0

Notes:
- Correlation files contain 50 s-bins (typically 3..198 Mpc/h).
- RascalC covariance files contain 45 s-bins per multipole (ell=0,2,4).
- We export the monopole ell=0 on the covariance s-grid (45 bins), with
  cov_xi0.txt as the top-left 45x45 monopole block.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve
import tempfile
import numpy as np
import h5py


BASE_DIR = Path(__file__).resolve().parent.parent
OUT_ROOT = BASE_DIR / "data" / "desi_dr1"

BASE_URL = "https://data.desi.lbl.gov/public/dr1/vac/dr1/full-shape-bao-clustering/v1.0"

BIN_MAP = [
    {
        "out_dir": "BGS_z0.1-0.4",
        "corr": "correlation-recsym-poles_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4.h5",
        "cov": "covariance_correlation-recsym-poles_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4.h5",
    },
    {
        "out_dir": "LRG_z0.4-0.6",
        "corr": "correlation-recsym-poles_LRG_GCcomb_z0.4-0.6.h5",
        "cov": "covariance_correlation-recsym-poles_LRG_GCcomb_z0.4-0.6.h5",
    },
    {
        "out_dir": "LRG_z0.6-0.8",
        "corr": "correlation-recsym-poles_LRG_GCcomb_z0.6-0.8.h5",
        "cov": "covariance_correlation-recsym-poles_LRG_GCcomb_z0.6-0.8.h5",
    },
    {
        "out_dir": "LRG+ELG_z0.8-1.1",
        "corr": "correlation-recsym-poles_LRG%2BELG_LOPnotqso_GCcomb_z0.8-1.1.h5",
        "cov": "covariance_correlation-recsym-poles_LRG%2BELG_LOPnotqso_GCcomb_z0.8-1.1.h5",
    },
    {
        "out_dir": "ELG_z1.1-1.6",
        "corr": "correlation-recsym-poles_ELG_LOPnotqso_GCcomb_z1.1-1.6.h5",
        "cov": "covariance_correlation-recsym-poles_ELG_LOPnotqso_GCcomb_z1.1-1.6.h5",
    },
    {
        "out_dir": "QSO_z0.8-2.1",
        "corr": "correlation-recsym-poles_QSO_GCcomb_z0.8-2.1.h5",
        "cov": "covariance_correlation-recsym-poles_QSO_GCcomb_z0.8-2.1.h5",
    },
]


def _read_and_align(corr_path: Path, cov_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(corr_path, "r") as fc:
        s_corr = fc["0/s"][:]
        xi_corr = fc["0/value"][:]

    with h5py.File(cov_path, "r") as fv:
        cov_full = fv["value"][:]
        s_cov = fv["observable/0/s"][:]

    n0 = len(s_cov)
    cov_xi0 = cov_full[:n0, :n0]

    match_idx = []
    for sc in s_cov:
        i = int(np.argmin(np.abs(s_corr - sc)))
        if np.abs(s_corr[i] - sc) > 1.0:
            raise ValueError(
                f"Could not match s={sc:.3f} between correlation and covariance grids"
            )
        match_idx.append(i)

    match_idx = np.array(match_idx, dtype=int)
    s = s_corr[match_idx]
    xi0 = xi_corr[match_idx]

    if cov_xi0.shape != (len(s), len(s)):
        raise ValueError(
            f"Covariance shape {cov_xi0.shape} does not match xi0 length {len(s)}"
        )

    return s, xi0, cov_xi0


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {OUT_ROOT}")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        for item in BIN_MAP:
            out_dir = OUT_ROOT / item["out_dir"]
            out_dir.mkdir(parents=True, exist_ok=True)

            corr_url = f"{BASE_URL}/data/recsym/correlation/{item['corr']}"
            cov_url = f"{BASE_URL}/data/covariance/RascalC/{item['cov']}"

            corr_h5 = tmpdir / f"{item['out_dir']}_corr.h5"
            cov_h5 = tmpdir / f"{item['out_dir']}_cov.h5"

            print(f"\n[{item['out_dir']}]")
            print(f"  downloading correlation...")
            urlretrieve(corr_url, corr_h5)
            print(f"  downloading covariance...")
            urlretrieve(cov_url, cov_h5)

            s, xi0, cov = _read_and_align(corr_h5, cov_h5)

            xi_file = out_dir / "xi0.txt"
            cov_file = out_dir / "cov_xi0.txt"

            np.savetxt(
                xi_file,
                np.column_stack([s, xi0]),
                fmt="%.8e",
                header="s_Mpc_per_h xi0",
            )
            np.savetxt(cov_file, cov, fmt="%.8e")

            print(
                f"  wrote {xi_file.name} ({len(s)} rows) and {cov_file.name} ({cov.shape[0]}x{cov.shape[1]})"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
