"""Process Pantheon+ data for first-pass cosmology analysis.

Steps:
1) Load full Pantheon+ table and full STAT+SYS covariance.
2) Apply cosmology cuts: IS_CALIBRATOR == 0 and zHD > 0.01.
3) Save filtered table and matched covariance subset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data_loader import load_covariance, load_pantheon_table


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    pantheon_dir = root / "data" / "pantheon"

    table_path = pantheon_dir / "Pantheon+SH0ES.dat"
    cov_path = pantheon_dir / "Pantheon+SH0ES_STAT+SYS.cov"

    out_table = pantheon_dir / "Pantheon+SH0ES_hubbleflow_zgt0p01.csv"
    out_cov = pantheon_dir / "Pantheon+SH0ES_STAT+SYS_hubbleflow_zgt0p01.npy"

    z_min = 0.01

    df = load_pantheon_table(table_path)
    cov = load_covariance(cov_path)

    if cov.shape[0] != len(df):
        raise ValueError(
            f"Covariance/table size mismatch: cov={cov.shape}, rows={len(df)}"
        )

    mask = (df["IS_CALIBRATOR"].to_numpy() == 0) & (df["zHD"].to_numpy() > z_min)
    filtered = df.loc[mask].reset_index(drop=True)
    cov_filtered = cov[np.ix_(mask, mask)]

    out_table.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(out_table, index=False)
    np.save(out_cov, cov_filtered)

    print("Pantheon+ processing complete")
    print(f"  Input rows: {len(df)}")
    print(f"  Output rows: {len(filtered)}")
    print(f"  z range: {filtered['zHD'].min():.5f} to {filtered['zHD'].max():.5f}")
    print(f"  Output table: {out_table}")
    print(f"  Output covariance: {out_cov}  shape={cov_filtered.shape}")


if __name__ == "__main__":
    main()
