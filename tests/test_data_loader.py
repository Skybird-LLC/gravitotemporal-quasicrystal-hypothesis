import numpy as np
import pandas as pd

from src.data_loader import apply_pantheon_cuts, select_mu_vectors


def test_apply_pantheon_cuts_filters_calibrators_and_lowz():
    df = pd.DataFrame(
        {
            "zHD": [0.005, 0.02, 0.5],
            "IS_CALIBRATOR": [0, 1, 0],
            "MU_SH0ES": [33.0, 34.0, 42.0],
        }
    )
    out = apply_pantheon_cuts(df, z_min=0.01)
    assert len(out) == 1
    assert np.isclose(out.iloc[0]["zHD"], 0.5)


def test_select_mu_vectors_returns_numpy_arrays():
    df = pd.DataFrame({"zHD": [0.1, 0.2], "MU_SH0ES": [38.0, 39.0]})
    z_data, mu_data = select_mu_vectors(df)
    assert isinstance(z_data, np.ndarray)
    assert isinstance(mu_data, np.ndarray)
    assert z_data.shape == (2,)
    assert mu_data.shape == (2,)
