import os
import numpy as np
import pandas as pd
import numpy.testing as npt
import pytest

from calcium_analysis.fitting import fit_exponential_decay_per_spike


def _golden_paths(out_dir):
    fit_xlsx = os.path.join(out_dir, "golden_fit_results_multiindex.xlsx")
    return fit_xlsx


def test_fit_per_spike_regression(rebase, tmp_path):
    out_dir = os.path.join("test_data", "peaks")

    fit_xlsx = _golden_paths(out_dir)

    # require peaks & signal golden files
    golden_timeseries = os.path.join(out_dir, "golden_peak_timeseries.xlsx")

    # Prefer loading precomputed spike timeseries golden; otherwise fall back
    # to computing it from signal + peaks if inputs are available.
    if os.path.exists(golden_timeseries):
        spike_timeseries = pd.read_excel(golden_timeseries)
        # restore MultiIndex written by to_excel
        spike_timeseries = spike_timeseries.set_index(
            ["Trace", "peak_index", "time_from_peak"]
        )
    else:
        pytest.skip(
            "Golden timeseries files missing; run tests with --rebase to create them"
        )

    # series with MultiIndex (peak_index, time_from_peak)
    series = spike_timeseries["signal_segment"]

    # compute fits — this uses the MultiIndex-aware decorator
    fits_df = fit_exponential_decay_per_spike(series)

    if rebase:
        os.makedirs(out_dir, exist_ok=True)
        # normalize fits_df to a flat table
        fits_df_clean = fits_df.reset_index()
        fits_df_clean.to_excel(fit_xlsx, index=False)
        pytest.skip("Rebased golden fit file.")

    if not os.path.exists(fit_xlsx):
        pytest.skip("Golden fit file missing; run tests with --rebase to create it")

    golden_fits = pd.read_excel(fit_xlsx)
    # assume peak_index is the index column written
    golden_fits = golden_fits.set_index(["Trace", "peak_index"])

    # align columns we expect
    expected_cols = [
        "peak_over_baseline",
        "tau",
        "baseline",
        "mean_square_error",
        "r2",
    ]

    for col in expected_cols:
        assert col in fits_df.columns, f"Missing column {col} in computed fits"
        assert col in golden_fits.columns, f"Missing column {col} in golden fits"

    # compare values (order by index)
    fits_df = fits_df.sort_index()
    golden_fits = golden_fits.sort_index()

    # ensure same index set
    npt.assert_array_equal(golden_fits.index.values, fits_df.index.values)

    for col in expected_cols:
        npt.assert_allclose(
            golden_fits[col].values, fits_df[col].values, rtol=1e-5, atol=1e-6
        )
