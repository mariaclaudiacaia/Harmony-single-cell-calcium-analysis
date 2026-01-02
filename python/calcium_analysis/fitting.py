import numpy as np
import pandas as pd
from typing import Tuple

from calcium_analysis.multiindex_decorators import (
    support_multiindex_signal_single_row_returns,
)


def _exponential_decay(x, A, k, C):
    return A * np.exp(-k * x) + C


def _fit_exponential_decay(
    x: np.ndarray,
    y: np.ndarray,
    p0: Tuple[float, float, float] | None = None,
    bounds: tuple | None = None,
):
    """Fit exponential decay using scipy.optimize.curve_fit.

    Returns (params, covariance) or raises the underlying exception.
    """
    from scipy.optimize import curve_fit

    if p0 is None:
        p0 = (y[0] - y[-1], 1.0, y[-1])
    if bounds is None:
        bounds = (-np.inf, np.inf)

    params, cov = curve_fit(_exponential_decay, x, y, p0=p0, bounds=bounds)
    return params, cov


@support_multiindex_signal_single_row_returns(
    signal_arg="signal", time_name="time_from_peak"
)
def fit_exponential_decay_per_spike(
    signal: pd.Series,
    only_positive: bool = True,
    p0: Tuple[float, float, float] | None = None,
    bounds: tuple | None = None,
) -> pd.DataFrame:
    """Fit an exponential decay reproducing the notebook logic for a spike segment.

    - If `only_positive` is True, only uses samples with time >= 0 (time axis provided
      by the series index).
    - Builds initial guesses similar to the notebook: C = last y, A = y_at_time0 - C,
      k = 1. Uses k lower bound 1e-6 by default.
    - Returns a Series with keys: `peak_over_baseline`, `tau`, `baseline`,
      `mean_square_error`, `r2`.

    If the fit fails (RuntimeError) or there is insufficient data, returns NaNs.
    """

    # prepare time and values
    x = signal.index.values.astype(float)
    y = signal.values

    if only_positive:
        mask = x >= 0
        x = x[mask]
        y = y[mask]

    if x.size == 0:
        return pd.DataFrame(
            [
                {
                    "peak_over_baseline": np.nan,
                    "tau": np.nan,
                    "baseline": np.nan,
                    "mean_square_error": np.nan,
                    "r2": np.nan,
                }
            ]
        )

    # initial guesses
    if p0 is None:
        p0_C = y[-1]
        # find value at time == 0, fallback to first value
        zero_idx = np.where(np.isclose(x, 0))[0]
        if zero_idx.size > 0:
            p0_y0 = y[zero_idx[0]]
        else:
            p0_y0 = y[0]
        p0_A = p0_y0 - p0_C
        p0_k = 1.0
        p0 = (p0_A, p0_k, p0_C)

    if bounds is None:
        bounds = ([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf])

    try:
        params, covariance = _fit_exponential_decay(x, y, p0=p0, bounds=bounds)
        A, k, C = params

        y_pred = _exponential_decay(x, A, k, C)
        mse = np.mean((y - y_pred) ** 2)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

        tau = 1.0 / k if k != 0 else np.nan

        return pd.DataFrame(
            [
                {
                    "peak_over_baseline": A,
                    "tau": tau,
                    "baseline": C,
                    "mean_square_error": mse,
                    "r2": r2,
                }
            ]
        )
    except RuntimeError:
        return pd.DataFrame(
            [
                {
                    "peak_over_baseline": np.nan,
                    "tau": np.nan,
                    "baseline": np.nan,
                    "mean_square_error": np.nan,
                    "r2": np.nan,
                }
            ]
        )
