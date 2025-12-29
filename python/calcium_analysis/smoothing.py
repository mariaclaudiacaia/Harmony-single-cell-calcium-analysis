import pandas as pd
from calcium_analysis.multiindex_decorators import (
    support_multiindex_signal,
)


@support_multiindex_signal(signal_arg="y")
def rolling_quantile(
    y: pd.Series,
    quantile: float,
    window_size: int,
    center: bool = True,
    min_periods: int | None = None,
) -> pd.Series:
    if not (0.0 <= quantile <= 1.0):
        raise ValueError("Quantile must be in [0, 1].")

    if min_periods is None:
        min_periods = window_size // 2

    return y.rolling(
        window=window_size, center=center, min_periods=min_periods
    ).quantile(quantile)


@support_multiindex_signal(signal_arg="y")
def rolling_gaussian_mean(
    y: pd.Series | pd.DataFrame,
    kernel_width=5,
    kernel_sigma=2,
) -> pd.Series:
    return y.rolling(
        window=kernel_width, win_type="gaussian", center=True, min_periods=1
    ).mean(std=kernel_sigma)
