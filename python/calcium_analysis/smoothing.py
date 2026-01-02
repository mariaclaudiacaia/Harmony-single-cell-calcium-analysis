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
    """
    Compute a rolling quantile over a 1D signal.

    This function applies a rolling window over the input series and computes
    the specified quantile within each window. By default, the window is
    centered and the minimum number of observations required in a window is
    set to half the window size.

    Parameters
    ----------
    y : pandas.Series
        Input signal over which to compute the rolling quantile.
    quantile : float
        Quantile to compute, must be between 0.0 and 1.0 inclusive.
    window_size : int
        Size of the moving window, in number of samples.
    center : bool, optional
        If True, set the labels at the center of the window; if False, set
        the labels at the right edge of the window. Defaults to True.
    min_periods : int or None, optional
        Minimum number of observations in the window required to have a
        value. If None (default), it is set to ``window_size // 2``.

    Returns
    -------
    pandas.Series
        Series of the same index as ``y`` containing the rolling quantile
        values.
    """
    if not (0.0 <= quantile <= 1.0):
        raise ValueError("Quantile must be in [0, 1].")

    if min_periods is None:
        min_periods = window_size

    return y.rolling(
        window=window_size, center=center, min_periods=min_periods
    ).quantile(quantile)


@support_multiindex_signal(signal_arg="y")
def rolling_gaussian_mean(
    y: pd.Series | pd.DataFrame,
    kernel_width=5,
    kernel_sigma=2,
) -> pd.Series:
    """
    Apply a centered Gaussian-weighted rolling mean to a time series or DataFrame.

    Parameters
    ----------
    y : pandas.Series or pandas.DataFrame
        Input signal to be smoothed. Can be a Series or a DataFrame; the
        rolling operation is applied independently to each column.
    kernel_width : int, optional
        Size of the rolling window, in samples. This is passed as the
        ``window`` argument to :meth:`pandas.Series.rolling`. Defaults to 5.
    kernel_sigma : float, optional
        Standard deviation of the Gaussian window, passed as the ``std``
        argument to :meth:`pandas.core.window.Rolling.mean`. Defaults to 2.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        The smoothed signal with the same type and index as ``y``, where each
        value is replaced by its Gaussian-weighted rolling mean.
    """
    return y.rolling(
        window=kernel_width, win_type="gaussian", center=True, min_periods=1
    ).mean(std=kernel_sigma)
