import pandas as pd

def rolling_window_norm(
    y: pd.Series,
    quant: float,
    wind: int,
    center: bool = True,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Returns a baseline using a centered rolling-quantile baseline. -> the baseline tracks the lower x-th quantile of your trace (robust for big-ish oscillations)
    quant is the quantile for the baseline (between 0 and 1, e.g., 0.1 = 10th percentile).
    wind is the window length (n of timepoints) for the rolling quantile.
    min_periods is the minimum number of recordings in the window (not NaN); defaults to 'wind' (i.e., edges become NaN until the window is full).
    """
    if not (0.0 <= quant <= 1.0):
      raise ValueError("Quantile must be in [0, 1].")

    if min_periods is None:
      min_periods = wind

    # Rolling baseline as lower-envelope tracker (robust to big-ish oscillations)
    return y.rolling(window=wind, center=center, min_periods=min_periods).quantile(quant)

def smooth_using_gaussian_kernel(
    y: pd.Series | pd.DataFrame,
) -> pd.Series:
    """
    Returns a smoothed version of y, using a Gaussian kernel.
    dt is the time difference between consecutive points in y
    nker is the number of points used to in the kernel
    """
    kernel_sigma = 2
    kernel_width = 5
    return (y.rolling(window=kernel_width, win_type='gaussian', center=True, min_periods=1).mean(std=kernel_sigma))

