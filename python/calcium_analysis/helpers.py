import numpy as np
import pandas as pd


def check_time_index(y: pd.Series):
    """
    Validate that a pandas Series has a suitable time-based float index.

    The function enforces the following conditions on ``y.index``:
    - The index dtype must be a float type, representing time in seconds.
    - The index values must be monotonically increasing.
    - All index differences must be finite.
    - The sampling interval must be constant (regularly spaced time points)
      within numerical tolerance.

    Parameters
    ----------
    y : pandas.Series
        Series whose index encodes time stamps in seconds as floating-point
        values.

    Raises
    ------
    ValueError
        If the index is not a float dtype, is not monotonically increasing,
        contains non-finite values, or does not have a constant sampling
        interval.
    """

    if not pd.api.types.is_float_dtype(y.index.dtype):
        raise ValueError(
            "Index of 'y' must be a float dtype representing time in seconds. "
            "Convert y.index to floats before calling this function."
        )
    if not y.index.is_monotonic_increasing:
        raise ValueError("Index of 'y' (time) must be monotonically increasing.")

    idx_values = y.index.values.astype(float)
    if len(idx_values) > 1:
        diffs = np.diff(idx_values)
        if not np.all(np.isfinite(diffs)):
            raise ValueError("Index of 'y' contains non-finite values.")
        if not np.allclose(diffs, diffs[0], rtol=1e-5, atol=1e-8):
            raise ValueError(
                "Index of 'y' (time) must be regularly spaced (constant sampling interval)."
            )
