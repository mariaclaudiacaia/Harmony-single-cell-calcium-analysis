import numpy as np
import pandas as pd


def check_time_index(y: pd.Series):

    if not pd.api.types.is_float_dtype(y.index.dtype):
        raise ValueError(
            "Series index must be a float dtype representing time in seconds. "
            "Convert your index to floats before calling this function."
        )
    if not y.index.is_monotonic_increasing:
        raise ValueError("Series index (time) must be monotonically increasing.")

    idx_values = y.index.values.astype(float)
    if len(idx_values) > 1:
        diffs = np.diff(idx_values)
        if not np.all(np.isfinite(diffs)):
            raise ValueError("Series index contains non-finite values.")
        if not np.allclose(diffs, diffs[0], rtol=1e-5, atol=1e-8):
            raise ValueError(
                "Series index (time) must be regularly spaced (constant sampling interval)."
            )
