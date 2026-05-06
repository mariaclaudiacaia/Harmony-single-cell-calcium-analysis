import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

from calcium_analysis.helpers import check_time_index
from calcium_analysis.multiindex_decorators import (
    support_multiindex_signal,
    support_multiindex_peaks_signal,
)

GAUSSIAN_SIGMA_OVER_MEDIAN_ABS_DEVIATIONS = 1.4826


def _as_tuple(key):
    if isinstance(key, tuple):
        return key
    return (key,)


def _normalize_group_levels(group_levels) -> list[str]:
    if isinstance(group_levels, str):
        return [group_levels]
    return list(group_levels)


def _robust_sigma_from_mad(values: np.ndarray) -> float:
    g_med = np.median(values)
    g_mad = np.median(np.abs(values - g_med))
    return g_mad * GAUSSIAN_SIGMA_OVER_MEDIAN_ABS_DEVIATIONS


def _find_peaks_positions_1d(
    y: pd.Series,
    height_z_score_threshold: float = 3.0,
    prominence_threshold_over_sigma: float = 2.0,
    min_delta_t: float = 0.5,
    absolute_height_threshold: float | None = None,
    absolute_prominence_threshold: float | None = None,
    robust_sigma: float | None = None,
) -> pd.DataFrame:
    check_time_index(y)

    v = y.values
    times = y.index.values

    dt = times[1] - times[0]

    g_med = np.median(v)
    g_sigma = _robust_sigma_from_mad(v) if robust_sigma is None else robust_sigma

    min_peak_distance_idx = int(np.ceil(min_delta_t / dt))
    height_threshold = g_med + height_z_score_threshold * g_sigma
    prominence_threshold = prominence_threshold_over_sigma * g_sigma

    if absolute_height_threshold is not None:
        height_threshold = max(height_threshold, absolute_height_threshold)
    if absolute_prominence_threshold is not None:
        prominence_threshold = max(prominence_threshold, absolute_prominence_threshold)

    peaks, properties = find_peaks(
        v,
        height=height_threshold,
        prominence=prominence_threshold,
        distance=min_peak_distance_idx,
    )

    # add time and index of peaks
    properties["peak_centers_seconds"] = times[peaks]
    properties["peak_centers_idx"] = peaks

    df = pd.DataFrame(properties)
    df.index.name = "peak_index"

    return df


def _mad_group_sigmas(
    y: pd.Series,
    mad_group_levels,
    time_name: str = "time",
) -> tuple[list[str], dict[tuple, float]]:
    if not isinstance(y.index, pd.MultiIndex):
        raise ValueError("mad_group_levels requires a MultiIndex signal.")

    mad_group_levels = _normalize_group_levels(mad_group_levels)
    if time_name in mad_group_levels:
        raise ValueError(
            f"mad_group_levels should identify replicate groups, not '{time_name}'."
        )

    names = list(y.index.names)
    missing = [level for level in mad_group_levels if level not in names]
    if missing:
        raise ValueError(f"mad_group_levels not found in signal index: {missing}")

    sigmas = {}
    for key, grp in y.groupby(level=mad_group_levels, sort=False):
        sigmas[_as_tuple(key)] = _robust_sigma_from_mad(grp.values)

    return mad_group_levels, sigmas


def _find_peaks_positions_with_grouped_mad(
    y: pd.Series,
    mad_group_levels,
    height_z_score_threshold: float = 3.0,
    prominence_threshold_over_sigma: float = 2.0,
    min_delta_t: float = 0.5,
    absolute_height_threshold: float | None = None,
    absolute_prominence_threshold: float | None = None,
    time_name: str = "time",
) -> pd.DataFrame:
    if not isinstance(y.index, pd.MultiIndex):
        raise ValueError("mad_group_levels requires a MultiIndex signal.")

    names = list(y.index.names)
    if time_name not in names:
        raise ValueError(
            f"MultiIndex signal must have a level named '{time_name}'. "
            "Do not rely on automatic inference; rename your index levels."
        )

    signal_group_levels = [name for name in names if name != time_name]
    mad_group_levels, sigmas = _mad_group_sigmas(
        y, mad_group_levels=mad_group_levels, time_name=time_name
    )

    pieces = []
    keys = []
    for signal_key, grp in y.groupby(level=signal_group_levels, sort=False):
        signal_key = _as_tuple(signal_key)
        key_by_name = dict(zip(signal_group_levels, signal_key))
        mad_key = tuple(key_by_name[level] for level in mad_group_levels)

        ts = grp.droplevel(signal_group_levels)
        peaks_df = _find_peaks_positions_1d(
            ts,
            height_z_score_threshold=height_z_score_threshold,
            prominence_threshold_over_sigma=prominence_threshold_over_sigma,
            min_delta_t=min_delta_t,
            absolute_height_threshold=absolute_height_threshold,
            absolute_prominence_threshold=absolute_prominence_threshold,
            robust_sigma=sigmas[mad_key],
        )
        if not peaks_df.empty:
            pieces.append(peaks_df)
            keys.append(signal_key if len(signal_group_levels) > 1 else signal_key[0])

    if not pieces:
        return pd.DataFrame()

    return pd.concat(pieces, keys=keys, names=signal_group_levels)


def get_peak_positions_and_properties(
    y: pd.Series,
    height_z_score_threshold: float = 3.0,
    prominence_threshold_over_sigma: float = 2.0,
    min_delta_t: float = 0.5,
    absolute_height_threshold: float | None = None,
    absolute_prominence_threshold: float | None = None,
    rel_prominences_for_widths: list[float] = [0.5, 0.75],
    mad_group_levels=None,
) -> pd.DataFrame:
    """
    Detect peak positions in a one-dimensional calcium (or similar) signal and
    compute peak properties, including widths at given relative prominences.

    This function is the main public entry point for peak detection. It combines
    `find_peaks_positions` (which locates peaks using robust, MAD-based
    thresholds) with `add_widths_to_peak_df` (which adds width-related
    measurements for each detected peak).

    Parameters
    ----------
    y : pandas.Series
        One-dimensional signal with a time-like index (e.g. seconds). The index
        must be monotonically increasing and evenly spaced; this is validated
        by :func:`calcium_analysis.helpers.check_time_index`. The series values
        are treated as fluorescence (or analogous) measurements.
    height_z_score_threshold : float, optional
        Z-score threshold applied to the signal height for peak detection.
        The signal's baseline mean and standard deviation are estimated
        robustly using the median and median absolute deviation (MAD). Peaks
        must exceed `baseline + height_z_score_threshold * sigma` to be
        considered. Default is 3.0.
    prominence_threshold_over_sigma : float, optional
        Minimum required peak prominence expressed in units of the robust
        standard deviation (sigma). The actual prominence threshold is
        computed as `prominence_threshold_over_sigma * sigma`. Default is 2.0.
    min_delta_t : float, optional
        Minimum temporal separation between two neighboring peaks, in the same
        time units as the index of `y` (typically seconds). This is converted
        to a minimum distance in samples and passed to
        :func:`scipy.signal.find_peaks`. Default is 0.5.
    absolute_height_threshold : float or None, optional
        Absolute minimum height a peak must reach on ``y`` to be considered.
        When provided, the effective height cutoff is the larger of the
        MAD-based threshold and this absolute threshold. Default is None.
    absolute_prominence_threshold : float or None, optional
        Absolute minimum prominence a peak must have on ``y`` to be considered.
        When provided, the effective prominence cutoff is the larger of the
        MAD-based threshold and this absolute threshold. Default is None.
    rel_prominences_for_widths : list of float, optional
        Relative prominence levels (between 0 and 1) at which peak widths are
        computed. For each value ``r`` in this list, additional columns such as
        ``width_{int(r*100)}``, ``width_{int(r*100)}_start_idx``,
        ``width_{int(r*100)}_end_idx``, ``width_{int(r*100)}_start_time`` and
        ``width_{int(r*100)}_end_time`` are added to the output. By default,
        widths are computed at 50 % and 75 % of the peak prominence.
    mad_group_levels : str, list of str, or None, optional
        MultiIndex level(s) used to calculate a shared MAD-based sigma for peak
        thresholds. Peak detection is still run per trace/object, but each trace
        receives the sigma calculated from all signal values in its MAD group.
        For well-level MAD in data indexed by ``Row``, ``Column``, ``Object ID``,
        and ``time``, pass ``["Row", "Column"]``. If None, MAD is calculated
        independently for each trace/object.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row per detected peak. At minimum, it contains
        the columns generated by :func:`find_peaks_positions`, including:

        - ``peak_centers_seconds``: time coordinate of the peak center.
        - ``peak_centers_idx``: integer sample index of the peak center.
        - Additional columns from :func:`scipy.signal.find_peaks` such as
          ``height``, ``prominence``, ``left_bases``, ``right_bases``, etc.

        It is then augmented by :func:`add_widths_to_peak_df` with width-related
        columns for each requested relative prominence level in
        `rel_prominences_for_widths`. If no peaks are detected, an empty
        DataFrame is returned.
    """
    if mad_group_levels is None:
        peaks_df = find_peaks_positions(
            y,
            height_z_score_threshold=height_z_score_threshold,
            prominence_threshold_over_sigma=prominence_threshold_over_sigma,
            min_delta_t=min_delta_t,
            absolute_height_threshold=absolute_height_threshold,
            absolute_prominence_threshold=absolute_prominence_threshold,
        )
    else:
        peaks_df = _find_peaks_positions_with_grouped_mad(
            y,
            mad_group_levels=mad_group_levels,
            height_z_score_threshold=height_z_score_threshold,
            prominence_threshold_over_sigma=prominence_threshold_over_sigma,
            min_delta_t=min_delta_t,
            absolute_height_threshold=absolute_height_threshold,
            absolute_prominence_threshold=absolute_prominence_threshold,
        )

    if peaks_df.empty:
        return peaks_df

    df_with_widths = add_widths_to_peak_df(peaks_df, y, rel_prominences_for_widths)

    return df_with_widths


@support_multiindex_signal(signal_arg="y")
def find_peaks_positions(
    y: pd.Series,
    height_z_score_threshold: float = 3.0,
    prominence_threshold_over_sigma: float = 2.0,
    min_delta_t: float = 0.5,
    absolute_height_threshold: float | None = None,
    absolute_prominence_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Detect peak positions and basic peak properties in a 1D time series.

    The function uses robust statistics (median and median absolute deviation)
    to estimate the signal noise level and derives absolute thresholds for the
    peak height and prominence. Peaks are required to be at least
    ``min_delta_t`` seconds apart.

    Parameters
    ----------
    y : pandas.Series
        One-dimensional signal with a monotonically increasing time index.
        The index is interpreted as time in seconds and is used to compute
        inter-peak distances.
    height_z_score_threshold : float, optional
        Number of robust standard deviations (MAD-based sigma) above the median
        the peak height must exceed to be considered a valid peak.
    prominence_threshold_over_sigma : float, optional
        Minimum required prominence expressed as a multiple of the robust
        standard deviation.
    min_delta_t : float, optional
        Minimum allowed time between neighboring peaks, in seconds. This is
        converted internally to a minimum index distance based on the sampling
        interval of ``y``.
    absolute_height_threshold : float or None, optional
        Absolute minimum height a peak must reach on ``y`` to be considered.
        If provided, the effective height cutoff becomes
        ``max(mad_based_height_threshold, absolute_height_threshold)``.
    absolute_prominence_threshold : float or None, optional
        Absolute minimum prominence a peak must have on ``y`` to be considered.
        If provided, the effective prominence cutoff becomes
        ``max(mad_based_prominence_threshold, absolute_prominence_threshold)``.

    Returns
    -------
    pandas.DataFrame
        A DataFrame where each row corresponds to a detected peak. It includes
        the properties returned by :func:`scipy.signal.find_peaks` (such as
        heights, prominences, and base indices), along with:

        - ``peak_centers_seconds``: time of the peak center (from the index of ``y``)
        - ``peak_centers_idx``: integer index position of the peak center

        The DataFrame index is named ``"peak_index"`` and is an arbitrary
        integer identifier for each peak.
    """

    return _find_peaks_positions_1d(
        y,
        height_z_score_threshold=height_z_score_threshold,
        prominence_threshold_over_sigma=prominence_threshold_over_sigma,
        min_delta_t=min_delta_t,
        absolute_height_threshold=absolute_height_threshold,
        absolute_prominence_threshold=absolute_prominence_threshold,
    )


@support_multiindex_peaks_signal(peaks_arg="peaks_df", other_df_args=("y",))
def add_widths_to_peak_df(
    peaks_df: pd.DataFrame,
    y: pd.Series,
    rel_prominences_for_widths: list[float] = [0.5, 0.75],
) -> pd.DataFrame:
    """
    Compute peak widths at specified relative prominence levels and append them to
    a peaks DataFrame.

    Parameters
    ----------
    peaks_df : pandas.DataFrame
        DataFrame describing detected peaks, expected to contain at least a
        ``'peak_centers_idx'`` column with the index positions of peaks in ``y``.
    y : pandas.Series
        One-dimensional signal from which the peaks were detected. Its index is
        interpreted as the time axis used to express peak widths in seconds.
    rel_prominences_for_widths : list of float, optional
        Relative prominence levels (between 0 and 1) at which to compute peak
        widths using :func:`scipy.signal.peak_widths`.

    Returns
    -------
    pandas.DataFrame
        The input ``peaks_df`` with additional columns describing the widths and
        segment bounds of each peak for the requested relative prominence levels.
    """

    if peaks_df.empty:
        return peaks_df

    v = y.values
    times = y.index.values

    peaks_idx = peaks_df["peak_centers_idx"].astype(int).values

    for rel_prominence in rel_prominences_for_widths:
        rel_prominence_str = int(round(rel_prominence * 100))

        width, _, start_idx, end_idx = peak_widths(
            v, peaks_idx, rel_height=rel_prominence
        )

        peaks_df[f"width_{rel_prominence_str}"] = width
        peaks_df[f"width_{rel_prominence_str}_start_idx"] = start_idx.astype(int)
        peaks_df[f"width_{rel_prominence_str}_end_idx"] = end_idx.astype(int)
        peaks_df[f"width_{rel_prominence_str}_start_time"] = times[
            start_idx.astype(int)
        ]
        peaks_df[f"width_{rel_prominence_str}_end_time"] = times[end_idx.astype(int)]
    return peaks_df


@support_multiindex_peaks_signal(peaks_arg="peaks_df", other_df_args=("signal",))
def append_segment_bounds_using_relative_prominence(
    peaks_df: pd.DataFrame, signal: pd.Series, rel_prominence: float = 0.75
) -> pd.DataFrame:
    """
    Append segment bounds derived from peak widths at a given relative prominence.

    This function uses the peak width indices computed at the specified
    ``rel_prominence`` (e.g., via :func:`add_widths_to_peak_df`) to define
    start/end indices for segments around each peak. The indices are clamped to
    the valid range of the provided signal, and a flag is added to indicate
    whether a segment was truncated by this clamping.

    Parameters
    ----------
    peaks_df:
        DataFrame containing peak information, including the columns
        ``width_{rel_prominence*100}_start_idx`` and
        ``width_{rel_prominence*100}_end_idx`` that specify the start and end
        indices of peak widths at the given relative prominence.
    signal:
        One-dimensional signal from which peaks were detected. Its length is
        used to clamp the segment index bounds.
    rel_prominence:
        Relative prominence (between 0 and 1) used to select which width
        columns to use when computing segment bounds. The value is converted to
        an integer percentage to form the expected column names in ``peaks_df``.

    Returns
    -------
    pd.DataFrame
        A copy of ``peaks_df`` with the following additional columns:

        * ``segment_start_idx``: clamped segment start index for each peak.
        * ``segment_end_idx``: clamped segment end index for each peak.
        * ``segment_truncated``: boolean flag indicating whether the original
          width bounds touched the start (index 0) or end (index ``len(signal)-1``)
          of the signal.
    """

    if peaks_df.empty:
        return peaks_df.copy()

    rel_str = int(round(rel_prominence * 100))
    src_start = f"width_{rel_str}_start_idx"
    src_end = f"width_{rel_str}_end_idx"

    if src_start not in peaks_df.columns or src_end not in peaks_df.columns:
        raise KeyError(
            f"Expected width columns '{src_start}' and '{src_end}' in peaks_df"
        )

    n = len(signal)

    new_df = peaks_df.copy()

    # convert and clamp indices
    start_idx = new_df[src_start].astype(int).to_numpy()
    end_idx = new_df[src_end].astype(int).to_numpy()

    start_idx_clamped = np.clip(start_idx, 0, n - 1)
    end_idx_clamped = np.clip(end_idx, 0, n - 1)

    new_df["segment_start_idx"] = start_idx_clamped
    new_df["segment_end_idx"] = end_idx_clamped

    new_df["segment_truncated"] = (start_idx == 0) | (end_idx == n - 1)

    return new_df


@support_multiindex_peaks_signal(peaks_arg="peaks_df", other_df_args=("signal",))
def append_segment_bounds_using_local_minima(
    peaks_df: pd.DataFrame, signal: pd.Series
) -> pd.DataFrame:
    """
    Append segment bounds around each peak using nearest local minima in the signal.

    For each peak specified in ``peaks_df`` (via the ``peak_centers_idx`` column),
    this function searches the 1D ``signal`` for local minima to the left and
    right of the peak center. The indices of these minima (or the signal
    boundaries if no interior minimum is found) are used as segment start and
    end indices. The function returns a copy of ``peaks_df`` with additional
    columns describing these segment bounds.

    Parameters
    ----------
    peaks_df : pandas.DataFrame
        DataFrame containing information about detected peaks. It must include
        a ``"peak_centers_idx"`` column with integer indices into ``signal``.
    signal : pandas.Series
        One-dimensional signal from which peaks were detected. Its length must
        be greater than 3 to allow detection of local minima.

    Returns
    -------
    pandas.DataFrame
        A copy of ``peaks_df`` with additional columns describing the segment
        bounds around each peak (for example, start and end indices, and any
        flags indicating truncated segments).
    """

    if peaks_df.empty:
        return peaks_df.copy()

    if "peak_centers_idx" not in peaks_df.columns:
        raise KeyError("Expected column 'peak_centers_idx' in peaks_df")

    v = signal.values
    n = len(v)
    assert n > 3, "Signal must have at least 4 samples to find local minima"

    new_df = peaks_df.copy()

    minima_mask = np.zeros(n, dtype=bool)
    interior = (v[:-2] > v[1:-1]) & (v[1:-1] < v[2:])
    minima_mask[1:-1] = interior

    minima_mask[0] = v[0] < v[1]
    minima_mask[-1] = v[-1] < v[-2]

    minima_idx = np.nonzero(minima_mask)[0]

    peaks_idx = new_df["peak_centers_idx"].values

    if minima_idx.size == 0:
        # no minima found: everything is truncated to full signal
        starts = np.zeros_like(peaks_idx, dtype=int)
        ends = np.full_like(peaks_idx, fill_value=n - 1, dtype=int)
    else:
        # insertion position of first minima > peak index
        pos = np.searchsorted(minima_idx, peaks_idx, side="right")

        # previous minima (pos-1) if pos>0 else start at 0
        prev_pos = pos - 1
        starts = np.zeros_like(peaks_idx, dtype=int)
        mask_prev = prev_pos >= 0
        if np.any(mask_prev):
            starts[mask_prev] = minima_idx[prev_pos[mask_prev]]

        # next minima is minima_idx[pos] if pos < len(minima_idx) else n-1
        ends = np.full_like(peaks_idx, fill_value=n - 1, dtype=int)
        mask_next = pos < minima_idx.size
        if np.any(mask_next):
            ends[mask_next] = minima_idx[pos[mask_next]]

    new_df["segment_start_idx"] = starts
    new_df["segment_end_idx"] = ends
    new_df["segment_truncated"] = (starts == 0) | (ends == n - 1)

    return new_df


@support_multiindex_peaks_signal(peaks_arg="peaks_df", other_df_args=("signal",))
def get_timeseries_per_spike_df(
    signal: pd.Series,
    peaks_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a time-aligned signal segment for each detected spike.

    Parameters
    ----------
    signal : pandas.Series
        One-dimensional signal trace. The index is assumed to represent time
        (in seconds or another consistent unit), and the values are the
        signal amplitudes.
    peaks_df : pandas.DataFrame
        DataFrame describing detected spikes. It must contain at least the
        following columns:
        - ``segment_start_idx``: integer index (position in ``signal``) of the
          first sample in the spike segment.
        - ``segment_end_idx``: integer index (position in ``signal``) of the
          last sample in the spike segment (inclusive).
        - ``peak_centers_seconds``: time (in the same units as the
          ``signal`` index) of the spike peak, used as the zero point for
          alignment.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing concatenated signal segments for all spikes.
        The index is a MultiIndex with levels:
        - ``peak_index``: the index of the corresponding row in ``peaks_df``.
        - ``time_from_peak``: time relative to the spike peak (signal index
          value minus ``peak_centers_seconds``).

        The DataFrame has at least one column:
        - ``signal_segment``: signal amplitude values for each time point
          within the segment.

        If ``peaks_df`` is empty or no segments can be constructed, an empty
        DataFrame is returned.
    """

    if peaks_df.empty:
        return pd.DataFrame()

    if (
        "segment_start_idx" not in peaks_df.columns
        or "segment_end_idx" not in peaks_df.columns
    ):
        raise KeyError(
            f"Expected columns 'segment_start_idx' and 'segment_end_idx' in peaks_df (got: {list(peaks_df.columns)})"
        )

    spike_data_list = []

    for idx, spike_props in peaks_df.iterrows():
        left_idx = int(spike_props["segment_start_idx"])
        right_idx = int(spike_props["segment_end_idx"])

        peak_time = spike_props["peak_centers_seconds"]

        temp_df = pd.DataFrame(
            {
                "peak_index": idx,
                "time_from_peak": signal.index.values[left_idx : right_idx + 1]
                - peak_time,
                "signal_segment": signal.iloc[left_idx : right_idx + 1].values,
            }
        )

        spike_data_list.append(temp_df)

    if len(spike_data_list) == 0:
        return pd.DataFrame()

    spike_data_df = pd.concat(spike_data_list, ignore_index=True)
    spike_data_df = spike_data_df.set_index(["peak_index", "time_from_peak"])

    return spike_data_df


def split_nested_peak_segments(
    peaks_df: pd.DataFrame,
    signal: pd.Series,
    inplace: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split segment bounds when one detected peak sits inside another peak segment.

    This is intended as a post-processing step after segment bounds have already
    been created, for example by
    :func:`append_segment_bounds_using_relative_prominence`.

    Peaks are processed independently inside each trace/object. For each pair of
    neighboring peaks sorted by ``peak_centers_idx``, if the second peak center
    falls inside the first peak segment and the second peak's segment starts
    inside that first segment:

    * the first peak's ``segment_end_idx`` is moved to the second peak's
      ``segment_start_idx``;
    * the second peak's ``segment_end_idx`` becomes the later of its original
      end and the first peak's original end.

    Parameters
    ----------
    peaks_df:
        Peak table with ``peak_centers_idx``, ``segment_start_idx`` and
        ``segment_end_idx`` columns. The index may be a simple ``peak_index`` or
        a MultiIndex with grouping levels plus ``peak_index``.
    signal:
        Signal used to compute the peak segments. It is only used to recompute
        ``segment_truncated`` when that column is present.
    inplace:
        If True, edit ``peaks_df`` directly. If False, return a corrected copy
        and leave ``peaks_df`` unchanged.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        ``(corrected_peaks_df, adjustments_df)``. The adjustments table records
        which nested peak pairs were changed and the old/new segment end
        indices.
    """
    if peaks_df.empty:
        return (peaks_df if inplace else peaks_df.copy()), pd.DataFrame()

    corrected = peaks_df if inplace else peaks_df.copy()
    required_columns = {"peak_centers_idx", "segment_start_idx", "segment_end_idx"}
    missing_columns = required_columns - set(corrected.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {sorted(missing_columns)}")

    group_levels = [name for name in corrected.index.names if name != "peak_index"]
    if group_levels:
        grouped_peaks = corrected.groupby(level=group_levels, sort=False)
    else:
        grouped_peaks = [((), corrected)]

    adjustments = []
    for group_key, group in grouped_peaks:
        group = group.sort_values("peak_centers_idx")
        for current_idx, next_idx in zip(group.index[:-1], group.index[1:]):
            current_start = int(corrected.loc[current_idx, "segment_start_idx"])
            current_end = int(corrected.loc[current_idx, "segment_end_idx"])
            next_start = int(corrected.loc[next_idx, "segment_start_idx"])
            next_end = int(corrected.loc[next_idx, "segment_end_idx"])
            next_center = int(corrected.loc[next_idx, "peak_centers_idx"])

            if (
                current_start <= next_center <= current_end
                and current_start < next_start < current_end
            ):
                corrected.loc[current_idx, "segment_end_idx"] = next_start
                corrected.loc[next_idx, "segment_end_idx"] = max(next_end, current_end)
                adjustments.append(
                    {
                        "containing_peak": current_idx,
                        "nested_peak": next_idx,
                        "containing_peak_old_end_idx": current_end,
                        "containing_peak_new_end_idx": next_start,
                        "nested_peak_old_end_idx": next_end,
                        "nested_peak_new_end_idx": max(next_end, current_end),
                    }
                )

    if "segment_truncated" in corrected.columns:
        if group_levels:
            if not isinstance(signal.index, pd.MultiIndex):
                raise ValueError(
                    "signal must have a MultiIndex when peaks_df has grouped "
                    "MultiIndex levels."
                )

            signal_names = list(signal.index.names)
            missing_signal_levels = [
                level for level in group_levels if level not in signal_names
            ]
            if missing_signal_levels:
                raise ValueError(
                    "signal is missing grouped index levels needed to recompute "
                    f"segment_truncated: {missing_signal_levels}"
                )

            for group_key, group in corrected.groupby(level=group_levels, sort=False):
                group_key = group_key if isinstance(group_key, tuple) else (group_key,)
                n_timepoints = len(
                    signal.xs(group_key, level=group_levels, drop_level=True)
                )
                corrected.loc[group.index, "segment_truncated"] = (
                    (corrected.loc[group.index, "segment_start_idx"].astype(int) == 0)
                    | (
                        corrected.loc[group.index, "segment_end_idx"].astype(int)
                        == n_timepoints - 1
                    )
                ).values
        else:
            corrected["segment_truncated"] = (
                corrected["segment_start_idx"].astype(int) == 0
            ) | (corrected["segment_end_idx"].astype(int) == len(signal) - 1)

    return corrected, pd.DataFrame(adjustments)
