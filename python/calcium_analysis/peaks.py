from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

from calcium_analysis.helpers import check_time_index
from calcium_analysis.multiindex_decorators import (
    support_multiindex_signal,
    support_multiindex_peaks_signal,
)

GAUSSIAN_SIGMA_OVER_MEDIAN_ABS_DEVIATIONS = 1.4826


def get_peak_positions_and_properties(
    y: pd.Series,
    height_z_score_threshold: float = 3.0,
    prominence_threshold_over_sigma: float = 2.0,
    min_delta_t: float = 0.5,
    rel_prominences_for_widths: list[float] = [0.5, 0.75],
) -> pd.DataFrame:
    peaks_df = find_peaks_positions(
        y,
        height_z_score_threshold=height_z_score_threshold,
        prominence_threshold_over_sigma=prominence_threshold_over_sigma,
        min_delta_t=min_delta_t,
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
) -> pd.DataFrame:

    check_time_index(y)

    v = y.values
    times = y.index.values

    dt = times[1] - times[0]

    g_med = np.median(v)
    g_mad = np.median(np.abs(v - g_med))
    g_sigma = g_mad * GAUSSIAN_SIGMA_OVER_MEDIAN_ABS_DEVIATIONS

    min_peak_distance_idx = int(np.ceil(min_delta_t / dt))
    height_threshold = g_med + height_z_score_threshold * g_sigma
    prominence_threshold = prominence_threshold_over_sigma * g_sigma

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


@support_multiindex_peaks_signal(peaks_arg="peaks_df", other_df_args=("y",))
def add_widths_to_peak_df(
    peaks_df: pd.DataFrame,
    y: pd.Series,
    rel_prominences_for_widths: list[float] = [0.5, 0.75],
) -> pd.DataFrame:

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
