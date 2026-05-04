import os
import numpy as np
import pandas as pd
import numpy.testing as npt
import pytest

from calcium_analysis.peaks import (
    get_peak_positions_and_properties,
    get_timeseries_per_spike_df,
    append_segment_bounds_using_relative_prominence,
)


def get_mock_signal():

    np.random.seed(123)
    t = np.linspace(0, 20, 201)

    # base signal: two Gaussian bumps
    def gauss(t, t0, t_width):
        return np.exp(-0.5 * ((t - t0) / t_width) ** 2)

    base = 5 * gauss(t, 2.0, 0.2) + 4.5 * gauss(t, 6.0, 0.3) + 0.2 * gauss(t, 8, 0.5)
    noise = np.random.normal(loc=0.0, scale=0.1, size=t.shape)
    signal = base + noise + 4
    return pd.Series(signal, index=pd.Index(t, name="time"), name="value")


def _golden_paths(out_dir):
    signal_xlsx = os.path.join(out_dir, "golden_signal.xlsx")
    peaks_xlsx = os.path.join(out_dir, "golden_peaks.xlsx")
    return signal_xlsx, peaks_xlsx


def test_find_peaks_positions_basic(rebase):
    signal = get_mock_signal()
    peaks_df = get_peak_positions_and_properties(signal, min_delta_t=0.1)
    assert not peaks_df.empty
    assert "peak_centers_idx" in peaks_df.columns

    assert peaks_df.shape[0] == 2

    out_dir = os.path.join("test_data", "peaks")
    os.makedirs(out_dir, exist_ok=True)

    signal_xlsx, peaks_xlsx = _golden_paths(out_dir)

    if rebase:
        # write golden files as flat tables (populate index columns)
        signal.reset_index().to_excel(signal_xlsx, index=False)
        peaks_df.reset_index().to_excel(peaks_xlsx, index=False)
        pytest.skip("Rebased golden files.")
    else:
        # load existing golden and compare
        golden_signal = pd.read_excel(signal_xlsx)
        golden_peaks = pd.read_excel(peaks_xlsx)
        npt.assert_allclose(
            golden_signal["value"].values, signal.values, rtol=1e-5, atol=1e-6
        )
        golden_peaks = golden_peaks.set_index("peak_index")

        npt.assert_allclose(golden_peaks.values, peaks_df.values, rtol=1e-5, atol=1e-6)


def test_absolute_height_threshold_filters_small_peaks():
    t = np.arange(11, dtype=float)
    signal = pd.Series(
        [0.0, 0.0, 0.03, 0.04, 0.03, 0.0, 0.0, 0.05, 0.08, 0.05, 0.0],
        index=pd.Index(t, name="time"),
        name="value",
    )

    peaks_df = get_peak_positions_and_properties(
        signal,
        height_z_score_threshold=0.0,
        prominence_threshold_over_sigma=0.0,
        min_delta_t=0.1,
    )
    assert peaks_df["peak_centers_idx"].tolist() == [3, 8]

    filtered_peaks_df = get_peak_positions_and_properties(
        signal,
        height_z_score_threshold=0.0,
        prominence_threshold_over_sigma=0.0,
        min_delta_t=0.1,
        absolute_height_threshold=0.05,
    )
    assert filtered_peaks_df["peak_centers_idx"].tolist() == [8]


def test_get_timeseries_per_spike_df_minimal():
    # simple signal and two spike segments
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    values = np.array([0, 10, 20, 30, 40])
    signal = pd.Series(values, index=times)

    peaks_df = pd.DataFrame(
        {
            "segment_start_idx": [0, 3],
            "segment_end_idx": [2, 4],
            "segment_truncated": [False, False],
            "peak_centers_seconds": [1.0, 3.0],
        },
        index=[0, 1],
    )

    out = get_timeseries_per_spike_df(signal, peaks_df)

    # build expected dataframe
    df1 = pd.DataFrame(
        {
            "peak_index": 0,
            "time_from_peak": np.array([0.0, 1.0, 2.0]) - 1.0,
            "signal_segment": np.array([0, 10, 20]),
        }
    )
    df2 = pd.DataFrame(
        {
            "peak_index": 1,
            "time_from_peak": np.array([3.0, 4.0]) - 3.0,
            "signal_segment": np.array([30, 40]),
        }
    )
    expected = pd.concat([df1, df2], ignore_index=True)
    expected = expected.set_index(["peak_index", "time_from_peak"])

    pd.testing.assert_frame_equal(out.sort_index(), expected.sort_index())


def test_get_timeseries_per_spike_df_golden():
    out_dir = os.path.join("test_data", "peaks")
    signal_xlsx, peaks_xlsx = _golden_paths(out_dir)

    # load golden files (signal saved without index)
    golden_peaks_df = pd.read_excel(peaks_xlsx)
    golden_peaks_df = golden_peaks_df.set_index(["peak_index"])

    signal = pd.read_excel(signal_xlsx)
    signal = signal.set_index("time")["value"]

    # compute segment bounds from golden peaks (expects width_* columns present)
    peaks_with_segments = append_segment_bounds_using_relative_prominence(
        golden_peaks_df, signal.index, rel_prominence=0.75
    )

    out = get_timeseries_per_spike_df(signal, peaks_with_segments)

    assert not out.empty

    # For each peak, verify rows correspond to exact slice of the original signal
    for idx, row in peaks_with_segments.iterrows():
        start = int(row["segment_start_idx"])
        end = int(row["segment_end_idx"])
        peak_time = row["peak_centers_seconds"]

        subset = out.loc[idx]

        expected_times = signal.index.values[start : end + 1] - peak_time
        expected_values = signal.iloc[start : end + 1].values

        npt.assert_allclose(subset.index.values, expected_times)
        npt.assert_allclose(subset["signal_segment"].values, expected_values)


def test_append_segment_bounds_local_minima_matches_reference():
    """Compare vectorized minima-based bounds with a simple reference impl."""

    from calcium_analysis.peaks import append_segment_bounds_using_local_minima

    def ref_append(peaks_df, signal):
        # naive reference: find minima indices by explicit loops
        v = signal.values
        n = len(v)

        # compute minima indices
        minima_idx = []
        for i in range(n):
            if i == 0:
                if n >= 2 and v[0] < v[1]:
                    minima_idx.append(0)
            elif i == n - 1:
                if v[-1] < v[-2]:
                    minima_idx.append(n - 1)
            else:
                if v[i - 1] > v[i] and v[i] < v[i + 1]:
                    minima_idx.append(i)

        minima_idx = np.array(minima_idx, dtype=int)

        out = peaks_df.copy()
        peaks_idx = out["peak_centers_idx"].astype(int).to_numpy()

        starts = []
        ends = []

        for p in peaks_idx:
            # previous minima: search from p downwards
            prev = None
            for m in minima_idx[::-1]:
                if m <= p:
                    prev = m
                    break
            if prev is None:
                s = 0
            else:
                s = int(prev)

            # next minima: search upwards
            nxt = None
            for m in minima_idx:
                if m > p:
                    nxt = m
                    break
            if nxt is None:
                e = n - 1
            else:
                e = int(nxt)

            starts.append(s)
            ends.append(e)

        out["segment_start_idx"] = np.array(starts, dtype=int)
        out["segment_end_idx"] = np.array(ends, dtype=int)
        out["segment_truncated"] = (out["segment_start_idx"] == 0) | (
            out["segment_end_idx"] == n - 1
        )

        return out

    rng = np.random.RandomState(42)

    # create several test signals
    signals = []
    # 1) random with bumps
    t = np.linspace(0, 10, 50)
    s1 = np.sin(t) + 0.2 * rng.normal(size=t.shape)
    signals.append(pd.Series(s1))

    # 2) monotonic increasing (no minima)
    s2 = np.linspace(0, 1, 10)
    signals.append(pd.Series(s2))

    # 3) simple valley at edges
    s3 = np.array([1.0, 0.0, 1.0, 2.0, 3.0])
    signals.append(pd.Series(s3))

    for signal in signals:
        n = len(signal)
        # pick some peak indices (within range)
        peaks_idx = np.unique(np.clip(rng.randint(0, n, size=5), 0, n - 1))

        peaks_df = pd.DataFrame(
            {"peak_centers_idx": peaks_idx}, index=np.arange(len(peaks_idx))
        )

        vec = append_segment_bounds_using_local_minima(peaks_df, signal)
        ref = ref_append(peaks_df, signal)

        # compare arrays
        npt.assert_array_equal(
            vec["segment_start_idx"].values, ref["segment_start_idx"].values
        )
        npt.assert_array_equal(
            vec["segment_end_idx"].values, ref["segment_end_idx"].values
        )
        npt.assert_array_equal(
            vec["segment_truncated"].values, ref["segment_truncated"].values
        )
