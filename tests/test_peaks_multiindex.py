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


def get_mock_signal(spikes=[(5.0, 2.0, 0.2), (4.5, 6.0, 0.3), (0.2, 8.0, 0.5)]):

    np.random.seed(123)
    t = np.linspace(0, 20, 201)

    # base signal: sudden jump + exponential decay
    def exponential_decay(t, t0, tau):
        y = np.zeros_like(t)
        mask = t >= t0
        y[mask] = np.exp(-(t[mask] - t0) / tau)
        return y

    base = t * 0.0
    for amp, center, width in spikes:
        base += amp * exponential_decay(t, center, width)
    noise = np.random.normal(loc=0.0, scale=0.1, size=t.shape)
    signal = base + noise + 4
    return pd.Series(signal, index=pd.Index(t, name="time"), name="value")


def _golden_paths(out_dir):
    signal_xlsx = os.path.join(out_dir, "golden_signal_multiindex.xlsx")
    peaks_xlsx = os.path.join(out_dir, "golden_peaks_multiindex.xlsx")
    return signal_xlsx, peaks_xlsx


def test_find_peaks_positions_basic(rebase):
    signal1 = get_mock_signal()
    signal2 = get_mock_signal(spikes=[(3.0, 3.0, 0.3), (4.0, 7.0, 0.2)])
    signal = pd.concat(
        [
            signal1.rename("value")
            .to_frame()
            .assign(Trace="A")
            .set_index("Trace", append=True),
            signal2.rename("value")
            .to_frame()
            .assign(Trace="B")
            .set_index("Trace", append=True),
        ]
    )
    # convert single-column DataFrame to Series to enforce strict input types
    signal = signal.iloc[:, 0]
    # ensure MultiIndex level for time is explicitly named so decorators don't infer
    signal.index.names = ["time", "Trace"]
    peaks_df = get_peak_positions_and_properties(signal, min_delta_t=0.1)
    assert not peaks_df.empty
    assert "peak_centers_idx" in peaks_df.columns

    # two traces each produce two peaks -> 4 rows expected
    assert peaks_df.shape[0] == 4

    out_dir = os.path.join("test_data", "peaks")
    os.makedirs(out_dir, exist_ok=True)

    signal_xlsx, peaks_xlsx = _golden_paths(out_dir)

    if rebase:
        # write golden files as flat tables (populate index columns)
        signal.reset_index().to_excel(signal_xlsx, index=False)
        peaks_df.reset_index().to_excel(peaks_xlsx, index=False)
        pytest.skip("Rebased golden files.")
    else:
        # load existing golden and compare (skip if golden files are missing)
        if not (os.path.exists(signal_xlsx) and os.path.exists(peaks_xlsx)):
            pytest.skip("Golden files missing; run tests with --rebase to create them")
        golden_signal = pd.read_excel(signal_xlsx)
        golden_peaks = pd.read_excel(peaks_xlsx)

        golden_signal = golden_signal.set_index(["time", "Trace"])
        golden_peaks = golden_peaks.set_index(["Trace", "peak_index"])

        npt.assert_allclose(
            golden_signal["value"].values, signal.values, rtol=1e-5, atol=1e-6
        )

        npt.assert_allclose(golden_peaks.values, peaks_df.values, rtol=1e-5, atol=1e-6)


def test_absolute_height_threshold_filters_small_peaks_multiindex():
    t = np.arange(11, dtype=float)
    signal_a = pd.Series(
        [0.0, 0.0, 0.03, 0.04, 0.03, 0.0, 0.0, 0.05, 0.08, 0.05, 0.0],
        index=pd.Index(t, name="time"),
        name="value",
    )
    signal_b = pd.Series(
        [0.0, 0.0, 0.06, 0.09, 0.06, 0.0, 0.0, 0.07, 0.11, 0.07, 0.0],
        index=pd.Index(t, name="time"),
        name="value",
    )
    signal = pd.concat(
        [
            signal_a.to_frame().assign(Trace="A").set_index("Trace", append=True),
            signal_b.to_frame().assign(Trace="B").set_index("Trace", append=True),
        ]
    ).iloc[:, 0]
    signal.index.names = ["time", "Trace"]

    filtered_peaks_df = get_peak_positions_and_properties(
        signal,
        height_z_score_threshold=0.0,
        prominence_threshold_over_sigma=0.0,
        min_delta_t=0.1,
        absolute_height_threshold=0.05,
    )

    peaks_a = filtered_peaks_df.xs("A", level="Trace")["peak_centers_idx"].tolist()
    peaks_b = filtered_peaks_df.xs("B", level="Trace")["peak_centers_idx"].tolist()
    assert peaks_a == [8]
    assert peaks_b == [3, 8]


def test_well_mad_group_levels_apply_shared_mad_to_each_object():
    t = np.arange(11, dtype=float)
    noisy_object = pd.Series(
        [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        index=pd.MultiIndex.from_product(
            [["A"], [1], ["noisy"], t],
            names=["Row", "Column", "Object ID", "time"],
        ),
        name="value",
    )
    quiet_object = pd.Series(
        [0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        index=pd.MultiIndex.from_product(
            [["A"], [1], ["quiet"], t],
            names=["Row", "Column", "Object ID", "time"],
        ),
        name="value",
    )
    signal = pd.concat([noisy_object, quiet_object])

    per_object_peaks = get_peak_positions_and_properties(
        signal,
        height_z_score_threshold=0.0,
        prominence_threshold_over_sigma=2.0,
        min_delta_t=0.1,
        absolute_height_threshold=2.0,
    )
    assert per_object_peaks.xs("quiet", level="Object ID")[
        "peak_centers_idx"
    ].tolist() == [5]

    well_mad_peaks = get_peak_positions_and_properties(
        signal,
        height_z_score_threshold=0.0,
        prominence_threshold_over_sigma=2.0,
        min_delta_t=0.1,
        absolute_height_threshold=2.0,
        mad_group_levels=["Row", "Column"],
    )
    assert well_mad_peaks.empty

    relaxed_well_mad_peaks = get_peak_positions_and_properties(
        signal,
        height_z_score_threshold=0.0,
        prominence_threshold_over_sigma=1.0,
        min_delta_t=0.1,
        absolute_height_threshold=2.0,
        mad_group_levels=["Row", "Column"],
    )
    assert relaxed_well_mad_peaks.index.names == [
        "Row",
        "Column",
        "Object ID",
        "peak_index",
    ]
    assert relaxed_well_mad_peaks.xs("quiet", level="Object ID")[
        "peak_centers_idx"
    ].tolist() == [5]


def test_get_timeseries_per_spike_df_golden(rebase):
    out_dir = os.path.join("test_data", "peaks")
    signal_xlsx, peaks_xlsx = _golden_paths(out_dir)

    # load golden files (signal saved without index)
    if not os.path.exists(peaks_xlsx):
        pytest.skip("Golden peaks file missing; run tests with --rebase to create it")
    golden_peaks_df = pd.read_excel(peaks_xlsx)
    golden_peaks_df = golden_peaks_df.set_index(["peak_index", "Trace"])

    signal = pd.read_excel(signal_xlsx)
    signal = signal.set_index(["time", "Trace"])["value"]

    # compute segment bounds from golden peaks (expects width_* columns present)
    peaks_with_segments = append_segment_bounds_using_relative_prominence(
        golden_peaks_df, signal.index, rel_prominence=0.75
    )

    out = get_timeseries_per_spike_df(signal, peaks_with_segments)

    assert not out.empty

    # If rebase requested, save the computed spike timeseries as a golden file
    if rebase:
        golden_timeseries = os.path.join(out_dir, "golden_peak_timeseries.xlsx")
        os.makedirs(out_dir, exist_ok=True)
        # save as a flat table (populate index columns) to avoid sparse
        # Excel representation that leaves blank cells for repeated index
        out.reset_index().to_excel(golden_timeseries, index=False)
        pytest.skip("Rebased golden peak timeseries file.")

    # For each peak, verify rows correspond to exact slice of the original signal
    for idx, row in peaks_with_segments.iterrows():
        start = int(row["segment_start_idx"])
        end = int(row["segment_end_idx"])
        peak_time = row["peak_centers_seconds"]

        subset = out.loc[idx]

        # when `signal` is MultiIndex (time, Trace) the index values are tuples;
        # extract the time level for arithmetic
        times = signal.index.get_level_values("time").values
        expected_times = times[start : end + 1] - peak_time

        # assume first level is 'time' and second is group ('Trace')
        trace_key = idx[0]
        grp = signal.xs(trace_key, level="Trace", drop_level=True)
        expected_values = grp.iloc[start : end + 1].values

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
