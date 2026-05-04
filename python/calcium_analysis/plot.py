from calcium_analysis.multiindex_decorators import support_multiindex_signal
from matplotlib import cm, colors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@support_multiindex_signal(
    signal_arg="y", group_levels=["Row", "Column", "Object ID"], time_name="time"
)
def _trace_to_frame(y: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"value": y})


def _limits_with_padding(y_min: float, y_max: float, pad_fraction: float = 0.05):
    if not np.isfinite(y_min):
        y_min = 0.0
    if not np.isfinite(y_max):
        y_max = 1.0
    if y_min == y_max:
        pad = max(abs(y_min) * pad_fraction, 1e-9)
        return y_min - pad, y_max + pad
    pad = (y_max - y_min) * pad_fraction
    return y_min - pad, y_max + pad


def plot_traces_by_rowcol(
    y: pd.Series,
    peaks_df: pd.DataFrame,
    peak_span_color: str = "#2F3131",
    peak_span_alpha: float = 0.18,
    fix_y_axis_to_global_peak: bool = False,
):
    traces = _trace_to_frame(y=y)
    if not isinstance(traces.index, pd.MultiIndex):
        raise ValueError("Expected MultiIndex with levels Row, Column, Object ID, time")

    pp = peaks_df.reset_index()
    global_signal_min = float(np.nanmin(traces["value"].values))
    global_signal_max = float(np.nanmax(traces["value"].values))

    if fix_y_axis_to_global_peak and "peak_heights" in pp.columns and len(pp):
        global_y_min, global_y_max = _limits_with_padding(
            global_signal_min, float(np.nanmax(pp["peak_heights"].to_numpy()))
        )
    elif fix_y_axis_to_global_peak:
        global_y_min, global_y_max = _limits_with_padding(
            global_signal_min, global_signal_max
        )

    for (row, col), grp in traces.groupby(level=["Row", "Column"]):
        obj_ids = grp.index.get_level_values("Object ID").unique()
        n = len(obj_ids)
        y_vals = grp["value"].values
        if fix_y_axis_to_global_peak:
            y_min, y_max = global_y_min, global_y_max
        else:
            local_y_min = float(np.nanmin(y_vals)) if y_vals.size else 0.0
            local_y_max = float(np.nanmax(y_vals)) if y_vals.size else 1.0
            y_min, y_max = _limits_with_padding(local_y_min, local_y_max)
        fig, axes = plt.subplots(
            n, 1, sharex=True, figsize=(3, max(1.5, 1 * n)), constrained_layout=True
        )
        if n == 1:
            axes = [axes]

        for ax, obj_id in zip(axes, obj_ids):
            s = grp.xs(obj_id, level="Object ID")["value"]
            t = s.index.get_level_values("time")
            ax.plot(t, s.values, color="black", lw=1)
            ax.axhline(0, color="#999999", lw=0.5, alpha=0.5)
            ax.text(
                0.01,
                0.88,
                f"Object ID: {obj_id}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#444444",
            )
            ax.set_ylim(y_min, y_max)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(left=True, labelleft=True, bottom=False)

            obj_peaks = pp[
                (pp["Row"] == row) & (pp["Column"] == col) & (pp["Object ID"] == obj_id)
            ]
            if len(obj_peaks):
                peak_times = None
                peak_values = None

                if "peak_centers_seconds" in obj_peaks.columns:
                    peak_times = obj_peaks["peak_centers_seconds"].to_numpy()
                elif "peak_centers_idx" in obj_peaks.columns:
                    peak_idx = obj_peaks["peak_centers_idx"].astype(int).to_numpy()
                    valid = (peak_idx >= 0) & (peak_idx < len(t))
                    peak_idx = peak_idx[valid]
                    peak_times = t.to_numpy()[peak_idx]

                if "peak_heights" in obj_peaks.columns:
                    peak_values = obj_peaks["peak_heights"].to_numpy()
                elif "peak_centers_idx" in obj_peaks.columns:
                    peak_idx = obj_peaks["peak_centers_idx"].astype(int).to_numpy()
                    valid = (peak_idx >= 0) & (peak_idx < len(s))
                    peak_idx = peak_idx[valid]
                    peak_values = s.to_numpy()[peak_idx]

                if peak_times is not None and peak_values is not None:
                    n_markers = min(len(peak_times), len(peak_values))
                    ax.scatter(
                        peak_times[:n_markers],
                        peak_values[:n_markers],
                        marker="x",
                        color="#B22222",
                        s=28,
                        linewidths=1.25,
                        zorder=3,
                    )
            if (
                len(obj_peaks)
                and "segment_start_idx" in obj_peaks.columns
                and "segment_end_idx" in obj_peaks.columns
            ):
                for peak in obj_peaks.itertuples(index=False):
                    left_idx = int(getattr(peak, "segment_start_idx"))
                    right_idx = int(getattr(peak, "segment_end_idx"))
                    if 0 <= left_idx < len(t) and 0 <= right_idx < len(t):
                        start_t = t.values[left_idx]
                        end_t = t.values[right_idx]
                        ax.axvspan(
                            start_t,
                            end_t,
                            color=peak_span_color,
                            alpha=peak_span_alpha,
                            lw=0,
                        )

        axes[-1].set_xlabel("time (s)")
        axes[len(axes) // 2].set_ylabel("value")
        fig.suptitle(f"Row {row} - Column {col}")
