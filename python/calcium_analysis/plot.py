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


def plot_traces_by_rowcol(
    y: pd.Series,
    peaks_df: pd.DataFrame,
    peak_span_color: str = "#2F3131",
    peak_span_alpha: float = 0.18,
):
    traces = _trace_to_frame(y=y)
    if not isinstance(traces.index, pd.MultiIndex):
        raise ValueError("Expected MultiIndex with levels Row, Column, Object ID, time")

    pp = peaks_df.reset_index()

    for (row, col), grp in traces.groupby(level=["Row", "Column"]):
        obj_ids = grp.index.get_level_values("Object ID").unique()
        n = len(obj_ids)
        y_vals = grp["value"].values
        y_min = float(np.nanmin(y_vals)) if y_vals.size else 0.0
        y_max = float(np.nanmax(y_vals)) if y_vals.size else 1.0
        if y_min == y_max:
            y_max = y_min + 1e-9
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
            ax.set_ylim(y_min, y_max)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(left=False, bottom=False)
            ax.yaxis.set_visible(False)

            obj_peaks = pp[
                (pp["Row"] == row) & (pp["Column"] == col) & (pp["Object ID"] == obj_id)
            ]
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
        fig.suptitle(f"Row {row} - Column {col}")
