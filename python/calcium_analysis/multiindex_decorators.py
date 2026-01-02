"""Decorators to add MultiIndex / grouped support to single-trace peak functions.

The decorators detect MultiIndex `signal` or `peaks_df` inputs and apply the
wrapped function per-group, concatenating results while preserving group
level names.
"""

from __future__ import annotations

import functools
import inspect
from typing import Tuple

import pandas as pd


def _bound_arguments(func, args, kwargs):
    sig = inspect.signature(func)
    bound = sig.bind_partial(*args, **kwargs)
    return bound


def _as_tuple(key):
    if isinstance(key, tuple):
        return key
    return (key,)


def support_multiindex_signal(
    signal_arg: str = "signal", *, group_levels=None, time_name="time"
):
    """Decorator to add MultiIndex grouping support to 1-D ``pd.Series`` signals.

    This decorator is intended for functions that operate on a single
    time-indexed trace, i.e. a 1-D ``pandas.Series`` whose index represents
    time. When the decorated function is called with a plain ``Series`` (no
    ``MultiIndex``), the original function is invoked unchanged. When the
    ``signal`` argument is a ``Series`` with a ``MultiIndex`` index, the
    decorator groups the series by one or more index levels, calls the wrapped
    function once per group on the corresponding 1-D time series, and
    concatenates the results while preserving the group keys in the index.

    Grouping behavior
    ------------------
    The time dimension must be represented by an index level named
    ``time_name`` (default: ``"time"``). All remaining levels are treated as
    grouping dimensions. For each unique combination of those group levels, a
    subgroup is extracted, reduced to a 1-D ``Series`` indexed only by the
    ``time_name`` level, and passed to the wrapped function as the
    ``signal_arg`` argument.

    If ``group_levels`` is provided, its values are used as the group level
    names instead of inferring them from all non-time index levels. In either
    case, if the required time level is missing, a :class:`ValueError` is
    raised to avoid ambiguous behavior.

    Parameters
    ----------
    signal_arg : str, optional
        Name of the keyword argument in the wrapped function that receives the
        signal series. By default, ``"signal"``. The decorator will inspect the
        call, extract this argument (from positional or keyword arguments), and
        use it for MultiIndex handling.
    group_levels : iterable of hashable, optional
        Explicit names of the index levels to group by. If ``None`` (the
        default), all index levels except the ``time_name`` level are used as
        grouping levels. If provided, the names must correspond to existing
        levels in the ``signal`` index.
    time_name : hashable, optional
        Name of the index level that represents time in the MultiIndex. The
        series passed to the wrapped function will be reindexed to have only
        this level. Defaults to ``"time"``.

    Returns
    -------
    callable
        A wrapped version of the original function. When called, it returns
        either:

        * The direct result of the original function if ``signal`` is not a
          MultiIndex ``Series``.
        * A ``DataFrame`` or other object resulting from applying the original
          function to each group and concatenating the per-group outputs when
          ``signal`` is a MultiIndex ``Series``. If the concatenated
          DataFrame is empty, an empty ``DataFrame`` is returned.

    Notes
    -----
    The decorator uses :meth:`pandas.Series.groupby` with the chosen group
    levels and :meth:`pandas.core.groupby.GroupBy.apply` to call the wrapped
    function on each subgroup. Pandas is responsible for attaching the group
    keys to the resulting index during this operation.

    Examples
    --------
    Basic usage with automatic grouping by all non-time levels::

        import pandas as pd

        @support_multiindex_signal()
        def compute_mean(signal: pd.Series) -> float:
            return signal.mean()

        # MultiIndex: (cell_id, trial_id, time)
        idx = pd.MultiIndex.from_product(
            [["cellA", "cellB"], [1, 2], range(3)],
            names=["cell", "trial", "time"],
        )
        signal = pd.Series(range(len(idx)), index=idx)

        # The decorator will group by (cell, trial) and call compute_mean
        # separately for each (cell, trial) combination.
        result = compute_mean(signal=signal)

    You can also specify explicit group levels, for example grouping only by
    ``"cell"`` and treating ``"trial"`` as part of the time series if encoded
    in the index or values::

        @support_multiindex_signal(group_levels=["cell"])
        def summarize_cell(signal: pd.Series):
            ...

        summary = summarize_cell(signal=signal)
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound = _bound_arguments(func, args, kwargs)
            if signal_arg not in bound.arguments:
                # fallback to kwargs
                signal = kwargs.get(signal_arg, None)
            else:
                signal = bound.arguments[signal_arg]

            # if not a MultiIndex Series, call original
            if not isinstance(signal, pd.Series) or not isinstance(
                signal.index, pd.MultiIndex
            ):
                return func(*args, **kwargs)

            # Require explicit 'time' level name. Do not attempt to infer levels.
            if group_levels is None:
                names = list(signal.index.names)
                if time_name not in names:
                    raise ValueError(
                        "MultiIndex signal must have a level named 'time'. "
                        "Do not rely on automatic inference; rename your index levels."
                    )
                group_level_names = [n for n in names if n != time_name]
            else:
                group_level_names = list(group_levels)

            # prepare call kwargs once
            call_kwargs = dict(bound.arguments)

            def _apply_fn(grp):
                # reduce subgroup to a time-indexed Series
                ts = grp
                if isinstance(ts.index, pd.MultiIndex):
                    names_ts = list(ts.index.names)
                    assert (
                        time_name in names_ts
                    ), "Subgroup missing required 'time' level"
                    drop_levels = [n for n in names_ts if n != time_name]
                    ts = ts.droplevel(drop_levels)

                local_kwargs = dict(call_kwargs)
                local_kwargs[signal_arg] = ts
                res = func(**local_kwargs)
                # pandas will automatically prefix the group key when using
                # groupby.apply, so return the inner DataFrame directly
                return res

            result = signal.groupby(level=group_level_names).apply(_apply_fn)
            # if result is empty or not a DataFrame, normalize
            if isinstance(result, pd.DataFrame):
                if result.empty:
                    return pd.DataFrame()
                return result
            return result

        return wrapper

    return decorator


def support_multiindex_peaks(
    peaks_arg: str = "peaks_df",
    other_df_args: Tuple[str, ...] = (),
):
    """Decorator factory adding MultiIndex / grouped support for peak tables.

    This decorator is intended for functions that operate on a single-group
    ``peaks_df`` (a regular :class:`pandas.DataFrame` indexed by
    ``"peak_index"`` only), but that may also be called with a MultiIndex
    ``peaks_df`` where the outer index levels encode grouping variables
    (e.g. ``cell_id``, ``trial``, etc.).

    Behavior
    --------
    The returned decorator wraps a function that has a keyword argument named
    ``peaks_arg`` (by default ``"peaks_df"``). When the wrapped function is
    called:

    * If the argument specified by ``peaks_arg`` is **not** a MultiIndex
      DataFrame, the function is invoked once and the result is returned
      unchanged.
    * If the argument specified by ``peaks_arg`` **is** a MultiIndex
      DataFrame, the code iterates over all group keys defined by the index
      levels **except** the innermost ``"peak_index"`` level. For each group:

      - The corresponding slice of ``peaks_df`` is passed to the wrapped
        function with all grouping levels dropped from its index, leaving
        only ``"peak_index"``.
      - For every argument name listed in ``other_df_args``, if the
        corresponding argument is a MultiIndex DataFrame, it is subset to the
        same group key and its grouping levels are dropped before being passed
        to the wrapped function.

    The per-group results are then concatenated along the index. Any grouping
    levels from the original ``peaks_df`` are re-attached as outer index
    levels in the combined result so that group membership is preserved.

    Parameters
    ----------
    peaks_arg:
        Name of the keyword argument in the wrapped function that provides
        the main peaks table (usually ``"peaks_df"``). This argument must be
        a :class:`pandas.DataFrame`. If it has a MultiIndex, the outer levels
        are interpreted as grouping levels and the last level is assumed to be
        ``"peak_index"``.
    other_df_args:
        Tuple of argument names for any *other* DataFrame parameters of the
        wrapped function that should be grouped in the same way as
        ``peaks_arg``. For each name in this tuple, if the corresponding
        argument is a MultiIndex DataFrame, it is sliced by the current group
        key and its grouping index levels are dropped before being passed to
        the wrapped function.

    Returns
    -------
    callable
        A decorator that can be applied to functions taking ``peaks_df`` and
        (optionally) additional DataFrame arguments. The decorated function
        accepts both grouped (MultiIndex) and ungrouped inputs.

    Examples
    --------
    Basic usage with a single ``peaks_df`` argument::

        @support_multiindex_peaks(peaks_arg="peaks_df")
        def compute_peak_stats(peaks_df: pd.DataFrame) -> pd.DataFrame:
            # expects peaks_df indexed only by "peak_index"
            return peaks_df.agg({"amplitude": "mean", "width": "mean"})

        # When peaks_df has a MultiIndex with (cell_id, peak_index),
        # compute_peak_stats is run once per cell_id, and the results are
        # concatenated with cell_id restored as an outer index level.
        stats = compute_peak_stats(peaks_df=multiindex_peaks_df)
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound = _bound_arguments(func, args, kwargs)
            if peaks_arg not in bound.arguments:
                peaks_df = kwargs.get(peaks_arg, None)
            else:
                peaks_df = bound.arguments[peaks_arg]

            # if not a MultiIndex DataFrame, call original
            if not isinstance(peaks_df, pd.DataFrame) or not isinstance(
                peaks_df.index, pd.MultiIndex
            ):
                return func(*args, **kwargs)

            # Require an explicit per-peak level name 'peak_index'. Do not infer.
            names = list(peaks_df.index.names)
            if "peak_index" not in names:
                raise ValueError(
                    "MultiIndex peaks_df must have a level named 'peak_index'. "
                    "Do not rely on automatic inference; rename your index levels."
                )
            group_level_names = [n for n in names if n != "peak_index"]

            # prepare call kwargs once
            call_kwargs = dict(bound.arguments)

            def _apply_fn(grp):
                # grp is the subgroup for one group key; grp.name is the group key
                group_key = grp.name
                key_tuple = _as_tuple(group_key)

                # drop the group levels from the peaks subgroup so the wrapped
                # function receives a peaks_df indexed by the per-spike level only
                peaks_local = grp
                if isinstance(peaks_local.index, pd.MultiIndex):
                    peaks_local = peaks_local.droplevel(group_level_names)

                local_kwargs = dict(call_kwargs)
                local_kwargs[peaks_arg] = peaks_local

                # for other dataframe/series args: if they are MultiIndex, subset
                # them to the same group key and drop the group levels
                for other_name in other_df_args:
                    other_df = call_kwargs.get(other_name, None)
                    if other_df is None:
                        continue
                    # if other_df is MultiIndex, require it has the same group levels
                    if isinstance(other_df, (pd.Series, pd.DataFrame)) and isinstance(
                        other_df.index, pd.MultiIndex
                    ):
                        other_names = list(other_df.index.names)
                        if not all(n in other_names for n in group_level_names):
                            raise ValueError(
                                f"Argument '{other_name}' must have group levels {group_level_names}"
                            )
                        subset = other_df.xs(
                            key_tuple, level=group_level_names, drop_level=True
                        )
                        local_kwargs[other_name] = subset

                res = func(**local_kwargs)
                return res

            result = peaks_df.groupby(level=group_level_names).apply(_apply_fn)

            # normalize empty-case
            if isinstance(result, pd.DataFrame):
                if result.empty:
                    return pd.DataFrame()
                return result
            return result

        return wrapper

    return decorator


def support_multiindex_signal_single_row_returns(
    signal_arg: str = "signal", *, group_levels=None, time_name="time"
):
    """Decorator for functions that accept a 1-D ``signal: pd.Series`` and return a
    single-row ``pd.DataFrame``.

    This decorator is a thin convenience wrapper around :func:`support_multiindex_signal`.
    It enables functions that operate on a single 1-D trace (a plain ``pd.Series``)
    and return a *single-row* DataFrame to be transparently applied to grouped or
    MultiIndex signals (for example, signals indexed by ``(cell_id, trial, time)``).

    When the wrapped function is applied group-wise via :meth:`pandas.Series.groupby.apply`,
    pandas will add an extra innermost index level corresponding to the row index of the
    returned DataFrame (typically ``0`` for a single-row result). This decorator removes
    that redundant innermost level so that the resulting index only reflects the group
    keys (and any existing index levels), making downstream operations and merges
    simpler.

    Parameters
    ----------
    signal_arg : str, optional
        Name of the keyword argument in the wrapped function that receives the
        signal. The signal is expected to be a 1-D ``pd.Series`` when called
        directly, or a (potentially MultiIndex) ``pd.Series`` when used in a
        grouped context. Defaults to ``"signal"``.
    group_levels : iterable of hashable, optional
        Names or integer positions of the index levels to treat as grouping
        levels when the input ``signal`` has a MultiIndex. If ``None``, all
        non-time levels are used as group levels (see
        :func:`support_multiindex_signal` for the exact behavior).
    time_name : hashable, optional
        Name of the index level that represents time. This is used by
        :func:`support_multiindex_signal` to distinguish time from grouping
        levels. Defaults to ``"time"``.

    Returns
    -------
    Callable
        A decorator that can be applied to a function ``f(signal: pd.Series, ...)``
        which returns a single-row ``pd.DataFrame``. The resulting wrapped
        function will:

        * Behave identically to ``f`` when passed a non-MultiIndex ``pd.Series``.
        * When given a MultiIndex ``pd.Series``, apply ``f`` per group and
          concatenate the per-group results into a single DataFrame with the
          redundant innermost index level removed.

    Examples
    --------
    Basic usage with a simple 1-D signal::

        @support_multiindex_signal_single_row_returns()
        def summarize_trace(signal: pd.Series) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "mean": [signal.mean()],
                    "std": [signal.std()],
                }
            )

        s = pd.Series([1.0, 2.0, 3.0], index=pd.Index([0, 1, 2], name="time"))
        summary = summarize_trace(signal=s)

    Usage with a MultiIndex signal grouped by ``cell_id`` and ``trial``::

        idx = pd.MultiIndex.from_product(
            [["cell_a", "cell_b"], [1, 2], [0, 1, 2]],
            names=["cell_id", "trial", "time"],
        )
        s = pd.Series(range(len(idx)), index=idx)

        # The result will be indexed by (cell_id, trial) only; the extra
        # innermost level created by groupby-apply is dropped.
        summary_by_group = summarize_trace(signal=s)
    """

    def decorator(func):
        # Apply the base decorator
        base_wrapper = support_multiindex_signal(
            signal_arg=signal_arg, group_levels=group_levels, time_name=time_name
        )(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = base_wrapper(*args, **kwargs)

            # If result is a MultiIndex DataFrame, drop the last level
            # This assumes the function returns a single row, so the last level
            # (added by the inner DataFrame's index) is redundant.
            if isinstance(result, pd.DataFrame) and isinstance(
                result.index, pd.MultiIndex
            ):
                return result.reset_index(level=-1, drop=True)

            return result

        return wrapper

    return decorator


# backward-compatible alias used by `peaks.py`
support_multiindex_peaks_signal = support_multiindex_peaks
