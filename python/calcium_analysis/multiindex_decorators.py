"""Decorators to add MultiIndex / grouped support to single-trace peak functions.

The decorators detect MultiIndex `signal` or `peaks_df` inputs and apply the
wrapped function per-group, concatenating results while preserving group
level names.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Iterable, Tuple

import numpy as np
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
    """Decorator for functions that accept a 1-D `signal: pd.Series`.

    If a MultiIndex Series is passed for `signal` the function is executed for
    each group (all index levels except the last), and results are concatenated
    with the group keys prefixed to the returned index.
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
    """Decorator for functions accepting both `peaks_df` (grouped) and other dataframes.

    - If `peaks_df` is a MultiIndex DataFrame, iterate over group keys (all
      levels, except for "peak_index"), calling the wrapped function for each group with group levels dropped
    - For all other DataFrame arguments named in `other_df_args`, if they are
      MultiIndex DataFrames, subset them to the same group key (dropping group
      levels) before passing to the wrapped function
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
    """Decorator for functions that accept a 1-D `signal: pd.Series` and return a single-row DataFrame.

    Wraps `support_multiindex_signal` but drops the innermost index level (usually 0)
    from the result, which `groupby.apply` adds when the applied function returns a DataFrame.
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
