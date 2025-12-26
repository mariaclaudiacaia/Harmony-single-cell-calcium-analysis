import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
from calcium_signal_analysis.signal_preprocessing import rolling_window_norm, smooth_using_gaussian_kernel

def get_peak_positions_and_properties(
    y: pd.Series,
    dt: float,
) -> pd.DataFrame:
  v = y.values
  g_med = np.median(v)
  g_mad = np.median(np.abs(v - g_med))
  g_sigma = g_mad * 1.4826 # Conversion factor for Gaussian distribution if noise is Gaussian

  # These values are derived from common usage in similar notebooks
  # and kernel state values (e.g., HEIGHT_SIGMA, PROM_SIGMA)
  MIN_DIST_S = 0.5 # Minimum peak distance in seconds
  HEIGHT_SIGMA = 3.0 # Multiplier for g_sigma to set height threshold
  PROM_SIGMA = 2.0   # Multiplier for g_sigma to set prominence threshold

  min_peak_distance_idx = int(round(MIN_DIST_S / dt))
  height_threshold = g_med + HEIGHT_SIGMA * g_sigma
  prominence_threshold = PROM_SIGMA * g_sigma

  peaks, properties = find_peaks(v, height=height_threshold, prominence=prominence_threshold, distance=min_peak_distance_idx)

  properties["peak_centers_seconds"] = peaks * dt
  properties["peak_centers_idx"] = peaks

  _, _, start_idx, end_idx = peak_widths(v, peaks, rel_height=0.75)
  properties["start_idx"] = start_idx.astype(int)
  properties["end_idx"] = end_idx.astype(int)
  properties["start_seconds"] = start_idx * dt
  properties["end_seconds"] = end_idx * dt

  properties["truncated"] = (properties["start_idx"] == 0) | (properties["end_idx"] == (len(y) - 1))

  # be explicit about which time variables are just indices, and which are in seconds
  #properties["left_bases_idx"] = properties.pop("left_bases")
  #properties["right_bases_idx"] = properties.pop("right_bases")


  # HACK: declare that basis points are width end points
  properties["left_bases_idx"] = properties["start_idx"]
  properties["right_bases_idx"] = properties["end_idx"]

  properties["left_bases_seconds"] = properties["left_bases_idx"] * dt
  properties["right_bases_seconds"] = properties["right_bases_idx"] * dt

  # TODO: add more properties: e.g. widths
  #e.g.

  width_50, _, _, _ = peak_widths(v, peaks, rel_height=0.5)

  properties["width_50_seconds"] = width_50 * dt

  df = pd.DataFrame(properties)
  df.index.name = "spike_index"

  return df

def get_timeseries_per_spike_df(signal, peaks_and_properties, original_df: pd.DataFrame, dt: float):
  spike_data_list = []

  for idx, spike_props in peaks_and_properties.iterrows():
      # Extract identifiers from MultiIndex
      row_val, col_val, obj_id_val, spike_idx_val = idx

      # Extract spike properties
      left_base = spike_props['left_bases_idx']
      right_base = spike_props['right_bases_idx']
      peak_center = spike_props['peak_centers_idx']

      group_intensity = signal.loc[(row_val, col_val, obj_id_val)]

      # Slice the signal using 0-based indices for the segment
      # right_base is inclusive, so we add 1 to include the last point
      segment_intensity = group_intensity.iloc[left_base : right_base + 1]

      # Calculate 'time from peak'
      # peak_center is an absolute index in the original group_intensity series
      # relative_peak_center is the peak's index within the segment_intensity series
      relative_peak_center = peak_center - left_base
      time_from_peak_array = (np.arange(len(segment_intensity)) - relative_peak_center) * dt

      # Create a temporary DataFrame for this spike segment
      temp_df = pd.DataFrame({
          'Row': row_val,
          'Column': col_val,
          'Object ID': obj_id_val,
          'spike_index': spike_idx_val,
          'time from peak': time_from_peak_array,
          'Intensity': segment_intensity.values
      })

      # Append to the list
      spike_data_list.append(temp_df)

  spike_data_df = pd.concat(spike_data_list, ignore_index=True)
  spike_data_df = spike_data_df.set_index(['Row', 'Column', 'Object ID', 'spike_index', 'time from peak'])

  metadata_to_add = original_df.groupby(level=['Row', 'Column', 'Object ID'])[['Compound', 'Concentration', 'Cell Type']].first()
  spike_data_df = spike_data_df.merge(
        metadata_to_add,
        left_index=True,
        right_index=True,
        how='left'
  )

  return spike_data_df

def get_spike_dataframes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Takes the raw dataframe with Intensity and metadata columns.
    Returns two dataframes with:
      - informations for each spike
      - the full time-series of each spike
    """


    baseline = df["Intensity"].groupby(["Row", "Column", "Object ID"], group_keys=False).apply(
        rolling_window_norm,
        quant = 0.10, #quantile
        wind = 100, #window size
        min_periods = 50 #can be passed to the function, if None, the standard is min_periods = wind
    )

    normalized_value = df["Intensity"] / baseline - 1

    smoothed_normalized = normalized_value.groupby(["Row", "Column", "Object ID"]).apply(
        smooth_using_gaussian_kernel,
        #add other args as needed
    )

    peaks_and_properties = smoothed_normalized.groupby(["Row", "Column", "Object ID"]).apply(
        get_peak_positions_and_properties
    )


    metadata_to_add = df.groupby(level=['Row', 'Column', 'Object ID'])[['Compound', 'Concentration', 'Cell Type']].first()
    peaks_and_properties = peaks_and_properties.merge(
        metadata_to_add,
        left_index=True,
        right_index=True,
        how='left'
    )

    spike_data_df = get_timeseries_per_spike_df(smoothed_normalized, peaks_and_properties)

    return smoothed_normalized, peaks_and_properties, spike_data_df

