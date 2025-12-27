import numpy as np
from scipy.optimize import curve_fit

def exponential_decay(peak_timeseries: np.ndarray, peak_position_idx: int):
    decay_data = peak_timeseries[peak_position_idx:]

    xdata = decay_data.index.get_level_values('time from peak').values
    ydata = decay_data['Intensity'].values

    # Initial guess for parameters (A, k, C)
    # C: last intensity value is a reasonable guess for the offset/plateau
    p0_C = ydata[-1]
    # y0: intensity at time from peak = 0 (or first point if xdata does not start at 0)
    # Find the index of xdata where time from peak is exactly 0
    idx_at_zero = np.where(xdata == 0)[0][0]
    p0_y0 = ydata[idx_at_zero]

    # A: Amplitude is initial value minus offset
    p0_A = p0_y0 - p0_C
    # k: decay constant, a small positive number
    p0_k = 1

    # Set bounds for parameters: A > 0, k > 0, C can be any real number
    # Lower bounds: A (0), k (small positive like 1e-6), C (-inf)
    # Upper bounds: A (inf), k (inf), C (inf)
    bounds = ([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf])

    try:
        params, covariance = curve_fit(exponential_decay, xdata, ydata, p0=[p0_A, p0_k, p0_C], bounds=bounds)
        A, k, C = params

        y_predicted = exponential_decay(xdata, A, k, C)
        mse = np.mean((ydata - y_predicted) ** 2)

        ss_res = np.sum((ydata - y_predicted) ** 2)
        ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        fitted_params.append({
            'Row': name[0],
            'Column': name[1],
            'Object ID': name[2],
            'spike_index': name[3],
            'peak_over_baseline': A,
            'tau': 1 / k,
            'baseline': C,
            "mean_square_error": mse,
            "r2": r2,
        })