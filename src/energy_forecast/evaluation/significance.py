import numpy as np
from scipy.stats import norm


def diebold_mariano_test(errors_a, errors_b, h=1, lags=None, power=2):
    """
    Diebold-Mariano test of equal forecast accuracy. A negative statistic means forecast A
    has the lower loss.

    power: loss exponent, 2 = squared error (matches RMSE)
    h: forecast horizon in steps
    lags: number of autocorrelation lags to correct for (Bartlett-weighted
          Newey-West style). Defaults to h-1 if not specified.
    """
    loss_a = np.abs(errors_a) ** power
    loss_b = np.abs(errors_b) ** power
    d = loss_a - loss_b

    if lags is None:
        lags = max(h - 1, 0)

    T = len(d)
    d_mean = np.mean(d)

    # Autocovariances of d about its overall mean, all divided by T (Newey-West estimator)
    d_centred = d - d_mean
    gamma_0 = np.sum(d_centred ** 2) / T
    gamma_sum = 0
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)  # Bartlett weight
        gamma_lag = np.sum(d_centred[lag:] * d_centred[:-lag]) / T
        gamma_sum += 2 * weight * gamma_lag
    var_d = (gamma_0 + gamma_sum) / T

    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * (1 - norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value
