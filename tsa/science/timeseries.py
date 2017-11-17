import numpy as np
import scipy
from tsa.science import numpy_ext as npx


# def binned_timeseries_1d(times, values, time_units_per_bin=1, time_unit='D', statistic='mean'):
#     '''
#     '''
#     first, last = npx.bounds(times)
#     bins = npx.datespace(first, last, time_units_per_bin, time_unit).astype('datetime64[s]')
#     # valid "statistic" strings: mean, median, count, sum
#     result = scipy.stats.binned_statistic(times.astype(float), values,
#         statistic=statistic, bins=bins.astype(float))
#     bin_statistics, bin_edges, bin_number = result
#     # the resulting statistic is 1 shorter than the bins
#     # return the left edges of the bins, and the resulting statistics
#     return bins[:-1], bin_statistics


def binned_timeseries(times, array, time_units_per_bin=1, time_unit='D', statistic='mean'):
    '''
    We want to end up with a matrix with the `array` scrunched vertically,
    but the same width.

    times: an array of datetime64s
    values: an array of numbers of whatever datatype

    Valid "statistic" strings:
      mean, median, count, or sum
      Can also be a function.
    '''
    first, last = npx.bounds(times)
    bins = npx.datespace(first, last, time_units_per_bin, time_unit).astype('datetime64[s]')
    # scipy.stats.binned_statistic requires floats
    times_floats = times.astype(float)
    bins_floats = bins.astype(float)

    # the resulting statistic is 1 shorter than the bins (a 12-inch ruler has 13 labels)
    bin_statistics_array = np.zeros((bins.size - 1, array.shape[1]))
    # column by column (feature by feature)
    for col in range(array.shape[1]):
        result = scipy.stats.binned_statistic(times_floats, array[:, col],
                                              statistic=statistic,
                                              bins=bins_floats)
        bin_statistics, _, _ = result
        bin_statistics_array[:, col] = bin_statistics

    # return the left edges of the bins (as the original datatype0, and the resulting statistics
    return bins[:-1], bin_statistics_array
