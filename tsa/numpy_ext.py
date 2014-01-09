import numpy as np


def _type_raise(prototype_array, force_dtype=None):
    '''
    For integer inputs, the default is float64;
    for floating point inputs, it is the same as the input dtype.
    '''
    if force_dtype is not None:
        return force_dtype

    if prototype_array.dtype in (np.int16, np.int32, np.int64):
        return np.float64

    return prototype_array.dtype


def head_indices(array, n):
    return range(0, n)

def tail_indices(array, n):
    return range(array.size - n, array.size)

def median_indices(array, n):
    median = array.size // 2
    half_n = n // 2
    return range(median - half_n, median + half_n + 1)

def edge_indices(array, n):
    return head_indices(array, n) + tail_indices(array, n)

def edge_and_median_indices(array, n):
    return head_indices(array, n) + median_indices(array, n) + tail_indices(array, n)


# def edgeindices(edgeitems):
#     # returns indices of the first <edgeitems> and the last <edgeitems> elements of an array
#     # (n) -> [0, 1, ..., (n - 1), -n, -(n + 1), ..., -(n - 1)]
#     # so, map(string.lowercase.__getitem__, margins(3))  ->  ['a', 'b', 'c', 'x', 'y', 'z']
#     # or, alphabet_array = np.array(list(string.lowercase))
#     #     alphabet_array[margins(3)]  ->  np.array(['a', 'b', 'c', 'x', 'y', 'z'])
#     # np.concatenate((indices[:edgeitems], indices[-edgeitems:]))
#     return range(0, edgeitems) + range(-edgeitems, 0)


# def edges(array, edgeitems):
#     return array[edgeindices(edgeitems)]


def bootstrap(n, n_iter, proportion=1.0):
    '''
    Like cross_validation.Bootstrap, but ignoring the test set

    Returns indices, all of which are between 0 and n
    '''
    size = int(n * proportion)
    for _ in range(n_iter):
        yield np.random.choice(n, size=size, replace=True), []


def bool_mask_to_indices(array):
    # return np.arange(array.size)[array]
    return np.where(array)[0]


def indices_to_bool_mask(indices, n=None):
    bools = np.zeros(n or indices.size, dtype=bool)
    bools[indices] = True
    return bools


def datespace(minimum, maximum, num, unit):
    # take the <unit>-floor of minimum
    # span = (maximum - minimum).astype('datetime64[%s]' % unit)
    delta = np.timedelta64(num, unit)
    start = minimum.astype('datetime64[%s]' % unit)
    end = maximum.astype('datetime64[%s]' % unit) + delta
    return np.arange(start, end + delta, delta).astype(minimum.dtype)


def table(ys, names=None):
    '''
    table(...) tabulates a list of item-count pairs, e.g.:
       [('For', 19118), ('NA', 0), ('Broken Link', 0), ('Against', 87584)]

    ys is generally a list of 0-indexed labels.
    names is generally a list of strings

    >>> np_table([0, 1, 0, 1, 2, 0], names=['a', 'b', 'c'])
    [('a', 3), ('b', 2), ('c', 1)]
    '''
    counts = np.bincount(ys)
    # list(enumerate(np.bincount(ys)))
    if names is None:
        names = range(len(counts))
    return zip(names, counts)


def datetime64(x):
    try:
        return np.datetime64(x)
    except ValueError:
        return np.datetime64()


def mean_accumulate(array, axis=0, dtype=None):
    '''
    Like np.mean.accumulate(...), except that doesn't work because np.mean is not a ufunc
    '''
    dtype = _type_raise(array, dtype)

    # ns = np.arange(1, array.shape[axis] + 1) -- only handles 1-D
    ns = np.add.accumulate(np.ones(array.shape), axis=axis)
    cumulative_sums = np.add.accumulate(array, axis=axis, dtype=dtype)
    # cumulative_means = cumulative_sums / ns
    return cumulative_sums / ns


def var_accumulate(array, axis=0, dtype=None):
    '''
    For example:

    >>> std1 = np.random.normal(scale=1, size=500)
    >>> hist(std1)
    >>> std5 = np.random.normal(scale=5, size=500)
    >>> hist(std5)
    >>> array = np.column_stack((std1, std5))

    >>> np_mean_accumulate(array, axis=0)
    >>> np_var_accumulate(array, axis=0)

    Check rows:
        last row = np.var(array, axis=0)
        penultimate row = np.var(array[:-1,], axis=0)
    etc.

    Also see alternatives on SO:
    * http://stackoverflow.com/questions/13828599/generalized-cumulative-functions-in-numpy-scipy
    * http://stackoverflow.com/questions/1174984/how-to-efficiently-calculate-a-running-standard-deviation
    '''
    dtype = _type_raise(array, dtype)

    cumulative_means = mean_accumulate(array, axis=axis, dtype=dtype)
    cumulative_means_squares = cumulative_means * cumulative_means
    '''
    >>> quadrants = np.array([[1, 2], [3, 4]])
    >>> quadrants * quadrants
    array([[ 1,  4],
           [ 9, 16]])
    '''
    squares = array * array
    squares_cumulative_means = mean_accumulate(squares, axis=axis, dtype=dtype)

    return squares_cumulative_means - cumulative_means_squares
