def whichmin(xs):
    # import operator
    # min_index, min_value = min(enumerate(values), key=operator.itemgetter(1))
    return min(enumerate(xs), key=lambda i_x: i_x[1])[0]


def whichmax(xs):
    return max(enumerate(xs), key=lambda i_x: i_x[1])[0]
