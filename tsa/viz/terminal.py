import os
import numpy as np

term_rows, term_columns = map(int, os.popen('stty size', 'r').read().split())


def fmt_float(x, max_width):
    '''fmt_float will ensure that a number's decimal part is truncated to fit within some bounds,
    unless the whole part is wider than max_width, which is a problem is a problem you need to sort out for yourself.
    '''
    # width of (whole part + 1 (to avoid zero)) + 1 because int floors, not ceils
    whole_width = int(np.log10(abs(x) + 1)) + 1
    # for +/- sign
    sign_width = 1 if x < 0 else 0
    # for . if we show it
    decimal_point_width = 1 if max_width >= whole_width else 0
    return '%.*f' % (max_width - whole_width - sign_width - decimal_point_width, x)


def hist(xs, range=None, margin=10, width=term_columns):
    '''Usage:
    import scipy.stats
    draws = scipy.stats.norm.rvs(size=100, loc=100, scale=10)
    hist(draws, margin=5)
    '''
    # I'm not sure why my font in iterm doesn't like \u2588, but it looks weird.
    #   It's too short and not the right width.
    chars = u' \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2589'
    # add 1 to each margin for the [ and ] brackets
    width = term_columns - (2 * (margin + 1))
    # compute the histogram values as floats, which is easier, even though we renormalize anyway
    hist_values, bin_edges = np.histogram(xs, bins=width, density=True, range=range)
    # we want the highest hist_height to be 1.0
    hist_heights = hist_values / max(hist_values)
    # np.array(...).astype(int) will floor each value
    hist_chars = (hist_heights * (len(chars) - 1)).astype(int)
    cells = [chars[hist_char] for hist_char in hist_chars]

    print '%s[%s]%s' % (
        fmt_float(bin_edges[0], margin).rjust(margin),
        u''.join(cells),
        fmt_float(bin_edges[-1], margin).ljust(margin))


if __name__ == '__main__':
    print 'running example ...'
    import scipy.stats
    draws = scipy.stats.norm.rvs(size=1000, loc=-100, scale=10)
    hist(draws, margin=10)
