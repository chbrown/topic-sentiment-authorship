import os
import numpy as np

from tsa import logging
logger = logging.getLogger(__name__)

from itertools import cycle, izip
import matplotlib.cm as colormap
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# import matplotlib.dates as mdates
# plt.rcParams['interactive'] = True
# plt.rcParams['axes.grid'] = True

qmargins = [0, 5, 10, 50, 90, 95, 100]

# plt.rcParams['ps.useafm'] = True
# plt.rcParams['pdf.use14corefonts'] = True
# plt.rcParams['text.usetex'] = True


def figure_path(name):
    '''
    name should be a full filename, like "issue2.pdf"
    This will never return a filename that exists at the time when the function returns.
    '''
    dirpath = os.path.expanduser('~/Dropbox/ut/qp/qp-2/figures')
    # make name filesystem-safe
    # also make latex-friendly, because latex doesn't like underscore in filenames (so weird!)
    name = name.replace('/', ' vs ').replace('  ', ' ').replace(':', '-').replace(' ', '-').replace('_', '-')

    base, ext = os.path.splitext(name)
    if len(ext) <= 1:
        ext = '.pdf'

    for index in range(100):
        filename = base + ('-%02d' % index if index > 0 else '') + ext
        filepath = os.path.join(dirpath, filename)
        if not os.path.exists(filepath):
            logger.info('Using filepath: %r', filepath)
            return filepath
    else:
        raise Exception('Could not find unused file! Last tried: %s' % filepath)


# def clear():
#     plt.cla()
#     plt.axes(aspect='auto')
#     # plt.axis('tight')
#     # plt.tight_layout()
#     plt.margins(0.025, tight=True)


# markers = {0: 'tickleft', 1: 'tickright', 2: 'tickup', 3: 'tickdown', 4: 'caretleft', 'D': 'diamond', 6: 'caretup', 7: 'caretdown', 's': 'square', '|': 'vline', '': 'nothing', 'None': 'nothing', 'x': 'x', 5: 'caretright', '_': 'hline', '^': 'triangle_up', None: 'nothing', 'd': 'thin_diamond', ' ': 'nothing', 'h': 'hexagon1', '+': 'plus', '*': 'star', ',': 'pixel', 'o': 'circle', '.': 'point', '1': 'tri_down', 'p': 'pentagon', '3': 'tri_left', '2': 'tri_up', '4': 'tri_right', 'H': 'hexagon2', 'v': 'triangle_down', '8': 'octagon', '<': 'triangle_left', '>': 'triangle_right'}

def _styles():
    # I can distinguish about six different colors from rainbow
    colors = colormap.rainbow(np.linspace(1, 0, 6))
    # and a few different linestyles
    linestyles = ['-', ':', '--', '-.']
    # and linewidth
    linewidths = [1, 2, 3]
    # this produces 6*4*3 = 72 styles, way more than you should really put on a single plot.
    for linewidth in linewidths:
        for linestyle in linestyles:
            for color in colors:
                yield dict(linewidth=linewidth, linestyle=linestyle, color=color)

# this will take those 72 style combos from style_gen() and loop indefinitely.
styles = cycle(_styles())


def distinct_styles():
    # e.g., for the colorblind
    linewidths = [1, 2, 3]
    linestyles = ['-', ':', '--', '-.']
    # array([0, 1, 2, 3, 4, 5])
    color_space = np.linspace(1, 0, 6)[np.array([0, 4, 2, 5, 3, 1])]
    colors = colormap.rainbow(color_space)
    # period = 60
    # markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']

    zipped = izip(cycle(linewidths), cycle(linestyles), cycle(colors))
    for linewidth, linestyle, color in zipped:
        yield dict(linewidth=linewidth, linestyle=linestyle, color=color)
