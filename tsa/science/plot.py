import os
import itertools
import numpy as np

from tsa import logging
logger = logging.getLogger(__name__)

import matplotlib.cm as colormap
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# import matplotlib.dates as mdates
plt.rcParams['interactive'] = True
plt.rcParams['axes.grid'] = True

qmargins = [0, 5, 10, 50, 90, 95, 100]

# plt.rcParams['ps.useafm'] = True
# plt.rcParams['pdf.use14corefonts'] = True
# plt.rcParams['text.usetex'] = True


def fig_path(name, index=0):
    dirpath = os.path.expanduser('~/Dropbox/ut/qp/figures-qp-2')
    base, ext = os.path.splitext(name)
    filename = base + ('-%02d' % index if index > 0 else '') + ext
    filepath = os.path.join(dirpath, filename)
    if os.path.exists(filepath):
        return fig_path(name, index + 1)
    logger.info('Using filepath: %r', filepath)
    return filepath


def clear():
    plt.cla()
    plt.axes(aspect='auto')
    # plt.axis('tight')
    # plt.tight_layout()
    plt.margins(0.025, tight=True)

def style_gen():
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

def style_loop():
    # this will take those 72 style combos from style_gen() and loop indefinitely.
    return itertools.cycle(style_gen())
