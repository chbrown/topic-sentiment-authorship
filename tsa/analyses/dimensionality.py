import math
import numpy as np
import IPython

from tsa.science import numpy_ext as npx


def nball(n):
    # returns the volume of the unit ball in n-dimensional space
    # going off the formula on Wikipedia
    return (math.pi ** (n / 2.0)) / math.gamma((n / 2.0) + 1)


def curse_of_dimensionality(analysis_options):
    for n in range(20):
        print('n', n)
        volume = nball(n)
        print('(nball) volume', volume)
        cube = 2.0**n
        print('containing cube', cube)
        print('volume/cube', volume / cube)
        print()
    # yes, the volume decreases as you go down, but say you hold volume constant, at 1
    # then the radius of this nball where n > 3 would start to become extremely spiky
    # like a pufferfish at first, then like a sea urchin (no?)


def harmonic_demo(analysis_options):
    print('This prints a sort of Dirichlet / joint probability space')
    print('and displays the value of the harmonic mean')

    ticks = np.arange(100) + 1.0
    grid = (ticks / ticks.max())
    edge = grid.reshape(1, -1)

    # 2-dimensional
    plt.cla()
    x = grid
    y = 1 - x
    pairs = np.column_stack((x.ravel(), y.ravel()))
    pos_pairs = pairs[y.ravel() > 0]
    # each row in `pairs` sums to 1
    xy_harmonic = npx.hmean(pos_pairs, axis=1)
    plt.scatter(pos_pairs[:, 0], xy_harmonic, c=xy_harmonic, linewidths=0)
    plt.xlabel('x, y = 1 - x')
    plt.ylabel('Harmonic mean of (x, y)')
    plt.title('2D (color and height are equivalent)')


    # 3-dimensional
    plt.cla()
    a = np.repeat(edge, edge.size, axis=0)
    b = a.T
    c = 1 - (a + b)
    triples = np.column_stack((a.ravel(), b.ravel(), c.ravel()))
    pos_triples = triples[c.ravel() > 0]
    abc_harmonic = npx.hmean(pos_triples, axis=1)
    plt.scatter(pos_triples[:, 0], pos_triples[:, 1], c=abc_harmonic, linewidths=0)
    plt.xlabel('a')
    plt.ylabel('b')
    plt.title('3D')
    plt.suptitle('color = c = 1 - a - b (where c > 0)')

    IPython.embed()
