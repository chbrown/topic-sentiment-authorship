import math


def nball(n):
    # returns the volume of the unit ball in n-dimensional space
    # going off the formula on Wikipedia
    return (math.pi ** (n / 2.0)) / math.gamma((n / 2.0) + 1)


def curse_of_dimensionality(analysis_options):
    for n in range(20):
        print 'n', n
        volume = nball(n)
        print '(nball) volume', volume
        cube = 2.0**n
        print 'containing cube', cube
        print 'volume/cube', volume / cube
        print
    # yes, the volume decreases as you go down, but say you hold volume constant, at 1
    # then the radius of this nball where n > 3 would start to become extremely spiky
    # like a pufferfish at first, then like a sea urchin (no?)
