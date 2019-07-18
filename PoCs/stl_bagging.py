import os
import sys

import numpy as np
from matplotlib import pyplot as plt


def plot_components(x, components, fignum=1):
    orginal, seasonal, trend, resid = components
    plt.figure(fignum)
    ax = plt.subplot(411)
    ax.plot(x, orginal)
    plt.subplot(412).plot(x, seasonal)
    plt.subplot(413).plot(x, trend)
    plt.subplot(414).plot(x, resid)


if __name__ == "__main__":
    PACKAGE_PARENT = '../'
    SCRIPT_DIR = os.path.dirname(os.path.realpath(
        os.path.join(os.getcwd(), os.path.expanduser(__file__))))
    sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
    from oversampling.seasonal_decompose import *

    x = np.arange(0, 30, 0.05)

    sin = 10 * np.sin(x*(2*np.pi))
    x_2 = x**2 / 50
    rand = np.random.rand(sin.size)
    y = sin + x_2 + rand

    plot_components(x, (y, sin, x_2, rand))

    decomposition = seasonal_decompose(
        y, model='additive', freq=20)  # frequency is period in samples WTF?!
    plot_components(x, decomposition, fignum=2)
    plt.show()
