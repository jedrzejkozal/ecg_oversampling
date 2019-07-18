import matplotlib.pyplot as plt
import numpy as np
from arch.bootstrap import MovingBlockBootstrap as MBB
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from statsmodels.nonparametric.smoothers_lowess import lowess
from seasonal_decompose import *


def stl_bagging(timeseries, num_samples=10):
    timeseries = preprocess(timeseries)
    bcox, boxcox_lambda = apply_boxcox(timeseries, boxcox_lambda=None)
    components = get_components(bcox)
    plot_components(bcox, components)
    return generate_additional_samples(timeseries, components, num_samples=num_samples, boxcox_lambda=boxcox_lambda)


def preprocess(timeseries):
    timeseries = timeseries.astype(np.float64)
    timeseries, removed_padding = remove_zero_padding(timeseries)
    return remove_zeros(timeseries)


def remove_zero_padding(timeseries):
    padding_begin = find_begining_of_padding(timeseries)
    timeseries_no_padding = timeseries[:padding_begin]
    return timeseries_no_padding, len(timeseries)-padding_begin


def find_begining_of_padding(timeseries):
    index = len(timeseries)-1
    while index > -1:
        if timeseries[index] != 0:
            return index+1
        index -= 1
    return 0


def remove_zeros(timeseries):
    eta = 1e-10
    return timeseries + eta


def apply_boxcox(timeseries, boxcox_lambda=0.9):
    return boxcox(timeseries.flatten(), lmbda=boxcox_lambda)


def get_components(bcox):
    is_seasonal = True
    if is_seasonal:
        _, seasonal, trend, remainder = seasonal_decompose(
            bcox, model='additive', freq=82, two_sided=False)
    else:
        = decomposition
        x = np.arange(len(bcox))
        trend = lowess(bcox, x, frac=0.6666666666666666, it=10,
                       is_sorted=True, return_sorted=False)
        remainder = bcox - trend
    return trend, seasonal, remainder


def plot_components(timeseries, components):
    trend, seasonal, remainder = components

    plt.figure(0, figsize=(10, 6))
    ax = plt.subplot(411)
    ax.plot(timeseries)
    plt.ylabel('timeseries')

    ax = plt.subplot(412)
    ax.plot(seasonal)
    plt.ylabel('seasonal')

    ax = plt.subplot(413)
    ax.plot(trend)
    plt.ylabel('trend')

    ax = plt.subplot(414)
    ax.plot(remainder)
    plt.ylabel('remainder')


def generate_additional_samples(timeseries, components, num_samples=10, boxcox_lambda=0.00001):
    trend, seasonal, remainder = components
    samples = [timeseries.reshape(1, len(timeseries))]
    bs = MBB(3, remainder)
    for data in bs.bootstrap(num_samples):
        bc = trend + seasonal + data[0][0]
        samples.append(inv_boxcox(
            bc, boxcox_lambda).reshape(1, len(timeseries)))
    return np.concatenate(samples, axis=0)
