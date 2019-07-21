import numpy as np
from scipy.signal import resample


def data_augumentation_generator(dataX, examples_to_generate):
    while dataX.shape[0] < examples_to_generate:
        additional_samples = np.apply_along_axis(
            augment, axis=1, arr=dataX).reshape(-1, 187)
        dataX = np.vstack([dataX, additional_samples])
    return choose_n_samples(dataX, examples_to_generate)


def augment(x):
    result = np.zeros(shape=(3, 187))
    result[0, :] = stretch(x)
    result[1, :] = amplify(x)
    result[2, :] = amplify(stretch(x))
    return result


def stretch(x):
    sum_samples = int(187 * (1 + (np.random.rand()-0.5)/3))
    y = resample(x, sum_samples)
    if sum_samples < 187:
        y_ = np.zeros(shape=(187, ))
        y_[:sum_samples] = y
    else:
        y_ = y[:187]
    return y_


def amplify(x):
    alpha = (np.random.rand()-0.5)
    factor = -alpha*x + (1+alpha)
    return x*factor


def choose_n_samples(dataX, n):
    all_indexes = np.arange(dataX.shape[0])
    chosen = np.random.choice(all_indexes, n, replace=False)
    del all_indexes

    return dataX[chosen]
