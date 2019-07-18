from scipy.signal import resample

import numpy as np


#data augumentation
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

def augment(x):
    result = np.zeros(shape= (3, 187))
    result[0, :] = stretch(x)
    result[1, :] = amplify(x)
    result[2, :] = amplify(stretch(x))
    return result
