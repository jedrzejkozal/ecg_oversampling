import numpy as np
from scipy.signal import resample

from oversampling.GeneratorBase import *


class DataAugumentationGenerator(GeneratorBase):

    def generate(self, dataX, dataY, examples_to_generate):
        while dataX.shape[0] < examples_to_generate:
            additional_samples = np.apply_along_axis(
                self.augment, axis=1, arr=dataX).reshape(-1, 187, 1)
            dataX = np.vstack([dataX, additional_samples])
        return self.choose_n_samples(dataX, examples_to_generate)

    def augment(self, x):
        result = np.zeros(shape=(3, 187))
        result[0, :] = self.stretch(x)
        result[1, :] = self.amplify(x)
        result[2, :] = self.amplify(self.stretch(x))
        return result

    def stretch(self, x):
        sum_samples = int(187 * (1 + (np.random.rand()-0.5)/3))
        y = resample(x, sum_samples)
        if sum_samples < 187:
            y_ = np.zeros(shape=(187, ))
            y_[:sum_samples] = y
        else:
            y_ = y[:187]
        return y_

    def amplify(self, x):
        alpha = (np.random.rand()-0.5)
        factor = -alpha*x + (1+alpha)
        return x*factor
