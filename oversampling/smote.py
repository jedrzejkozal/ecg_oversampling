import numpy as np
from imblearn.over_sampling import SMOTE
from oversampling.DataAugumentationGenerator import *


# smote cannot be implemented as plug-in for reduce_imbalance
# it needs all labels to produce additional examples
def smote(dataX, dataY):
    sm = SMOTE(random_state=42)
    resX, resY = sm.fit_resample(np.squeeze(dataX), dataY)
    resX = np.expand_dims(resX, axis=2)
    return resX, resY


def smote_with_data_agumentation(dataX, dataY):
    num_samples = dataX.shape[0]
    generatedX, generatedY = smote(dataX, dataY)
    augX = generatedX[num_samples:]  # choose samples generated by smote
    augX = augument_data(augX)
    return np.vstack([dataX, augX]), generatedY


def augument_data(generatedX):
    signal_shape = generatedX.shape[1:]
    random_noise = 0.1*np.random.rand(*signal_shape) - 0.05
    generatedX = generatedX+random_noise
    generatedX = np.apply_along_axis(
        DataAugumentationGenerator.stretch, axis=1, arr=generatedX)
    return generatedX
