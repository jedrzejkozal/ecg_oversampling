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
    print("dataX.shape = ", dataX.shape)
    num_samples = dataX.shape[0]
    augX, augY = smote(dataX, dataY)
    generatedX = augX[num_samples:]
    generatedX = augument_data(generatedX)
    print(dataX.shape)
    print(generatedX.shape)
    return np.vstack([dataX, generatedX]), augY


def augument_data(generatedX):
    random_noise = 0.1*np.random.rand() - 0.05
    generatedX = generatedX+random_noise
    generatedX = np.apply_along_axis(
        DataAugumentationGenerator.stretch, axis=1, arr=generatedX)
    return generatedX
