from imblearn.over_sampling import SMOTE
from oversampling.GeneratorBase import *
from keras.utils import to_categorical


class SmoteGenerator(GeneratorBase):

    def generate(self, dataX, dataY, examples_to_generate):
        sm = SMOTE(random_state=42)
        dataX = np.squeeze(dataX)
        dataY = dataY.argmax(axis=1)
        while dataX.shape[0] < examples_to_generate:
            dataX, dataY = sm.fit_resample(dataX, dataY)
        dataX = np.expand_dims(dataX, axis=2)
        dataY = to_categorical(dataY)
        return choose_n_samples(dataX, examples_to_generate)
