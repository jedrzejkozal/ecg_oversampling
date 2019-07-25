from imblearn.over_sampling import SMOTE
from oversampling.GeneratorBase import *


class SmoteGenerator(GeneratorBase):

    def generate(self, dataX, dataY, examples_to_generate):
        sm = SMOTE(random_state=42)
        while dataX.shape[0] < examples_to_generate:
            dataX, dataY = sm.fit_resample(dataX, dataY)

        return choose_n_samples(dataX, examples_to_generate)
