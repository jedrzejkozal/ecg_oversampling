import numpy as np
from oversampling.stl_bagging import *


def stl_bagging_generator(dataX, num_examples_we_want):
    num_examples = dataX.shape[0]
    examples_to_generate = (num_examples_we_want - num_examples)
    examples_per_signal = examples_to_generate // num_examples
    rest = examples_to_generate - num_examples * examples_per_signal
    assert examples_per_signal * num_examples + \
        num_examples + rest == num_examples_we_want

    result = np.copy(dataX)
    for i in range(num_examples):
        if i < rest:
            generate = examples_per_signal+1
        else:
            generate = examples_per_signal
        if generate > 0:
            additional_samples = stl_bagging(dataX[i], num_samples=generate)
            additional_samples = np.expand_dims(additional_samples, axis=2)
            result = np.vstack([result, additional_samples])
    return result
