import numpy as np
from oversampling.stl_bagging import *


def stl_bagging_generator(dataX, examples_to_generate):
    num_examples = dataX.shape[0]
    examples_per_class = examples_to_generate // num_examples
    rest = num_examples % examples_per_class
    for i in range(num_examples):
        additional_samples = additional_samples_for_signal(
            dataX[i], i, rest, examples_per_class)
        additional_samples = np.expand_dims(additional_samples, axis=2)
        dataX = np.vstack([dataX, additional_samples])
    return dataX


def additional_samples_for_signal(signal, i, rest, examples_per_class):
    num_additional_samples = how_many_to_generate(
        i, rest, examples_per_class)
    return stl_bagging(
        signal, num_samples=num_additional_samples)


def how_many_to_generate(i, rest, examples_per_class):
    if i <= rest:
        return examples_per_class + 1
    else:
        return examples_per_class
