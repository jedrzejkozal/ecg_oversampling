import numpy as np


def reduce_imbalance(dataX, dataY, examples_generator, num_examples=6000):
    resultX = []
    resultY = []

    for class_index, class_x, _ in get_classes_examples(dataX, dataY):
        x = oversample_or_undersample(
            class_x, examples_generator, num_examples)
        resultX.append(x)
        y = np.ones((num_examples,)) * class_index
        resultY.append(y)

    return np.vstack(resultX), np.hstack(resultY)


def oversample_or_undersample(class_x, examples_generator, num_examples):
    num_class_samples = class_x.shape[0]
    if num_class_samples >= num_examples:  # undersampling
        x = choose_n_samples(class_x, num_examples)
    else:  # oversampling
        x = examples_generator(class_x, num_examples)
    return x


def get_classes_examples(dataX, dataY):
    num_classes = 5
    for i in range(num_classes):
        yield (i,) + get_class_examples(dataX, dataY, i)


def get_class_examples(dataX, dataY, class_index):
    indexes = get_class_indexes(dataY, class_index)
    return dataX[indexes], dataY[indexes]


def get_class_indexes(dataY, class_index):
    return np.argwhere(dataY.flatten() == class_index).flatten()


def choose_n_samples(dataX, n):
    all_indexes = np.arange(dataX.shape[0])
    chosen = np.random.choice(all_indexes, n, replace=False)
    del all_indexes

    return dataX[chosen]
