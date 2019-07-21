import numpy as np
from keras.utils.np_utils import to_categorical
from pandas import read_csv
from scipy.signal import resample


# data augumentation
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
    result = np.zeros(shape=(3, 187))
    result[0, :] = stretch(x)
    result[1, :] = amplify(x)
    result[2, :] = amplify(stretch(x))
    return result


def load_data(filename):
    dataframe = read_csv(filename, engine='python', encoding='ascii')
    dataset = dataframe.values
    return dataset.astype('float32')


def oversample_class(trainX, trainY, class_index):
    class_exaples = np.argwhere(trainY == class_index).flatten()

    additional_samples = np.apply_along_axis(
        augment, axis=1, arr=trainX[class_exaples]).reshape(-1, 187)
    additional_labels = np.ones(
        shape=(additional_samples.shape[0],), dtype=int)*class_index
    trainX = np.vstack([trainX, additional_samples])
    trainY = np.hstack([trainY, additional_labels])
    return trainX, trainY


def load():
    trainX = load_data('dataset/mitbih_train.csv')[:, :-1]
    trainY = load_data('dataset/mitbih_train.csv')[:, -1]
    testX = load_data('dataset/mitbih_test.csv')[:, :-1]
    testY = load_data('dataset/mitbih_test.csv')[:, -1]

    print("train set shape: ", trainX.shape)
    print("train set shape: ", trainY.shape)
    print("test set shape: ", testX.shape)
    print("test set shape: ", testY.shape)

    trainX, trainY = oversample_class(trainX, trainY, 1)
    trainX, trainY = oversample_class(trainX, trainY, 2)
    trainX, trainY = oversample_class(trainX, trainY, 3)
    trainX, trainY = oversample_class(trainX, trainY, 3)

    print("bincount before choosing 800 samples from each class: ",
          np.bincount(np.array(trainY, dtype=np.int64)))

    trainX, trainY, _, _ = choose_n_samples_from_each_class(
        trainX, trainY, 6000)

    print(np.bincount(np.array(trainY, dtype=np.int64)))

    print("------------------")
    print("train set shape: ", trainX.shape)
    print("train set shape: ", trainY.shape)
    print("test set shape: ", testX.shape)
    print("test set shape: ", testY.shape)

    return trainX, trainY, testX, testY


def choose_n_samples_from_each_class(dataX, dataY, n):
    # classes indexes
    C0 = np.argwhere(dataY == 0).flatten()
    C1 = np.argwhere(dataY == 1).flatten()
    C2 = np.argwhere(dataY == 2).flatten()
    C3 = np.argwhere(dataY == 3).flatten()
    C4 = np.argwhere(dataY == 4).flatten()

    # random choice of indexes
    subC0 = np.random.choice(C0, n, replace=False)
    subC1 = np.random.choice(C1, n, replace=False)
    subC2 = np.random.choice(C2, n, replace=False)
    subC3 = np.random.choice(C3, n, replace=False)
    subC4 = np.random.choice(C4, n, replace=False)

    remainingC0 = get_diff(C0, subC0)
    remainingC1 = get_diff(C1, subC1)
    remainingC2 = get_diff(C2, subC2)
    remainingC3 = get_diff(C3, subC3)
    remainingC4 = get_diff(C4, subC4)

    dataX_result = np.vstack(
        [dataX[subC0], dataX[subC1], dataX[subC2], dataX[subC3], dataX[subC4]])
    dataY_result = np.hstack(
        [dataY[subC0], dataY[subC1], dataY[subC2], dataY[subC3], dataY[subC4]])

    dataX_remaining = np.vstack([dataX[remainingC0], dataX[remainingC1],
                                 dataX[remainingC2], dataX[remainingC3], dataX[remainingC4]])
    dataY_remaining = np.hstack([dataY[remainingC0], dataY[remainingC1],
                                 dataY[remainingC2], dataY[remainingC3], dataY[remainingC4]])

    return dataX_result, dataY_result, dataX_remaining, dataY_remaining


def get_diff(a, b):
    a_set = frozenset(a)
    b_set = frozenset(b)

    return np.array(list(a_set - b_set))


def load_whole_dataset():
    trainX, trainY, testX, testY = load()

    trainX = np.expand_dims(trainX, axis=3)
    trainY = np.expand_dims(trainY, axis=2)
    testX = np.expand_dims(testX, axis=3)
    testY = np.expand_dims(testY, axis=2)

    trainY, testY = change_y_to_categorical(trainY, testY)
    return trainX, trainY, testX, testY


def load_validation_dataset(split=0.7):
    trainX_whole, trainY_whole, _, _ = load()
    trainX, trainY, validX, validY = choose_n_samples_from_each_class(trainX_whole, trainY_whole,
                                                                      int(6000*split))

    print(np.bincount(np.array(trainY, dtype=np.int64)))
    print("------------------")
    print("train set shape: ", trainX.shape)
    print("train set shape: ", trainY.shape)
    print("valid set shape: ", validX.shape)
    print("valid set shape: ", validY.shape)

    trainX = np.expand_dims(trainX, axis=3)
    trainY = np.expand_dims(trainY, axis=2)
    validX = np.expand_dims(validX, axis=3)
    validY = np.expand_dims(validY, axis=2)

    trainY, validY = change_y_to_categorical(trainY, validY)
    return trainX, trainY, validX, validY


def change_y_to_categorical(trainY, testY):
    num_classes = 5
    trainY = to_categorical(trainY, num_classes)
    testY = to_categorical(testY, num_classes)
    return trainY, testY
