import numpy as np
from sklearn.model_selection import train_test_split

from dataset_utils.load_data import *


def load_train():
    train = load_data('dataset/mitbih_train.csv')
    trainX = train[:, :-1]
    trainY = train[:, -1]

    return expand_dims(trainX, trainY)


def load_test():
    test = load_data('dataset/mitbih_test.csv')
    testX = test[:, :-1]
    testY = test[:, -1]

    return expand_dims(testX, testY)


def expand_dims(trainX, trainY):
    trainX = np.expand_dims(trainX, axis=3)
    trainY = np.expand_dims(trainY, axis=2)
    return trainX, trainY


def load_testing_dataset():
    trainX, trainY = load_train()
    testX, testY = load_test()

    return trainX, trainY, testX, testY


def load_validation_dataset(split=0.3):
    trainX, trainY = load_test()
    trainX, validX, trainY, validY = train_test_split(
        trainX, trainY, test_size=split, random_state=42, stratify=trainY)

    return trainX, trainY, validX, validY
