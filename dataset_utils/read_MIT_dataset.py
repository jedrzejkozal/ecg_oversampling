import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

from dataset_utils.load_data import *


def load():
    train = load_data('dataset/mitbih_train.csv')
    trainX = train[:, :-1]
    trainY = train[:, -1]
    test = load_data('dataset/mitbih_test.csv')
    testX = test[:, :-1]
    testY = test[:, -1]

    print("train set shape: ", trainX.shape)
    print("train set shape: ", trainY.shape)
    print("test set shape: ", testX.shape)
    print("test set shape: ", testY.shape)

    return trainX, trainY, testX, testY


def load_testing_dataset():
    trainX, trainY, testX, testY = load()

    return trainX, trainY, testX, testY


def load_validation_dataset(split=0.7):
    trainX, trainY, _, _ = load()
    trainX, validX, trainY, validY = train_test_split(
        trainX, trainY, test_size=split, random_state=42, stratify=trainY)

    return trainX, trainY, validX, validY
