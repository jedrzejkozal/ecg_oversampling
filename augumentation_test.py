from oversampling.old_impl import *
import numpy as np
import pytest

from dataset_utils.read_MIT_dataset import *
from oversampling.data_augumentation_generator import *
from oversampling.reduce_imbalance import *


def test_400_samples_count_of_learning_examples_for_all_classes_is_equal():
    _, y_aug = get_dataset(400)
    check_bincount(y_aug)


def test_3000_samples_count_of_learning_examples_for_all_classes_is_equal():
    _, y_aug = get_dataset(3000)
    check_bincount(y_aug)


def test_6000_samples_count_of_learning_examples_for_all_classes_is_equal():
    _, y_aug = get_dataset(6000)
    check_bincount(y_aug)


def get_dataset(num_examples):
    x_train, y_train, _, _ = load_testing_dataset()
    print_bincount(y_train)
    x_aug, y_aug = reduce_imbalance(
        x_train, y_train, data_augumentation_generator, num_examples=num_examples)
    print_bincount(y_aug)
    return x_aug, y_aug


def print_bincount(labels):
    print("bincount = {}".format(bincount(labels)))


def bincount(labels):
    return np.bincount(labels.astype('int64').flatten())


def check_bincount(labels):
    bin = bincount(labels)
    for i in range(1, len(bin)):
        assert bin[i-1] == bin[i]


def old_impl_test():
    num_classes = 5
    x_train, y_train, x_valid, y_valid = load_whole_dataset()
    return x_train, y_train


def test_regression_shape_is_the_same():
    x_old, _ = old_impl_test()
    x_aug, _ = get_dataset(6000)
    assert x_aug.shape == x_old.shape


def test_regression_bincount_is_the_same():
    _, y_old = old_impl_test()
    _, y_aug = get_dataset(6000)
    assert (bincount(y_old) == bincount(y_aug)).all()
