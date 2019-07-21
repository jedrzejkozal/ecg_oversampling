#from oversampling.old_impl import *
import numpy as np
import pytest

from dataset_utils.read_MIT_dataset import *
from oversampling.data_augumentation_generator import *
from oversampling.reduce_imbalance import *


def bincount(labels):
    return np.bincount(labels.astype('int64'))


def print_bincount(labels):
    print("bincount = {}".format(bincount(labels)))


def check_bincount(labels):
    bin = bincount(labels)
    for i in range(1, len(bin)):
        assert bin[i-1] == bin[i]


def test_new_impl_400_samples_count_of_learning_examples_for_all_classes_is_equal():
    x_train, y_train, _, _ = load_testing_dataset()
    print_bincount(y_train)
    x_aug, y_aug = reduce_imbalance(
        x_train, y_train, data_augumentation_generator, num_examples=400)
    print_bincount(y_aug)
    check_bincount(y_aug)


def test_new_impl_3000_samples_count_of_learning_examples_for_all_classes_is_equal():
    x_train, y_train, _, _ = load_testing_dataset()
    print_bincount(y_train)
    x_aug, y_aug = reduce_imbalance(
        x_train, y_train, data_augumentation_generator, num_examples=3000)
    print_bincount(y_aug)
    check_bincount(y_aug)


def test_new_impl_6000_samples_count_of_learning_examples_for_all_classes_is_equal():
    x_train, y_train, _, _ = load_testing_dataset()
    print_bincount(y_train)
    x_aug, y_aug = reduce_imbalance(
        x_train, y_train, data_augumentation_generator, num_examples=6000)
    print_bincount(y_aug)
    check_bincount(y_aug)


def old_impl_test():
    num_classes = 5
    x_train, y_train, x_valid, y_valid = load_whole_dataset()


if __name__ == "__main__":
    new_impl_test()
