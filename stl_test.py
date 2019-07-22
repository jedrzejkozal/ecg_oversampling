from oversampling.old_impl import *
import numpy as np
import pytest

from dataset_utils.read_MIT_dataset import *
from oversampling.stl_bagging_generator import *
from oversampling.reduce_imbalance import *


def test_400_samples_number_of_learning_examples_for_all_classes_is_equal():
    _, y_aug = get_dataset(400)
    check_if_bincount_is_equal(y_aug)


def test_400_samples_number_of_learning_examples_is_rgiht():
    _, y_aug = get_dataset(400)
    check_if_bincount_have_right_number_of_examples(y_aug, 400)


def test_400_samples_number_of_learning_examples_is_the_same_for_x_and_y():
    x_aug, y_aug = get_dataset(400)
    assert x_aug.shape[0] == y_aug.shape[0]


def test_400_samples_no_nans_or_infs():
    x_aug, y_aug = get_dataset(400)
    checks_for_nans_and_inf(x_aug)
    checks_for_nans_and_inf(y_aug)


def test_3000_samples_number_of_learning_examples_for_all_classes_is_equal():
    _, y_aug = get_dataset(3000)
    check_if_bincount_is_equal(y_aug)


def test_3000_samples_number_of_learning_examples_is_rgiht():
    _, y_aug = get_dataset(3000)
    check_if_bincount_have_right_number_of_examples(y_aug, 3000)


def test_3000_samples_number_of_learning_examples_is_the_same_for_x_and_y():
    x_aug, y_aug = get_dataset(3000)
    assert x_aug.shape[0] == y_aug.shape[0]


def test_3000_samples_no_nans_or_infs():
    x_aug, y_aug = get_dataset(3000)

    checks_for_nans_and_inf(x_aug)
    checks_for_nans_and_inf(y_aug)


def test_6000_samples_number_of_learning_examples_for_all_classes_is_equal():
    _, y_aug = get_dataset(6000)
    check_if_bincount_is_equal(y_aug)


def test_6000_samples_number_of_learning_examples_is_rgiht():
    _, y_aug = get_dataset(6000)
    check_if_bincount_have_right_number_of_examples(y_aug, 6000)


def test_6000_samples_number_of_learning_examples_is_the_same_for_x_and_y():
    x_aug, y_aug = get_dataset(6000)
    assert x_aug.shape[0] == y_aug.shape[0]


def get_dataset(num_examples):
    x_train, y_train, _, _ = load_testing_dataset()
    print_bincount(y_train)
    x_aug, y_aug = reduce_imbalance(
        x_train, y_train, stl_bagging_generator, num_examples=num_examples)
    print_bincount(y_aug)
    return x_aug, y_aug


def print_bincount(labels):
    print("bincount = {}".format(bincount(labels)))


def bincount(labels):
    return np.bincount(labels.astype('int64').flatten())


def check_if_bincount_is_equal(labels):
    bin = bincount(labels)
    for i in range(1, len(bin)):
        assert bin[i-1] == bin[i]


def check_if_bincount_have_right_number_of_examples(labels, num_examples):
    bin = bincount(labels)
    for b in bin:
        assert b == num_examples


def checks_for_nans_and_inf(x):
    assert not np.isnan(x).any()
    assert not np.isinf(x).any()
    assert np.isfinite(x).all()
