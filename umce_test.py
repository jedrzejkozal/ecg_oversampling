import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from ensamble.umce import *


def simple_dataset():
    all_samples = 30
    x = np.arange(all_samples)
    class_a = np.full((5,), 0)
    class_b = np.full((10,), 1)
    class_c = np.full((15,), 2)

    return x, np.hstack([class_a, class_b, class_c])


def sample_dummy_classifier():
    return DummyClassifier()


def test_get_samples_in_classes_for_simple_dataset_all_returned_y_arrays_contain_only_one_label():
    x_train, y_train = simple_dataset()
    sut = MuticlassUMCE(sample_dummy_classifier)
    _, y_classes = sut.get_samples_in_classes(x_train, y_train)
    for y in y_classes:
        array_from_first_label = np.full(y.shape, y[0])
        assert np.array_equal(y, array_from_first_label)


def test_get_samples_in_classes_for_simple_dataset_all_returned_x_y_have_same_number_of_examples():
    x_train, y_train = simple_dataset()
    sut = MuticlassUMCE(sample_dummy_classifier)
    x_train, y_classes = sut.get_samples_in_classes(x_train, y_train)
    for x, y in zip(x_train, y_classes):
        assert x.shape[0] == y.shape[0]


def test_get_samples_in_classes_for_simple_dataset_all_classes_found():
    x_train, y_train = simple_dataset()
    sut = MuticlassUMCE(sample_dummy_classifier)
    _, y_classes = sut.get_samples_in_classes(x_train, y_train)
    assert len(y_classes) == 3


def test_get_num_samples_in_each_class_returns_right_number():
    x_train, y_train = simple_dataset()
    sut = MuticlassUMCE(sample_dummy_classifier)
    arg = [np.array([0, 0, 0]), np.array([1, 1])]
    num_samples = sut.get_num_samples_in_each_class(arg)
    assert num_samples == [3, 2]


def test_after_fit_simple_dataset_k_is_3():
    x_train, y_train = simple_dataset()
    sut = MuticlassUMCE(sample_dummy_classifier)
    sut.fit(x_train, y_train)
    assert sut.k == 3
