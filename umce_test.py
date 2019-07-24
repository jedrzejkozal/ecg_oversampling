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


def test_after_fit_simple_dataset_num_classifiers_is_3():
    x_train, y_train = simple_dataset()
    sut = MuticlassUMCE(sample_dummy_classifier)
    sut.fit(x_train, y_train)
    assert sut.num_classifiers == 3


def test_k_fold_for_each_class_returns_right_number_of_folds():
    arg_x = [np.array([0, 1, 2, 3]), np.array([4, 5])]
    arg_y = [np.array([0, 0, 0, 0]), np.array([1, 1])]
    arg_k_i = [2, 1]

    sut = MuticlassUMCE(sample_dummy_classifier)
    folds_x, folds_y = sut.k_fold_for_each_class(arg_k_i, arg_x, arg_y)
    class_0_folds_x, class_1_folds_x = folds_x
    class_0_folds_y, class_1_folds_y = folds_y

    assert len(class_0_folds_x) == 2
    assert len(class_0_folds_y) == 2
    assert len(class_1_folds_x) == 1
    assert len(class_1_folds_y) == 1


def test_k_fold_for_each_class_number_samples_in_each_fold_close_to_minimal_num_of_samples():
    arg_x = [np.array([0, 1, 2, 3]), np.array([4, 5])]
    arg_y = [np.array([0, 0, 0, 0]), np.array([1, 1])]
    arg_k_i = [2, 1]
    minimal = min(list(map(lambda x: len(x), arg_y)))

    sut = MuticlassUMCE(sample_dummy_classifier)
    folds_x, folds_y = sut.k_fold_for_each_class(arg_k_i, arg_x, arg_y)
    class_0_folds_x, class_1_folds_x = folds_x
    class_0_folds_y, class_1_folds_y = folds_y

    for class_folds_lists in folds_x:
        for fold in class_folds_lists:
            # atol due to floor in during calculation of k_i
            assert np.isclose(len(fold), minimal, atol=1)
    for class_folds_lists in folds_y:
        for fold in class_folds_lists:
            assert np.isclose(len(fold), minimal, atol=1)


def test_k_fold_for_each_class_number_each_fold_for_x_and_y_have_the_same_len():
    arg_x = [np.array([0, 1, 2, 3]), np.array([4, 5])]
    arg_y = [np.array([0, 0, 0, 0]), np.array([1, 1])]
    arg_k_i = [2, 1]
    minimal = min(list(map(lambda x: len(x), arg_y)))

    sut = MuticlassUMCE(sample_dummy_classifier)
    folds_x, folds_y = sut.k_fold_for_each_class(arg_k_i, arg_x, arg_y)
    class_0_folds_x, class_1_folds_x = folds_x
    class_0_folds_y, class_1_folds_y = folds_y

    for class_folds_x_list, class_folds_y_list in zip(folds_x, folds_y):
        for fold_x, fold_y in zip(class_folds_x_list, class_folds_y_list):
            assert len(fold_x) == len(fold_y)
