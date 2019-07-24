import numpy as np
from sklearn.model_selection import KFold


class MuticlassUMCE(object):

    def __init__(self, base_clasif_sampling):
        self.base_clasif_sampling = base_clasif_sampling

    def fit(self, x, y):
        x_classes, y_classes = self.get_samples_in_classes(x, y)
        num_samples_in_each_class = self.get_num_samples_in_each_class(
            y_classes)

        min_samples = min(num_samples_in_each_class)
        max_samples = max(num_samples_in_each_class)
        ir = max_samples // min_samples
        self.num_classifiers = min(10, ir)

        k_i = self.imbalance_ratio_for_each_class(min_samples,
                                                  num_samples_in_each_class)

        self.k_fold_for_each_class(k_i, x_classes, y_classes)

    def get_samples_in_classes(self, x, y):
        if y.ndim == 2 and y.shape[1] > 1:
            y = self.one_hot_to_labels(y)
        num_classes = y.max()+1  # labels usualy start from 0

        classes_x, classes_y = [], []
        for i in range(num_classes):
            class_indexes = np.argwhere(y == i)
            classes_x.append(x[class_indexes])
            classes_y.append(y[class_indexes])
        return classes_x, classes_y

    def one_hot_to_labels(self, y):
        return y.argmax(axis=1)

    def get_num_samples_in_each_class(self, y):
        return list(map(lambda x: x.size, y))

    def imbalance_ratio_for_each_class(self, min_samples, num_samples_in_each_class):
        return list(map(lambda x: x // min_samples,
                        num_samples_in_each_class))

    def k_fold_for_each_class(self, k_i, x_classes, y_classes):
        folds_x, folds_y = [], []
        for k, x_class, y_class in zip(k_i, x_classes, y_classes):
            folds_x.append([])
            folds_y.append([])

            self.split_into_folds(k, x_class, y_class, folds_x, folds_y)

        return tuple(folds_x), tuple(folds_y)

    def split_into_folds(self, k, x_class, y_class, folds_x, folds_y):
        if k > 1:
            kf = KFold(n_splits=k)
            for _, fold_index in kf.split(x_class):
                folds_x[-1].append(x_class[fold_index])
                folds_y[-1].append(y_class[fold_index])
        else:  # minority class
            folds_x[-1].append(x_class)
            folds_y[-1].append(y_class)
