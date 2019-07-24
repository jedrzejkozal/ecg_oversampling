import numpy as np
from sklearn.model_selection import KFold
from keras.utils import to_categorical


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

        folds_x, folds_y = self.k_fold_for_each_class(
            k_i, x_classes, y_classes)

        self.create_models()
        self.fit_all_models(k_i, folds_x, folds_y)

    def get_samples_in_classes(self, x, y):
        if y.ndim == 2 and y.shape[1] > 1:
            y = self.one_hot_to_labels(y)
        else:
            y = y.flatten()
        self.num_classes = y.max()+1  # labels usualy start from 0

        classes_x, classes_y = [], []
        for i in range(self.num_classes):
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

            self.add_folds_for_class(k, x_class, y_class, folds_x, folds_y)

        return tuple(folds_x), tuple(folds_y)

    def add_folds_for_class(self, k, x_class, y_class, folds_x, folds_y):
        if k > 1:
            kf = KFold(n_splits=k)
            for _, fold_index in kf.split(x_class):
                folds_x[-1].append(x_class[fold_index])
                folds_y[-1].append(y_class[fold_index])
        else:  # minority class
            folds_x[-1].append(x_class)
            folds_y[-1].append(y_class)

    def create_models(self):
        self.models = []
        for i in range(self.num_classifiers):
            self.models.append(self.base_clasif_sampling())

    def fit_all_models(self, k_i, folds_x, folds_y):
        for model in self.models:
            x_train, y_train = self.get_dataset(k_i, folds_x, folds_y)
            y_train = to_categorical(y_train, num_classes=self.num_classes)
            print("y_Train.shape = ", y_train.shape)
            model.fit(x_train, y_train)

    def get_dataset(self, k_i, folds_x, folds_y):
        x_train, y_train = [], []
        for k, folds_for_class_x, folds_for_class_y in zip(k_i, folds_x, folds_y):
            fold_index = self.get_random_fold_index(k)
            x_train.append(folds_for_class_x[fold_index])
            y_train.append(folds_for_class_y[fold_index].flatten())
        return np.vstack(x_train), np.hstack(y_train)

    def get_random_fold_index(self, k):
        if k == 1:
            return 0  # only one fold
        return np.random.randint(0, k-1)
