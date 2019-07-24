import numpy as np


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

        self.k = min(10, ir)

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
