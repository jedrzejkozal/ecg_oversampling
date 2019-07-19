import numpy as np

from dataset_utils.read_MIT_dataset import *
from oversampling.data_augumentation_generator import *
from oversampling.reduce_imbalance import *

x_train, y_train, _, _ = load_validation_dataset()
print("bincount = {}".format(np.bincount(y_train.astype('int64'))))
x_aug, y_aug = reduce_imbalance(
    x_train, y_train, data_augumentation_generator, num_examples=400)
print("bincount = {}".format(np.bincount(y_aug.astype('int64'))))
