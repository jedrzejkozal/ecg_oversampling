from dataset_utils.read_MIT_dataset import *
from oversampling.SmoteGenerator import *
from oversampling.reduce_imbalance import *
from oversampling_test_base import *


class TestSmote(OversamplingTestBase):

    @classmethod
    def get_dataset(self, num_examples):
        x_train, y_train, _, _ = load_testing_dataset()
        x_aug, y_aug = reduce_imbalance(
            x_train, y_train, SmoteGenerator(), num_examples=num_examples)
        return x_aug, y_aug
