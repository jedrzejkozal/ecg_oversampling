import numpy as np


def expand_dims(input, target):
    input = np.expand_dims(input, axis=3)
    target = np.expand_dims(target, axis=2)
    return input, target


def imbalance_analysis(filename):
    df_train = pd.read_csv(filename)

    target_count = df_train.iloc[:, -1].value_counts()
    print("Number of traning examples in each class:")
    for i in range(len(target_count)):
        print('Class {}:'.format(i), target_count[i])

    print("Imbalance ratio for classes 1-4:")
    for i in range(1, len(target_count)):
        print('IR(class {}) ='.format(i), round(
            target_count[0] / target_count[i], 2), ': 1')

    target_count.plot(kind='bar', title='Count (target)')


def sets_shapes_report(x_train, y_train):
    print("train set shape: ", x_train.shape)
    print("train set shape: ", y_train.shape)

    print("train classes sample count:")
    print(np.bincount(y_train.astype('int32').flatten()))
