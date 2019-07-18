import pandas as pd


def load_data(filename):
    dataframe = pd.read_csv(filename, engine='python')
    dataset = dataframe.values
    return dataset.astype('float32')


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
