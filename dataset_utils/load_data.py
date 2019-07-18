from pandas import read_csv

def load_data(filename):
    dataframe = read_csv(filename, engine='python')
    dataset = dataframe.values
    return dataset.astype('float32')
