from pandas import read_csv

import numpy as np
import pandas as ps
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from scipy.signal import resample


def load_data(filename):
    dataframe = read_csv(filename, engine='python')
    dataset = dataframe.values
    return dataset.astype('float32')


trainX = load_data('dataset/mitbih_train.csv')[:, :-1]
trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))

trainY = load_data('dataset/mitbih_train.csv')[:, -1]
trainY = trainY.reshape((trainY.shape[0]))

testX = load_data('dataset/mitbih_test.csv')[:, :-1]
testX = testX.reshape((testX.shape[0], testX.shape[1], 1))

testY = load_data('dataset/mitbih_test.csv')[:, -1]
testY = testY.reshape((testY.shape[0]))

print("train set shape: ", trainX.shape)
print("train set shape: ", trainY.shape)
print("test set shape: ", testX.shape)
print("test set shape: ", testY.shape)


#imballance analysis
train_counts = np.bincount(trainY.astype('int64'))
train_perc = train_counts / sum(train_counts)
print("\ntrain set classes counts: ", train_counts, ", sum: ", sum(train_counts))
print("train set classes percentage: ", train_perc, ", sum: ", sum(train_perc))

test_counts = np.bincount(testY.astype('int64'))
test_perc = test_counts / sum(test_counts)
print("\ntest set classes counts: ", test_counts, ", sum: ", sum(test_counts))
print("test set classes percentage: ", test_perc, ", sum: ", sum(test_perc), "\n")

class1_train_index = train_counts[0]

num_classes = 5
trainY = to_categorical(trainY, num_classes)
testY = to_categorical(testY, num_classes)


#visualisation
plt.figure(1, figsize=(10, 6))
plt.plot(trainX[0], label='pacjent 0')
plt.plot(trainX[1], label='pacjent 1')
plt.plot(trainX[2], label='pacjent 2')
plt.plot(trainX[3], label='pacjent 3')
plt.legend()
plt.xlabel('próbki')
plt.ylabel('sygnał EKG')
plt.savefig('doc/img/ECG1.png')


plt.figure(2, figsize=(10, 6))
plt.plot(trainX[0], label='klasa 1')
plt.plot(trainX[train_counts[0]+1], label='klasa 2')
plt.plot(trainX[train_counts[1]+1], label='klasa 3')
plt.plot(trainX[train_counts[2]+1], label='klasa 4')
plt.plot(trainX[train_counts[3]+1], label='klasa 5')
plt.legend()
plt.xlabel('próbki')
plt.ylabel('sygnał EKG')
plt.savefig('doc/img/ECG2.png')
#plt.show()


#data augumentation
def stretch(x):
    sum_samples = int(187 * (1 + (np.random.rand()-0.5)/3))
    y = resample(x, sum_samples)
    if sum_samples < 187:
        y_ = np.zeros(shape=(187, ))
        y_[:sum_samples] = y
    else:
        y_ = y[:187]
    return y_

def amplify(x):
    alpha = (np.random.rand()-0.5)
    factor = -alpha*x + (1+alpha)
    return x*factor

def augment(x):
    result = np.zeros(shape= (3, 187))
    result[0, :] = stretch(x)
    result[1, :] = amplify(x)
    result[2, :] = amplify(stretch(x))
    return result

plt.figure(3, figsize=(10, 6))
plt.plot(trainX[0, :], label='oryginalny sygnał')
plt.plot(amplify(trainX[0, :]), label='sygnał po ponownym próbkowaniu')
plt.plot(stretch(trainX[0, :]), label='rozciągnięty sygnał')
plt.legend()
plt.xlabel('próbki')
plt.ylabel('sygnał EKG')
plt.savefig('doc/img/ECG_augument.png')
#plt.show()
