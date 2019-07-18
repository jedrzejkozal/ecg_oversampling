from keras.utils.np_utils import to_categorical

import numpy as np

from dataset_utils.load_data import *
from dataset_utils.data_augumentation import *


normalX = load_data('dataset/ptbdb_normal.csv')[:, :-1]
normalY = load_data('dataset/ptbdb_normal.csv')[:, -1]
abnormalX = load_data('dataset/ptbdb_abnormal.csv')[:, :-1]
abnormalY = load_data('dataset/ptbdb_abnormal.csv')[:, -1]


print("normal set shape: ", normalX.shape)
print("normal set shape: ", normalY.shape)
print("abnormal set shape: ", abnormalX.shape)
print("abnormal set shape: ", abnormalY.shape)

split_normal = int(normalX.shape[0]*0.8)
split_abnormal = int(abnormalX.shape[0]*0.8)
print("split_normal: ", split_normal)
print("split_abnormal: ", split_abnormal)
trainX = np.vstack([normalX[:split_normal], abnormalX[:split_abnormal]])
trainY = np.hstack([normalY[:split_normal], abnormalY[:split_abnormal]])
testX = np.vstack([normalX[split_normal:], abnormalX[split_abnormal:]])
testY = np.hstack([normalY[split_normal:], abnormalY[split_abnormal:]])

print("train set shape: ", trainX.shape)
print("train set shape: ", trainY.shape)
print("test set shape: ", testX.shape)
print("test set shape: ", testY.shape)


#classes indexes
C0 = np.argwhere(trainY == 0).flatten()
C1 = np.argwhere(trainY == 1).flatten()


#class 3 oversampling
additional_samples = np.apply_along_axis(augment, axis=1, arr=trainX[C1]).reshape(-1, 187)
additional_labels = np.ones(shape=(additional_samples.shape[0],), dtype=int)*1
trainX = np.vstack([trainX, additional_samples])
trainY = np.hstack([trainY, additional_labels])

C1 = np.argwhere(trainY == 1).flatten()

#random choice of indexes
subC0 = np.random.choice(C0, 5000)
subC1 = np.random.choice(C1, 5000)


trainX = np.vstack([trainX[subC0], trainX[subC1]])
trainY = np.hstack([trainY[subC0], trainY[subC1]])

print("------------------")
print("train set shape: ", trainX.shape)
print("train set shape: ", trainY.shape)
print("test set shape: ", testX.shape)
print("test set shape: ", testY.shape)

print(trainY)
print(np.bincount(np.array(trainY, dtype=np.int64)))

num_classes = 2
trainY = to_categorical(trainY, num_classes)
testY = to_categorical(testY, num_classes)
