import numpy as np
from sklearn.svm import SVC
import pandas as pd

from kernels.substring_kernel import substring_kernel
from kernels.spectrum_kernel import spectrum_kernel


TRAIN_DATA_PATH = 'data/Xtr1.csv'
TRAIN_LABEL_PATH = 'data/Ytr1.csv'
TEST_DATA_PATH = 'data/Xte0.csv'

n_training, n_test = 1600, 400

data = np.loadtxt(TRAIN_DATA_PATH, dtype=str,
                  delimiter=',', skiprows=1, usecols=1)
labels = pd.read_csv(TRAIN_LABEL_PATH, index_col=0).to_numpy().squeeze()

training_data, test_data = data[:n_training], data[n_training:n_training+n_test]
training_labels, test_labels = labels[:n_training], labels[n_training:n_training+n_test]


k = 10
lambd = 0.5
index = 1
K_train = np.zeros((n_training, n_training))
for i in range(n_training):
    for j in range(i+1):
        source, target = training_data[i], training_data[j]
        K_train[i, j] = substring_kernel(source, target, lambd, k)
        # K_train[i, j] = spectrum_kernel(source, target, k)
        K_train[j, i] = K_train[i, j]
        print(index)
        index += 1

print('kernel')
classifier = SVC(kernel='precomputed')
print('fit')
classifier.fit(K_train, training_labels)

K_test = np.zeros((n_test, n_training))
for i in range(n_test):
    for j in range(n_training):
        source, target = test_data[i], training_data[j]
        K_test[i, j] = substring_kernel(source, target, lambd, k)
        # K_test[i, j] = spectrum_kernel(source, target, k)

print('predict')
test_predictions = classifier.predict(K_test)

accuracy = np.mean(test_predictions == test_labels)
print(f'accuracy = {accuracy}')
