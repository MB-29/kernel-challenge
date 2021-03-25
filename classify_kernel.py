import numpy as np
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt

from kernels.substring_kernel import substring_kernel
from kernels.spectrum_kernel import spectrum_kernel
from models.kernel_SVM import KernelSVM


TRAIN_DATA_PATH = 'data/Xtr0.csv'
TRAIN_LABEL_PATH = 'data/Ytr0.csv'
TEST_DATA_PATH = 'data/Xte0.csv'

n_training, n_test = 1600, 400

data = np.loadtxt(TRAIN_DATA_PATH, dtype=str,
                  delimiter=',', skiprows=1, usecols=1)
labels = pd.read_csv(TRAIN_LABEL_PATH, index_col=0).to_numpy().squeeze()

training_data, test_data = data[:n_training], data[n_training:n_training+n_test]
training_labels, test_labels = labels[:n_training], labels[n_training:n_training+n_test]


k = 12
m = 1
lambd = 0.5
index = 1
regularization = 1

print('kernel')
K_training = np.zeros((n_training, n_training))
for i in range(n_training):
    for j in range(i+1):
        source, target = training_data[i], training_data[j]
        # K_training[i, j] = substring_kernel(source, target, lambd, k)
        K_training[i, j] = spectrum_kernel(source, target, k)
        K_training[j, i] = K_training[i, j]
        index += 1

K_test_training = np.zeros((n_test, n_training))
for i in range(n_test):
    for j in range(n_training):
        source, target = test_data[i], training_data[j]
        # K_test_training[i, j] = substring_kernel(source, target, lambd, k)
        K_test_training[i, j] = spectrum_kernel(source, target, k)

print('fit')
classifier = SVC(kernel='precomputed')
classifier.fit(K_training, training_labels)
print('predict')
test_predictions = classifier.predict(K_test_training)

# print('fit')
# classifier = KernelSVM(
#     training_data, training_labels, regularization, K_training)
# classifier.train()
# print('predict')
# test_predictions = classifier.classify(K_test_training)



accuracy = np.mean(test_predictions == test_labels)
print(f'data {TRAIN_DATA_PATH}, k={k}, accuracy = {accuracy}')
