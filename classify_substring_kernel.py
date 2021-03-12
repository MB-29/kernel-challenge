import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pandas as pd

from utils import plot_sequence_colours
from substring_kernel import fill_kernel_table

TRAIN_DATA_PATH = 'data/Xtr0.csv'
TRAIN_EMBEDDING_PATH = 'data/Xtr0_mat100.csv'
TRAIN_LABEL_PATH = 'data/Ytr0.csv'


TEST_DATA_PATH = 'data/Xte0.csv'
TEST_EMBEDDING_PATH = 'data/Xte0_mat100.csv'

n_train, n_test = 100, 50

data = np.loadtxt(TRAIN_DATA_PATH, dtype=str,
                  delimiter=',', skiprows=1, usecols=1)
embeddings = np.loadtxt(TRAIN_EMBEDDING_PATH)
labels = pd.read_csv(TRAIN_LABEL_PATH, index_col=0).to_numpy().squeeze()

train_data, test_data = data[:n_train], data[n_train:n_train+n_test]
train_labels, test_labels = labels[:n_train], labels[n_train:n_train+n_test]

test_embedding = np.loadtxt(TEST_EMBEDDING_PATH)

positive_train_data = train_data[train_labels == 1]
negative_train_data = train_data[train_labels == 0]
n_positive = len(positive_train_data)

k = 4
lambd = 0.5

K_train = np.zeros((n_train, n_train))
for i in range(n_train):
    for j in range(i+1):
        kernel_values = fill_kernel_table(
            train_data[i], train_data[j], lambd, k)
        K_train[i, j] = kernel_values[-1, -1]
        K_train[j, i] = K_train[i, j]

print('kernel')
classifier = SVC(kernel='precomputed')
print('fit')
classifier.fit(K_train, train_labels)

K_test = np.zeros((n_test, n_train))
for i in range(n_test):
    for j in range(n_train):
        kernel_values = fill_kernel_table(
            test_data[i], train_data[j], lambd, k)
        K_test[i, j] = kernel_values[-1, -1]

print('predict')
test_predictions = classifier.predict(K_test)

print('report')
report = classification_report(test_labels, test_predictions)
print(report)
