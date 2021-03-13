import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pandas as pd

from models.logistic_regression import LogisticRegression

TRAIN_DATA_PATH = 'data/Xtr0.csv'
TRAIN_EMBEDDING_PATH = 'data/Xtr0_mat100.csv'
TRAIN_LABEL_PATH = 'data/Ytr0.csv'


TEST_DATA_PATH = 'data/Xte0.csv'
TEST_EMBEDDING_PATH = 'data/Xte0_mat100.csv'

n_train, n_test = 1500, 500

data = np.loadtxt(TRAIN_DATA_PATH, dtype=str,
                  delimiter=',', skiprows=1, usecols=1)

embeddings = np.loadtxt(TRAIN_EMBEDDING_PATH)
# test_embedding = np.loadtxt(TEST_EMBEDDING_PATH)
train_embeddings, test_embeddings = embeddings[:n_train], embeddings[n_train:n_train+n_test]
    
labels = pd.read_csv(TRAIN_LABEL_PATH, index_col=0).to_numpy().squeeze()
train_labels, test_labels = labels[:n_train], labels[n_train:n_train+n_test]

n_steps = 50
regularization_tradeoff = 0.1

classifier = LogisticRegression(train_embeddings, train_labels, regularization_tradeoff)
classifier.train(n_steps)

predictions = classifier.predict(test_embeddings)

accuracy = np.mean(test_labels == predictions)
print(f'Accuracy = {accuracy}')
