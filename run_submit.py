import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.kernel_rid_reg import GaussKRR
from models.kernel_SVM import GaussKSVM

test_predictions = np.array([], dtype=int)

for k in range(3):
    TRAIN_DATA_PATH = f'data/Xtr{k}.csv'
    TRAIN_EMBEDDING_PATH = f'data/Xtr{k}_mat100.csv'
    TRAIN_LABEL_PATH = f'data/Ytr{k}.csv'
    TEST_EMBEDDING_PATH = f'data/Xte{k}_mat100.csv'

    n_train = 2000

    data = np.loadtxt(TRAIN_DATA_PATH, dtype=str,
                      delimiter=',', skiprows=1, usecols=1)

    embeddings = np.loadtxt(TRAIN_EMBEDDING_PATH)
    test_embeddings = np.loadtxt(TEST_EMBEDDING_PATH)
    train_embeddings = embeddings[:n_train]

    labels = pd.read_csv(TRAIN_LABEL_PATH, index_col=0).to_numpy().squeeze()
    train_labels = labels[:n_train]

    classifier = GaussKRR(train_embeddings, train_labels, 0.1, 7e-4)
    classifier.fit()
    predictions = classifier.predict_test(test_embeddings)
    test_predictions = np.append(test_predictions, predictions, 0)

dt = pd.DataFrame(data=test_predictions)
dt.to_csv('data/pred_krr01.csv', index=True, header=['Bound'], index_label='Id')