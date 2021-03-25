import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.kernel_SVM import KSVM, spectrum_kernel, fill_spectrum_kernel

test_predictions = np.array([], dtype=int)
k_list = [11, 6, 10]

for i in range(3):
    TRAIN_DATA_PATH = f'data/Xtr{i}.csv'
    TRAIN_LABEL_PATH = f'data/Ytr{i}.csv'
    TEST_DATA_PATH = f'data/Xte{i}.csv'

    data = np.loadtxt(TRAIN_DATA_PATH, dtype=str,
                      delimiter=',', skiprows=1, usecols=1)

    test_data = np.loadtxt(TEST_DATA_PATH, dtype=str,
                           delimiter=',', skiprows=1, usecols=1)

    labels = pd.read_csv(TRAIN_LABEL_PATH, index_col=0).to_numpy().squeeze()

    training_data, test_data = data, test_data
    n_training, n_test = training_data.shape[0], test_data.shape[0]
    training_labels = labels
    print(n_training, n_test)

    k = k_list[i]
    lambd = 1.0
    K_train = np.zeros((n_training, n_training))
    for i in range(n_training):
        for j in range(i+1):
            source, target = training_data[i], training_data[j]
            K_train[i, j] = spectrum_kernel(source, target, k)
            K_train[j, i] = K_train[i, j]

    print('kernel')
    classifier = KSVM(training_data, training_labels, lambd, kernel=K_train)
    print('fit')
    classifier.fit()

    K_test = np.zeros((n_test, n_training))
    for i in range(n_test):
        for j in range(n_training):
            source, target = test_data[i], training_data[j]
            K_test[i, j] = spectrum_kernel(source, target, k)

    print('predict')
    predictions = classifier.predict(K_test)
    test_predictions = np.append(test_predictions, predictions, 0)

dt = pd.DataFrame(data=test_predictions)
dt.to_csv('data/pred_spectrum1.csv', index=True, header=['Bound'], index_label='Id')