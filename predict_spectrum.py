import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse 

from models.kernel_SVM import KSVM
from kernels.spectrum_kernel import spectrum_kernel


def run_prediction(DATA_PATH):

    test_predictions = np.array([], dtype=int)

    # hyperparameters
    k_list = [11, 6, 10]
    
    for i in range(3):

        print(f'processing dataset {i+1}')

        TRAIN_DATA_PATH = os.path.join(DATA_PATH, f'Xtr{i}.csv')
        TRAIN_LABEL_PATH = os.path.join(DATA_PATH, f'Ytr{i}.csv')
        TEST_DATA_PATH = os.path.join(DATA_PATH, f'Xte{i}.csv')

        data = np.loadtxt(TRAIN_DATA_PATH, dtype=str,
                        delimiter=',', skiprows=1, usecols=1)

        test_data = np.loadtxt(TEST_DATA_PATH, dtype=str,
                            delimiter=',', skiprows=1, usecols=1)

        labels = pd.read_csv(TRAIN_LABEL_PATH, index_col=0).to_numpy().squeeze()

        training_data, test_data = data, test_data
        n_training, n_test = training_data.shape[0], test_data.shape[0]
        training_labels = labels

        k = k_list[i]
        lambd = 1.0

        # training kernel matrix
        K_train = np.zeros((n_training, n_training))
        for i in range(n_training):
            for j in range(i+1):
                source, target = training_data[i], training_data[j]
                K_train[i, j] = spectrum_kernel(source, target, k)
                K_train[j, i] = K_train[i, j]

        print('computing the kernel matrix')
        classifier = KSVM(training_data, training_labels, lambd, kernel=K_train)

        print('fiting')
        classifier.fit()

        # # test kernel matrix
        K_test = np.zeros((n_test, n_training))
        for i in range(n_test):
            for j in range(n_training):
                source, target = test_data[i], training_data[j]
                K_test[i, j] = spectrum_kernel(source, target, k)

        print('predicting')
        predictions = classifier.predict(K_test)
        test_predictions = np.append(test_predictions, predictions, 0)

    dt = pd.DataFrame(data=test_predictions)
    dt.to_csv('pred_spectrum.csv', index=True, header=['Bound'], index_label='Id')


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('data_dir_path', type=str,
                            help='relative path to the data directory')
    args = arg_parser.parse_args()
    run_prediction(args.data_dir_path)
