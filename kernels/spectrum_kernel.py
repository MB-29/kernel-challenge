import numpy as np
from collections import Counter


def spectrum_kernel(x, y, k):
    x_substrings = [x[index:index+k] for index in range(len(x)-k+1)]
    y_substrings = [y[index:index+k] for index in range(len(y)-k+1)]
    x_counts = Counter(x_substrings)
    y_counts = Counter(y_substrings)
    substrings = set(x_counts.keys()).intersection(set(y_counts.keys()))
    K = 0
    for string in substrings:
        K += x_counts[string] * y_counts[string]
    return K

def fill_spectrum_kernel(X, Y, k):
    assert X.shape == Y.shape
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            K[i, j] = spectrum_kernel(X[i], Y[j], k)
    return K

    