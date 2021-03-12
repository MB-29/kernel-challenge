import numpy as np
from numpy.linalg import solve


class LogisticRegression:

    def __init__(self, train_data, train_labels, tradeoff):

        self.n, self.d = train_data.shape
        self.train_data = train_data
        self.extended_train_data = np.zeros((self.n, self.d+1))
        self.extended_train_data[:, 0] = 1
        self.train_labels = train_labels
        self.weight = np.random.rand(self.d+1)
        self.tradeoff = tradeoff

    def train(self, n_steps=100):
        
        # Newton-Raphson for L2-regularized loss
        for step_index in range(n_steps):
            activation = sigmoid(self.extended_train_data @ self.weight)
            first_derivative = (self.train_labels - activation).reshape((self.n, -1)) * self.extended_train_data
            gradient =  np.sum(first_derivative, axis=0) - 2 * self.tradeoff * self.weight

            hessian = 0
            for i in range(self.n):
                hessian -= activation[i] * (1 - activation[i]) * self.extended_train_data[i].T @ self.extended_train_data[i]
            hessian -= 2 * self.tradeoff * np.eye(self.d+1)
            increment = - solve(hessian, gradient)
            self.weight += increment

            likelihood = self.compute_likelihood()

    def classify(self, sample):
        probability = sigmoid(self.weight[1:] @ sample + self.weight[0])
        return np.heaviside(probability - 1/2, 0)

    def predict(self, test_data):
        return np.array([self.classify(sample) for sample in test_data])

    def compute_likelihood(self):
        overlap = self.extended_train_data @ self.weight
        regularisation = - self.tradeoff * self.weight @ self.weight
        return np.sum(self.train_labels * overlap - np.log(1 + np.exp(overlap))) + regularisation
        
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
