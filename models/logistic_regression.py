import numpy as np
from numpy.linalg import solve
from scipy.optimize import minimize

class LogisticRegression:

    def __init__(self, train_data, train_labels, tradeoff):

        self.n, self.d = train_data.shape
        self.train_data = train_data
        self.extended_train_data = np.zeros((self.n, self.d+1))
        self.extended_train_data[:, 0] = 1
        self.extended_train_data[:, 1:] = self.train_data
        
        self.train_labels = train_labels
        self.weight = np.random.randn(self.d+1)
        self.tradeoff = tradeoff

    def _train(self, n_steps=20):
        
        # Newton-Raphson for L2-regularized likelihood maximization
        for step in range(n_steps):
            activation = sigmoid(self.extended_train_data @ self.weight)
            # first_derivative = (self.train_labels - activation).reshape((self.n, -1)) * self.extended_train_data
            # gradient =  np.sum(first_derivative, axis=0) - 2 * self.n * self.tradeoff * self.weight
            gradient = self.extended_train_data.T @ (self.train_labels - activation) - 2 * self.n * self.tradeoff * self.weight

            hessian = np.zeros((self.d+1, self.d+1))
            for i in range(self.n): 
                hessian -= activation[i] * (1 - activation[i]) * self.extended_train_data[i].T @ self.extended_train_data[i]
            hessian -= 2 * self.n * self.tradeoff * np.eye(self.d+1)
            increment = -solve(hessian, gradient)
            self.weight += increment

            likelihood = self.compute_likelihood(self.weight)
            print(likelihood)
    
    def train(self):
        objective = lambda x : - self.compute_likelihood(x)
        jac = lambda x : - self.compute_gradient(x)
        hess= lambda x : - self.compute_hessian(x)
        self.result = minimize(objective, self.weight, method='Newton-CG', jac=jac, hess=hess)
        self.weight = self.result.x
        print(self.weight)


    def classify(self, sample):
        log_ratio = self.weight[1:] @ sample + self.weight[0]
        return np.heaviside(log_ratio, 0)

    def predict(self, test_data):
        return np.array([self.classify(sample) for sample in test_data])

    def compute_likelihood(self, weight):
        overlap = self.extended_train_data @ weight
        regularisation = - self.n * self.tradeoff * weight @ weight
        return np.sum(self.train_labels * overlap - np.log(1 + np.exp(overlap))) + regularisation
    
    def compute_gradient(self, weight):
        activation = sigmoid(self.extended_train_data @ weight)
        first_derivative = (
            self.train_labels - activation).reshape((self.n, -1)) * self.extended_train_data
        gradient = np.sum(first_derivative, axis=0) - 2 * \
            self.n * self.tradeoff * weight
        return gradient
    
    def compute_hessian(self, weight):
        activation = sigmoid(self.extended_train_data @ weight)
        hessian = np.zeros((self.d+1, self.d+1))
        for i in range(self.n):
            hessian -= activation[i] * (1 - activation[i]) * \
                self.extended_train_data[i].T @ self.extended_train_data[i]
        hessian -= 2 * self.n * self.tradeoff * np.eye(self.d+1)
        return hessian

    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
