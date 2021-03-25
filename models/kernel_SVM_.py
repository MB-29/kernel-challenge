import numpy as np
import cvxpy

class KernelSVM:

    def __init__(self, train_data, training_labels, regularization, training_gram=None):
        self.n = len(train_data)
        self.train_data = train_data
        self.training_labels = 2*np.array(training_labels)-1
        self.regularization = regularization
        self.gram = training_gram
        if self.gram is None:
            self.compute_kernel

    def compute_kernel(self):
        raise NotImplementedError('Please implement kernel method')
    
    def train(self):
        alpha = cvxpy.Variable(self.n)
        objective = cvxpy.Minimize(
            cvxpy.quad_form(alpha, self.gram) - 2*self.training_labels@alpha
            )
        constraints = [
            cvxpy.multiply(alpha, self.training_labels) <= 1/(2*self.regularization*self.n),
            cvxpy.multiply(alpha, self.training_labels) >= 0
            ]
        problem = cvxpy.Problem(objective, constraints)
        problem.solve()
        self.alpha = alpha.value
        

    def classify(self, K_test_training):
        decision_values = K_test_training @ self.alpha
        return (np.sign(decision_values) + 1)//2
        
    


    
