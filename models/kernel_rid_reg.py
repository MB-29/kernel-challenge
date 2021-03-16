import numpy as np

from scipy.spatial.distance import pdist, squareform

class KRR:

    def __init__(self, train_data, train_labels, reg_lambda):
        self.n, self.d = train_data.shape
        self.train_data = train_data
        self.train_labels = 2*np.array(train_labels)-1
        self.weight = np.random.randn(self.d)
        self.reg_lambda = reg_lambda
        self.kernel = lambda x, y: None
        self.gram = np.zeros((self.n, self.n))
        self.compute_kernel()

    def compute_kernel(self):
        raise NotImplementedError('Please implement kernel method')
    
    def fit(self):
        self.weight = np.linalg.inv(self.gram + self.reg_lambda*np.eye(self.n)) @ self.train_labels

    def classify(self, sample):
        K_star = np.array([self.kernel(x, sample) for x in self.train_data])
        
        return np.heaviside(self.weight @ K_star, 0)
    
    def predict_test(self, test_data):
        return np.array([self.classify(sample) for sample in test_data])


class LinKRR(KRR):

    def __init__(self, train_data, train_labels, reg_lambda):
        super().__init__(train_data, train_labels, reg_lambda)

    def compute_kernel(self):
        self.kernel = lambda x,y: x @ y
        self.gram = self.train_data  @ self.train_data.T


class GaussKRR(KRR):

    def __init__(self, train_data, train_labels, reg_lambda, sigma_coef):
        self.sigma_coef = sigma_coef
        super().__init__(train_data, train_labels, reg_lambda)

    def compute_kernel(self):
        self.kernel = lambda x,y: np.exp(-np.linalg.norm(x-y)**2 / (2*self.sigma_coef))
        pairwise_sq_dists = squareform(pdist(self.train_data, 'sqeuclidean'))
        self.gram = np.exp(-pairwise_sq_dists / (2*self.sigma_coef**2))
    
