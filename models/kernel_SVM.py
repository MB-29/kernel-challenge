import numpy as np

from scipy.spatial.distance import pdist, squareform
from cvxopt import matrix, solvers

from models.mismatch import MismatchKernelForBioSeq


class KSVM:

    def __init__(self, train_data, train_labels, regularization, kernel=None):
        self.n_train = train_data.shape[0]
        self.train_data = train_data
        self.train_labels = 2*np.array(train_labels)-1
        self.regularization = regularization
        self.kernel = kernel
        self.gram = kernel
        if kernel is None:
            self.compute_kernel()

    def compute_kernel(self):
        raise NotImplementedError('Please implement kernel method')
    
    def fit(self):
        P = matrix(self.gram, tc='d')
        q = matrix(-self.train_labels, tc='d')
        G_array = np.diag(self.train_labels)
        G_array = np.concatenate((G_array, np.diag(-self.train_labels)), axis=0)
        G = matrix(G_array, tc='d')
        h_array = np.array([1/(2*self.regularization*self.n_train)]*self.n_train + [0]*self.n_train)
        h = matrix(h_array, tc='d')
        sol = solvers.qp(P, q, G, h)
        self.alpha = np.squeeze(np.array(sol['x']), axis=1)

    def classify(self, sample):
        K_star = np.array([self.kernel(x, sample) for x in self.train_data])

        return int(np.heaviside(self.alpha @ K_star, 0))
    
    def predict_test(self, test_data):
        return np.array([self.classify(sample) for sample in test_data])
    
    def predict(self, K_test):
        return np.heaviside(K_test @ self.alpha, 0).astype(int)


class LinKSVM(KSVM):

    def __init__(self, train_data, train_labels, reg_lambda):
        super().__init__(train_data, train_labels, reg_lambda)

    def compute_kernel(self):
        self.kernel = lambda x,y: x @ y
        self.gram = self.train_data  @ self.train_data.T


class GaussKSVM(KSVM):

    def __init__(self, train_data, train_labels, reg_lambda, sigma_coef):
        self.sigma_coef = sigma_coef
        super().__init__(train_data, train_labels, reg_lambda)

    def compute_kernel(self):
        self.kernel = lambda x,y: np.exp(-np.linalg.norm(x-y)**2 / (2*self.sigma_coef))
        pairwise_sq_dists = squareform(pdist(self.train_data, 'sqeuclidean'))
        self.gram = np.exp(-pairwise_sq_dists / (2*self.sigma_coef**2))


class PolyKSVM(KSVM):

    def __init__(self, train_data, train_labels, reg_lambda, poly_c, poly_d):
        self.poly_c = poly_c
        self.poly_d = poly_d
        super().__init__(train_data, train_labels, reg_lambda)

    def compute_kernel(self):
        self.kernel = lambda x,y: (x @ y + self.poly_c)**self.poly_d
        self.gram = (self.train_data  @ self.train_data.T + self.poly_c)**self.poly_d


class MismatchSVM(KSVM):

    def __init__(self, train_data, test_data, train_labels, reg_lambda, k, m, l):
        # train test have same dim
        self.n_train, self.n_test = train_data.shape[0], test_data.shape[0]
        self.train_labels = 2*np.array(train_labels)-1
        self.regularization = reg_lambda
        self.train_test_data = np.concatenate((train_data, test_data), axis=0)
        self.mismatchtree = MismatchKernelForBioSeq(self.train_test_data, k, m, l)
        kernel = self.mismatchtree.compute_kernel()
        self.kernel = kernel
        self.gram = self.kernel[:self.n_train, :self.n_train]

    def classify(self):
        K_star = self.kernel[-self.n_test:, :self.n_train]

        return np.heaviside(K_star @ self.alpha, 0).astype(int)

