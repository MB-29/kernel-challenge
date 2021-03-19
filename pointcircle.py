import numpy as np
import matplotlib.pyplot as plt

from models.kernel_SVM import KernelSVM

n_training, n_test = 100, 100
n, d = n_training + n_test, 2
R = 5

X1 = np.random.multivariate_normal(
    mean=[0, 0], cov=[[1, 0], [0, 1]], size=n//2)
U = np.random.uniform(0, 2*np.pi, n//2)
X2 = np.zeros((n//2, 2))
X2[:, 0], X2[:, 1] = R*np.cos(U), R*np.sin(U)
# X1 = np.random.multivariate_normal(
#     mean=[-2, -2], cov=[[1, 0], [0, 1]], size=n//2)
# X2 = np.random.multivariate_normal(
#     mean=[1, 1], cov=[[1, 0], [0, 1]], size=n//2)
data = np.concatenate([X1, X2])
indices = np.random.permutation(n)
training_indices, test_indices = indices[:
                                      n_training], indices[n_training:n_training+n_test]
training_data, test_data = data[training_indices],  data[test_indices]
labels = np.concatenate([-np.ones(n//2), np.ones(n//2)])
training_labels, test_labels = labels[training_indices], labels[test_indices]



def kernel(x, y):
    return (x@y)**2
    # return np.exp(-(x-y)@(x-y))

regularization = 100

K_training = np.zeros((n_training, n_training))
K_test_training = np.zeros((n_test, n_training))
for i in range(n_training):
    for j in range(i+1):
        K_training[i, j] = kernel(training_data[i], training_data[j])
        K_training[j, i] = K_training[i, j]
    for j in range(n_test):
        K_test_training[j, i] = kernel(test_data[j], training_data[i])


classifier = KernelSVM(
    training_data, training_labels, regularization, K_training)
classifier.train()
predictions = classifier.classify(K_test_training)
accuracy = np.mean(test_labels == predictions)
print(f'Accuracy = {accuracy}')

# # Plot


color_map = {
    1: 'red',
    -1: 'blue'
}
test_colors = [color_map[label] for label in test_labels]
predicted_colors = [color_map[label] for label in predictions]

plt.scatter(test_data[:, 0], test_data[:, 1], color=predicted_colors)
plt.show()
