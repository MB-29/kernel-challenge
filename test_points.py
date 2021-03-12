import numpy as np
import matplotlib.pyplot as plt 

from models.logistic_regression import LogisticRegression


n, d = 100, 2
n_train, n_test = 50, 50
X1 = np.random.multivariate_normal(
    mean=[0.8, 0.8], cov=[[1, 0], [0, 1]], size=n//2)
X2 = np.random.multivariate_normal(
    mean=[-0.8, -0.8], cov=[[1, 0], [0, 1]], size=n//2)
data = np.concatenate([X1, X2])
indices = np.random.permutation(n)
train_indices, test_indices = indices[:n_train], indices[n_train:n_train+n_test]
train_data, test_data = data[train_indices],  data[test_indices]
labels = np.concatenate([np.zeros(n//2), np.ones(n//2)])
train_labels, test_labels = labels[train_indices], labels[test_indices]

color_map = {
    1: 'red',
    0: 'blue'
}


classifier = LogisticRegression(train_data, train_labels, 1)
classifier.train(10)
predictions = classifier.predict(test_data)

test_colors = [color_map[label] for label in test_labels]
predicted_colors = [color_map[label] for label in predictions]

direction = classifier.weight[1:] / np.linalg.norm(classifier.weight[1:])

plt.scatter(test_data[:, 0], test_data[:, 1], color=test_colors)
plt.plot([0, direction[0]], [0, direction[1]])
plt.show()
