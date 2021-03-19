import numpy as np
import matplotlib.pyplot as plt 

from models.logistic_regression import LogisticRegression


n, d = 500, 2
n_train, n_test = n//2, n//2
X1 = np.random.multivariate_normal(
    mean=[-1, 2], cov=[[1, 0], [0, 1]], size=n//2)
X2 = np.random.multivariate_normal(
    mean=[1, -1], cov=[[1, 0], [0, 1]], size=n//2)
data = np.concatenate([X1, X2])
indices = np.random.permutation(n)
train_indices, test_indices = indices[:n_train], indices[n_train:n_train+n_test]
train_data, test_data = data[train_indices],  data[test_indices]
labels = np.concatenate([np.ones(n//2), np.ones(n//2)])
train_labels, test_labels = labels[train_indices], labels[test_indices]


regularization_tradeoff = 100
n_steps = 20

classifier = LogisticRegression(train_data, train_labels, regularization_tradeoff)
classifier.train()
predictions = classifier.predict(test_data)
accuracy = np.mean(test_labels == predictions)
print(f'Accuracy = {accuracy}')

# Plot

direction = classifier.weight[1:] / np.linalg.norm(classifier.weight[1:])

color_map = {
    1: 'red',
    0: 'blue'
}
test_colors = [color_map[label] for label in test_labels]
predicted_colors = [color_map[label] for label in predictions]

plt.scatter(test_data[:, 0], test_data[:, 1], color=predicted_colors)
plt.plot([0, direction[0]], [0, direction[1]])
plt.show()
