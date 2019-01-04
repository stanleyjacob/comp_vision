import numpy as np
import sklearn

np.random.seed(0)
X = np.random.random((10, 3))
tree = sklearn.neighbors.KDTree(X, leaf_size=2)
dist, ind = tree.query(X[:1], k=3)
print(ind)
print(dist)
