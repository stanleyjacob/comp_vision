import numpy as np
np.random.seed(0)
X = np.random.random((10, 3))
tree = KDTree(X, leaf_size=2)
dist, ind = tree.query(X[:1], k=3)
print(ind)
print(dist)
