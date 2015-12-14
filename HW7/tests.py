import Kernels as k
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, rbf_kernel, polynomial_kernel, cosine_similarity
import numpy as np
import Consts as c
a = np.array([[1, 2, 3], [1, 2, 3]]) / 10
aa = np.array([[1, 2, 3]]) / 10
b = np.array([[4, 5, 6]]) / 10

x = [1, 2]
y = [1, 2]

kern = k.Kernels(c.POLY)
res1 = kern.get_value([x], [y])
res2 = polynomial_kernel(x, y, degree=2, gamma=1, coef0=0)
# res2 = euclidean_distances(a, b)

print('{} \n {}'.format(res1, res2))



