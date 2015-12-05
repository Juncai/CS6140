import Kernels as k
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, rbf_kernel, polynomial_kernel, cosine_similarity
import numpy as np
import Consts as c
a = np.array([[1, 2, 3], [1, 2, 3]]) / 10
b = np.array([[4, 5, 6]]) / 10

kern = k.Kernels(c.EUCLIDIAN)
res1 = kern.get_value(a, b)
# res2 = polynomial_kernel(a, a, degree=2)
res2 = euclidean_distances(a, b)

print('{} \n {}'.format(res1, res2))



