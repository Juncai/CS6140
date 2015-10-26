import EM as em
import DataLoader as loader
import numpy as np


GAUSS2 = 'data/2gaussian.txt'
GAUSS3 = 'data/3gaussian.txt'

path = GAUSS3
k = 3

x = np.matrix(loader.load_arrays(path))
x = np.transpose(x)
label, model, llh = em.em(x, k)

print model
