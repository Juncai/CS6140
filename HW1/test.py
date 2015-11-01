import Utilities as u

a = [1, 2, 2, 3]
b = [10, 12, 14, 15]
c = a + b
d = [1, 1, 1, 1, 3, 3 ,3, 3]
print u.mse_all(d, c, 2)