import numpy as np

from classes import ActivationSoftmax

# define a 2x3 matrix
a = np.array([[1, 2, 3], [4, 5, 6]])
# define a 1x3 matrix 2d array
b = np.array([[2, 3, 4]])


# multiply a by b
# c = np.dot(a, b)
# print(c)


e = ActivationSoftmax()
o = e.forward(b)
