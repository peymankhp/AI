# NumPy is capable of finding roots for polynomials and linear equations, but it can not find roots for non linear equations, like this one:
# x + cos(x)
# For that you can use SciPy's optimze.root function.
# This function takes two required arguments:
# fun - a function representing an equation.
# x0 - an initial guess for the root.
# The function returns an object with information regarding the solution.
# The actual solution is given under attribute x of the returned object:

from scipy.optimize import root
from math import cos
def eqn(x):
  return x + cos(x)
myroot = root(eqn, 0)
print(myroot.x)
#Answer:   [-0.73908513]
print()

# We can use scipy.optimize.minimize() function to minimize the function.
# The minimize() function takes the following arguments:
# fun - a function representing an equation.
# x0 - an initial guess for the root.
# method - name of the method to use. Legal values:
#     'CG'
#     'BFGS'
#     'Newton-CG'
#     'L-BFGS-B'
#     'TNC'
#     'COBYLA'
#     'SLSQP'
# callback - function called after each iteration of optimization.
# options - a dictionary defining extra params:
# {
#      "disp": boolean - print detailed description
#      "gtol": number - the tolerance of the error
#   } 

from scipy.optimize import minimize
def eqn(x):
  return x**2 + x + 2
mymin = minimize(eqn, 0, method='BFGS')
print(mymin)
#Answer:         fun: 1.75
#  hess_inv: array([[ 0.50000001]])
#       jac: array([ 0.])
#   message: 'Optimization terminated successfully.'
#      nfev: 12
#       nit: 2
#      njev: 4
#    status: 0
#   success: True
#         x: array([-0.50000001])
print()

# How to Work With Sparse Data
# SciPy has a module, scipy.sparse that provides functions to deal with sparse data.
# There are primarily two types of sparse matrices that we use:
# CSC - Compressed Sparse Column. For efficient arithmetic, fast column slicing.
# CSR - Compressed Sparse Row. For fast row slicing, faster matrix vector products
# We will use the CSR matrix in this tutorial.
import numpy as np
from scipy.sparse import csr_matrix
arr = np.array([0, 0, 0, 0, 0, 1, 1, 0, 2])
print(csr_matrix(arr))
#Answer:     (0, 5)	1     (0, 6)	1     (0, 8)	2
print()

import numpy as np
from scipy.sparse import csr_matrix
arr = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 2]])
print(csr_matrix(arr).data)
#Answer:     [1 1 2]
print()

import numpy as np
from scipy.sparse import csr_matrix
arr = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 2]])
print(csr_matrix(arr).count_nonzero())
#Answer:     3
print()

import numpy as np
from scipy.sparse import csr_matrix
arr = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 2]])
mat = csr_matrix(arr)
mat.eliminate_zeros()
print(mat)
#Answer:       (1, 2)	1      (2, 0)	1      (2, 2)	2
print()

import numpy as np
from scipy.sparse import csr_matrix
arr = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 2]])
mat = csr_matrix(arr)
mat.sum_duplicates()
print(mat)
#Answer:         (1, 2)	1      (2, 0)	1       (2, 2)	2
print()

import numpy as np
from scipy.sparse import csr_matrix
arr = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 2]])
newarr = csr_matrix(arr).tocsc()
print(newarr)
#Answer:           (2, 0)	1      (1, 2)	1      (2, 2)	2
print()