import numpy as np

"""
  Compute Pascal triangle up to size n
  Returns: Numpy matrix with binomial coefficients
  - Matrix represents linear map P(X) \mapsto P(X+1) in the canonical basis
  - Convention is that first columns are the image of monomials of higher degree
"""
def pascal(n):
    # Initialize diagonal and first row
    mat       = np.identity(n, dtype=int)
    mat[:, 0] = 1
    # Loop using the standard binomial recurrence
    for i in range(2,n):
        mat[i, 1:-1] = mat[i-1, 0:-2] + mat[i-1, 1:-1]
    # Satisfy indexing conventions
    mat = np.flip(mat, axis=0)
    mat = mat.transpose()
    mat = np.flip(mat, axis=0)
    return mat

"""
 If | P  (X) = \sum c_i X^i = \sum d_i (X-a)^i
    | P_a(X) = \sum d_i X^i = P(X+a)
Then:
    taylor_expand computes P_a = Taylor expansion of P at a
    which is nothing by the linear map
    P(X) \mapsto P_a(X) = P(X+a)
    X^n \mapsto \sum (n,k) a^k X^{n-k}
 """
def taylor_expand(p, a, pascal_triangle=None):
    #
    degree = len(p)-1
    if pascal_triangle is None:
        pascal_triangle = pascal(degree+1)
    exponents = np.array( range(0, degree+1, 1) )
    powers    = a**exponents
    diag      = np.diag( powers )
    diag_inv  = np.diag( 1/powers )
    weighted_pascal = diag.dot( pascal_triangle )
    weighted_pascal = weighted_pascal.dot( diag_inv )
    return weighted_pascal.dot(p)