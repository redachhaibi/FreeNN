import cython
import numpy as np
cimport numpy as np

a=None

cpdef reset_counters():
    return

cpdef complex compute_G_adaptative( complex z_objective, Polynomial_Kantorovich_Wrapper function_wrapper, proxy=None):
    cdef complex j = complex(0,1)
    cdef complex z
    cdef complex g
    #
    # If no proxy is available, find high enough z in basin of attraction and compute associated w
    # This search uses a doubling strategy
    if proxy is None: 
        z = z_objective
        g = 1/z
        while not is_in_basin_ZG(z, g, function_wrapper = function_wrapper ):
            z = z + j*z.imag
            g = 1/z
        g = newton_raphson_ZG(z, function_wrapper = function_wrapper, guess=g)
    else:
        z, g = proxy
    #
    # Starts heading towards the objective z
    while abs(z- z_objective)>0:
        dz = z_objective-z
        while not is_in_basin_ZG(z+dz, g, function_wrapper=function_wrapper):
            dz = 0.5*dz
        z = z+dz
        g = newton_raphson_ZG(z, function_wrapper=function_wrapper, guess=g)
    # end while
    return g

"""
    Computes the inverse of a function thanks to the Newton-Raphson scheme
    Input:
    - z: Complex value
    - function_wrapper: Wrapper defining the function whose zero is searched
    - guess: starting point
    WARNING: If 'guess' is not in basin of attraction, then infinite chaotic loop
""" 
DEFAULT_PRECISION = 1e-12
cdef inline complex newton_raphson( complex z, Polynomial_Kantorovich_Wrapper function_wrapper, complex guess, double error=DEFAULT_PRECISION):
    cdef complex value
    cdef complex grad
    cdef complex m = guess
    #
    while True:
        value = f(function_wrapper, m, z)
        if ( abs(value) < error ):
            break
        grad = f_prime(function_wrapper, m, z)
        m = m - value/grad
    # end while
    return m

# Same after transform m \mapsto g i.e finds the g = G(z) such that:
# m = z g - 1
cdef inline complex newton_raphson_ZG( complex z, Polynomial_Kantorovich_Wrapper function_wrapper, complex guess, double error=DEFAULT_PRECISION):
    cdef complex m
    cdef complex g=guess
    m = newton_raphson( z, function_wrapper, guess=(z*g-1), error=error)
    return (m+1)/z

cdef inline bint is_in_basin_ZG( complex z, complex g, Polynomial_Kantorovich_Wrapper function_wrapper):
    return is_in_basin(z, z*g-1, function_wrapper)

cdef inline bint is_in_basin( complex z, complex m, Polynomial_Kantorovich_Wrapper function_wrapper):
    cdef complex value
    cdef complex derivative
    cdef complex step
    cdef complex new_m
    cdef float ball_max_y
    cdef float bound_f_2
    cdef float criterion
    value      = f(function_wrapper, m, z)
    derivative = f_prime(function_wrapper, m, z)
    # Compute w value after one step
    step  = -value / derivative
    new_m = m + step
    # Check if new_m in domain
    #if new_m.imag >= 0:
    #   return False
    # Check if NR ball is in domain
    ball_max_y = new_m.imag + abs(step.imag)
    #if ball_max_y >= 0:
    #    return False
    # Compute bound on second derivative
    bound_f_2 = f_second_bound(function_wrapper, new_m, step, z)
    criterion = abs(step/derivative)*bound_f_2
    return criterion < 0.5

# Computes the rational function
cdef inline complex phi(Polynomial_Kantorovich_Wrapper wrapper, complex m):
    return my_polyval(wrapper.numerator,m) / my_polyval(wrapper.denominator,m)

# Computes value of f_z(m)
cdef inline complex f(Polynomial_Kantorovich_Wrapper wrapper, complex m, complex z):
    return my_polyval(wrapper.numerator,m)/z - my_polyval(wrapper.denominator,m)

# Computes value of f_z'(m)
cdef inline complex f_prime(Polynomial_Kantorovich_Wrapper wrapper, complex m, complex z):
    return my_polyval(wrapper.numerator1,m)/z - my_polyval(wrapper.denominator1,m)

# Computes bound on ball for f_z
# Input:
# - (m,z)   : Center
# - step: Radius
cdef inline float f_second_bound(Polynomial_Kantorovich_Wrapper wrapper, complex m, complex step, complex z):
    cdef np.ndarray p
    p = wrapper.numerator2/z - wrapper.denominator2
    p = taylor_expand( p, m, wrapper.pascal_triangle)
    p = abs(p)
    return my_polyval_real(p, abs(step))

ctypedef double DTYPE_t
cdef inline complex my_polyval(np.ndarray p, complex x):
    cdef int i
    cdef int length = p.shape[0]
    cdef complex y = complex(0,0)
    #
    for i in range(length):
        y = x * y + p[i]
    return y

cdef inline float my_polyval_real(np.ndarray p, float x):
    cdef int i
    cdef float y = 0
    #
    for i in range(len(p)):
        y = x * y + p[i]
    return y


"""
    Rational Wrapper for function m \mapsto f_z(m) 

We are searching for m such that f_z(m) = 0
    Use case: 
    - Inverting a rational function phi(m) = num(m)/den(m)
    - This is reduced to m \mapsto f_z(m) polynomial
    - Away from axis f_z( w=0 ) \approx 0 as z \approx \infty
    - Excellent bound on second derivative via Taylor expansion
"""
cdef class Polynomial_Kantorovich_Wrapper:

    numerator   : np.ndarray
    denominator : np.ndarray
    numerator1  : np.ndarray
    denominator1: np.ndarray
    numerator2  : np.ndarray
    denominator2: np.ndarray
    pascal_triangle: nd.array

    # Constructor
    # Params: Numerator and Denominator of rational function
    #   - Coefficients are numpy arrays
    #   - Highest degree comes first
    def __init__(self, numerator, denominator):
        # Copy
        self.numerator   = numerator
        self.denominator = denominator
        # Derivatives
        self.numerator1   = np.polyder(self.numerator   )
        self.denominator1 = np.polyder(self.denominator )
        self.numerator2   = np.polyder(self.numerator1  )
        self.denominator2 = np.polyder(self.denominator1)
        # Maximal degree for second derivative
        max_degree2 = max( len(self.numerator2), len(self.denominator2)) - 1
        # Extends coeffs of second derivative, in order to have same length
        self.numerator2   = np.append( [0.0]*(max_degree2+1-len(self.numerator2  )), self.numerator2  )
        self.denominator2 = np.append( [0.0]*(max_degree2+1-len(self.denominator2)), self.denominator2)
        # Precompute Pascal triangle
        self.pascal_triangle = pascal( max_degree2 + 1)

"""
  Compute Pascal triangle up to size n
  Returns: Numpy matrix with binomial coefficients
  - Matrix represents linear map P(X) \mapsto P(X+1) in the canonical basis
  - Convention is that first columns are the image of monomials of higher degree
"""
cdef np.ndarray pascal(int n):
    cdef np.ndarray mat
    cdef int i
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
cdef inline np.ndarray taylor_expand(np.ndarray p, complex a, np.ndarray pascal_triangle):
    #
    cdef int degree
    cdef np.ndarray exponents
    cdef np.ndarray powers
    cdef np.ndarray weighted_pascal
    cdef i
    #
    degree = len(p)-1
    #
    exponents = np.array( range(0, degree+1, 1)  )
    powers    = a**exponents # Not so slow
    #
    # powers = np.full( (degree+1,) , a )
    # powers[0] = 1
    # powers = np.cumprod(powers)
    #
    #diag      = np.diag( powers )
    #diag_inv  = np.diag( 1/powers )
    # weighted_pascal = diag.dot( pascal_triangle )
    # weighted_pascal = weighted_pascal.dot( diag_inv )
    weighted_pascal = powers[:,None] * pascal_triangle * ( 1/powers)[None,:]
    return weighted_pascal.dot(p)