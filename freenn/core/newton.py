import numpy as np
from freenn.utils import polynomials

"""
    Computes the inverse of a function thanks to the Newton-Raphson scheme
    Input:
    - z: Complex value
    - function_wrapper: Wrapper defining the function whose zero is searched
    - guess: starting point
    WARNING: If 'guess' is not in basin of attraction, then infinite chaotic loop
""" 
DEFAULT_PRECISION = 1e-12
def newton_raphson( z, function_wrapper, guess=None, error=None):
    if guess is None:
        m = 0
    else:
        m = guess
    #
    if error is None:
        error = DEFAULT_PRECISION
    #
    while True:
        value = function_wrapper.f(m, z)
        if ( abs(value) < error ):
            break
        grad = function_wrapper.f_prime(m, z)
#        if ( abs(grad) < error ):
#            print("Gradient too small!!")
#            print("value: ", value)
#            print("grad: ", grad)
#            return None
#            break
        # Newton-Raphson iteration
        m = m - value/grad
    # end while
    return m

# Same after transform m \mapsto g i.e finds the g = G(z) such that:
# m = z g - 1
def newton_raphson_ZG( z, function_wrapper, guess=None, error=None):
    if guess is None:
        g = 1/z
    else:
        g = guess
    #
    m = newton_raphson( z, function_wrapper, guess=(z*g-1), error=error)
    return (m+1)/z

def is_in_basin(z, m, function_wrapper, debug=False):
    value      = function_wrapper.f(m, z)
    derivative = function_wrapper.f_prime(m, z)
    # Compute w value after one step
    step  = -value / derivative
    new_m = m + step
    if debug:
        print("")
        print("Call is_in_basin for z=",z)
        print("value:     ", value)
        print("derivative:", derivative)
        print("m:    ", m)
        print("new_m:", new_m)
        print("Im(m + h_0): ", new_m.imag)
    # Check if new_m in domain
    if new_m.imag >= 0:
       return False
    # Check if NR ball is in domain
    ball_max_y = new_m.imag + abs(step.imag)
    if debug:
        print("Im(m + h_0) + |Im(h_0)|: ", ball_max_y)
    #if ball_max_y >= 0:
    #    return False
    # Compute bound on second derivative
    bound_f_2 = function_wrapper.f_second_bound(new_m, step, z)
    criterion = abs(step/derivative)*bound_f_2
    if debug:
        print("Kantorovich criterion: ", criterion)
    return criterion < 0.5

def is_in_basin_ZG(z, g, function_wrapper, debug=False):
    return is_in_basin(z, z*g-1, function_wrapper, debug)

"""
    Rational Wrapper for function m \mapsto f_z(m) 

We are searching for m such that f_z(m) = 0
    Use case: 
    - Inverting a rational function phi(m) = num(m)/den(m)
    - This is reduced to m \mapsto f_z(m) polynomial
    - Away from axis f_z( w=0 ) \approx 0 as z \approx \infty
    - Excellent bound on second derivative via Taylor expansion
"""
class Polynomial_Kantorovich_Wrapper:

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

    # Computes the rational function
    def phi(self, m):
        return np.polyval(self.numerator,m) / np.polyval(self.denominator,m)

    # Computes value of f_z(m)
    def f(self, m, z):
        return np.polyval(self.numerator,m)/z - np.polyval(self.denominator,m)

    # Computes value of f_z'(m)
    def f_prime(self, m, z):
        return np.polyval(self.numerator1,m)/z - np.polyval(self.denominator1,m)

    # Computes bound on ball for f_z
    # Input:
    # - (m,z)   : Center
    # - step: Radius
    def f_second_bound(self, m, step, z):
        p = self.numerator2/z - self.denominator2
        p = polynomials.taylor_expand( p, m)
        p = abs(p)
        return np.polyval(p, abs(step))