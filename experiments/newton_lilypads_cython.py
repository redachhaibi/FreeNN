
"""
Here we consider the $S$-transforms of Marchenko-Pastur distributions, which are of the form:
$$ S_{W_l^T W_l}(m) = \frac{1}{\sigma_l^2} \frac{1}{1+\lambda_l m} \ ,$$
where the $\lambda_i$ are the scale pameters and $\sigma_l^2$ are the variances. 

We have computed in Section 3, Theorem 3.1 of the paper:
$$ S_{J^T J}(m)
 = \frac{m+1}{m M^{\langle -1 \rangle}(m)}
 = \prod_{\ell=1}^L \left[ S_{D_\ell^2}(\Lambda_\ell m) \frac{1}{\sigma_l^2} \frac{1}{1+\Lambda_l m} \right] \ .$$
$$ M_{J^T J}^{\langle -1 \rangle}(m)
 = \frac{m+1}{m}
   \prod_{\ell=1}^L \left[ \frac{\sigma_\ell^2 (1+\Lambda_\ell m)}{S_{D^2_\ell(\Lambda_\ell m)}} \right] \ .$$
"""

import time
import math
import numpy as np
import matplotlib.pyplot as plt

# Compute q^\ell by recurrence
from scipy import special
def compute_C(q):
    return 0.5*special.erf(1/np.sqrt(2*q))
  
def compute_g( q, non_linearity_string ):
    switcher = {
        'Linear'  : (lambda q: q),
        'Relu'    : (lambda q: 0.5*q),
        'HardTanh': (lambda q: 2*q*compute_C(q) - np.sqrt(2*q/math.pi)*np.exp(-1/(2*q)) ),
        'HardSine': (lambda q: 1/3 + (4/(math.pi**2)*( - np.exp(-0.5*q*math.pi**2) + 0.25*np.exp(-2*q*math.pi**2) ) ) ),
    }
    # Get the function from switcher dictionary
    g = switcher.get(non_linearity_string, lambda: "Invalid non-linearity name")
    # Execute the function
    return g(q)

# Returns P,Q such that P/Q = 1/(z-1)
def compute_M_basic():
    return ( np.array([1]), np.array([1, -1]) )

def compute_M_Linear(q):
    return compute_basic()
 
def compute_M_Relu(q):
    P, Q = compute_M_basic()
    return ( 0.5*P, Q )

def compute_M_HardTanh(q):
    P, Q = compute_M_basic()
    C    = compute_C(q)
    return ( C*P, Q )

def compute_M_Triangle(q):
    return compute_basic()
  
def compute_M_D( q, non_linearity_string ):
    switcher = {
        'Linear'  : compute_M_Linear,
        'Relu'    : compute_M_Relu,
        'HardTanh': compute_M_HardTanh,
        'HardSine': compute_M_Triangle,
    }
    # Get the function from switcher dictionary
    non_linearity = switcher.get(non_linearity_string, lambda: "Invalid non-linearity name")
    # Execute the function
    return non_linearity(q)

def compute_S_D( q, non_linearity_string ):
    C = compute_C(q)
    switcher = {
        'Linear'  : ( [1], [1] ),
        'Relu'    : ( [1, 1], [1, 0.5] ),
        'HardTanh': ( [1, 1], [1, C] ),
        'HardSine': ( [1], [1] ),
    }
    # Get the function from switcher dictionary
    non_linearity = switcher.get(non_linearity_string, lambda: "Invalid non-linearity name")
    # Execute the function
    return non_linearity

"""
    Input: Polynomial P as numpy array (Larger degree coeff first)
           Scalar s
    Ouput: Polynomial X \mapsto P(sX)
"""
def scale_polynomial( P, s):
    degree    = len(P)-1
    exponents = np.linspace(degree, 0, num=degree+1)
    return P*( s**exponents )

#N: Space mesh
# N=Number of points
# a=Left edge
# b=Right edge
import cython
from freenn.core import adaptative_cython

def run_as_module(lambdas, sigma2s, verbose=False, plots=False,
                    imaginary_parts = [0.1, 0.01, 1e-3, 1e-4],
                    interval = (-3, 20), N=1000, ignoreNonLinearity=False):

    # Input = Array of \lambda_l's, width ratios
    Lambdas = np.cumprod(lambdas)
    non_linearities = ['Relu']*len(lambdas)
    L       = len(lambdas)


    # Compute coefficients of M_inverse (numerator and denominator)
    # Conventions:
    #   - Coefficients are numpy arrays
    #   - Highest degree comes first
    #
    if verbose:
        print("Initialization...")
    # Compute product of S_D's
    num_S, den_S = [1], [1]
    q            = 1
    for index in range(L):
        non_linearity = non_linearities[index]
        q    = sigma2s[index]*compute_g(q, non_linearity)
        # Compute X \mapsto M_D(X)
        P, Q = compute_S_D(q, non_linearity)
        # Compute X \mapsto M_D( Lambda*X )
        Lambda = Lambdas[index]
        P = scale_polynomial(P, Lambda)
        Q = scale_polynomial(Q, Lambda)
        # Multiply numerators and denominators
        num_S  = np.polymul(num_S, P)
        den_S  = np.polymul(den_S, Q)

    # General non-linearity
    # # Numerator of M_inverse
    # roots         = np.append(-1, -1/Lambdas)
    # leading_coeff = np.prod(Lambdas)*np.prod(sigma2s)
    # coeff_num     = np.poly( roots ) * leading_coeff
    # # Denominator of M_inverse = m
    # coeff_den     = np.array( [1, 0] )
    # # Incorporate the product of S_D's by dividing by them
    # coeff_num  = np.polymul(coeff_num, den_S)  # Numerator of M_inverse
    # coeff_den  = np.polymul(coeff_den, num_S)  # Denominator of M_inverse

    # Valid for ReLu only
    # Numerator of M_inverse
    if ignoreNonLinearity:
        roots         = np.append(-1, -1/(Lambdas) )
    else:
        roots         = np.append(-1, -1/(2*Lambdas) )
    leading_coeff = np.prod(Lambdas)*np.prod(sigma2s)
    coeff_num     = np.poly( roots ) * leading_coeff
    # Denominator of M_inverse = m
    coeff_den     = np.array( [1, 0] )
    if verbose:
        print( "Numerator coefficients :", coeff_num)

    # Setup Wrapper
    wrapper = adaptative_cython.Polynomial_Kantorovich_Wrapper( coeff_num, coeff_den)

    def mean( coeff_num, coeff_den):
        mean = np.polyval(coeff_num, 0) / np.polyval( coeff_den[:-1], 0 )
        return mean

    # 
    # Stabilization trick:
    # M(z=0) = \mu({0}) - 1
    # and 
    # M_inverse has root closest to -1 which is -\max(1/\Lambda_i)
    # Hence M(z=0) = -\max(1/(2*\Lambda_i)) if > -1
    #              = -1 otherwise
    if ignoreNonLinearity:
        M_zero = -np.max(1/(Lambdas))
    else:
        M_zero = -np.max(1/(2*Lambdas))
    M_zero = np.max( [M_zero, -1])
    mass_at_zero = M_zero + 1 

    #
    # Computation of measure
    if verbose:
        print("Computation of measure...")

    #Init
    A,B = interval
    space_grid = np.linspace(A, B, N)
    dx = (B-A)/N

    #Multiple passes for the number of iterations
    mass_in_window  = 0
    pass_counter    = 0

    j = complex(0,1)
    measure_mean = mean(coeff_num, coeff_den)

    if plots:
        fig = plt.figure( figsize = (24,12) )
        ax  = fig.add_subplot( 121 )
        ax2 = fig.add_subplot( 122 )
    y_proxy = None
    guess   = None
    G       = np.array( np.zeros_like(space_grid) - 1e-3*complex(0,1) )
    for y in imaginary_parts:
        pass_counter += 1
        if verbose:
            print("")
            print ('Pass [{}/{}]:' 
                    .format(pass_counter, len(imaginary_parts)))
            print(f'|- Im z: {y}')
            print(f'|- Mean: {measure_mean}')
        start = time.time()
        adaptative_cython.reset_counters()
        # Compute
        z = np.array( space_grid + y*complex(0,1) )
        # Problematic region is around zero
        if y<=0.01:
            l2 = np.searchsorted(space_grid, 0.02)
            l1 = np.searchsorted(space_grid, -0.02)
        else:
            l2 = np.searchsorted(space_grid, 0)-1
            l1 = l2+1
        if verbose:
            print(f"Problematic indices: [{l1},{l2}]")
        # Sweep right
        if verbose:
            print("Sweeping left of zero")
        # guess = precomputed_proxy
        guess = None
        #for i in range(mean_index, N):
        for i in range(l1):
            G[i]  = adaptative_cython.compute_G_adaptative(z[i], function_wrapper = wrapper, proxy=guess)
            guess = ( z[i], G[i] )
        # Sweep left
        if verbose:
            print("Sweeping right of zero")
        # guess = precomputed_proxy
        guess = None
        for i in range(N-1, l2, -1):
            G[i] = adaptative_cython.compute_G_adaptative(z[i], function_wrapper = wrapper, proxy=guess)
            guess = ( z[i], G[i] )
        #
        # Statistics
        timing        = time.time() - start
        if verbose:
            print ('Pass [{}/{}], Duration: {:.1f} ms' 
                    .format(pass_counter, len(imaginary_parts), 1000*timing))
        if verbose:
            print("Number of calls to subroutine:")
            print("'Newton-Raphson'  :", adaptative_cython.call_counter_NR)
            print("'Attraction basin':", adaptative_cython.call_counter_failed_basin)
        #
        # Compute density
        #density    = G-mass_at_zero/(space_grid+y*j) # This line removes mass at zero
        density    = G
        density    = -np.imag(density)/np.pi
        density    = density*(density>0) # For safety
        # Compute cumulative, including mass at zero
        cumulative = np.cumsum(density)*dx
        if verbose:
            print(f"Mass at zero: {mass_at_zero}")
        mass_in_window = max( mass_in_window, np.max(cumulative) )
        if verbose:
            print(f"Captured total mass in window: {np.max(cumulative)}" )
        mass_missing = mass_in_window-np.max(cumulative)
        if verbose:
            print(f"Missing due to instability at zero: {mass_missing}" )
            print(f"Total inflated mass: {mass_missing+np.max(cumulative)}" )
        cumulative = (mass_missing)*(space_grid>0) + cumulative
        cumulative = cumulative*(cumulative<1.0) + 1.0*(cumulative>=1.0)
        cumulative[-1] = 1.0
        # Plot
        if plots:
            # Plot for densities
            ax.plot(space_grid, density, '--', label="y=%.5f"%y)
            ax.set(xlabel='Space (x)', ylabel='Value',
                    title='Density')
            ax.grid()
            ax.set_ylim(0,0.5)
            ax.legend()
            # Plot for cumulative distributions
            ax2.plot(space_grid, cumulative, '--', label="y=%.5f"%y)
            ax2.set(xlabel='Space (x)', ylabel='Value',
                    title='Cumulative')
            ax2.grid()
            ax2.set_ylim(0,1.1)
            ax2.legend()
        #
        cumulative = cumulative/np.max(cumulative)
        # Update proxy
        y_proxy = y
    #
    # Compute quantiles
    quantiles = np.zeros(11)
    for i in range(11):
        a = 0.1*i
        quantiles[i] = max( space_grid[ np.searchsorted( cumulative, a) ], 0)
    if verbose:
        print('Quantiles (from 0 to 1, with 0.1 steps):')
        print(quantiles)
    # Final plot
    if plots:
        plt.show()

    return {
        'space_grid': space_grid,
        'density'   : density,
        'cumulative': cumulative,
        'quantiles' : quantiles
    }


import click
@click.command()
#@click.option('-j', '--json_file', required=True, help='The JSON file to run the experiment')
def run():
    lambdas = [1]*5
    sigma2s = [1]*len(lambdas)
    return run_as_module(lambdas, sigma2s, verbose=True, plots=True)

if __name__ == '__main__':
    run()