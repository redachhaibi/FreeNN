
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


def run_as_module(lambdas, sigma2s, verbose=False, plots=False):
    from freenn.core import newton, adaptative

    # Input = Array of \lambda_l's, width ratios
    Lambdas = np.cumprod(lambdas)
    non_linearities = ['Relu']*len(lambdas)
    L       = len(lambdas)


    # Compute coefficients of M_inverse (numerator and denominator)
    # Conventions:
    #   - Coefficients are numpy arrays
    #   - Highest degree comes first
    #
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

    # Numerator of M_inverse
    roots         = np.append(-1, -1/Lambdas)
    leading_coeff = np.prod(Lambdas)*np.prod(sigma2s)
    coeff_num     = np.poly( roots ) * leading_coeff
    # Of denominator of M_inverse = w
    coeff_den     = np.array( [1, 0] )
    # Incorporate the inverse product of M_D's
    coeff_num  = np.polymul(coeff_num, den_S)
    coeff_den  = np.polymul(coeff_den, num_S)

    # Setup Wrapper
    wrapper = newton.Polynomial_Kantorovich_Wrapper( coeff_num, coeff_den)

    def mean( coeff_num, coeff_den):
        mean = np.polyval(coeff_num, 0) / np.polyval( coeff_den[:-1], 0 )
        return mean

    #
    # Computation of measure
    print("Computation of measure...")

    #N: Space mesh
    N=1000
    a=-1
    b=6

    #Init
    space_grid = np.linspace(a, b, N)
    dx = (b-a)/N

    #Multiple passes for the number of iterations
    #imaginary_parts = [1.0, 0.01, 1e-4]
    imaginary_parts = [1e-4]
    densities       = []
    hilbert_transf  = []
    pass_counter    = 0
    iter_count      = [ [] for i in space_grid ]
    errors1         = [ [] for i in space_grid ]
    errors2         = [ [] for i in space_grid ]
    choices         = [ [] for i in space_grid ]

    j = complex(0,1)
    measure_mean = mean(coeff_num, coeff_den)

    fig = plt.figure( figsize = (12,7) )
    ax = fig.add_subplot( 111 )
    y_proxy = None
    guess   = None
    G       = np.array( space_grid + complex(0,1) )
    for y in imaginary_parts:
        start = time.time()
        adaptative.reset_counters()
        # Compute
        if y_proxy is not None:
            z_proxy = z
        z = np.array( space_grid + y*complex(0,1) )
        mean_index = int(0.5*N) #np.searchsorted(space_grid, measure_mean)
        if y_proxy is None: #First pass is special
            g                 = adaptative.compute_G_adaptative(measure_mean+j, function_wrapper = wrapper, proxy=None)
            precomputed_proxy = (measure_mean+j,  g)
        # Sweep right
        guess = precomputed_proxy
        for i in range(mean_index, N):
            G[i]  = adaptative.compute_G_adaptative(z[i], function_wrapper = wrapper, proxy=guess)
            guess = ( z[i], G[i] )
        # Sweep left
        guess = precomputed_proxy
        for i in range(mean_index, -1, -1):
            G[i] = adaptative.compute_G_adaptative(z[i], function_wrapper = wrapper, proxy=guess)
            guess = ( z[i], G[i] )
        # else:
        #     for i in range(0, N):
        #         guess = ( z_proxy[i], G[i] )
        #         G[i]  = adaptative.compute_G_adaptative(z[i], function_wrapper = wrapper, proxy=guess)
        # Statistics
        pass_counter += 1
        timing        = time.time() - start
        if verbose:
            print ('Pass [{}/{}], Duration: {:.1f} ms' 
                    .format(pass_counter, len(imaginary_parts), 1000*timing))
            print("Number of calls to subroutine:")
            print("'Newton-Raphson'  :", adaptative.call_counter_NR)
            print("'Attraction basin':", adaptative.call_counter_failed_basin)
            print("")
        # Plot
        ax.plot(space_grid, -np.imag(G)/np.pi, '--', label="y=%.5f"%y)
        ax.set(xlabel='Space (x)', ylabel='Value',
                title='Density')
        ax.grid()
        # Update proxy
        y_proxy = y
        precomputed_proxy = ( z[mean_index], G[mean_index])
    #
    density    = -np.imag(G)/np.pi
    cumulative = np.cumsum(density)
    cumulative = cumulative/max(cumulative)
    # Final plot
    if plots:
        plt.ylim(0,0.5)
        plt.legend()
        plt.savefig('multiscale.png')
        plt.show()
        #
        fig = plt.figure( figsize = (12,7) )
        ax = fig.add_subplot( 111 )
        ax.plot(space_grid, cumulative, '--', label="y=%.5f"%y)
        ax.set(xlabel='Space (x)', ylabel='Value',
                title='Cumulative')
        ax.grid()
        plt.ylim(0,1.1)
        plt.legend()
        plt.show()
    #
    quantiles = np.zeros(11)
    for i in range(11):
        a = 0.1*i
        quantiles[i] = space_grid[ np.searchsorted( cumulative, a) ]
    print('Quantiles (from 0 to 1, with 0.1 steps):')
    print(quantiles)
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