import numpy as np
from freenn.core import newton

call_counter_failed_basin = 0
call_counter_NR           = 0

def reset_counters():
    global call_counter_failed_basin
    global call_counter_NR
    call_counter_failed_basin = 0
    call_counter_NR    = 0

def compute_G_adaptative( z_objective, function_wrapper, proxy=None, debug=False):
    global call_counter_failed_basin
    global call_counter_NR
    j = complex(0,1)
    #
    # If no proxy is available, find high enough z in basin of attraction and compute associated w
    # This search uses a doubling strategy
    if proxy is None: 
        z = z_objective
        g = 1/z
        while not newton.is_in_basin_ZG(z, g, function_wrapper = function_wrapper ):
            call_counter_failed_basin += 1
            z = z + j*z.imag
            g = 1/z
        g = newton.newton_raphson_ZG(z, function_wrapper = function_wrapper, guess=g)
        if debug:
            print("Valid z: ", z)
            print("Guess g: ", 1/z)
            print("G(t,z) = ", g )
    else:
        z, g = proxy
    #
    if debug:
        print("Proxy (z,g): ", z, g)
    #
    # Starts heading towards the objective z
    while abs(z- z_objective)>0:
        dz = z_objective-z
        while not newton.is_in_basin_ZG(z+dz, g, function_wrapper=function_wrapper):
            call_counter_failed_basin += 1
            dz = 0.5*dz
        z = z+dz
        g = newton.newton_raphson_ZG(z, function_wrapper=function_wrapper, guess=g)
        call_counter_NR += 1
        if debug:
            print("Valid z: ", z)
            print("G(t,z) = ", g )
    # end while
    return g