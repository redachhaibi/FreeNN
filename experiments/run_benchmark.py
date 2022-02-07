import os
import numpy as np

import experiments.newton_lilypads        as newton_lilypads
import experiments.newton_lilypads_cython as newton_lilypads_cython
import experiments.pennington_algorithm1  as pennington_algorithm1
import time

def launch_method_with_repeat( method, width_ratios, variances, N, repeat ):
    results = []
    for i in range(repeat):
        result = launch_method( method, width_ratios, variances, N)
        result['pass'] = i
        print(f"Duration {i+1}/{repeat} Pass: {1000*result['time']} ms")
        results.append( result )
    return results

def launch_method( method, width_ratios, variances, N):
    #
    # Define switcher for all methods
    interval=(-1, 6)
    switcher = {
        'lilypads'             : (lambda q: newton_lilypads.run_as_module( width_ratios, variances, verbose=False, plots=False, imaginary_parts=[1e-4], interval=interval, N=N, ignoreNonLinearity=True )),
        'lilypads_cython'      : (lambda q: newton_lilypads_cython.run_as_module( width_ratios, variances, verbose=False, plots=False, imaginary_parts=[1e-4], interval=interval, N=N, ignoreNonLinearity=True )),
        'pennington_algorithm1': (lambda q: pennington_algorithm1.run_as_module( width_ratios, variances, verbose=False, plots=False, interval=interval, N=N, ignoreNonLinearity=True ) ),
    }
    # Get the function from switcher dictionary
    selection = switcher.get(method, lambda: "Invalid method name")
    # Launch
    start = time.time()
    analysis = selection("")
    timing = time.time()-start
    # Record result
    result = {'method' : method,
              'depth'  : len(width_ratios),
              'N'      : N,
              'time'   : timing}
    return result

def launch():
    # Depths 
    depths = np.arange(2,42, step=2)
    sampling_points = np.arange(100, 2000, step=100)
    # Numer of repetitions for benchmark
    repeat = 5

    timings = []

    # N=1000
    # for L in depths:

    L=20
    for N in sampling_points:

        # FPT hyperparameters
        # Easy case for benchmark:
        # - no spread out measure
        # - no mass at zero
        width_ratios  = np.array( [0.8]*L )
        gain          = 1.0
        variances     = gain*gain*2/(1+width_ratios)         # Variance of Xavier (Glorot) initialization
        variances     = np.array( [1]*L )
        # print("--> FPT hyperparameters")
        # print("Width ratios: ", width_ratios)
        # print("Variances   : ", variances)
    
        print("")
        print(f"Neural network's depth: {L}")
        print(f"Sampling points : {N}")
        print("")

        #
        # Analysis via Newton lilypads
        print("|-> Analysis via Newton lilypads (Pure python)...")
        method  = 'lilypads'
        timings = timings + launch_method_with_repeat( method, width_ratios, variances, N, repeat )
        print("")

        #
        # Analysis via Newton lilypads (Cython)
        print(f"|-> Analysis via Newton lilypads (Cython)...")
        method  = 'lilypads_cython'
        timings = timings + launch_method_with_repeat( method, width_ratios, variances, N, repeat )
        print("")

        #
        # Analysis via Pennington et al.'s Algorithm 1
        print("|-> Analysis via Pennington et al.'s Algorithm 1...")
        method = 'pennington_algorithm1'
        timings = timings + launch_method_with_repeat( method, width_ratios, variances, N, repeat )
        print("")

    # #
    # # Save benchmark result
    print("")
    print("Recording data")
    import csv
    filename = "benchmark.csv"
    with open( os.path.join('./', filename), 'w') as csvfile:
        fieldnames = ['method', 'depth', 'pass', 'time', 'N']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in timings:
            writer.writerow( entry )
        csvfile.close()

if __name__ == '__main__':

    launch()

    # import pstats, cProfile

    # import pyximport
    # pyximport.install()

    # cProfile.runctx("launch()", globals(), locals(), "Profile.prof")

    # s = pstats.Stats("Profile.prof")
    # s.strip_dirs().sort_stats("tottime").print_stats()