import os
import numpy as np

import experiments.newton_lilypads        as newton_lilypads
import experiments.newton_lilypads_cython as newton_lilypads_cython
import experiments.newton_lilypads_numba  as newton_lilypads_numba
import experiments.pennington_algorithm1  as pennington_algorithm1
import time

def launch():
    # Depths 
    depths = np.arange(2,42, step=2)
    sampling_points = np.arange(100, 2000, step=100)
    # Numer of repetitions for benchmark
    repeat = 10

    timings = []

    for L in depths:

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
        interval=(-1, 6)

        #
        # Analysis via Newton lilypads
        print("|-> Analysis via Newton lilypads (Pure python)...")
        method  = 'lilypads'
        for i in range(repeat):
            start = time.time()
            analysis = newton_lilypads.run_as_module( width_ratios, variances, verbose=False, plots=False, imaginary_parts=[1e-4], interval=interval, N=1000, ignoreNonLinearity=True )
            timing = time.time()-start
            print(f"Duration {i+1}/{repeat} Pass: {1000*timing} ms")
            timings.append( {'method' : method,
                             'depth'  : L,
                             'pass'   : i,
                             'time'   : timing} )

        #
        # Analysis via Newton lilypads (Cython)
        print(f"|-> Analysis via Newton lilypads (Cython)...")
        method = 'lilypads_cython'
        for i in range(repeat):
            start = time.time()
            analysis = newton_lilypads_cython.run_as_module( width_ratios, variances, verbose=False, plots=False, imaginary_parts=[1e-4], interval=interval, N=1000, ignoreNonLinearity=True )
            timing = time.time()-start
            print(f"Duration {i+1}/{repeat} Pass: {1000*timing} ms")
            timings.append( {'method' : method,
                             'depth'  : L,
                             'pass'   : i,
                             'time'   : timing} )
        #
        # Analysis via Newton lilypads (Numba JIT) -- Does not work. Numba is a mess
        # print("--> Analysis via Newton lilypads (Numba)...")
        # for i in range(repeat):
        #     start = time.time()
        #     analysis = newton_lilypads_numba.run_as_module( width_ratios, variances, verbose=False, plots=False, imaginary_parts=[1e-4], interval=interval, N=1000, ignoreNonLinearity=True )
        #     timing = time.time()-start
        #     print(f"Duration {i+1}/{repeat} Pass: {1000*timing} ms")
        #     print("")

        #
        # Analysis via Pennington et al.'s Algorithm 1
        print("|-> Analysis via Pennington et al.'s Algorithm 1...")
        method = 'pennington_algorithm1'
        for i in range(repeat):
            start = time.time()
            analysis = pennington_algorithm1.run_as_module( width_ratios, variances, verbose=False, plots=False, interval=interval, N=1000, ignoreNonLinearity=True )
            timing = time.time()-start
            print(f"Duration {i+1}/{repeat}: {1000*timing} ms")
            timings.append( {'method' : method,
                             'depth'  : L,
                             'pass'   : i,
                             'time'   : timing} )

    # #
    # # Save benchmark result
    print("")
    print("Recording data")
    import csv
    filename = "benchmark.csv"
    with open( os.path.join('./', filename), 'w') as csvfile:
        fieldnames = ['method', 'depth', 'pass', 'time']
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