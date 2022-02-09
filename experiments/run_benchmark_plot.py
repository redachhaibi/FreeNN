import csv
import numpy as np
import matplotlib.pyplot as plt

def launch():
    with open('./experiments/benchmark_depth.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        data   = {}
        for row in reader:
            if row['method'] not in data:
                data[ row['method'] ] = []    
            #
            data[ row['method'] ].append( row )
    #
    fig, (ax1, ax2) = plt.subplots( 1, 2, sharey=True, figsize=(16,6) )
    fig.subplots_adjust(wspace=0)
    #
    # First plot: Depth
    for method in data:
        values = data[method]
        # Plot for densities
        x = [ e['depth'] for e in values]
        y = [ float(e['time'])*1000 for e in values]
        ax1.scatter(x, y, label=method)
        ax1.set(xlabel='Number of layers (L)', ylabel='Computational time (ms)',
                title='')
    ax1.grid()
    ax1.set_yscale('log')
    ax1.legend()
    #
    with open('./experiments/benchmark_N.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        data   = {}
        for row in reader:
            if row['method'] not in data:
                data[ row['method'] ] = []    
            #
            data[ row['method'] ].append( row )
    #
    # Second plot: Depth
    for method in data:
        values = data[method]
        # Plot for densities
        x = [ e['N'] for e in values]
        y = [ float(e['time'])*1000 for e in values]
        ax2.scatter(x, y, label=method)
    ax2.set(xlabel='Number of density points (N)', ylabel='',
            title='')
    ax2.grid()
    ax2.set_yscale('log')
    every_nth = 2
    for n, label in enumerate(ax2.xaxis.get_ticklabels()):
        if n % every_nth != 1:
            label.set_visible(False)
    #
    # Saving
    plt.savefig("benchmark_plot.png")
    plt.show()


if __name__ == '__main__':
    launch()
