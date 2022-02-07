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
    fig = plt.figure( figsize = (12,12) )
    ax  = fig.add_subplot( 111 )
    for method in data:
        values = data[method]
        # Plot for densities
        x = [ e['depth'] for e in values]
        y = [ float(e['time'])*1000 for e in values]
        ax.scatter(x, y, label=method)
        ax.set(xlabel='Number of layers (L)', ylabel='Computational time (ms)',
                title='')
        ax.grid()
        ax.set_yscale('log')
        #ax.set_ylim(0,0.5)
        ax.legend()
    #
    plt.savefig("benchmark_plot_depth.png")
    plt.show()


if __name__ == '__main__':
    launch()
