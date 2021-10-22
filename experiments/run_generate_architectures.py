
import json
import os
import numpy as np

import run_experiment
import newton_lilypads

if __name__ == '__main__':
    config = json.load( open('./experiment_jsons/mlp_randomized_template.json') )
    #
    # Loop
    run_count = 10
    for i in range(run_count):
        print("")
        print(f'''Launching experiment number {i}...''')

        #
        # Randomize hyperparameters at initialization
        print("--> Generating random architecture...")
        description = config['network']['build_kwargs']['layer_description']
        layer_count = len(description)
        out_size = description[0]['in_size']
        width_ratios = []
        for i in range(layer_count-1):
            layer = description[i]
            #
            layer['in_size'] = out_size
            random_factor = np.random.uniform()
            random_factor = 1.5*random_factor+0.2
            width_ratios.append( random_factor )
            out_size = max(int( random_factor*out_size + 0.5), 1)
            layer['out_size'] = out_size
        # Set last in_size
        description[-1]['in_size'] = out_size
        # Gain
        gain = 2.0 #np.random.exponential()
        config['initialization']['torch_kwargs']['gain'] = gain

        #
        # Analysis via Free Probability
        print("--> Analysis via FPT...")
        width_ratios  = np.array( width_ratios )
        variances     = gain*gain*2/(1+width_ratios)         # Variance of Xavier initialization
        print("Width ratios   : ", width_ratios)
        print("Gains (squared): ", variances)
        analysis = newton_lilypads.run_as_module( width_ratios, variances, verbose=True, plots=True )

        #
        # Save
        import time
        date = time.ctime().replace(":", "-").replace(" ", "_").replace("__", "_")
        filename = "random_"+ date+".json"
        config['FTP'] = { 'quantiles': list(analysis['quantiles']) }
        with open( os.path.join('./random_jsons', filename), 'w') as outfile:
            json.dump(config, outfile, indent=4)
            outfile.close()
        # for key in analysis:
        #     filename = "random_"+date+"_"+key+".npy"
        #     path = os.path.join('./random_jsons', filename)
        #     np.save( path, analysis[key])