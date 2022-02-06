
import json
import os
import numpy as np

import run_experiment
import newton_lilypads

if __name__ == '__main__':
    config = json.load( open('./experiment_jsons/mlp_randomized_template.json') )
    need_generation = True
    #
    # Comment if not test
    #config = json.load( open('./random_jsons/random_Thu_Feb_3_14-21-22_2022.json') )
    #config = json.load( open('./random_jsons/random_Sun_Feb_6_00-27-47_2022.json') )
    #config = json.load( open('./random_jsons/random_Sun_Feb_6_01-00-47_2022.json') )
    #config = json.load( open('./random_jsons/random_Sun_Feb_6_01-15-13_2022.json') )
    #config = json.load( open('./random_jsons/random_Sun_Feb_6_01-27-06_2022.json') )
    #config = json.load( open('./random_jsons/random_Sun_Feb_6_02-48-33_2022.json') )
    #config = json.load( open('./random_jsons/random_Sun_Feb_6_02-57-10_2022.json') )
    #config = json.load( open('./random_jsons/random_Sun_Feb_6_03-18-46_2022.json') )
    #config = json.load( open('./random_jsons/random_Sun_Feb_6_13-58-31_2022.json') )
    #config = json.load( open('./random_jsons/random_Sun_Feb_6_16-30-04_2022.json') )
    need_generation = not 'FPT' in config

    #
    # Loop
    if not need_generation:
        run_count=1
    else:
        run_count = 10
    for i in range(run_count):
        print("")
        print(f'''Launching experiment number {i}...''')

        #
        # Identifier
        import time
        date = time.ctime().replace(":", "-").replace(" ", "_").replace("__", "_")
        identifier = "random_"+ date
        print(f"--> Identifier: {identifier}")
        # Check if architecture is already available
        if need_generation:
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
                random_factor = 1.5*random_factor+0.25
                width_ratios.append( random_factor )
                out_size = max(int( random_factor*out_size + 0.5), 1)
                layer['out_size'] = out_size
            # Set last in_size
            description[-1]['in_size'] = out_size
            # Gain
            gain = min(np.random.exponential(), 5)
            config['initialization']['torch_kwargs']['gain'] = gain
            # FPT hyperparameters
            width_ratios  = np.array( width_ratios )
            variances     = gain*gain*2/(1+width_ratios)         # Variance of Xavier (Glorot) initialization
            config['FPT'] = { 'width_ratios': list(width_ratios),
                    'variances'   : list(variances),
                    'quantiles': [] }
        else:
            print("--> Found already existing FPT hyperparameters...")
            width_ratios  = np.array( config['FPT']['width_ratios'] )
            variances     = np.array( config['FPT']['variances'] )


        #
        # Save architecture
        filename = identifier+".json"
        with open( os.path.join('./random_jsons', filename), 'w') as outfile:
            json.dump(config, outfile, indent=4)
            outfile.close()

        #
        # Analysis via Free Probability
        print("--> Analysis via FPT...")
        print("Width ratios: ", width_ratios)
        print("Variances   : ", variances)
        analysis = newton_lilypads.run_as_module( width_ratios, variances, verbose=True, plots=True )

        #
        # Save FTP result
        filename = identifier+".json"
        config['FPT']['quantiles'] = list(analysis['quantiles']) 
        with open( os.path.join('./random_jsons', filename), 'w') as outfile:
            json.dump(config, outfile, indent=4)
            outfile.close()
        # for key in analysis:
        #     filename = "random_"+date+"_"+key+".npy"
        #     path = os.path.join('./random_jsons', filename)
        #     np.save( path, analysis[key])