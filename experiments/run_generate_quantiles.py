
import json
import os
import numpy as np

import run_experiment
import newton_lilypads

if __name__ == '__main__':
    abspath  = os.path.abspath(__file__)
    dirname  = os.path.dirname( abspath )

    experiment_dir = "random_architectures_with_loss/fusion_raw"
    experiment_dir = os.path.join( dirname, experiment_dir )

    out_dir = "random_architectures_with_loss"
    out_dir = os.path.join( dirname, out_dir )

    # Get list of json files
    filenames = [f for f in os.listdir(experiment_dir) if os.path.isfile(os.path.join(experiment_dir, f))]
    data = []

    #
    # Loop
    print(f"Input path {experiment_dir}")
    for filename in filenames:
        print("")
        print(f" |-Processing file {filename}...")

        # Open json
        full_path = os.path.join(experiment_dir, filename)
        config = json.load( open( full_path) )
        # Fix for old files where:
        # - there was a typo
        # - width_ratios and variances were not saved
        if 'FPT' in config:
            # Do nothing
            None
        elif 'FTP' in config:
            FPT_data = config.pop('FTP')
            #
            description = config['network']['build_kwargs']['layer_description']
            layer_count = len(description)
            out_size = description[0]['in_size']
            width_ratios = []
            for i in range(layer_count-1):
                layer = description[i]
                #
                in_size = layer['in_size']
                out_size = layer['out_size']
                width_ratios.append( out_size*1.0/in_size )
            # Gain
            gain = config['initialization']['torch_kwargs']['gain']
            # FPT hyperparameters
            width_ratios  = np.array( width_ratios )
            variances     = gain*gain*2/(1+width_ratios)         # Variance of Xavier (Glorot) initialization
            config['FPT'] = { 'width_ratios': list(width_ratios),
                    'variances'   : list(variances),
                    'quantiles'   : FPT_data['quantiles'] }
            print("Width ratios: ", width_ratios)
            print("Variances   : ", variances)
        else:
            assert(False)

        #
        # Analysis via Free Probability
        print(" |-> Analysis via FPT...")
        width_ratios = config['FPT']['width_ratios']
        variances    = config['FPT']['variances']
        print("Width ratios: ", width_ratios)
        print("Variances   : ", variances)

        analysis = newton_lilypads.run_as_module( width_ratios, variances, interval=(-3,100), N=4000, verbose=False, plots=False )

        #
        # Save FTP result
        config['FPT']['cumulative'] = list(analysis['cumulative']) 
        config['FPT']['density']    = list(analysis['density']) 
        config['FPT']['quantiles']  = list(analysis['quantiles']) 
        fullpath = os.path.join(out_dir, filename)
        print(f" |-> Saving to {fullpath}")
        with open( fullpath, 'w') as outfile:
            json.dump(config, outfile, indent=4)
            outfile.close()
    # End for
    print("Done.")