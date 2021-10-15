
import json
import os
import numpy as np

import run_experiment

if __name__ == '__main__':
    config = json.load( open('./experiment_jsons/mlp_from_list.json') )
    #
    # Loop
    run_count = 10
    run_per_architecture = 1
    for i in range(run_count):
        print("")
        print(f'''Launching experiment number {i}''')
        # Randomize using exponential random variables
        description = config['network']['build_kwargs']['layer_description']
        layer_count = len(description)
        out_size = description[0]['in_size']
        for i in range(layer_count-1):
            layer = description[i]
            #
            layer['in_size'] = out_size
            random_factor = np.random.exponential()
            out_size = max(int( random_factor*out_size + 0.5), 1)
            layer['out_size'] = out_size
        # Set last in_size
        description[-1]['in_size'] = out_size
        # Save new json to disk
        with open('./experiment_jsons/mlp_randomized.json', 'w') as outfile:
            json.dump(config, outfile, indent=4)

        # Analysis via Free Probability
        from freenn.core import newton, adaptative

        assert(False)
        # Run experiment
        for j in range(run_per_architecture):
            path = run_experiment.run_as_module("./experiment_jsons/mlp_randomized.json")
            print(path)

        #os.system( 'python run_experiment.py --json_file=./experiment_jsons/mlp_randomized.json' )
        #file = open('./run_experiment.py --json_file=./experiment_jsons/mlp_randomized.json').read()
        #exec(file)
        #close(file)