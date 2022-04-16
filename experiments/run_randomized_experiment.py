
import json
import os
import numpy as np

import run_experiment

if __name__ == '__main__':
    #
    # Loop
    run_per_architecture = 10

    # Directory
    abspath  = os.path.abspath(__file__)
    dirname  = os.path.dirname( abspath )
    # Json config MNIST
    #jsonfilename = os.path.join( dirname, 'experiment_jsons/mlp_randomized_template.json')
    # Experiment dir
    experiment_dir = os.path.join( dirname, "random_architectures_with_loss")
    

    for i in range(run_per_architecture):
        print("")
        print(f'''Launching experiment number {i}...''')
        filenames = [f for f in os.listdir(experiment_dir) if os.path.isfile(os.path.join(experiment_dir, f))]

        for filename in filenames:
            full_path = os.path.join(experiment_dir, filename)
            #
            # Run architecture
            print(" |- Running experiment ", full_path)
            path       = run_experiment.run_as_module(full_path)
            #path      = "random_Fri_Oct_15_09-02-15_2021_Fri_Oct_15_09-42-33_2021"
            # 
            # Get learning curves
            train_path =  os.path.join(path, "learning_curve_train.npy")
            test_path  =  os.path.join(path, "learning_curve_test.npy")
            acc_path   =  os.path.join(path, "learning_curve_acc.npy")
            learning_curve_train = np.load(train_path)
            learning_curve_test  = np.load(test_path)
            learning_curve_acc   = np.load(acc_path)
            print("   Final loss (train)   :", learning_curve_train[-1])
            print("   Final loss (test )   :", learning_curve_test[-1] )
            print("   Final accuracy (test):", learning_curve_acc[-1] )
            #
            # Record:
            config = json.load( open( full_path) )
            if not 'Results' in config:
               config['Results'] = {
                   'learning_curve_train': [],
                   'learning_curve_test': [],
                   'learning_curve_acc' : [],
               }
            elif not 'learning_curve_acc' in config['Results']:
                config['Results']['learning_curve_acc'] = []
            config['Results']['learning_curve_train'].append( float(learning_curve_train[-1]) )
            config['Results']['learning_curve_test' ].append( float(learning_curve_test[-1] ) )
            config['Results']['learning_curve_acc'].append( float(learning_curve_acc[-1] ) )
            #
            # Dump
            with open( full_path, 'w') as outfile:
                json.dump(config, outfile, indent=4)
                outfile.close()