# FreeNN

## Content

The repository is structured as follows. We only describe the most important files for a new user.
```bash
./
|-- experiments
|  |-- run_generate_architectures.py: Samples random architectures and compute FPT metrics.
|  |-- run_randomized_experiment.py: Trains several neural networks based on architectures described in json files.
|-- freenn: Contains the numerical routines aimed at computing FPT densities.
|-- ipynb: Contains Python notebooks which demonstrate how the code works
|  |-- DemoFPT.ipynb: Illustrates the measure concentration in FPT.
|  |-- mc_vs_lilypads.ipynb: Benchmarks our method compared to Monte-Carlo sampling. Used to generate the figure of Section 5.
|  |-- LossStatistics.ipynb: Once all the experiments have been run, this notebook computes correlation statistics and gives a scatter plot.
|-- tests: Unit tests
|-- README.md: This file
```

Note that the dependencies have been left to a bare minimum in order to run the package freenn. Running the experiments however requires the installation of torch, torchvision and click via:
```bash
$ pip install torch torchvision click
```

## Installation

1. Create new virtual environment

```bash
$ python3 -m venv .venv
```

(Do
sudo apt install python3-venv
if needed)

3. Activate virtual environment

```bash
$ source .venv/bin/activate
```

4. Upgrade pip, wheel and setuptools 

```bash
$ pip install --upgrade pip
$ pip install --upgrade setuptools
$ pip install wheel
```

5. Install the `freenn` package.

```bash
python setup.py develop
```

6. (Optional) In order to use Jupyter with this virtual environment .venv
```bash
pip install --user ipykernel
python -m ipykernel install --user --name=myenv
```
(see https://janakiev.com/blog/jupyter-virtual-envs/ for details)

## Configuration
Nothing to do

## Credits
Later