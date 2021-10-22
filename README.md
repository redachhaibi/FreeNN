# FreeNN

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