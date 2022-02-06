from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='freenn',

    version='0.01',

    description='Deterministic models for Jacobians of NN using Free Probability Theory',
    long_description=""" Jacobians of Neural Networks using Free Probability Theory (FreeNN).
    Thanks to the tools of Free Probability Theory, there is a deterministic model describing the spectral properties of Jacobians for large neural networks.
    This is an implementation of adaptative Newton-Raphson schemes, which effectively compute such spectral densities.""",
    url='',

    author='Anonymous',
    author_email='Anonymous',

    license='ApacheV2',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Programming Language :: Python :: 3',
    ],

    install_requires=["numpy==1.19", "matplotlib", "scipy==1.7"],

    keywords='',

    packages=find_packages(),

    entry_points={
        'console_scripts': [
            'sample=sample:main',
        ],
    },
)
