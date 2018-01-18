"""A Python library for Dynamic Ensemble Selection.

``DESlib`` is a library containig the implementation of the state-of-the art 
dynamic classifier and ensemble selection techniques. The library also provides
some static ensemble methods that are used as baseline comparison.

Subpackages
-----------
des
    The mplementation of several DES techniques.

dcs
    The mplementation of several DCS techniques.

static
    The Implementation of baseline ensemble methods.

util
    A collection of aggregation functions and diversity measures for ensemble of classifiers.
"""

from ._version import __version__

# list of all modules available in the library
__all__ = ['des', 'dcs', 'static', 'util']
