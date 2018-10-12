"""A Python library for Dynamic Ensemble Selection.

``DESlib`` is a library containing the implementation of the state-of-the art
dynamic classifier and ensemble selection techniques. The library also provides
some static ensemble methods that are used as baseline comparison.

Subpackages
-----------
des
    The implementation of several DES techniques.

dcs
    The implementation of several DCS techniques.

static
    The implementation of baseline ensemble methods.

util
    A collection of aggregation functions and diversity measures for ensemble
    of classifiers.
"""

# list of all modules available in the library
__all__ = ['des', 'dcs', 'static', 'util', 'tests']

__version__ = '0.3.dev'
