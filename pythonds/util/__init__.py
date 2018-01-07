"""
The :mod:`pythonds.util` This module includes various utilities. They are divided into three parts:

pythonds.util.aggregation - Implementation of aggregation functions such as majority voting and averaging.
Such functions can be applied to any list of classifiers.

pythonds.util.diversity - Implementation of different measures of diversity between classifiers.

pythonds.util.prob_functions - Functions to estimate the competence of a base classifier based on the
probability estimates.
"""

from .aggregation import *
from .diversity import *
from .prob_functions import *
