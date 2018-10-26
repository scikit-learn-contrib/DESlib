"""
The :mod:`deslib.util` This module includes various utilities. They are divided
into four parts:

deslib.util.aggregation - Implementation of aggregation functions such as
majority voting and averaging. Such functions can be applied to any list of
classifiers.

deslib.util.diversity - Implementation of different measures of diversity
between classifiers.

deslib.util.prob_functions - Functions to estimate the competence of a base
classifier based on the
probability estimates.

deslib.util.instance_hardness - Functions to measure the hardness level of a
given instance

deslib.util.faiss_knn_wrapper - Wrapper for Facebook AI fast similarity search
on GPU

deslib.util.datasets - Provides methods to generate synthetic data.
"""

from .aggregation import *
from .diversity import *
from .instance_hardness import *
from .prob_functions import *
from .datasets import *
