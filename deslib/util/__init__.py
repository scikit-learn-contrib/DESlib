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

deslib.util.knne - Implementation of the K-Nearest Neighbors Equality
technique

deslib.util.aggregation.dfp - General Dynamic Frienemy Pruning (DFP)
implementation. This implementation allows using the DFP method to any ensemble
model, not only dynamic ones.

deslib.util.bpso - V and S shaped Binary Particle Swarm Optimization for
 used feature selection.
"""

from .aggregation import *
from .bpso import BPSO
from .datasets import *
from .dfp import frienemy_pruning
from .dfp import frienemy_pruning_preprocessed
from .diversity import *
from .faiss_knn_wrapper import FaissKNNClassifier
from .instance_hardness import *
from .knne import KNNE
from .prob_functions import *
