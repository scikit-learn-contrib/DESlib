"""
The :mod:`deslib.des` provides a set of key dynamic ensemble selection algorithms (DES). DES techniques by default
selects all base classifiers that attain a certain competence level.
"""

from .base import DES
from .des_clustering import DESClustering
from .des_knn import DESKNN
from .des_p import DESP
from .knop import KNOP
from .knora_e import KNORAE
from .knora_u import KNORAU
from .meta_des import METADES
from .probabilistic import Probabilistic, RRC, Exponential, DESKL, MinimumDifference

__all__ = ['DES',
           'METADES',
           'KNORAE',
           'KNORAU',
           'KNOP',
           'DESP',
           'DESKNN',
           'DESClustering',
           'Probabilistic',
           'RRC',
           'DESKL',
           'MinimumDifference',
           'Exponential']
