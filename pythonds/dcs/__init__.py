"""
The :mod:`pythonds.dcs` provides a set of key dynamic classifier selection algorithms (DCS).
"""

from .a_posteriori import APosteriori
from .a_priori import APriori
from .base import DCS
from .lca import LCA
from .mcb import MCB
from .mla import MLA
from .ola import OLA
from .rank import Rank

__all__ = ['DCS',
           'APosteriori',
           'APriori',
           'LCA',
           'OLA',
           'MLA',
           'MCB',
           'Rank']
