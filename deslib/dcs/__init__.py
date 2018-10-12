"""
The :mod:`deslib.dcs` provides a set of key dynamic classifier selection
algorithms (DCS).
"""

from .a_posteriori import APosteriori
from .a_priori import APriori
from .base import BaseDCS
from .lca import LCA
from .mcb import MCB
from .mla import MLA
from .ola import OLA
from .rank import Rank

__all__ = ['BaseDCS',
           'APosteriori',
           'APriori',
           'LCA',
           'OLA',
           'MLA',
           'MCB',
           'Rank']
