import pytest

from deslib.dcs.rank import Rank
from deslib.tests.examples_test import *


@pytest.mark.parametrize('index, expected', [(0, [1, 5, 0]),
                                             (1, [1, 1, 2]),
                                             (2, [0, 0, 1])])
def test_estimate_competence(index, expected):
    rank_test = Rank(create_pool_classifiers())
    rank_test.processed_dsel = dsel_processed_ex1
    rank_test.neighbors = neighbors_ex1[index, :]
    rank_test.distances = distances_ex1[index, :]
    rank_test.DFP_mask = [1, 1, 1]
    query = np.array([1, 1])
    competences = rank_test.estimate_competence(query.reshape(1, -1))
    assert np.isclose(competences, expected).all()


