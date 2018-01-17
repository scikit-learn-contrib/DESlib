import pytest

from deslib.dcs.ola import OLA
from deslib.tests.examples_test import *


@pytest.mark.parametrize('index, expected', [(0, [0.57142857,  0.71428571,  0.71428571]),
                                             (1, [0.71428571,  0.85714286,  0.71428571]),
                                             (2, [0.57142857,  0.71428571,  0.57142857])])
def test_estimate_competence(index, expected):
    ola_test = OLA(create_pool_classifiers())
    ola_test.processed_dsel = dsel_processed_ex1
    ola_test.neighbors = neighbors_ex1[index, :]
    ola_test.distances = distances_ex1[index, :]
    ola_test.DFP_mask = [1, 1, 1]
    query = np.array([1, 1])
    competences = ola_test.estimate_competence(query.reshape(1, -1))
    assert np.isclose(competences, expected).all()
