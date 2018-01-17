import pytest

from deslib.dcs.lca import LCA
from deslib.tests.examples_test import *


@pytest.mark.parametrize('index, expected', [(0, [0.75000000,  0.66666667,  0.75000000]),
                                             (1, [0.80000000,  1.00000000,  0.80000000]),
                                             (2, [1.00000000,  0.60000000,  0.50000000])])
def test_estimate_competence_woods(index, expected):
    lca_test = LCA(create_pool_classifiers())
    lca_test.processed_dsel = dsel_processed_ex1
    lca_test.neighbors = neighbors_ex1[index, :]
    lca_test.distances = distances_ex1[index, :]
    lca_test.DFP_mask = [1, 1, 1]
    lca_test.DSEL_target = y_dsel_ex1
    query = np.array([1, 1])
    competences = lca_test.estimate_competence(query.reshape(1, -1))
    assert np.isclose(competences, expected).all()


# in this test case, the target of the neighbors is always different than the predicted. So
# the estimation of competence should always be zero
@pytest.mark.parametrize('index', [0, 1, 2])
def test_estimate_competence_diff_target(index):
    query = np.array([1, 1])

    lca = LCA(create_pool_classifiers())

    lca.processed_dsel = dsel_processed_ex1
    lca.DSEL_target = np.ones(15, dtype=int) * 3

    lca.neighbors = neighbors_ex1[index, :]
    lca.distances = distances_ex1[index, :]
    lca.DFP_mask = [1, 1, 1]

    expected = [0.0, 0.0, 0.0]

    competences = lca.estimate_competence(query.reshape(1, -1))
    assert np.isclose(competences, expected).all()
