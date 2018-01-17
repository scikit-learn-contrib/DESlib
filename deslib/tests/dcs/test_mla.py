import pytest

from deslib.dcs.mla import MLA
from deslib.tests.examples_test import *


# Should always be 1.0 since the supports for the correct class is always 1.
@pytest.mark.parametrize('index', [0, 1, 2])
def test_estimate_competence_all_ones(index):
    query = np.array([1, 1])

    mla_test = MLA(create_pool_classifiers())

    mla_test.processed_dsel = dsel_processed_all_ones
    mla_test.dsel_scores = dsel_scores_all_ones
    mla_test.DSEL_target = y_dsel_ex1
    mla_test.n_classes = 2

    mla_test.neighbors = neighbors_ex1[index, :]
    mla_test.distances = distances_all_ones[index, :]
    mla_test.DFP_mask = [1, 1, 1]

    expected = [1.0, 1.0, 1.0]

    competences = mla_test.estimate_competence(query.reshape(1, -1))
    assert np.isclose(competences, expected).all()


# Should always be 1.0 since the supports for the correct class is always 1.
@pytest.mark.parametrize('index, expected', [(0, [0.75000,  0.66666,  0.75000]),
                                             (1, [0.80000,  1.00000,  0.80000]),
                                             (2, [1.00000,  0.60000,  0.50000])])
def test_estimate_competence(index, expected):
    query = np.array([1, 1])

    mla_test = MLA(create_pool_classifiers())

    mla_test.processed_dsel = dsel_processed_ex1
    mla_test.dsel_scores = dsel_scores_all_ones
    mla_test.DSEL_target = y_dsel_ex1
    mla_test.n_classes = 2

    mla_test.neighbors = neighbors_ex1[index, :]
    mla_test.distances = distances_all_ones[index, :]
    mla_test.DFP_mask = [1, 1, 1]

    competences = mla_test.estimate_competence(query.reshape(1, -1))
    assert np.isclose(competences, expected).all()


# Testing example from kuncheva's book (combining pattern classifiers)
def test_estimate_competence_kuncheva_ex():
    query = np.array([1, 1])

    mla_test = MLA([create_base_classifier(return_value=1)], k=k_ex_kuncheva)

    mla_test.processed_dsel = dsel_processed_kuncheva
    mla_test.dsel_scores = dsel_scores_ex_kuncheva
    mla_test.DSEL_target = y_dsel_ex_kuncheva_dependent
    mla_test.n_classes = n_classes_ex_kuncheva

    mla_test.neighbors = neighbors_ex_kuncheva
    mla_test.distances = distances_ex_kuncheva
    mla_test.DFP_mask = [1]

    competences = mla_test.estimate_competence(query.reshape(1, -1))
    assert np.isclose(competences, 0.95, atol=0.01)


# in this test case, the target of the neighbors is always different than the predicted. So
# the estimation of competence should always be zero
@pytest.mark.parametrize('index', [0, 1, 2])
def test_estimate_competence_diff_target(index):
    query = np.array([1, 1])

    mla_test = MLA(create_pool_classifiers())

    mla_test.processed_dsel = dsel_processed_ex1
    mla_test.DSEL_target = np.ones(15, dtype=int) * 3

    mla_test.neighbors = neighbors_ex1[index, :]
    mla_test.distances = distances_ex1[index, :]
    mla_test.DFP_mask = [1, 1, 1]

    expected = [0.0, 0.0, 0.0]

    competences = mla_test.estimate_competence(query.reshape(1, -1))
    assert np.isclose(competences, expected).all()