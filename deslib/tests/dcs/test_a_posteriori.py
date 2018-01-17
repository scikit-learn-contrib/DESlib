import pytest

from deslib.dcs.a_posteriori import APosteriori
from deslib.tests.examples_test import *


# Should always be 1.0 since the supports for the correct class is always 1.
@pytest.mark.parametrize('index', [0, 1, 2])
def test_estimate_competence_all_ones(index):
    query = np.array([1, 1])

    a_posteriori_test = APosteriori(create_pool_classifiers())

    a_posteriori_test.processed_dsel = dsel_processed_ex1
    a_posteriori_test.dsel_scores = dsel_scores_all_ones
    a_posteriori_test.DSEL_target = y_dsel_ex1
    a_posteriori_test.n_classes = 2

    a_posteriori_test.neighbors = neighbors_ex1[index, :]
    a_posteriori_test.distances = distances_all_ones[index, :]
    a_posteriori_test.DFP_mask = [1, 1, 1]

    expected = [1.0, 1.0, 1.0]

    competences = a_posteriori_test.estimate_competence(query.reshape(1, -1))
    assert np.isclose(competences, expected).all()


# Testing example from kuncheva's book (combining pattern classifiers)
def test_estimate_competence_kuncheva_ex():
    query = np.array([1, 1])

    a_posteriori_test = APosteriori([create_base_classifier(return_value=1)], k=k_ex_kuncheva)

    a_posteriori_test.processed_dsel = dsel_processed_kuncheva
    a_posteriori_test.dsel_scores = dsel_scores_ex_kuncheva
    a_posteriori_test.DSEL_target = y_dsel_ex_kuncheva_dependent
    a_posteriori_test.n_classes = n_classes_ex_kuncheva

    a_posteriori_test.neighbors = neighbors_ex_kuncheva
    a_posteriori_test.distances = distances_ex_kuncheva
    a_posteriori_test.DFP_mask = [1]

    competences = a_posteriori_test.estimate_competence(query.reshape(1, -1))
    assert np.isclose(competences, 0.95, atol=0.01)


# in this test case, the target of the neighbors is always different than the predicted. So
# the estimation of competence should always be zero
@pytest.mark.parametrize('index', [0, 1, 2])
def test_estimate_competence_diff_target(index):
    query = np.array([1, 1])

    a_posteriori_test = APosteriori(create_pool_classifiers())

    a_posteriori_test.processed_dsel = dsel_processed_ex1
    a_posteriori_test.dsel_scores = dsel_scores_all_ones
    a_posteriori_test.DSEL_target = np.ones(15, dtype=int) * 3
    a_posteriori_test.n_classes = 2

    a_posteriori_test.neighbors = neighbors_ex1[index, :]
    a_posteriori_test.distances = distances_all_ones[index, :]
    a_posteriori_test.DFP_mask = [1, 1, 1]

    expected = [0.0, 0.0, 0.0]

    competences = a_posteriori_test.estimate_competence(query.reshape(1, -1))
    assert np.isclose(competences, expected).all()


def test_fit():
    a_posteriori_test = APosteriori(create_pool_classifiers())
    a_posteriori_test.fit(X_dsel_ex1, y_dsel_ex1)
    assert np.isclose(a_posteriori_test.dsel_scores, [0.5, 0.5, 1.0, 0.0, 0.33, 0.67]).all()
