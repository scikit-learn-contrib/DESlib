import pytest

from deslib.dcs.a_priori import APriori
from deslib.tests.examples_test import *


# Should always be 1.0 since the supports for the correct class is always 1.
@pytest.mark.parametrize('index, expected', [(0, [1.0,  1.0,  1.0]),
                                             (1, [1.0,  1.0,  1.0]),
                                             (2, [1.0,  1.0,  1.0])])
def test_estimate_competence_all_ones(index, expected):
    query = np.array([1, 1])

    a_priori_test = APriori(create_pool_classifiers())

    a_priori_test.processed_dsel = dsel_processed_ex1
    a_priori_test.dsel_scores = dsel_scores_all_ones
    a_priori_test.DSEL_target = y_dsel_ex1
    a_priori_test.n_classes = 2

    a_priori_test.neighbors = neighbors_ex1[index, :]
    a_priori_test.distances = distances_all_ones[index, :]
    a_priori_test.DFP_mask = [1, 1, 1]

    competences = a_priori_test.estimate_competence(query.reshape(1, -1))
    assert np.isclose(competences, expected).all()


# Testing example from kuncheva's book (combining pattern classifiers)
def test_estimate_competence_kuncheva_ex():
    query = np.array([1, 1])

    a_priori_test = APriori([create_base_classifier(return_value=1)], k=k_ex_kuncheva)

    a_priori_test.processed_dsel = dsel_processed_kuncheva
    a_priori_test.dsel_scores = dsel_scores_ex_kuncheva
    a_priori_test.DSEL_target = y_dsel_ex_kuncheva_independent
    a_priori_test.n_classes = n_classes_ex_kuncheva

    a_priori_test.neighbors = neighbors_ex_kuncheva
    a_priori_test.distances = distances_ex_kuncheva
    a_priori_test.DFP_mask = [1]

    competences = a_priori_test.estimate_competence(query.reshape(1, -1))
    assert np.isclose(competences, 0.70, atol=0.01)


@pytest.mark.parametrize('index, expected', [(0, [0.333333,  0.50000,  0.40000]),
                                             (1, [0.666666,  0.50000,  0.60000]),
                                             (2, [0.000000,  0.50000,  0.20000])])
def test_estimate_competence2(index, expected):
    query = np.array([1, 1])

    # Using 3 neighbors to facilitate the calculations
    a_priori_test = APriori(create_pool_classifiers(), 3)

    a_priori_test.processed_dsel = dsel_processed_ex1
    a_priori_test.dsel_scores = dsel_scores_ex1
    a_priori_test.DSEL_target = y_dsel_ex1
    a_priori_test.n_classes = 2

    a_priori_test.neighbors = neighbors_ex1[index, 0:3]
    a_priori_test.distances = distances_all_ones[index, 0:3]
    a_priori_test.DFP_mask = [1, 1, 1]

    competences = a_priori_test.estimate_competence(query.reshape(1, -1))
    assert np.isclose(competences, expected).all()


def test_fit():
    a_priori_test = APriori(create_pool_classifiers())
    a_priori_test.fit(X_dsel_ex1, y_dsel_ex1)
    assert np.isclose(a_priori_test.dsel_scores, [0.5, 0.5, 1.0, 0.0, 0.33, 0.67]).all()
