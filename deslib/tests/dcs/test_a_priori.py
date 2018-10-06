import numpy as np
import pytest
from sklearn.linear_model import Perceptron
from deslib.dcs.a_priori import APriori
from deslib.tests.examples_test import (setup_example1,
                                        setup_example_all_ones,
                                        create_pool_classifiers,
                                        create_base_classifier,
                                        setup_example_kuncheva)

from sklearn.utils.estimator_checks import check_estimator


def test_check_estimator():
    check_estimator(APriori(selection_method='best'))


# Should always be 1.0 since the supports for the correct class is always 1.
@pytest.mark.parametrize('index, expected', [(0, [1.0,  1.0,  1.0]),
                                             (1, [1.0,  1.0,  1.0]),
                                             (2, [1.0,  1.0,  1.0])])
def test_estimate_competence_all_ones(index, expected):
    X, y, neighbors, distances, dsel_processed, dsel_scores = setup_example_all_ones()

    query = np.array([1, 1])

    a_priori_test = APriori(create_pool_classifiers())

    a_priori_test.DSEL_processed_ = dsel_processed
    a_priori_test.dsel_scores_ = dsel_scores
    a_priori_test.DSEL_target_ = y
    a_priori_test.n_classes_ = 2

    neighbors = neighbors[index, :].reshape(1, -1)
    distances = distances[index, :].reshape(1, -1)

    competences = a_priori_test.estimate_competence(query.reshape(1, -1), neighbors, distances)
    assert np.isclose(competences, expected).all()


# Testing example from kuncheva's book (combining pattern classifiers)
def test_estimate_competence_kuncheva_ex():
    query = np.array([1, 1])
    example_kuncheva = setup_example_kuncheva()

    a_priori_test = APriori([create_base_classifier(return_value=1)], k=example_kuncheva['k'])

    a_priori_test.DSEL_processed_ = example_kuncheva['dsel_processed']
    a_priori_test.dsel_scores_ = example_kuncheva['dsel_scores']
    a_priori_test.DSEL_target_ = example_kuncheva['y_independent']
    a_priori_test.n_classes_ = example_kuncheva['n_classes']

    neighbors = example_kuncheva['neighbors'].reshape(1, -1)
    distances = example_kuncheva['distances'].reshape(1, -1)

    competences = a_priori_test.estimate_competence(query.reshape(1, -1), neighbors, distances)
    assert np.isclose(competences, 0.70, atol=0.01)


# Test the estimate competence method receiving n samples as input
def test_estimate_competence_batch():
    _, y, neighbors, _, dsel_processed, dsel_scores = setup_example1()

    query = np.ones((3, 2))
    expected = np.array([[0.333333,  0.50000,  0.40000],
                         [0.666666, 0.50000, 0.60000],
                         [0.000000, 0.50000, 0.20000]])

    # Using 3 neighbors to facilitate the calculations
    a_priori_test = APriori(create_pool_classifiers(), 3)

    a_priori_test.DSEL_processed_ = dsel_processed
    a_priori_test.dsel_scores_ = dsel_scores
    a_priori_test.DSEL_target_ = y
    a_priori_test.n_classes_ = 2

    neighbors = neighbors[:, 0:3]
    distances = np.ones((3, 3))

    competences = a_priori_test.estimate_competence(query, neighbors, distances)
    assert np.allclose(competences, expected, atol=0.01)


def test_fit():
    X, y = setup_example1()[0:2]

    a_priori_test = APriori(create_pool_classifiers())
    a_priori_test.fit(X, y)
    expected = np.array([[0.5, 0.5], [1.0, 0.0], [0.33, 0.67]])
    expected = np.tile(expected, (15, 1, 1))
    assert np.array_equal(a_priori_test.dsel_scores_, expected)


# Test if the class is raising an error when the base classifiers do not implements the predict_proba method.
# Should raise an exception when the base classifier cannot estimate posterior probabilities (predict_proba)
# Using Perceptron classifier as it does not implements the predict_proba method.
def test_not_predict_proba():
    X, y = setup_example1()[0:2]

    clf1 = Perceptron()
    clf1.fit(X, y)
    with pytest.raises(ValueError):
        APriori([clf1, clf1]).fit(X, y)
