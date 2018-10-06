import numpy as np
import pytest
from sklearn.linear_model import Perceptron
from unittest.mock import Mock

from deslib.des.knop import KNOP
from deslib.tests.examples_test import create_pool_classifiers, setup_example1
from sklearn.utils.estimator_checks import check_estimator


def test_check_estimator():
    check_estimator(KNOP)


@pytest.mark.parametrize('index, expected', [(0, [4.0, 3.0, 4.0]),
                                             (1, [5.0, 2.0, 5.0]),
                                             (2, [2.0, 5.0, 2.0])])
def test_estimate_competence(index, expected):
    X, y, neighbors, distances, _, _ = setup_example1()
    query = np.atleast_2d([1, 1])

    knop_test = KNOP(create_pool_classifiers())
    knop_test.fit(X, y)

    knop_test.DFP_mask = np.ones(knop_test .n_classifiers_)
    knop_test.neighbors = neighbors[index, :]
    knop_test._get_similar_out_profiles = Mock(return_value=(None, neighbors[index, :].reshape(1, -1)))
    knop_test.distances = distances[index, :]

    probabilities = []
    for clf in knop_test.pool_classifiers:
        probabilities.append(clf.predict_proba(query))

    probabilities = np.array(probabilities).transpose((1, 0, 2))

    competences = knop_test.estimate_competence_from_proba(query, probabilities)
    assert np.allclose(competences, expected, atol=0.01)


# Test the estimate competence method receiving n samples as input
def test_estimate_competence_batch():
    X, y, neighbors, distances, _, _ = setup_example1()
    query = np.ones((3, 2))
    expected = np.array([[4.0, 3.0, 4.0],
                         [5.0, 2.0, 5.0],
                         [2.0, 5.0, 2.0]])

    knop_test = KNOP(create_pool_classifiers())
    knop_test.fit(X, y)

    knop_test.DFP_mask = np.ones(knop_test .n_classifiers_)
    knop_test.neighbors = neighbors
    knop_test._get_similar_out_profiles = Mock(return_value=(None, neighbors))
    knop_test.distances = distances
    probabilities = np.zeros((3, 6)) # not used in this test

    competences = knop_test.estimate_competence_from_proba(query, probabilities)
    assert np.allclose(competences, expected, atol=0.01)


def test_weights_zero():
    X, y = setup_example1()[0:2]

    knop_test = KNOP(create_pool_classifiers())
    knop_test.fit(X, y)
    competences = np.zeros((1, 3))
    result = knop_test.select(competences)

    assert np.all(result)


def test_fit():
    X, y = setup_example1()[0:2]

    knop_test = KNOP(create_pool_classifiers())
    knop_test.fit(X, y)
    expected_scores = np.array([[0.5, 0.5], [1.0, 0.0], [0.33, 0.67]])
    expected_scores = np.tile(expected_scores, (15, 1, 1))

    assert np.array_equal(expected_scores, knop_test.dsel_scores_)

    # Assert the roc_algorithm_ is fitted to the scores (decision space) rather than the features (feature space)
    expected_roc_data = knop_test.dsel_scores_[:, :, 0]
    assert np.array_equal(knop_test.op_knn_._fit_X, expected_roc_data)


# Test if the class is raising an error when the base classifiers do not implements the predict_proba method.
# Should raise an exception when the base classifier cannot estimate posterior probabilities (predict_proba)
# Using Perceptron classifier as it does not implements the predict_proba method.
def test_not_predict_proba():
    X, y = setup_example1()[0:2]

    clf1 = Perceptron()
    clf1.fit(X, y)
    with pytest.raises(ValueError):
        knop = KNOP([clf1, clf1])
        knop.fit(X, y)


def test_select():
    knop_test = KNOP(create_pool_classifiers())
    competences = np.ones(3)
    competences[0] = 0
    expected = np.atleast_2d([False, True, True])
    selected = knop_test.select(competences)
    assert np.array_equal(expected, selected)