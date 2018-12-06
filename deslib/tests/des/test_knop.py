from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.linear_model import Perceptron
from sklearn.utils.estimator_checks import check_estimator

from deslib.des.knop import KNOP


def test_check_estimator():
    check_estimator(KNOP)


# Test the estimate competence method receiving n samples as input
def test_estimate_competence_batch(example_estimate_competence,
                                   create_pool_classifiers):
    X, y, neighbors, distances, _, _ = example_estimate_competence
    query = np.ones((3, 2))
    expected = np.array([[4.0, 3.0, 4.0],
                         [5.0, 2.0, 5.0],
                         [2.0, 5.0, 2.0]])

    knop_test = KNOP(create_pool_classifiers)
    knop_test.fit(X, y)
    knop_test.neighbors = neighbors
    knop_test.distances = distances

    knop_test._get_similar_out_profiles = Mock(return_value=(None, neighbors))
    probabilities = np.zeros((3, 6))

    competences = knop_test.estimate_competence_from_proba(query,
                                                           probabilities)
    assert np.allclose(competences, expected, atol=0.01)


def test_weights_zero():
    knop_test = KNOP()
    competences = np.zeros((1, 3))
    result = knop_test.select(competences)

    assert np.all(result)


def test_fit(example_estimate_competence, create_pool_classifiers):
    X, y = example_estimate_competence[0:2]

    knop_test = KNOP(create_pool_classifiers)
    knop_test.fit(X, y)
    expected_scores = np.array([[0.5, 0.5], [1.0, 0.0], [0.33, 0.67]])
    expected_scores = np.tile(expected_scores, (15, 1, 1))

    assert np.array_equal(expected_scores, knop_test.dsel_scores_)

    # Assert the roc_algorithm_ is fitted to the scores (decision space)
    # rather than the features (feature space)
    expected_roc_data = knop_test.dsel_scores_[:, :, 0]
    assert np.array_equal(knop_test.op_knn_._fit_X, expected_roc_data)


# Test if the class is raising an error when the base classifiers do not
# implements the predict_proba method. Should raise an exception when the
# base classifier cannot estimate posterior probabilities (predict_proba)
# Using Perceptron classifier as it does not implements predict_proba.
def test_not_predict_proba(create_X_y):
    X, y = create_X_y

    clf1 = Perceptron()
    clf1.fit(X, y)
    with pytest.raises(ValueError):
        knop = KNOP([clf1, clf1])
        knop.fit(X, y)


def test_select():
    knop_test = KNOP()
    competences = np.ones(3)
    competences[0] = 0
    expected = np.atleast_2d([False, True, True])
    selected = knop_test.select(competences)
    assert np.array_equal(expected, selected)
