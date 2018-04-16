from unittest.mock import Mock

import pytest
from sklearn.linear_model import Perceptron

from deslib.des.knop import KNOP
from deslib.tests.examples_test import *


@pytest.mark.parametrize('index, expected', [(0, [4.0, 3.0, 4.0]),
                                             (1, [5.0, 2.0, 5.0]),
                                             (2, [2.0, 5.0, 2.0])])
def test_estimate_competence(index, expected):
    query = np.atleast_2d([1, 1])

    knop_test = KNOP(create_pool_classifiers())
    knop_test.fit(X_dsel_ex1, y_dsel_ex1)

    knop_test.DFP_mask = np.ones(knop_test .n_classifiers)
    knop_test.neighbors = neighbors_ex1[index, :]
    knop_test._get_similar_out_profiles = Mock(return_value=(None, np.atleast_2d(neighbors_ex1[index, :])))
    knop_test.distances = distances_ex1[index, :]

    probabilities = []
    for clf in knop_test.pool_classifiers:
        probabilities.append(clf.predict_proba(query))

    probabilities = np.array(probabilities).transpose((1, 0, 2))

    competences = knop_test.estimate_competence_from_proba(query, probabilities)
    assert np.allclose(competences, expected, atol=0.01)


# Test the estimate competence method receiving n samples as input
def test_estimate_competence_batch():
    query = np.ones((3, 2))
    expected = np.array([[4.0, 3.0, 4.0],
                          [5.0, 2.0, 5.0],
                          [2.0, 5.0, 2.0]])

    knop_test = KNOP(create_pool_classifiers())
    knop_test.fit(X_dsel_ex1, y_dsel_ex1)

    knop_test.DFP_mask = np.ones(knop_test .n_classifiers)
    knop_test.neighbors = neighbors_ex1
    knop_test._get_similar_out_profiles = Mock(return_value=(None, neighbors_ex1))
    knop_test.distances = distances_ex1

    probabilities = np.zeros((3, 6)) # not used in this test

    competences = knop_test.estimate_competence_from_proba(query, probabilities)
    assert np.allclose(competences, expected, atol=0.01)


def test_weights_zero():

    knop_test = KNOP(create_pool_classifiers())
    knop_test.fit(X_dsel_ex1, y_dsel_ex1)
    competences = np.zeros((1, 3))
    result = knop_test.select(competences)

    assert np.all(result)


def test_fit():
    knop_test = KNOP(create_pool_classifiers())
    knop_test.fit(X_dsel_ex1, y_dsel_ex1)
    expected_scores = np.array([[0.5, 0.5], [1.0, 0.0], [0.33, 0.67]])
    expected_scores = np.tile(expected_scores, (15, 1, 1))

    assert np.array_equal(expected_scores, knop_test.dsel_scores)

    # Assert the roc_algorithm is fitted to the scores (decision space) rather than the features (feature space)
    expected_roc_data = knop_test.dsel_scores[:, :, 0]
    assert np.array_equal(knop_test.op_knn._fit_X, expected_roc_data)


# Test if the class is raising an error when the base classifiers do not implements the predict_proba method.
# Should raise an exception when the base classifier cannot estimate posterior probabilities (predict_proba)
# Using Perceptron classifier as it does not implements the predict_proba method.
def test_not_predict_proba():
    X = X_dsel_ex1
    y = y_dsel_ex1
    clf1 = Perceptron()
    clf1.fit(X, y)
    with pytest.raises(ValueError):
        KNOP([clf1, clf1])


def test_select():
    knop_test = KNOP(create_pool_classifiers())
    competences = np.ones(3)
    competences[0] = 0
    expected = np.atleast_2d([False, True, True])
    selected = knop_test.select(competences)
    assert np.array_equal(expected, selected)