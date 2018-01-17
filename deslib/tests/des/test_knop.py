import pytest

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
    knop_test.distances = distances_ex1[index, :]
    competences = knop_test.estimate_competence(query)
    assert np.isclose(competences, expected, atol=0.01).all()


@pytest.mark.parametrize('index, expected', [(0, 0),
                                             (1, 0),
                                             (2, 1)])
def test_classify(index, expected):
    query = np.atleast_2d([1, 1])

    knop_test = KNOP(create_pool_classifiers())
    knop_test.fit(X_dsel_ex1, y_dsel_ex1)

    knop_test.DFP_mask = np.ones(knop_test .n_classifiers)
    knop_test.neighbors = neighbors_ex1[index, :]
    knop_test.distances = distances_ex1[index, :]
    prediction = knop_test.classify_instance(query)

    assert prediction == expected


def test_weights_zero():
    query = np.atleast_2d([1, 1])

    knop_test = KNOP(create_pool_classifiers())
    knop_test.fit(X_dsel_ex1, y_dsel_ex1)

    knop_test.estimate_competence = MagicMock(return_value=np.zeros(3))

    result = knop_test.select(query)
    assert np.array_equal(result, np.array([0, 1, 0]))


def test_fit():
    knop_test = KNOP(create_pool_classifiers())
    knop_test.fit(X_dsel_ex1, y_dsel_ex1)
    expected_scores = np.ones((15, 6)) * np.array([0.5, 0.5, 1.0, 0.0, 0.33, 0.67])
    assert np.array_equal(expected_scores, knop_test.dsel_scores)
    # Assert the roc_algorithm is fitted to the scores (decision space) rather than the features (feature space)
    assert np.array_equal(knop_test.roc_algorithm._fit_X, knop_test.dsel_scores)




