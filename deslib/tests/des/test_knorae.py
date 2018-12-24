import numpy as np
import pytest
from sklearn.linear_model import Perceptron
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.testing import ignore_warnings

from deslib.des.knora_e import KNORAE


def test_check_estimator():
    with ignore_warnings(category=RuntimeWarning):
        check_estimator(KNORAE)


def test_estimate_competence_batch(example_estimate_competence,
                                   create_pool_classifiers):
    X, y, neighbors, distances, _, _ = example_estimate_competence

    query = np.ones((3, 2))
    expected = np.array([[1.0, 0.0, 1.0],
                         [2.0, 0.0, 2.0],
                         [0.0, 3.0, 0.0]])

    knora_e_test = KNORAE(create_pool_classifiers)
    knora_e_test.fit(X, y)

    competences = knora_e_test.estimate_competence(query, neighbors,
                                                   distances=distances)
    assert np.allclose(competences, expected)


@pytest.mark.parametrize('index, expected', [(0, [[True, False, True]]),
                                             (1, [[True, False, True]]),
                                             (2, [[False, True, False]])])
def test_select(index, expected, create_pool_classifiers,
                example_estimate_competence):
    X, y, neighbors, distances, _, _ = example_estimate_competence

    query = np.atleast_2d([1, 1])

    knora_e_test = KNORAE(create_pool_classifiers)
    knora_e_test.fit(X, y)
    neighbors = neighbors[index, :].reshape(1, -1)
    distances = distances[index, :].reshape(1, -1)
    competences = knora_e_test.estimate_competence(query,
                                                   neighbors,
                                                   distances=distances)
    selected = knora_e_test.select(competences)

    assert np.array_equal(selected, expected)


# No classifier here is selected, since the always predict class 2 where there
# are only samples labeled as class 0 and 1
# in the region of competence
def test_select_none_competent():
    knora_e_test = KNORAE()
    competences = np.zeros(100)
    selected = knora_e_test.select(competences)
    expected = np.atleast_2d([True] * 100)

    assert np.array_equal(expected, selected)


# Test if the class is raising an error when the base classifiers do not
# implements the predict_proba method. In this case the test should not raise
# an error since this class does not require base classifiers that
# can estimate probabilities
def test_predict_proba(create_X_y):
    X, y = create_X_y

    clf1 = Perceptron()
    clf1.fit(X, y)
    KNORAE([clf1, clf1]).fit(X, y)
