import pytest
from sklearn.linear_model import Perceptron

from deslib.des.knora_e import KNORAE
from deslib.tests.examples_test import *


@pytest.mark.parametrize('index, expected', [(0, [1.0, 0.0, 1.0]),
                                             (1, [2.0, 0.0, 2.0]),
                                             (2, [0.0, 3.0, 0.0])])
def test_estimate_competence(index, expected):
    query = np.atleast_2d([1, 1])

    knora_e_test = KNORAE(create_pool_classifiers())
    knora_e_test.fit(X_dsel_ex1, y_dsel_ex1)
    knora_e_test.DFP_mask = np.ones(knora_e_test .n_classifiers)
    knora_e_test.neighbors = neighbors_ex1[index, :]
    knora_e_test.distances = distances_ex1[index, :]
    competences = knora_e_test.estimate_competence(query)
    assert np.isclose(competences, expected).all()


def test_estimate_competence_batch():
    query = np.ones((3, 2))
    expected = np.array([[1.0, 0.0, 1.0],
                         [2.0, 0.0, 2.0],
                         [0.0, 3.0, 0.0]])

    knora_e_test = KNORAE(create_pool_classifiers())
    knora_e_test.fit(X_dsel_ex1, y_dsel_ex1)
    knora_e_test.DFP_mask = np.ones(knora_e_test .n_classifiers)
    knora_e_test.neighbors = neighbors_ex1
    knora_e_test.distances = distances_ex1
    competences = knora_e_test.estimate_competence(query)
    assert np.allclose(competences, expected)


@pytest.mark.parametrize('index, expected', [(0, [[True, False, True]]),
                                             (1, [[True, False, True]]),
                                             (2, [[False, True, False]])])
def test_select(index, expected):
    query = np.atleast_2d([1, 1])

    knora_e_test = KNORAE(create_pool_classifiers())
    knora_e_test.fit(X_dsel_ex1, y_dsel_ex1)
    knora_e_test.DFP_mask = np.ones(knora_e_test .n_classifiers)
    knora_e_test.neighbors = neighbors_ex1[index, :]
    knora_e_test.distances = distances_ex1[index, :]
    competences = knora_e_test.estimate_competence(query)
    selected = knora_e_test.select(competences)

    assert np.array_equal(selected, expected)


# No classifier here is selected, since the always predict class 2 where there are only samples labeled as class 0 and 1
# in the region of competence
def test_select_none_competent():

    knora_e_test = KNORAE(create_pool_all_agree(1, 100))
    competences = np.zeros(100)
    selected = knora_e_test.select(competences)
    expected = np.atleast_2d([True] * 100)

    assert np.array_equal(expected, selected)


# Test if the class is raising an error when the base classifiers do not implements the predict_proba method.
# In this case the test should not raise an error since this class does not require base classifiers that
# can estimate probabilities
def test_predict_proba():
    X = X_dsel_ex1
    y = y_dsel_ex1
    clf1 = Perceptron()
    clf1.fit(X, y)
    KNORAE([clf1, clf1])



