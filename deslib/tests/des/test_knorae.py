import pytest

from deslib.des.knora_e import KNORAE
from deslib.tests.examples_test import *
from sklearn.linear_model import Perceptron


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
    assert np.isclose(competences, expected, atol=0.01).all()


@pytest.mark.parametrize('index, expected', [(0, [0, 2]),
                                             (1, [0, 2]),
                                             (2, [1])])
def test_select(index, expected):
    query = np.atleast_2d([1, 1])

    knora_e_test = KNORAE(create_pool_classifiers())
    knora_e_test.fit(X_dsel_ex1, y_dsel_ex1)
    knora_e_test.DFP_mask = np.ones(knora_e_test .n_classifiers)
    knora_e_test.neighbors = neighbors_ex1[index, :]
    knora_e_test.distances = distances_ex1[index, :]
    competences = knora_e_test.estimate_competence(query)
    selected = knora_e_test.select(competences)

    assert selected == expected


# No classifier here is selected, since the always predict class 2 where there are only samples labeled as class 0 and 1
# in the region of competence
def test_select_none_competent():
    query = np.atleast_2d([1, 1])

    y_dsel = np.array([2, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    neighbors = [8, 11, 4, 7, 13, 10, 1]

    knora_e_test = KNORAE(create_pool_all_agree(2, 100))
    knora_e_test.fit(X_dsel_ex1, y_dsel)

    knora_e_test.neighbors = neighbors
    knora_e_test.distances = distances_ex1[0, :]
    knora_e_test.DFP_mask = np.ones(knora_e_test.n_classifiers)

    competences = knora_e_test.estimate_competence(query)
    indices = knora_e_test.select(competences)

    assert indices == list(range(knora_e_test.n_classifiers))

# Test if the class is raising an error when the base classifiers do not implements the predict_proba method.
# In this case the test should not raise an error since this class does not require base classifiers that
# can estimate probabilities
def test_predict_proba():
    X = X_dsel_ex1
    y = y_dsel_ex1
    clf1 = Perceptron()
    clf1.fit(X, y)
    KNORAE([clf1, clf1])



