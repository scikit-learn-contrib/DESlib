from sklearn.linear_model import Perceptron

from deslib.des.knora_u import KNORAU
from deslib.tests.examples_test import *
from sklearn.utils.estimator_checks import check_estimator


def test_check_estimator():
    check_estimator(KNORAU)


# Test the estimate competence method receiving n samples as input
def test_estimate_competence_batch():
    query = np.ones((3, 2))
    expected = np.array([[4.0, 3.0, 4.0],
                         [5.0, 2.0, 5.0],
                         [2.0, 5.0, 2.0]])
    knora_u_test = KNORAU(create_pool_classifiers())
    knora_u_test.fit(X_dsel_ex1, y_dsel_ex1)
    neighbors = neighbors_ex1

    competences = knora_u_test.estimate_competence(query, neighbors)
    assert np.allclose(competences, expected, atol=0.01)


def test_weights_zero():

    knorau_test = KNORAU(create_pool_classifiers())
    knorau_test.fit(X_dsel_ex1, y_dsel_ex1)
    competences = np.zeros((1, 3))
    result = knorau_test.select(competences)

    assert np.all(result)


# Test if the class is raising an error when the base classifiers do not implements the predict_proba method.
# In this case the test should not raise an error since this class does not require base classifiers that
# can estimate probabilities
def test_predict_proba():
    X = X_dsel_ex1
    y = y_dsel_ex1
    clf1 = Perceptron()
    clf1.fit(X, y)
    KNORAU([clf1, clf1]).fit(X, y)


def test_select():
    knorau_test = KNORAU(create_pool_classifiers())
    competences = np.ones(3)
    competences[0] = 0
    expected = np.atleast_2d([False, True, True])
    selected = knorau_test.select(competences)
    assert np.array_equal(expected, selected)
