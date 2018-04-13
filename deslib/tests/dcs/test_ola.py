import pytest
from sklearn.linear_model import Perceptron

from deslib.dcs.ola import OLA
from deslib.tests.examples_test import *


@pytest.mark.parametrize('index, expected', [(0, [0.57142857,  0.71428571,  0.71428571]),
                                             (1, [0.71428571,  0.85714286,  0.71428571]),
                                             (2, [0.57142857,  0.71428571,  0.57142857])])
def test_estimate_competence(index, expected):
    ola_test = OLA(create_pool_classifiers())
    ola_test.processed_dsel = dsel_processed_ex1
    ola_test.neighbors = neighbors_ex1[index, :]
    ola_test.distances = distances_ex1[index, :]
    ola_test.DFP_mask = [1, 1, 1]
    query = np.array([1, 1])
    competences = ola_test.estimate_competence(query.reshape(1, -1))
    assert np.isclose(competences, expected).all()


def test_estimate_competence_batch():
    expected = np.array([[0.57142857,  0.71428571,  0.71428571],
                         [0.71428571,  0.85714286,  0.71428571],
                         [0.57142857, 0.71428571, 0.57142857]])

    ola_test = OLA(create_pool_classifiers())
    ola_test.processed_dsel = dsel_processed_ex1
    ola_test.neighbors = neighbors_ex1
    ola_test.distances = distances_ex1
    ola_test.DFP_mask = np.ones((3, 3))
    query = np.array([[1, 1], [1, 1], [1, 1]])
    competences = ola_test.estimate_competence(query)
    assert np.allclose(competences, expected)

# Test if the class is raising an error when the base classifiers do not implements the predict_proba method.
# In this case the test should not raise an error since this class does not require base classifiers that
# can estimate probabilities
def test_predict_proba():
    X = X_dsel_ex1
    y = y_dsel_ex1
    clf1 = Perceptron()
    clf1.fit(X, y)
    OLA([clf1, clf1])
