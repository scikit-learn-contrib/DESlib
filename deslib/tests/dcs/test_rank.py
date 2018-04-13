import pytest
from sklearn.linear_model import Perceptron

from deslib.dcs.rank import Rank
from deslib.tests.examples_test import *


@pytest.mark.parametrize('index, expected', [(0, [1, 5, 0]),
                                             (1, [1, 1, 2]),
                                             (2, [0, 0, 1])])
def test_estimate_competence(index, expected):
    rank_test = Rank(create_pool_classifiers())
    rank_test.processed_dsel = dsel_processed_ex1
    rank_test.neighbors = neighbors_ex1[index, :]
    rank_test.distances = distances_ex1[index, :]
    rank_test.DFP_mask = [1, 1, 1]
    query = np.array([1, 1])
    competences = rank_test.estimate_competence(query.reshape(1, -1))
    assert np.isclose(competences, expected).all()


def test_estimate_competence_batch():
    expected = np.array([[1, 5, 0],
                         [1, 1, 2],
                         [0, 0, 1]])
    rank_test = Rank(create_pool_classifiers())
    rank_test.processed_dsel = dsel_processed_ex1
    rank_test.neighbors = neighbors_ex1
    rank_test.distances = distances_ex1
    rank_test.DFP_mask = np.ones((3, 3))
    query = np.array([[1, 1], [1, 1], [1, 1]])
    competences = rank_test.estimate_competence(query)
    assert np.allclose(competences, expected)

# Test if the class is raising an error when the base classifiers do not implements the predict_proba method.
# In this case the test should not raise an error since this class does not require base classifiers that
# can estimate probabilities
def test_predict_proba():
    X = X_dsel_ex1
    y = y_dsel_ex1
    clf1 = Perceptron()
    clf1.fit(X, y)
    Rank([clf1, clf1])



