from sklearn.linear_model import Perceptron

from deslib.dcs.rank import Rank
from deslib.tests.examples_test import *
from sklearn.utils.estimator_checks import check_estimator


def test_check_estimator():
    check_estimator(Rank)


def test_estimate_competence_batch():
    expected = np.array([[1, 5, 0],
                         [1, 1, 2],
                         [0, 0, 1]])
    rank_test = Rank(create_pool_classifiers())
    rank_test.DSEL_processed_ = dsel_processed_ex1
    neighbors = neighbors_ex1
    distances = distances_ex1
    query = np.array([[1, 1], [1, 1], [1, 1]])
    competences = rank_test.estimate_competence(query, neighbors, distances=distances)
    assert np.allclose(competences, expected)


# Test if the class is raising an error when the base classifiers do not implements the predict_proba method.
# In this case the test should not raise an error since this class does not require base classifiers that
# can estimate probabilities
def test_predict_proba():
    X = X_dsel_ex1
    y = y_dsel_ex1
    clf1 = Perceptron()
    clf1.fit(X, y)
    Rank([clf1, clf1]).fit(X, y)



