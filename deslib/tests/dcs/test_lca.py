import numpy as np
import pytest
from sklearn.linear_model import Perceptron

from deslib.dcs.lca import LCA
from sklearn.utils.estimator_checks import check_estimator


def test_check_estimator():
    check_estimator(LCA)


def test_estimate_competence_batch(example_estimate_competence):
    _, y, neighbors, distances, dsel_processed, _ = example_estimate_competence

    expected = np.array([[0.75000000,  0.66666667,  0.75000000],
                         [0.80000000, 1.00000000, 0.80000000],
                         [1.00000000, 0.60000000, 0.50000000]])
    lca_test = LCA()
    lca_test.DSEL_processed_ = dsel_processed
    lca_test.DSEL_target_ = y

    query = np.ones((3, 2))

    predictions = np.array([[0, 1, 0]])
    competences = lca_test.estimate_competence(query, neighbors,
                                               distances=distances,
                                               predictions=np.array(
                                                   predictions))

    assert np.isclose(competences, expected).all()


# in this test case, the target of the neighbors is always different than
# the predicted class. So the estimation of competence should always be zero
@pytest.mark.parametrize('index', [0, 1, 2])
def test_estimate_competence_diff_target(index,
                                         example_estimate_competence,
                                         create_pool_classifiers):
    _, y, neighbors, distances, dsel_processed, _ = example_estimate_competence

    lca_test = LCA(create_pool_classifiers)
    lca_test.DSEL_processed_ = dsel_processed
    lca_test.DSEL_target_ = np.ones(15, dtype=int) * 3

    neighbors = neighbors[index, :].reshape(1, -1)
    distances = distances[index, :].reshape(1, -1)

    query = np.atleast_2d([1, 1])
    expected = [0.0, 0.0, 0.0]

    predictions = np.array([[0, 1, 0]])
    competences = lca_test.estimate_competence(query,
                                               neighbors,
                                               distances=distances,
                                               predictions=np.array(
                                                   predictions))

    assert np.isclose(competences, expected).all()


# Test if the class is raising an error when the base classifiers do not
# implements the predict_proba method. In this case the test should not raise
# an error since this class does not require base classifiers that
# can estimate probabilities
def test_predict_proba(create_X_y):
    X, y = create_X_y

    clf1 = Perceptron()
    clf1.fit(X, y)
    LCA([clf1, clf1]).fit(X, y)
