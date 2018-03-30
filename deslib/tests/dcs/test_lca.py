import pytest
from sklearn.linear_model import Perceptron
from deslib.dcs.lca import LCA
from deslib.tests.examples_test import *


@pytest.mark.parametrize('index, expected', [(0, [0.75000000,  0.66666667,  0.75000000]),
                                             (1, [0.80000000,  1.00000000,  0.80000000]),
                                             (2, [1.00000000,  0.60000000,  0.50000000])])
def test_estimate_competence_woods(index, expected):
    lca_test = LCA(create_pool_classifiers())
    lca_test.processed_dsel = dsel_processed_ex1
    lca_test.neighbors = neighbors_ex1[index, :]
    lca_test.distances = distances_ex1[index, :]
    lca_test.DFP_mask = [1, 1, 1]
    lca_test.DSEL_target = y_dsel_ex1

    query = np.atleast_2d([1, 1])

    predictions = []
    for clf in lca_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])
    competences = lca_test.estimate_competence(query, predictions=np.array(predictions))

    assert np.allclose(competences, expected)


def test_estimate_competence_batch():
    expected = np.array([[0.75000000,  0.66666667,  0.75000000],
                         [0.80000000, 1.00000000, 0.80000000],
                         [1.00000000, 0.60000000, 0.50000000]])
    lca_test = LCA(create_pool_classifiers())
    lca_test.processed_dsel = dsel_processed_ex1
    lca_test.neighbors = neighbors_ex1
    lca_test.distances = distances_ex1
    lca_test.DFP_mask = np.ones((3, 3))
    lca_test.DSEL_target = y_dsel_ex1

    query = np.ones((3, 2))

    predictions = []
    for clf in lca_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])
    competences = lca_test.estimate_competence(query, predictions=np.array(predictions))

    assert np.isclose(competences, expected).all()


# in this test case, the target of the neighbors is always different than the predicted class. So
# the estimation of competence should always be zero
@pytest.mark.parametrize('index', [0, 1, 2])
def test_estimate_competence_diff_target(index):
    lca_test = LCA(create_pool_classifiers())
    lca_test.processed_dsel = dsel_processed_ex1
    lca_test.DSEL_target = np.ones(15, dtype=int) * 3
    lca_test.neighbors = neighbors_ex1[index, :]
    lca_test.distances = distances_ex1[index, :]
    lca_test.DFP_mask = [1, 1, 1]

    query = np.atleast_2d([1, 1])
    expected = [0.0, 0.0, 0.0]

    predictions = []
    for clf in lca_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])
    competences = lca_test.estimate_competence(query, predictions=np.array(predictions))

    assert np.isclose(competences, expected).all()


# Test if the class is raising an error when the base classifiers do not implements the predict_proba method.
# In this case the test should not raise an error since this class does not require base classifiers that
# can estimate probabilities
def test_predict_proba():
    X = X_dsel_ex1
    y = y_dsel_ex1
    clf1 = Perceptron()
    clf1.fit(X, y)
    LCA([clf1, clf1])
