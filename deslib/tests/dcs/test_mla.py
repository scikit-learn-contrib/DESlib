import pytest
from sklearn.linear_model import Perceptron

from deslib.dcs.mla import MLA
from deslib.tests.examples_test import *
from sklearn.utils.estimator_checks import check_estimator


def test_check_estimator():
    check_estimator(MLA)


# Should always be 1.0 since the supports for the correct class is always 1.
@pytest.mark.parametrize('index', [0, 1, 2])
def test_estimate_competence_all_ones(index):
    query = np.atleast_2d([1, 1])

    mla_test = MLA(create_pool_classifiers())
    mla_test.n_classifiers_ = 3

    mla_test.DSEL_processed_ = dsel_processed_all_ones
    mla_test.DSEL_scores = dsel_scores_all_ones
    mla_test.DSEL_target_ = y_dsel_ex1
    mla_test.n_classes_ = 2

    neighbors = neighbors_ex1[index, :].reshape(1, -1)
    distances = distances_all_ones[index, :].reshape(1, -1)

    expected = [1.0, 1.0, 1.0]

    predictions = []
    for clf in mla_test.pool_classifiers:
        predictions.append(clf.predict(query))
    competences = mla_test.estimate_competence(query,
                                               neighbors,
                                               distances=distances,
                                               predictions=np.array(predictions))

    assert np.isclose(competences, expected).all()


def test_estimate_competence_batch():
    query = np.array([[1, 1], [1, 1], [1, 1]])
    expected = np.array([[0.750,  0.666,  0.750],
                         [0.800,  1.000,  0.800],
                         [1.000,  0.600,  0.500]])

    mla_test = MLA(create_pool_classifiers())
    mla_test.n_classifiers_ = 3
    mla_test.DSEL_processed_ = dsel_processed_ex1
    mla_test.DSEL_scores = dsel_scores_all_ones
    mla_test.DSEL_target_ = y_dsel_ex1
    mla_test.n_classes_ = 2

    neighbors = neighbors_ex1
    distances = distances_all_ones

    predictions = []
    for clf in mla_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])
    competences = mla_test.estimate_competence(query,
                                               neighbors,
                                               distances=distances,
                                               predictions=np.array(predictions))

    assert np.allclose(competences, expected, atol=0.01)


# in this test case, the target of the neighbors is always different than the predicted. So
# the estimation of competence should always be zero
@pytest.mark.parametrize('index', [0, 1, 2])
def test_estimate_competence_diff_target(index):
    query = np.atleast_2d([1, 1])

    mla_test = MLA(create_pool_classifiers())
    mla_test.n_classifiers_ = 3

    mla_test.DSEL_processed_ = dsel_processed_ex1
    mla_test.DSEL_target_ = np.ones(15, dtype=int) * 3

    neighbors = neighbors_ex1[index, :].reshape(1, -1)
    distances = distances_ex1[index, :].reshape(1, -1)

    expected = [0.0, 0.0, 0.0]

    predictions = []
    for clf in mla_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])
    competences = mla_test.estimate_competence(query,
                                               neighbors,
                                               distances=distances,
                                               predictions=np.array(predictions))

    assert np.isclose(competences, expected).all()


# Testing example from kuncheva's book (combining pattern classifiers)
def test_estimate_competence_kuncheva_ex():
    query = np.atleast_2d([1, 1])

    mla_test = MLA([create_base_classifier(return_value=1)]*2, k=k_ex_kuncheva)
    mla_test.n_classifiers_ = 2

    mla_test.DSEL_processed_ = np.repeat(dsel_processed_kuncheva, 2, axis=1)
    mla_test.DSEL_scores = dsel_scores_ex_kuncheva
    mla_test.DSEL_target_ = y_dsel_ex_kuncheva_dependent
    mla_test.n_classes_ = n_classes_ex_kuncheva

    neighbors = neighbors_ex_kuncheva.reshape(1, -1)
    distances = distances_ex_kuncheva.reshape(1, -1)

    predictions = []
    for clf in mla_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])
    competences = mla_test.estimate_competence(query,
                                               neighbors,
                                               distances=distances,
                                               predictions=np.array(predictions))

    assert np.allclose(competences, [0.95, 0.95], atol=0.01)


# Test if the class is raising an error when the base classifiers do not implements the predict_proba method.
# In this case the test should not raise an error since this class does not require base classifiers that
# can estimate probabilities
def test_predict_proba():
    X = X_dsel_ex1
    y = y_dsel_ex1
    clf1 = Perceptron()
    clf1.fit(X, y)
    MLA([clf1, clf1]).fit(X, y)
