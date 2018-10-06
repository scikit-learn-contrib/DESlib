import pytest
import numpy as np
from sklearn.linear_model import Perceptron

from deslib.dcs.a_posteriori import APosteriori
from deslib.tests.examples_test import (setup_example1,
                                        setup_example_kuncheva,
                                        create_pool_classifiers,
                                        setup_example_all_ones,
                                        create_base_classifier)

from sklearn.utils.estimator_checks import check_estimator


def test_check_estimator():
    check_estimator(APosteriori)


# Should always be 1.0 since the supports for the correct class is always 1.
@pytest.mark.parametrize('index', [0, 1, 2])
def test_estimate_competence_all_ones(index):
    X, y, neighbors, distances, dsel_processed, dsel_scores = setup_example_all_ones()

    query = np.atleast_2d([1, 1])

    a_posteriori_test = APosteriori(create_pool_classifiers())
    a_posteriori_test.fit(X, y)
    a_posteriori_test.DSEL_processed_ = dsel_processed
    a_posteriori_test.dsel_scores_ = dsel_scores

    neighbors = neighbors[index, :].reshape(1, -1)
    distances = distances[index, :].reshape(1, -1)

    expected = [1.0, 1.0, 1.0]

    predictions = []
    for clf in a_posteriori_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])

    competences = a_posteriori_test.estimate_competence(query, neighbors, distances, predictions=np.array(predictions))
    assert np.isclose(competences, expected).all()


# Testing example from kuncheva's book (combining pattern classifiers)
def test_estimate_competence_kuncheva_ex():
    example_kuncheva = setup_example_kuncheva()
    query = np.atleast_2d([1, 1])

    a_posteriori_test = APosteriori([create_base_classifier(return_value=1)], k=example_kuncheva['k'])
    a_posteriori_test.n_classifiers_ = 1

    a_posteriori_test.DSEL_processed_ = example_kuncheva['dsel_processed']
    a_posteriori_test.dsel_scores_ = example_kuncheva['dsel_scores']
    a_posteriori_test.DSEL_target_ = example_kuncheva['y_dependent']
    a_posteriori_test.n_classes_ = example_kuncheva['n_classes']

    neighbors = example_kuncheva['neighbors'].reshape(1, -1)
    distances = example_kuncheva['distances'].reshape(1, -1)

    predictions = []
    for clf in a_posteriori_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])
    competences = a_posteriori_test.estimate_competence(query, neighbors, distances, predictions=np.array(predictions))
    assert np.isclose(competences, 0.95, atol=0.01)


# Testing example from kuncheva's book (combining pattern classifiers)
def test_estimate_competence_kuncheva_ex_batch():
    # considering a batch composed of 10 samples
    query = np.ones((10, 2))
    example_kuncheva = setup_example_kuncheva()

    a_posteriori_test = APosteriori([create_base_classifier(return_value=1)], k=example_kuncheva['k'])
    a_posteriori_test.fit(example_kuncheva['dsel_processed'], example_kuncheva['y_dependent'])
    a_posteriori_test.DSEL_processed_ = example_kuncheva['dsel_processed']
    a_posteriori_test.dsel_scores_ = example_kuncheva['dsel_scores']

    a_posteriori_test.n_classes_ = example_kuncheva['n_classes']

    # repeating the same matrix in a new axis to simulate a batch input.
    neighbors = np.tile(example_kuncheva['neighbors'], (10, 1))
    distances = np.tile(example_kuncheva['distances'], (10, 1))

    predictions = []
    for clf in a_posteriori_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])
    competences = a_posteriori_test.estimate_competence(query, neighbors, distances, predictions=np.array(predictions))
    assert np.allclose(competences, 0.95, atol=0.01)


# in this test case, the target of the neighbors is always different than the predicted. So
# the estimation of competence should always be zero
@pytest.mark.parametrize('index', [0, 1, 2])
def test_estimate_competence_diff_target(index):
    _, _, neighbors, distances, dsel_processed, _ = setup_example_all_ones()

    query = np.atleast_2d([1, 1])
    pool_classifiers = create_pool_classifiers()
    a_posteriori_test = APosteriori(pool_classifiers=pool_classifiers)
    a_posteriori_test.n_classifiers_ = len(pool_classifiers)
    a_posteriori_test.DSEL_processed_ = dsel_processed
    a_posteriori_test.dsel_scores_ = np.ones((15, 3, 3))
    a_posteriori_test.DSEL_target_ = np.ones(15, dtype=int) * 2
    a_posteriori_test.n_classes_ = 2

    neighbors = neighbors[index, :].reshape(1, -1)
    distances = distances[index, :].reshape(1, -1)

    expected = [0.0, 0.0, 0.0]

    predictions = []
    for clf in a_posteriori_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])
    competences = a_posteriori_test.estimate_competence(query, neighbors, distances, predictions=np.array(predictions))
    assert np.isclose(competences, expected).all()


# Check if the fit method is pre-calculating the classifier scores correctly
def test_fit():
    X, y = setup_example1()[0:2]
    a_posteriori_test = APosteriori(create_pool_classifiers())
    a_posteriori_test.fit(X, y)
    expected = np.array([[0.5, 0.5], [1.0, 0.0], [0.33, 0.67]])
    expected = np.tile(expected, (15, 1, 1))
    assert np.array_equal(a_posteriori_test.dsel_scores_, expected)


# Test if the class is raising an error when the base classifiers do not implements the predict_proba method.
# Should raise an exception when the base classifier cannot estimate posterior probabilities (predict_proba)
# Using Perceptron classifier as it does not implements the predict_proba method.
def test_not_predict_proba():
    X, y = setup_example1()[0:2]
    clf1 = Perceptron()
    clf1.fit(X, y)
    with pytest.raises(ValueError):
        APosteriori([clf1, clf1]).fit(X, y)
