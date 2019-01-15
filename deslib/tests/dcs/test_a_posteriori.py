import numpy as np
import pytest
from sklearn.linear_model import Perceptron
from sklearn.utils.estimator_checks import check_estimator

from deslib.dcs.a_posteriori import APosteriori


def test_check_estimator():
    check_estimator(APosteriori(selection_method='best'))


# Should always be 1.0 since the supports for the correct class is always 1.
@pytest.mark.parametrize('index', [0, 1, 2])
def test_estimate_competence_all_ones(index, example_all_ones):
    _, y, neighbors, distances, dsel_processed, dsel_scores = example_all_ones

    query = np.atleast_2d([1, 1])

    a_posteriori_test = APosteriori()
    a_posteriori_test.n_classifiers_ = 3
    a_posteriori_test.DSEL_processed_ = dsel_processed
    a_posteriori_test.dsel_scores_ = dsel_scores
    a_posteriori_test.DSEL_target_ = y

    neighbors = neighbors[index, :].reshape(1, -1)
    distances = distances[index, :].reshape(1, -1)

    expected = [1.0, 1.0, 1.0]
    predictions = np.array([0, 1, 0])

    competences = a_posteriori_test.estimate_competence(query, neighbors,
                                                        distances,
                                                        predictions=np.array(
                                                            predictions))
    assert np.isclose(competences, expected).all()


# Testing example from kuncheva's book (combining pattern classifiers)
def test_estimate_competence_kuncheva_ex(example_kuncheva):
    query = np.atleast_2d([1, 1])

    a_posteriori_test = APosteriori(k=example_kuncheva['k'])
    a_posteriori_test.n_classifiers_ = 1

    a_posteriori_test.DSEL_processed_ = example_kuncheva['dsel_processed']
    a_posteriori_test.dsel_scores_ = example_kuncheva['dsel_scores']
    a_posteriori_test.DSEL_target_ = example_kuncheva['y_dependent']
    a_posteriori_test.n_classes_ = example_kuncheva['n_classes']

    neighbors = example_kuncheva['neighbors'].reshape(1, -1)
    distances = example_kuncheva['distances'].reshape(1, -1)

    predictions = np.array([[1]])

    competences = a_posteriori_test.estimate_competence(query, neighbors,
                                                        distances,
                                                        predictions=np.array(
                                                            predictions))
    assert np.isclose(competences, 0.95, atol=0.01)


# Testing example from kuncheva's book (combining pattern classifiers)
def test_estimate_competence_kuncheva_ex_batch(example_kuncheva):
    # considering a batch composed of 10 samples
    query = np.ones((10, 2))

    a_posteriori_test = APosteriori(k=example_kuncheva['k'])
    a_posteriori_test.fit(example_kuncheva['dsel_processed'],
                          example_kuncheva['y_dependent'])
    a_posteriori_test.DSEL_processed_ = example_kuncheva['dsel_processed']
    a_posteriori_test.dsel_scores_ = example_kuncheva['dsel_scores']

    a_posteriori_test.n_classes_ = example_kuncheva['n_classes']

    # repeating the same matrix in a new axis to simulate a batch input.
    neighbors = np.tile(example_kuncheva['neighbors'], (10, 1))
    distances = np.tile(example_kuncheva['distances'], (10, 1))

    predictions = np.ones((1, 10))
    competences = a_posteriori_test.estimate_competence(query, neighbors,
                                                        distances,
                                                        predictions=np.array(
                                                            predictions))
    assert np.allclose(competences, 0.95, atol=0.01)


# in this test case, the target of the neighbors is always different
# than the predicted. So
# the estimation of competence should always be zero
@pytest.mark.parametrize('index', [0, 1, 2])
def test_estimate_competence_diff_target(index, example_all_ones):
    _, _, neighbors, distances, dsel_processed, _ = example_all_ones

    query = np.atleast_2d([1, 1])
    a_posteriori_test = APosteriori()
    a_posteriori_test.n_classifiers_ = 3
    a_posteriori_test.DSEL_processed_ = dsel_processed
    a_posteriori_test.dsel_scores_ = np.ones((15, 3, 3))
    a_posteriori_test.DSEL_target_ = np.ones(15, dtype=int) * 2
    a_posteriori_test.n_classes_ = 2

    neighbors = neighbors[index, :].reshape(1, -1)
    distances = distances[index, :].reshape(1, -1)

    expected = [0.0, 0.0, 0.0]

    predictions = np.array([0, 1, 0])
    competences = a_posteriori_test.estimate_competence(query, neighbors,
                                                        distances,
                                                        predictions=np.array(
                                                            predictions))
    assert np.isclose(competences, expected).all()


# Check if the fit method is pre-calculating the classifier scores correctly
def test_fit(create_X_y, create_pool_classifiers):
    X, y = create_X_y
    a_posteriori_test = APosteriori(create_pool_classifiers)
    a_posteriori_test.fit(X, y)
    expected = np.array([[0.5, 0.5], [1.0, 0.0], [0.33, 0.67]])
    expected = np.tile(expected, (15, 1, 1))
    assert np.array_equal(a_posteriori_test.dsel_scores_, expected)


# Test if the class is raising an error when the base classifiers do not
# implements the predict_proba method. Should raise an exception when the
# base classifier cannot estimate posterior probabilities (predict_proba)
# Using Perceptron classifier as it does not implements predict_proba.
def test_not_predict_proba(create_X_y):
    X, y = create_X_y
    clf1 = Perceptron()
    clf1.fit(X, y)
    with pytest.raises(ValueError):
        APosteriori([clf1, clf1]).fit(X, y)
