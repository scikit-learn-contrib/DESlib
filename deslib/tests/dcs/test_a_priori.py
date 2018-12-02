import numpy as np
import pytest
from sklearn.linear_model import Perceptron
from sklearn.utils.estimator_checks import check_estimator

from deslib.dcs.a_priori import APriori


def test_check_estimator():
    check_estimator(APriori(selection_method='best'))


# Should always be 1.0 since the supports for the correct class is always 1.
@pytest.mark.parametrize('index, expected', [(0, [1.0, 1.0, 1.0]),
                                             (1, [1.0, 1.0, 1.0]),
                                             (2, [1.0, 1.0, 1.0])])
def test_estimate_competence_all_ones(index, expected, example_all_ones):
    X, y, neighbors, distances, dsel_processed, dsel_scores = example_all_ones
    query = np.array([1, 1])

    a_priori_test = APriori()

    a_priori_test.DSEL_processed_ = dsel_processed
    a_priori_test.dsel_scores_ = dsel_scores
    a_priori_test.DSEL_target_ = y
    a_priori_test.n_classes_ = 2

    neighbors = neighbors[index, :].reshape(1, -1)
    distances = distances[index, :].reshape(1, -1)

    competences = a_priori_test.estimate_competence(query.reshape(1, -1),
                                                    neighbors, distances)
    assert np.isclose(competences, expected).all()


# Testing example from kuncheva's book (combining pattern classifiers)
def test_estimate_competence_kuncheva_ex(example_kuncheva):
    query = np.array([1, 1])
    a_priori_test = APriori(k=example_kuncheva['k'])
    test_example = example_kuncheva
    a_priori_test.DSEL_processed_ = test_example['dsel_processed']
    a_priori_test.dsel_scores_ = test_example['dsel_scores']
    a_priori_test.DSEL_target_ = test_example['y_independent']
    a_priori_test.n_classes_ = test_example['n_classes']

    neighbors = test_example['neighbors'].reshape(1, -1)
    distances = test_example['distances'].reshape(1, -1)

    competences = a_priori_test.estimate_competence(query.reshape(1, -1),
                                                    neighbors, distances)
    assert np.isclose(competences, 0.70, atol=0.01)


# Test the estimate competence method receiving n samples as input
def test_estimate_competence_batch(example_estimate_competence):
    _, y, nn, _, dsel_processed, dsel_scores = example_estimate_competence

    query = np.ones((3, 2))
    expected = np.array([[0.333333, 0.50000, 0.40000],
                         [0.666666, 0.50000, 0.60000],
                         [0.000000, 0.50000, 0.20000]])

    # Using 3 neighbors to facilitate the calculations
    a_priori_test = APriori(k=3)

    a_priori_test.DSEL_processed_ = dsel_processed
    a_priori_test.dsel_scores_ = dsel_scores
    a_priori_test.DSEL_target_ = y
    a_priori_test.n_classes_ = 2

    nn = nn[:, 0:3]
    distances = np.ones((3, 3))

    competences = a_priori_test.estimate_competence(query, nn,
                                                    distances)
    assert np.allclose(competences, expected, atol=0.01)


def test_fit(create_pool_classifiers, create_X_y):
    X, y = create_X_y

    a_priori_test = APriori(create_pool_classifiers)
    a_priori_test.fit(X, y)
    expected = np.array([[0.5, 0.5], [1.0, 0.0], [0.33, 0.67]])
    expected = np.tile(expected, (15, 1, 1))
    assert np.array_equal(a_priori_test.dsel_scores_, expected)


# Test if the class is raising an error when the base classifiers do not
# implements the predict_proba method. Should raise an exception when the
# base classifier cannot estimate posterior probabilities (predict_proba)
# Using Perceptron classifier as it does not implements predict_proba.
def test_not_predict_proba(create_X_y):
    X, y = create_X_y

    clf1 = Perceptron()
    clf1.fit(X, y)
    with pytest.raises(ValueError):
        APriori([clf1, clf1]).fit(X, y)
