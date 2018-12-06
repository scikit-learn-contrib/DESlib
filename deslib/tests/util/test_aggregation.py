import pytest

from deslib.util.aggregation import *
from ..conftest import create_base_classifier


def test_majority_voting():
    query = np.array([[1, -1], [0, 0], [3, -1]])

    ensemble_classifiers = [create_base_classifier(return_value=0)] * 10 + [
        create_base_classifier(return_value=1)] * 9

    predicted = majority_voting(ensemble_classifiers, query)
    assert predicted.all() == 0 and predicted.size == 3


def test_majority_voting_multi_class():
    query = np.array([1, -1])
    ensemble_classifiers = ([create_base_classifier(return_value=0)] * 10) + \
        [create_base_classifier(return_value=2)] * 9 + \
        [create_base_classifier(return_value=1)] * 20

    predicted = majority_voting(ensemble_classifiers, query)
    assert predicted.all() == 1 and predicted.size == 1


def test_weighted_majority_voting():
    query = np.array([[1, -1], [0, 0], [3, -1]])
    ensemble_classifiers = ([create_base_classifier(return_value=0)] * 10) + \
        [create_base_classifier(return_value=2)] * 9
    weights = np.array([([0.5] * 10) + ([0.8] * 9), ([0.5] * 10) + ([0.8] * 9),
                        ([0.5] * 10) + ([0.8] * 9)])
    predicted = weighted_majority_voting(ensemble_classifiers, weights, query)
    assert predicted.all() == 1 and predicted.size == 3


def test_weighted_majority_voting_single_sample():
    query = np.array([1, -1])
    clf_1 = create_base_classifier(return_value=1)
    clf_2 = create_base_classifier(return_value=1)
    clf_3 = create_base_classifier(return_value=2)
    ensemble_classifiers = [clf_2, clf_1, clf_3]
    weights = np.atleast_2d([0.2, 0.5, 1.0])
    predicted = weighted_majority_voting(ensemble_classifiers, weights, query)
    assert predicted == 2 and predicted.size == 1


def test_predict_proba(create_pool_classifiers):
    query = np.array([[1, -1]])
    ensemble_classifiers = create_pool_classifiers
    predicted_proba = predict_proba_ensemble(ensemble_classifiers, query)
    assert np.isclose(predicted_proba, [0.61, 0.39]).all()


# This experiment should raise an error since we have 3 base classifiers
# and 4 weights.
def test_wrong_weights_votes(create_pool_classifiers):
    query = np.array([[1, -1]])
    ensemble_classifiers = create_pool_classifiers
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        weighted_majority_voting(ensemble_classifiers, weights, query)


# This experiment should raise an error since we have 3 base classifiers
# and 4 weights.
def test_wrong_weights_proba(create_pool_classifiers):
    query = np.array([[1, -1]])
    ensemble_classifiers = create_pool_classifiers
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        predict_proba_ensemble_weighted(ensemble_classifiers, weights, query)


# -------Test routines for the ensemble combination methods-------
# These routines calculates the matrix with the supports given for
# each class for each base classifier and them Aggregates the supports

def test_product_combiner(create_pool_classifiers):
    query = np.array([[1, -1]])
    ensemble_classifiers = create_pool_classifiers
    expected = 0
    result = product_combiner(ensemble_classifiers, query)
    assert np.allclose(expected, result)


def test_average_combiner(create_pool_classifiers):
    query = np.array([[1, -1]])
    ensemble_classifiers = create_pool_classifiers
    expected = 0
    result = average_combiner(ensemble_classifiers, query)
    assert result == expected


def test_minimum_combiner(create_pool_classifiers):
    query = np.array([[1, -1]])
    ensemble_classifiers = create_pool_classifiers
    expected = 0
    result = minimum_combiner(ensemble_classifiers, query)
    assert np.allclose(expected, result)


def test_maximum_combiner(create_pool_classifiers):
    query = np.array([[1, -1]])
    ensemble_classifiers = create_pool_classifiers
    expected = 0
    result = maximum_combiner(ensemble_classifiers, query)
    assert np.allclose(expected, result)


def test_median_combiner(create_pool_classifiers):
    query = np.array([[1, -1]])
    ensemble_classifiers = create_pool_classifiers
    expected = 0
    result = median_combiner(ensemble_classifiers, query)
    assert np.allclose(expected, result)


def test_check_predictions():
    predictions = example_kuncheva
    with pytest.raises(ValueError):
        average_rule(predictions)


# -------Test routines for the fusion rules receiving prediction directly------
# These receives the matrix with the supports given for each class and
# returns the class labels (max score)

# Test example taken from Kuncheva's book: Combining pattern classifiers


example_kuncheva = np.array(
    [[0.1, 0.5, 0.4], [0.0, 0.0, 1.0], [0.4, 0.3, 0.4], [0.2, 0.7, 0.1],
     [0.1, 0.8, 0.2]])
example_kuncheva_batch = np.expand_dims(example_kuncheva, axis=0)
example_kuncheva_batch = np.repeat(example_kuncheva_batch, 10, axis=0)


def test_product_rule():
    expected = 2
    result = product_rule(example_kuncheva_batch)
    assert np.allclose(expected, result)


def test_average_rule():
    expected = 1
    result = average_rule(example_kuncheva_batch)
    assert np.allclose(expected, result)


def test_minimum_rule():
    expected = 2
    result = minimum_rule(example_kuncheva_batch)
    assert np.allclose(expected, result)


def test_maximum_rule():
    expected = 2
    result = maximum_rule(example_kuncheva_batch)
    assert np.allclose(expected, result)


def test_median_rule():
    expected = 1
    result = median_rule(example_kuncheva_batch)
    assert np.allclose(expected, result)
