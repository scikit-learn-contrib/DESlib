import numpy as np
import pytest

from deslib.tests.examples_test import create_pool_all_agree, create_base_classifier, create_pool_classifiers
from deslib.util.aggregation import majority_voting, weighted_majority_voting, predict_proba_ensemble, \
    predict_proba_ensemble_weighted, average_rule


def test_majority_voting():
    query = np.array([[1, -1], [0, 0], [3, -1]])
    ensemble_classifiers = create_pool_all_agree(return_value=0, size=10)
    ensemble_classifiers += create_pool_all_agree(return_value=2, size=9)
    predicted = majority_voting(ensemble_classifiers, query)
    assert predicted.all() == 0 and predicted.size == 3


def test_majority_voting_single_sample():
    query = np.array([1, -1])
    ensemble_classifiers = create_pool_all_agree(return_value=0, size=10)
    ensemble_classifiers += create_pool_all_agree(return_value=2, size=9)
    ensemble_classifiers += create_pool_all_agree(return_value=1, size=20)
    predicted = majority_voting(ensemble_classifiers, query)
    assert predicted.all() == 1 and predicted.size == 1


def test_weighted_majority_voting():
    query = np.array([[1, -1], [0, 0], [3, -1]])
    ensemble_classifiers = create_pool_all_agree(return_value=0, size=10)
    ensemble_classifiers += create_pool_all_agree(return_value=2, size=9)
    weights = np.array([([0.5] * 10) + ([0.8] * 9), ([0.5] * 10) + ([0.8] * 9), ([0.5] * 10) + ([0.8] * 9)])
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


def test_predict_proba():
    query = np.array([[1, -1]])
    ensemble_classifiers = create_pool_classifiers()
    predicted_proba = predict_proba_ensemble(ensemble_classifiers, query)
    assert np.isclose(predicted_proba, [0.61, 0.39]).all()


def test_average():
    query = np.array([[1, -1]])
    ensemble_classifiers = create_pool_classifiers()
    predicted_label = average_rule(ensemble_classifiers, query)
    assert predicted_label == 0


# This experiment should raise an error since we have 3 base classifiers and 4 weights.
def test_wrong_weights_votes():
    query = np.array([[1, -1]])
    ensemble_classifiers = create_pool_classifiers()
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        weighted_majority_voting(ensemble_classifiers, weights, query)


# This experiment should raise an error since we have 3 base classifiers and 4 weights.
def test_wrong_weights_probas():
    query = np.array([[1, -1]])
    ensemble_classifiers = create_pool_classifiers()
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        predict_proba_ensemble_weighted(ensemble_classifiers, weights, query)
