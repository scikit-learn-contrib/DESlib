import numpy as np

from pythonds.tests.examples_test import create_pool_all_agree, create_base_classifier
from pythonds.util.aggregation import majority_voting, weighted_majority_voting


def tests_majority_voting():
    query = np.array([[1, -1], [0, 0], [3, -1]])
    ensemble_classifiers = create_pool_all_agree(return_value=0, size=10)
    ensemble_classifiers += create_pool_all_agree(return_value=2, size=9)
    predicted = majority_voting(ensemble_classifiers, query)
    assert predicted.all() == 0 and len(predicted) == 3


def tests_majority_voting_single_sample():
    query = np.array([1, -1])
    ensemble_classifiers = create_pool_all_agree(return_value=0, size=10)
    ensemble_classifiers += create_pool_all_agree(return_value=2, size=9)
    ensemble_classifiers += create_pool_all_agree(return_value=1, size=20)
    predicted = majority_voting(ensemble_classifiers, query)
    assert predicted.all() == 1


def tests_weighted_majority_voting():
    query = np.array([[1, -1], [0, 0], [3, -1]])
    ensemble_classifiers = create_pool_all_agree(return_value=0, size=10)
    ensemble_classifiers += create_pool_all_agree(return_value=2, size=9)
    weights = np.array([([0.5] * 10) + ([0.8] * 9), ([0.5] * 10) + ([0.8] * 9), ([0.5] * 10) + ([0.8] * 9)])
    predicted = weighted_majority_voting(ensemble_classifiers, weights, query)
    assert predicted.all() == 1 and len(predicted) == 3


def tests_weighted_majority_voting_single_sample():
    query = np.array([1, -1])
    clf_1 = create_base_classifier(return_value=1)
    clf_2 = create_base_classifier(return_value=1)
    clf_3 = create_base_classifier(return_value=2)
    ensemble_classifiers = [clf_2, clf_1, clf_3]
    weights = np.atleast_2d([0.2, 0.5, 1.0])
    predicted = weighted_majority_voting(ensemble_classifiers, weights, query)
    assert predicted == 2
