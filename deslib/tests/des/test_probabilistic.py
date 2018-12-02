import pytest
import numpy as np
from sklearn.linear_model import Perceptron
from deslib.des.probabilistic import (BaseProbabilistic,
                                      Logarithmic,
                                      Exponential,
                                      RRC,
                                      DESKL,
                                      MinimumDifference)

from sklearn.utils.estimator_checks import check_estimator


def test_check_estimator_RRC():
    check_estimator(RRC)


def test_check_estimator_DESKL():
    check_estimator(DESKL)


def test_check_estimator_Logarithmic():
    check_estimator(Logarithmic)


def test_check_estimator_Exponential():
    check_estimator(Exponential)


def test_check_estimator_MinimumDifference():
    check_estimator(MinimumDifference)


# Test if the class is raising an error when the base classifiers do not
# implements the predict_proba method. Should raise an exception when the
# base classifier cannot estimate posterior probabilities (predict_proba)
# Using Perceptron classifier as it does not implements predict_proba.
def test_not_predict_proba(create_X_y):
    X, y = create_X_y

    clf1 = Perceptron()
    clf1.fit(X, y)
    with pytest.raises(ValueError):
        BaseProbabilistic([clf1, clf1]).fit(X, y)


# Being all ones, all base classifiers are deemed competent
def test_select_all_ones():
    competences = np.ones(100)
    probabilistic_test = BaseProbabilistic()
    probabilistic_test.n_classes_ = 2
    selected_matrix = probabilistic_test.select(competences)
    assert selected_matrix.all()


# Being all zeros, no base classifier is deemed competent, so the system
# selects all of them
def test_select_all_zeros():
    competences = np.zeros(100)
    probabilistic_test = BaseProbabilistic()
    probabilistic_test.n_classes_ = 2
    selected_matrix = probabilistic_test.select(competences)
    assert selected_matrix.all()


# Being all zeros, no base classifier is deemed competent, so the system
# selects all of them
def test_select_random_classifier():
    competences = np.random.rand(1, 100)
    expected = (competences > 0.25)
    probabilistic_test = BaseProbabilistic()
    probabilistic_test.n_classes_ = 4
    indices = probabilistic_test.select(competences)
    assert np.array_equal(indices, expected)


# Being all zeros, no base classifier is deemed competent, so the system
# selects all of them
def test_select_threshold():
    competences = np.random.rand(1, 100)
    expected = (competences > 0.5)

    probabilistic_test = BaseProbabilistic()
    probabilistic_test.selection_threshold = 0.5
    indices = probabilistic_test.select(competences)
    assert np.array_equal(indices, expected)


# Test the potential function calculation. The return value should be zero in
# this test.
def test_potential_function_zeros():
    dists = np.zeros(10)
    value = BaseProbabilistic.potential_func(dists)
    assert np.array_equal(value, np.ones(10))


# Test the potential function calculation. Higher values for distances should
# obtain a lower value in the results
def test_potential_function():
    dists = np.array([1.0, 0.5, 2, 0.33])
    value = BaseProbabilistic.potential_func(dists)
    assert np.allclose(value, [0.3679, 0.7788, 0.0183, 0.8968], atol=0.001)


# Test the potential function calculation. Higher values for distances should
# obtain a lower value in the results
def test_potential_function_batch():
    dists = np.tile([1.0, 0.5, 2, 0.33], (10, 1))
    value = BaseProbabilistic.potential_func(dists)
    expected = np.tile([0.3679, 0.7788, 0.0183, 0.8968], (10, 1))
    assert np.allclose(value, expected, atol=0.001)


def test_estimate_competence_batch():
    n_samples = 10
    query = np.ones((n_samples, 2))
    probabilistic_test = BaseProbabilistic()
    probabilistic_test.k_ = 7
    distances = np.tile([0.5, 1.0, 2.0], (n_samples, 1))
    neighbors = np.tile([0, 1, 2], (n_samples, 1))

    probabilistic_test.C_src_ = np.array(
        [[0.5, 0.2, 0.8], [1.0, 1.0, 1.0], [1.0, 0.6, 0.3]])
    expected = np.tile([0.665, 0.458, 0.855], (n_samples, 1))
    competence = probabilistic_test.estimate_competence(query,
                                                        neighbors=neighbors,
                                                        distances=distances)
    assert np.allclose(competence, expected, atol=0.01)


# Test the estimate competence function when the competence source is equal
# to zero. The competence should also be zero.
def test_estimate_competence_zeros(example_estimate_competence):
    distances = example_estimate_competence[3]
    query = np.atleast_2d([1, 1])
    probabilistic_test = BaseProbabilistic()
    probabilistic_test.k_ = 7

    distances = distances[0, 0:3].reshape(1, -1)
    neighbors = np.array([[0, 2, 1]])
    probabilistic_test.C_src_ = np.zeros((3, 3))
    competence = probabilistic_test.estimate_competence(query,
                                                        neighbors=neighbors,
                                                        distances=distances)
    assert np.sum(competence) == 0.0


# Test the estimate competence function when the competence source is equal
# to one. The competence should also be ones.
def test_estimate_competence_ones(example_estimate_competence):
    distances = example_estimate_competence[3]
    query = np.atleast_2d([1, 1])
    probabilistic_test = BaseProbabilistic()
    probabilistic_test.k_ = 7

    distances = distances[0, 0:3].reshape(1, -1)
    neighbors = np.array([[0, 2, 1]])
    probabilistic_test.C_src_ = np.ones((3, 3))
    competence = probabilistic_test.estimate_competence(query, neighbors,
                                                        distances)
    assert (competence == 1.0).all()


""" Test the source_competence using the rrc method. Here we consider the same
values from the example applied in the test_prob_functions.py to assert if the
source_competence function call the ccprmod correctly and fill the competence
source (C_src) with the correct results.

The scores used are: [[0.3, 0.6, 0.1], [1.0 / 3, 1.0 / 3, 1.0 / 3],
                      [0.5, 0.2, 0.3], [0.5, 0.2, 0.3]]
The correct labels are: [1, 0, 0, 1]
The expected value should be: an np.array (4,1) with
the values = [[0.7849], [0.3328], [0.6428], [0.1194]]
"""


def test_source_competence_rrc():
    rrc_test = RRC()
    rrc_test.n_classifiers_ = 1
    rrc_test.dsel_scores_ = np.array([[[0.3, 0.6, 0.1],
                                       [1.0 / 3, 1.0 / 3, 1.0 / 3],
                                       [0.5, 0.2, 0.3],
                                       [0.5, 0.2, 0.3]]]).reshape(4, 1, 3)
    rrc_test.DSEL_target_ = [1, 0, 0, 1]
    rrc_test.n_classes_ = 3
    rrc_test.n_samples_ = 4
    C_src = rrc_test.source_competence()
    expected = np.array([[0.7849], [0.3328], [0.6428], [0.1194]])
    assert np.allclose(C_src, expected, atol=0.01)


# Test the source_competence estimation for the Kullback-Leibler method. Here
# we consider the same values applied in the test_prob_functions.py to assert
# if the source_competence function fill the competence source (C_src) with
# the correct results.
#
# The scores used are: [[0.33, 0.33, 0.33], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
# The matrix with correct predictions is: [False, True, False]
# The expected value should be: an np.array (3,1) with
# the values = [[0.0], [1.0], [-1.0]]
def test_source_competence_kl():
    entropy_test = DESKL()
    entropy_test.n_classifiers_ = 1
    entropy_test.dsel_scores_ = np.array([[[0.33, 0.33, 0.33],
                                           [1.0, 0.0, 0.0],
                                           [1.0, 0.0, 0.0]]]).reshape(3, 1, 3)
    entropy_test.DSEL_processed_ = np.array([[False], [True], [False]])
    entropy_test.n_classes_ = 3
    entropy_test.n_samples_ = 3
    C_src = entropy_test.source_competence()
    expected = np.array([[0.0], [1.0], [-1.0]])
    assert np.allclose(C_src, expected, atol=0.01)


# Test the source_competence estimation for the Minimum difference  method.
# Here we consider the same values applied in the test_prob_functions.py to
# assert if the source_competence function fill the competence source
# (C_src) with the correct results.
#
# The scores used are: [[0.3, 0.6, 0.1], [1.0 / 3, 1.0 / 3, 1.0 / 3],
#                       [0.5, 0.2, 0.3], [0.5, 0.2, 0.3]]
# The correct labels are: [1, 0, 0, 1]
# The expected value should be: an np.array (4,1) with
# the values = [[0.7849], [0.3328], [0.6428], [0.1194]]
def test_source_competence_minimum_difference():
    md_test = MinimumDifference()
    md_test.n_classifiers_ = 1
    md_test.dsel_scores_ = np.array([[[0.3, 0.6, 0.1],
                                      [1.0 / 3, 1.0 / 3, 1.0 / 3],
                                      [0.5, 0.2, 0.3],
                                      [0.5, 0.2, 0.3]]]).reshape(4, 1, 3)

    md_test.DSEL_target_ = [1, 0, 0, 1]
    md_test.n_classes_ = 3
    md_test.n_samples_ = 4
    C_src = md_test.source_competence()
    expected = np.array([[0.3], [0.0], [0.2], [-0.3]])
    assert np.allclose(C_src, expected, atol=0.01)


# Test the source_competence using the logarithmic method. Here we consider
# the same values applied in the test_prob_functions.py to assert if the source
# competence function fill the competence source
# (C_src) with the correct results.
#
# The scores used are: [[0.67, 0.33, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
# The correct labels are: [1, 1, 1], so the supports for the correct
#  class are: [0.33, 0.0, 1.0]
# The expected value should be: an np.array (3,1) with
# the values = [[0.0], [-1.0], [1.0]]]
def test_source_competence_logarithmic():
    log_test = Logarithmic()
    log_test.n_classifiers_ = 1
    log_test.dsel_scores_ = np.array([[[0.67, 0.33, 0.0],
                                       [1.0, 0.0, 0.0],
                                       [0.0, 1.0, 0.0]]]).reshape(3, 1, 3)

    log_test.DSEL_target_ = [1, 1, 1]
    log_test.n_classes_ = 3
    log_test.n_samples_ = 3
    C_src = log_test.source_competence()
    expected = np.array([[0.0], [-1.0], [1.0]])
    assert np.allclose(C_src, expected, atol=0.01)


# Test the source_competence using the exponential method. Here we consider
# the same values  applied in the test_prob_functions.py to assert if the
# source_competence function fill the competence source (C_src) with the
# correct results.
#
# Only two classes_ are considered in this example.
# The scores used are: [[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]].
# The correct labels are: [1, 1, 1], so the supports for the correct
# class are: [0.5, 0.0, 1.0].
# The expected value should be: an np.array (3,1) with
# the values = [[0.0], [-1.0], [1.0]]].
def test_source_competence_exponential():
    exp_test = Exponential()
    exp_test.n_classifiers_ = 1
    exp_test.dsel_scores_ = np.array([[[0.5, 0.5],
                                       [1.0, 0.0],
                                       [0.0, 1.0]]]).reshape(3, 1, 2)

    exp_test.DSEL_target_ = [1, 1, 1]
    exp_test.n_classes_ = 2
    exp_test.n_samples_ = 3
    C_src = exp_test.source_competence()
    expected = np.array([[0.0], [-1.0], [1.0]])
    assert np.allclose(C_src, expected, atol=0.01)
