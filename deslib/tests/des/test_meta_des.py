from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.estimator_checks import check_estimator

from deslib.des.meta_des import METADES


def test_check_estimator():
    check_estimator(METADES)


# ---------------------- Testing Hyper-parameters -----------------------
def test_meta_classifier_not_predict_proba(create_pool_classifiers):
    X = np.random.rand(10, 2)
    y = np.ones(10)
    with pytest.raises(ValueError):
        meta = METADES(create_pool_classifiers, Perceptron())
        meta.fit(X, y)


@pytest.mark.parametrize('Hc', ['a', None, 0.2, -1])
def test_parameter_Hc(Hc, create_pool_classifiers):
    X = np.random.rand(10, 2)
    y = np.ones(10)
    with pytest.raises((ValueError, TypeError)):
        meta = METADES(create_pool_classifiers, Hc=Hc)
        meta.fit(X, y)


@pytest.mark.parametrize('selection_threshold', ['a', None, 0, -1, 0.45])
def test_parameter_gamma(selection_threshold, create_pool_classifiers):
    X = np.random.rand(10, 2)
    y = np.ones(10)
    with pytest.raises((ValueError, TypeError)):
        meta = METADES(create_pool_classifiers,
                       selection_threshold=selection_threshold)
        meta.fit(X, y)


# -------------------------- Testing Methods -----------------------
def test_compute_meta_features(example_estimate_competence,
                               create_pool_classifiers):

    X, y, nn, _, dsel_processed, dsel_scores = example_estimate_competence

    query = np.ones((1, 2))
    pool = create_pool_classifiers
    meta_test = METADES(pool_classifiers=[pool[0]])
    meta_test.n_classifiers_ = 1
    meta_test.k_ = 7
    meta_test.Kp_ = 5
    # Considering only one classifier in the pool (index = 0)
    meta_test.DSEL_processed_ = dsel_processed[:, 0].reshape(-1, 1)
    meta_test.dsel_scores_ = dsel_scores[:, 0, :].reshape(15, 1, 2)
    meta_test.DSEL_target_ = y
    meta_test.n_classes_ = 2

    neighbors_op = nn[2, 0:meta_test.Kp]

    # Expected values for each meta feature based on the data of ex1.
    expected_f1 = [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    expected_f2 = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    expected_f3 = [4.0 / 7.0]
    expected_f4 = [0.0, 1.0, 1.0, 1.0, 0.0]
    expected_f5 = [0.5]

    scores = np.empty(
        (query.shape[0], meta_test.n_classifiers_, meta_test.n_classes_))
    for index, clf in enumerate(meta_test.pool_classifiers):
        scores[:, index, :] = clf.predict_proba(query)

    meta_features = meta_test.compute_meta_features(scores, nn[0, :],
                                                    neighbors_op)
    expected = np.asarray(
        expected_f1 + expected_f2 + expected_f3 + expected_f4 + expected_f5)
    assert np.array_equal(meta_features, expected.reshape(1, -1))


# Test the estimate competence function considering
# 3 base classifiers and 1 test sample
def test_estimate_competence(example_estimate_competence,
                             create_pool_classifiers):
    _, y, nn, _, dsel_processed, dsel_scores = example_estimate_competence

    query = np.ones((1, 2))
    meta_test = METADES(create_pool_classifiers)
    meta_test.n_classifiers_ = 3
    meta_test.k_ = 7
    meta_test.Kp_ = 5
    # Set the state of the system which is set by the fit method.
    meta_test.DSEL_processed_ = dsel_processed
    meta_test.dsel_scores_ = dsel_scores
    meta_test.DSEL_target_ = y
    meta_test.n_classes_ = 2

    meta_test.meta_classifier_ = GaussianNB()

    meta_test._get_similar_out_profiles = MagicMock(
        return_value=(None, nn[0, 0:meta_test.Kp]))
    meta_test.meta_classifier_.predict_proba = MagicMock(
        return_value=np.array([[0.2, 0.8], [1.0, 0.0], [0.2, 0.8]]))

    probabilities = []
    for clf in meta_test.pool_classifiers:
        probabilities.append(clf.predict_proba(query))

    probabilities = np.array(probabilities).transpose((1, 0, 2))

    expected = np.array([[0.8, 0.0, 0.8]])
    competences = meta_test.estimate_competence_from_proba(query,
                                                           nn[0, :],
                                                           probabilities)
    assert np.array_equal(competences, expected)


# Test the estimate competence function considering 3 base classifiers
# and 3 test samples.
def test_estimate_competence_batch(example_estimate_competence,
                                   create_pool_classifiers):
    _, y, nn, _, dsel_processed, dsel_scores = example_estimate_competence

    query = np.ones((3, 1))
    meta_test = METADES(pool_classifiers=create_pool_classifiers)
    meta_test.n_classifiers_ = 3
    n_meta_features = 21
    meta_test.meta_classifier_ = GaussianNB
    # Set the state of the system which is set by the fit method.
    meta_test.DSEL_processed_ = dsel_processed
    meta_test.dsel_scores_ = dsel_scores
    meta_test.DSEL_target_ = y
    nn = nn

    meta_test._get_similar_out_profiles = MagicMock(
        return_value=(None, nn[:, 0:meta_test.Kp]))
    meta_test.compute_meta_features = MagicMock(
        return_value=np.ones((9, n_meta_features)))
    meta_test.meta_classifier_.predict_proba = MagicMock(
        return_value=np.tile([0.0, 0.8], (9, 1)))

    probabilities = []
    for clf in meta_test.pool_classifiers:
        probabilities.append(clf.predict_proba(query))

    probabilities = np.array(probabilities).transpose((1, 0, 2))

    expected = np.ones((3, 3)) * 0.8
    competences = meta_test.estimate_competence_from_proba(query, nn,
                                                           probabilities)
    assert np.array_equal(competences, expected)


# Test select passing a single sample
def test_select():
    meta_test = METADES()
    competences = np.asarray(
        [0.8, 0.6, 0.7, 0.2, 0.3, 0.4, 0.6, 0.1, 1.0, 0.98])
    expected = np.asarray(
        [True, True, True, False, False, False, True, False, True, True])
    selected_matrix = meta_test.select(competences)
    assert np.array_equal(selected_matrix, expected.reshape(1, -1))


# test select passing 10 samples
def test_select_batch():
    meta_test = METADES()
    competences = np.tile(
        np.array([0.8, 0.6, 0.7, 0.2, 0.3, 0.4, 0.6, 0.1, 1.0, 0.98]), (10, 1))
    expected = np.tile(
        [True, True, True, False, False, False, True, False, True, True],
        (10, 1))
    selected_matrix = meta_test.select(competences)
    assert np.array_equal(selected_matrix, expected)


# 10 samples, no classifier is selected so the array should return all
# True (10 x 3)
def test_select_no_competent_classifiers_batch():
    meta_test = METADES()
    meta_test.n_classifiers_ = 3
    competences = np.zeros((10, meta_test.n_classifiers_))
    selected_matrix = meta_test.select(competences)
    assert np.all(selected_matrix)


# Test the sample selection mechanism considering 5 test samples and 15 base
# classifiers. The agreement is computed
# for all samples at the same time
def test_sample_selection():
    meta_test = METADES()
    meta_test.n_classifiers_ = 15
    meta_test.DSEL_processed_ = np.ones((5, 15))
    meta_test.DSEL_processed_[(1, 3, 4), 5:] = 0
    expected = np.asarray([1, 1 / 3, 1, 1 / 3, 1 / 3])
    value = meta_test._sample_selection_agreement()
    assert np.array_equal(value, expected)


def test_sample_selection_working():
    meta_test = METADES()
    meta_test.n_classifiers_ = 15
    meta_test.DSEL_processed_ = np.ones((5, 15))
    meta_test.DSEL_processed_[(1, 3, 4), 5:] = 0
    expected = np.asarray([1, 1 / 3, 1, 1 / 3, 1 / 3])
    value = meta_test._sample_selection_agreement()
    assert np.array_equal(value, expected)


# Test if the class is raising an error when the base classifiers do not
# implements the predict_proba method. Should raise an exception when the
# base classifier cannot estimate posterior probabilities (predict_proba)
# Using Perceptron classifier as it does not implements predict_proba.
def test_not_predict_proba(create_X_y):
    X, y = create_X_y

    clf1 = Perceptron()
    clf1.fit(X, y)
    with pytest.raises(ValueError):
        meta = METADES([clf1, clf1])
        meta.fit(X, y)
