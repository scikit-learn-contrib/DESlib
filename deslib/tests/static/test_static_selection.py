from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from deslib.static.static_selection import StaticSelection
from sklearn.utils.estimator_checks import check_estimator


def test_check_estimator():
    check_estimator(StaticSelection)

#
# def create_example_static(create_X_y):
#     for clf in pool:
#         clf.score = MagicMock(return_value=score)
#     return pool


# Testing if the fit function selects the correct classifiers.
# The 50 last classifiers should be selected.
def test_fit(example_static_selection):
    X, y, pool = example_static_selection
    static_selection_test = StaticSelection(pool, 0.5)
    static_selection_test.fit(X, y)

    assert static_selection_test.n_classifiers_ensemble_ == 50
    assert static_selection_test.n_classifiers_ensemble_ == len(
        static_selection_test.clf_indices_)
    assert np.array_equal(np.sort(static_selection_test.clf_indices_),
                          list(range(50, 100)))


# The classifier with highest accuracy always predicts 0. So the expected
# prediction should always be equal zero.
def test_predict(example_static_selection, create_pool_classifiers):
    X, y, _ = example_static_selection

    static_selection_test = StaticSelection(create_pool_classifiers*10, 0.25)
    static_selection_test.fit(X, y)

    predicted_labels = static_selection_test.predict(X)
    assert np.equal(predicted_labels, 0).all()


# Classifiers predicting different labels are selected
def test_predict_diff(example_static_selection):
    X, y, pool = example_static_selection

    static_selection_test = StaticSelection(pool, 0.75)
    static_selection_test.fit(X, y)

    predicted_labels = static_selection_test.predict(X)
    assert np.equal(predicted_labels, 1).all()


def test_not_fitted():
    static_selection_test = StaticSelection()
    with pytest.raises(NotFittedError):
        static_selection_test.predict(np.array([[1, -1]]))


def test_invalid_pct():
    with pytest.raises(TypeError):
        test = StaticSelection(pct_classifiers='something')
        test.fit(np.random.rand(10, 2), np.ones(10))


def test_invalid_pct2():
    with pytest.raises(ValueError):
        test = StaticSelection(pct_classifiers=1.2)
        test.fit(np.random.rand(10, 2), np.ones(10))
