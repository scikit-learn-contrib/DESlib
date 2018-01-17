from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from deslib.static.static_selection import StaticSelection
from deslib.tests.examples_test import create_pool_classifiers, create_pool_all_agree, X_dsel_ex1, y_dsel_ex1


def create_pool_classifiers_score(prediction, size, score):
    pool = create_pool_all_agree(return_value=prediction, size=size)
    for clf in pool:
        clf.score = MagicMock(return_value=score)
    return pool
# Testing if the fit function selects the correct classifiers. The 50 last classifiers should be selected.


def test_fit():
    X = X_dsel_ex1
    y = y_dsel_ex1
    pool_classifiers = create_pool_classifiers_score(1, 50, 0.5) + create_pool_classifiers_score(1, 50, 1.0)
    static_selection_test = StaticSelection(pool_classifiers, 0.5)
    static_selection_test.fit(X, y)

    assert static_selection_test.n_classifiers_ensemble == 50
    assert static_selection_test.n_classifiers_ensemble == len(static_selection_test.clf_indices)
    assert np.array_equal(np.sort(static_selection_test.clf_indices), list(range(50, 100)))


# The classifier with highest accuracy always predicts 0. So the expected prediction should always be equal zero.
def test_predict():
    X = X_dsel_ex1
    y = y_dsel_ex1
    pool_classifiers = create_pool_classifiers_score(1, 25, 0.5) + create_pool_classifiers_score(0, 25, 1.0)
    static_selection_test = StaticSelection(pool_classifiers, 0.25)
    static_selection_test.fit(X, y)

    predicted_labels = static_selection_test.predict(X)
    assert np.equal(predicted_labels, 0).all()


# Classifiers predicting different labels are selected
def test_predict_diff():
    X = X_dsel_ex1
    y = y_dsel_ex1
    pool_classifiers = create_pool_classifiers_score(1, 25, 0.5) + create_pool_classifiers_score(0, 25, 0.5)
    pool_classifiers += create_pool_classifiers_score(1, 25, 0.75)
    static_selection_test = StaticSelection(pool_classifiers, 0.33)
    static_selection_test.fit(X, y)

    predicted_labels = static_selection_test.predict(X)
    assert np.equal(predicted_labels, 1).all()


def test_not_fitted():
    static_selection_test = StaticSelection(create_pool_classifiers(), 0.25)
    with pytest.raises(NotFittedError):
        static_selection_test.predict(np.array([1, -1]))


def test_invalid_pct():
    with pytest.raises(TypeError):
        StaticSelection(create_pool_classifiers(), pct_classifiers='something')

    with pytest.raises(ValueError):
        StaticSelection(create_pool_classifiers(), pct_classifiers=1.2)