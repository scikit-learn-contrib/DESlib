from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from deslib.static.single_best import SingleBest
from sklearn.utils.estimator_checks import check_estimator


def test_check_estimator():
    check_estimator(SingleBest)


# Testing if the fit function selects the correct classifier (the one with
# highest classification accuracy). # Note: clf[0] and clf[2] have the
# same accuracy since they always predict the same label.
def test_fit(create_X_y, create_pool_classifiers):
    X, y = create_X_y

    pool_classifiers = create_pool_classifiers
    single_best_test = SingleBest(pool_classifiers)
    single_best_test.fit(X, y)

    assert (single_best_test.best_clf_index_ == 0 or
            single_best_test.best_clf_index_ == 2)


# The classifier with highest accuracy always predicts 0. So the expected
# prediction should always be equal zero.
def test_predict(create_X_y, create_pool_classifiers):
    X, y = create_X_y

    pool_classifiers = create_pool_classifiers
    single_best_test = SingleBest(pool_classifiers=pool_classifiers)
    single_best_test.fit(X, y)

    predicted_labels = single_best_test.predict(X)
    assert np.equal(predicted_labels, 0).all()


# The probabilities predicted must always be equals to the probabilities
# predicted by the base classifier with index 0.
def test_predict_proba(create_X_y, create_pool_classifiers):
    X, y = create_X_y

    pool_classifiers = create_pool_classifiers
    single_best_test = SingleBest(pool_classifiers)
    single_best_test.fit(X, y)

    predicted_proba = single_best_test.predict_proba(X)
    assert np.equal(predicted_proba,
                    pool_classifiers[0].predict_proba(X)).all()


def test_not_fitted():
    single_best_test = SingleBest()
    with pytest.raises(NotFittedError):
        single_best_test.predict(np.array([[1, -1]]))


# Test calling the predict_proba function with classifiers that do not
# implement the predict_proba
def test_not_predict_proba(create_X_y):
    X, y = create_X_y

    classifier = MagicMock()
    classifier.predict.return_value = [0]
    single_best_test = SingleBest([classifier] * 10)
    single_best_test.fit(X, y)
    with pytest.raises(ValueError):
        single_best_test.predict_proba(X)
