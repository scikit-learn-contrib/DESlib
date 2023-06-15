from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score
from sklearn.utils.estimator_checks import check_estimator

from deslib.static.single_best import SingleBest


def test_check_estimator():
    check_estimator(SingleBest())


# Testing if the fit function selects the correct classifier (the one with
# highest classification accuracy). # Note: clf[0] and clf[2] have the
# same accuracy since they always predict the same label.
def test_fit(create_X_y, create_pool_classifiers):
    X, y = create_X_y

    pool_classifiers = create_pool_classifiers
    single_best_test = SingleBest(pool_classifiers)
    single_best_test._estimate_performances = MagicMock(
        return_value=[1.0, 0.5, 0.99])

    single_best_test.fit(X, y)

    assert single_best_test.best_clf_index_ == 0


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


def test_label_encoder(create_label_encoder_test):
    X, y, pool = create_label_encoder_test
    sb = SingleBest(pool).fit(X, y)
    pred = sb.predict(X)
    assert np.array_equal(pred, y)


def test_label_encoder_base_ensemble():
    from sklearn.ensemble import RandomForestClassifier
    X, y = make_classification()
    y[y == 1] = 2
    y = y.astype(float)
    pool = RandomForestClassifier().fit(X, y)
    sb = SingleBest(pool)
    sb.fit(X, y)
    pred = sb.predict(X)
    assert np.isin(sb.classes_, pred).all()


def test_different_scorer():
    X, y = make_classification(n_samples=100, random_state=42)
    X_val, y_val = make_classification(n_samples=25, random_state=123)
    pool = AdaBoostClassifier(n_estimators=10).fit(X, y)
    performances = []
    for clf in pool:
        preds = clf.predict_proba(X_val)
        performances.append(roc_auc_score(y_val.ravel(), preds[:, -1]))
    id_best = np.argmax(performances)
    sb = SingleBest(pool_classifiers=pool, scoring='roc_auc')
    sb.fit(X_val, y_val)
    assert id_best == sb.best_clf_index_
