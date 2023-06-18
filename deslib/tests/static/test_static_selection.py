import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import log_loss
from sklearn.utils.estimator_checks import check_estimator

from deslib.static.static_selection import StaticSelection


def test_check_estimator():
    check_estimator(StaticSelection())


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


def test_label_encoder(create_label_encoder_test):
    X, y, pool = create_label_encoder_test
    static = StaticSelection(pool).fit(X, y)
    pred = static.predict(X)
    assert np.array_equal(pred, y)


def test_label_encoder_base_ensemble():
    from sklearn.ensemble import RandomForestClassifier
    X, y = make_classification()
    y[y == 1] = 2
    y = y.astype(float)
    pool = RandomForestClassifier().fit(X, y)
    ss = StaticSelection(pool)
    ss.fit(X, y)
    pred = ss.predict(X)
    assert np.isin(ss.classes_, pred).all()


def test_predict_proba(example_static_selection):
    X, y, pool = example_static_selection
    expected = np.tile([0.52, 0.48], (y.size, 1))
    static_selection_test = StaticSelection(pool, 0.5)
    static_selection_test.fit(X, y)
    proba = static_selection_test.predict_proba(X)
    assert np.allclose(proba, expected)


# Test if static_selection can select the best classifier according to a
# metric that needs to be minimized.
def test_different_scorer():
    X, y = make_classification(n_samples=100, random_state=42)
    X_val, y_val = make_classification(n_samples=25, random_state=123)
    pool = AdaBoostClassifier(n_estimators=10).fit(X, y)
    performances = []
    for clf in pool:
        preds = clf.predict_proba(X_val)
        performances.append(log_loss(y_val.ravel(), preds[:, -1]))
    id_best = np.argsort(performances)
    ss = StaticSelection(pool_classifiers=pool, scoring='neg_log_loss')
    ss.fit(X_val, y_val)
    assert (id_best[:ss.n_classifiers_ensemble_] == ss.clf_indices_).all()


# Test if static_selection can select the best classifier according to a
# metric that needs to be minimized.
def test_different_scorer():
    X, y = make_classification(n_samples=100, random_state=42)
    X_val, y_val = make_classification(n_samples=25, random_state=123)
    pool = AdaBoostClassifier(n_estimators=10).fit(X, y)
    performances = []
    for clf in pool:
        preds = clf.predict_proba(X_val)
        performances.append(log_loss(y_val.ravel(), preds[:, -1]))
    id_best = np.argsort(performances)
    ss = StaticSelection(pool_classifiers=pool, scoring='neg_log_loss')
    ss.fit(X_val, y_val)
    assert (id_best[:ss.n_classifiers_ensemble_] == ss.clf_indices_).all()
