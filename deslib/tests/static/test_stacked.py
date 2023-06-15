import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.estimator_checks import check_estimator

from deslib.static.stacked import StackedClassifier


def test_check_estimator():
    check_estimator(StackedClassifier())


# Test if the class is raising an error when the base classifiers do not
# implements the predict_proba method. Should raise an exception when the
# base classifier cannot estimate posterior probabilities (predict_proba)
# Using Perceptron classifier as it does not implements predict_proba.
def test_not_predict_proba(create_X_y):
    X, y = create_X_y

    clf1 = Perceptron()
    clf1.fit(X, y)
    with pytest.raises(ValueError):
        StackedClassifier([clf1, clf1]).fit(X, y)


# Test if the class is raising an error when the meta classifiers do not
# implements the predict_proba method. Should raise an exception when the
# base classifier cannot estimate posterior probabilities (predict_proba)
# Using Perceptron classifier as it does not implements predict_proba.
def test_not_predict_proba_meta(create_X_y, create_pool_classifiers):
    X, y = create_X_y

    pool = create_pool_classifiers
    with pytest.raises(ValueError):
        meta_clf = StackedClassifier(pool_classifiers=pool,
                                     meta_classifier=Perceptron())
        meta_clf.fit(X, y)
        meta_clf.predict_proba(X)


def test_label_encoder():
    y = ['one', 'one', 'one', 'zero', 'zero', 'two']
    X = np.random.rand(6, 3)
    pool = [DecisionTreeClassifier().fit(X, y) for _ in range(5)]
    stacked = StackedClassifier(pool).fit(X, y)
    pred = stacked.predict(X)
    assert np.array_equal(pred, y)


def test_label_encoder_base_ensemble():
    from sklearn.ensemble import RandomForestClassifier
    X, y = make_classification()
    y[y == 1] = 2
    y = y.astype(float)
    pool = RandomForestClassifier().fit(X, y)
    st = StackedClassifier(pool)
    st.fit(X, y)
    pred = st.predict(X)
    assert np.isin(st.classes_, pred).all()


def test_one_class_meta_dataset(create_X_y):
    X, y = create_X_y
    pool = [DecisionTreeClassifier().fit(X, y) for _ in range(5)]
    stacked = StackedClassifier(pool)
    X_meta = np.random.rand(10, 2)
    y_meta = np.zeros(10, dtype=int)
    with pytest.raises(ValueError):
        stacked.fit(X_meta, y_meta)


def test_passthrough_true(create_X_y):
    X, y = create_X_y
    pool = [DecisionTreeClassifier().fit(X, y) for _ in range(5)]
    stacked = StackedClassifier(pool, passthrough=True)
    stacked.fit(X, y)
    assert stacked.meta_classifier_.coef_.shape == (1, 7)


def test_passthrough_false(create_X_y):
    X, y = create_X_y
    pool = [DecisionTreeClassifier().fit(X, y) for _ in range(5)]
    stacked = StackedClassifier(pool, passthrough=False)
    stacked.fit(X, y)
    assert stacked.meta_classifier_.coef_.shape == (1, 5)


def test_single_model_pool(create_X_y):
    X, y = create_X_y
    pool = [DecisionTreeClassifier().fit(X, y)]
    with pytest.raises(ValueError):
        StackedClassifier(pool_classifiers=pool).fit(X, y)
