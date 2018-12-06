import pytest
from deslib.static.stacked import StackedClassifier
from sklearn.utils.estimator_checks import check_estimator
from sklearn.linear_model import Perceptron


def test_check_estimator():
    check_estimator(StackedClassifier)


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
