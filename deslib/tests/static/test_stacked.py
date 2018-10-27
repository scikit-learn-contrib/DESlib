from deslib.static.stacked import StackedClassifier
from sklearn.utils.estimator_checks import check_estimator


def test_check_estimator():
    check_estimator(StackedClassifier)
