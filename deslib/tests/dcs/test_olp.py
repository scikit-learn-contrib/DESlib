from deslib.dcs.olp import OLP
from sklearn.utils.estimator_checks import check_estimator


def test_check_estimator():
    check_estimator(OLP)
