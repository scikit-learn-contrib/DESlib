from deslib.dcs.olp import OLP
from sklearn.utils.estimator_checks import check_estimator
import pytest


@pytest.mark.skip(reason='Waiting batch processing implementation (See issue #101)')
def test_check_estimator():
    check_estimator(OLP)
