from unittest.mock import MagicMock
from pythonds.base import DS
import numpy as np


def test_all_classifiers_agree():
    # 10 classifiers that return 1
    pool_classifiers = [create_base_classifier(return_value=1)] * 10
    ds = DS(pool_classifiers)

    x = np.ones((1, 10))
    assert ds._all_classifier_agree(x)


def test_not_all_classifiers_agree():
    # 10 classifiers that return 1, and one that returns 2
    pool_classifiers = [create_base_classifier(return_value=1)] * 10
    pool_classifiers.append(create_base_classifier(return_value=2))
    ds = DS(pool_classifiers)

    x = np.ones((1, 10))
    assert not ds._all_classifier_agree(x)


def create_base_classifier(return_value, return_prob=None):
    classifier = MagicMock()
    classifier.predict.return_value = [return_value]
    classifier.predict_proba.return_value = [return_prob]
    return classifier