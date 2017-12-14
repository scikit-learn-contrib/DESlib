from unittest.mock import MagicMock
from pythonds.base import DS
import numpy as np
import pytest


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


def create_some_classifiers():
    return [create_base_classifier(return_value=1)] * 10


@pytest.mark.parametrize('k', [0, 1, -1])
def test_check_k(k):
    pool_classifiers = create_some_classifiers()

    with pytest.raises(ValueError):
        DS(pool_classifiers, k=k)


@pytest.mark.parametrize('k', ['a', 2.5])
def test_check_k(k):
    pool_classifiers = create_some_classifiers()

    with pytest.raises(TypeError):
        DS(pool_classifiers, k=k)


@pytest.mark.parametrize('k, ih_k', [(2, 3), (5, 7)])
def test_valid_IH_k(k, ih_k):
    with pytest.raises(ValueError):
        DS([create_base_classifier(1)], k=k, safe_k=ih_k)


def create_base_classifier(return_value, return_prob=None):
    classifier = MagicMock()
    classifier.predict.return_value = [return_value]
    classifier.predict_proba.return_value = [return_prob]
    return classifier
