import pytest
import numpy as np
from deslib.util.sgh import SGH
from deslib.static.oracle import Oracle
from sklearn.datasets import load_iris


def test_build_perceptron_equal_centroids():
    X = np.ones((10, 2))
    y = np.ones(10)
    y[:5] = 0
    sgh_test = SGH()
    with pytest.raises(Warning):
        sgh_test.fit(X, y)


# Check if technique generate a pool of classifiers with 100% accuracy oracle
def test_generate_pool_oracle():
    data = load_iris()
    sgh_test = SGH()
    sgh_test.fit(data.data, data.target)
    pool_classifiers = sgh_test.estimators_
    oracle = Oracle(pool_classifiers).fit(data.data, data.target)
    assert oracle.score(data.data, data.target) == 1.0
