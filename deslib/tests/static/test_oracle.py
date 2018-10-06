import numpy as np

from deslib.static.oracle import Oracle
from deslib.tests.examples_test import setup_example1, create_pool_classifiers, create_pool_all_agree


def test_predict():
    X, y = setup_example1()[0:2]

    oracle_test = Oracle(create_pool_classifiers())
    oracle_test.fit(X, y)
    predicted_labels = oracle_test.predict(X, y)
    assert np.equal(predicted_labels, y).all()

    assert oracle_test.score(X, y) == 1.0


# All classifiers predicts the same label. This test only the samples with label == 0
# are correctly classified by the Oracle. The misclassified samples are set to -1.
def test_predict_all_same():
    X, y = setup_example1()[0:2]

    expected = y
    oracle_test = Oracle(create_pool_all_agree(return_value=0, size=10))
    oracle_test.fit(X, y)
    expected[expected == 1] = 0
    predicted_labels = oracle_test.predict(X, y)
    assert np.equal(predicted_labels, expected).all()

