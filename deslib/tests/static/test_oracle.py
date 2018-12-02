import numpy as np

from deslib.static.oracle import Oracle


def test_predict(create_X_y, create_pool_classifiers):
    X, y = create_X_y

    oracle_test = Oracle(create_pool_classifiers)
    oracle_test.fit(X, y)
    predicted_labels = oracle_test.predict(X, y)
    assert np.equal(predicted_labels, y).all()

    assert oracle_test.score(X, y) == 1.0


# All classifiers predicts the same label. This test only the samples
# with label == 0 are correctly classified by the Oracle.
# The misclassified samples are set to -1.
def test_predict_all_same(create_X_y, create_pool_all_agree):
    X, y = create_X_y

    expected = y
    oracle_test = Oracle(create_pool_all_agree)
    oracle_test.fit(X, y)
    expected[expected == 1] = 0
    predicted_labels = oracle_test.predict(X, y)
    assert np.equal(predicted_labels, expected).all()
