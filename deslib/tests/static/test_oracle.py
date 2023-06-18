import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

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


def test_predict_proba_shape():
    n_test_samples = 200
    X, y = make_classification(n_samples=1000)
    X_test, y_test = make_classification(n_samples=n_test_samples)
    pool = RandomForestClassifier(max_depth=3).fit(X, y)
    oracle = Oracle(pool_classifiers=pool).fit(X, y)

    proba = oracle.predict_proba(X_test, y_test)
    assert proba.shape == (n_test_samples, 2)


def test_predict_proba_right_class():
    n_test_samples = 200
    X, y = make_classification(n_samples=1000)
    X_test, y_test = make_classification(n_samples=n_test_samples)
    pool = RandomForestClassifier(max_depth=3).fit(X, y)
    oracle = Oracle(pool_classifiers=pool).fit(X, y)

    preds = oracle.predict(X_test, y_test)
    proba = oracle.predict_proba(X_test, y_test)
    probas_max = np.argmax(proba, axis=1)
    assert np.allclose(probas_max, preds)


def test_label_encoder_base_ensemble():
    from sklearn.ensemble import RandomForestClassifier
    X, y = make_classification()
    y[y == 1] = 2
    y = y.astype(float)
    pool = RandomForestClassifier().fit(X, y)
    oracle = Oracle(pool)
    oracle.fit(X, y)
    pred = oracle.predict(X, y)
    assert np.isin(oracle.classes_, pred).all()
