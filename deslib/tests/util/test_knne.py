import numpy as np
from deslib.util import KNNE


def test_equal_classes():
    X = np.tile(np.arange(15).reshape(-1, 1), 3)
    y = np.array(5 * [0] + 5 * [1] + 5 * [2])

    knne = KNNE(n_neighbors=6)
    knne.fit(X, y)
    dist, inds = knne.kneighbors()
    classes = y[inds]
    b = np.apply_along_axis(np.bincount, 1, classes)
    assert np.equal(b, 2).all()


def test_predict():
    X = np.tile(np.arange(10).reshape(-1, 1), 3)
    y = np.array(5 * [0] + 5 * [1])
    knne = KNNE(n_neighbors=6)
    knne.fit(X, y)
    prediction = knne.predict(X)
    assert np.equal(prediction, y).all()


def test_predict_proba():
    X = np.tile(np.arange(10).reshape(-1, 1), 3)
    y = np.array(5 * [0] + 5 * [1])

    knne = KNNE(n_neighbors=6)
    knne.fit(X, y)
    probas = knne.predict_proba(X)
    assert np.all(probas[0:5, 0] > probas[0:5, 1])
    assert np.all(probas[5:-1, 1] > probas[5:-1, 0])
