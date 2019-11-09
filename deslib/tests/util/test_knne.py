import pytest
import numpy as np
from deslib.util import faiss_knn_wrapper
from deslib.util import KNNE


def setup_test(n_neighbors, knn_classifier='sklearn'):
    X = np.tile(np.arange(15).reshape(-1, 1), 3)
    y = np.array(5 * [0] + 5 * [1] + 5 * [2])
    knne = KNNE(n_neighbors=n_neighbors, knn_classifier=knn_classifier)
    knne.fit(X, y)
    return X, y, knne


def test_equal_classes():
    X, y, knne = setup_test(n_neighbors=6)
    dist, inds = knne.kneighbors()
    classes = y[inds]
    b = np.apply_along_axis(np.bincount, 1, classes)
    assert np.equal(b, 2).all()


def test_uneven_k():
    X, y, knne = setup_test(n_neighbors=7)
    test = X[0, :].reshape(1, -1)
    dist, inds = knne.kneighbors(test, 7)
    assert np.allclose([0, 1, 5, 6, 10, 11, 2], inds)


def test_predict():
    X, y, knne = setup_test(n_neighbors=6)
    prediction = knne.predict(X)
    assert np.equal(prediction, y).all()


def test_predict_proba():
    X, y, knne = setup_test(n_neighbors=6)
    probas = knne.predict_proba(X)
    assert np.all(probas[0:5, 0] > probas[0:5, 1])
    assert np.all(probas[5:-1, 1] > probas[5:-1, 0])


def test_labels_not_encoded():
    X = np.tile(np.arange(10).reshape(-1, 1), 3)
    y = np.array(5 * ['cat'] + 5 * ['dog'])
    knne = KNNE(n_neighbors=6)
    knne.fit(X, y)
    dist, inds = knne.kneighbors()
    classes = y[inds]
    a = np.sum(classes == 'dog')
    b = np.sum(classes == 'cat')
    assert np.equal(b, a).all() and a == 30


def test_n_neighbors_none():
    X = np.tile(np.arange(10).reshape(-1, 1), 3)
    y = np.array(5 * ['cat'] + 5 * ['dog'])
    knne = KNNE(n_neighbors=None)
    with pytest.raises(ValueError):
        knne.fit(X, y)


def test_return_indices_only():
    X, y, knne = setup_test(n_neighbors=6)
    inds = knne.kneighbors(X, return_distance=False)
    assert inds.shape == (15, 6)


def test_n_neighbors_less_n_classes():
    with pytest.raises(ValueError):
        setup_test(n_neighbors=2)


def test_n_neighbors_not_integer():
    with pytest.raises(TypeError):
        setup_test(n_neighbors=5.5)


def test_n_neighbors_not_multiple_raise_warning():
    with pytest.warns(Warning):
        setup_test(n_neighbors=7)


# ------Tests using KNNE using faiss for similarity search------------------
@pytest.mark.skipif(not faiss_knn_wrapper.is_available(),
                    reason="requires the faiss library")
def test_faiss_knne():
    X, y, knne = setup_test(n_neighbors=6, knn_classifier='faiss')
    y_pred = knne.predict(X)
    assert np.allclose(y, y_pred)


@pytest.mark.skipif(not faiss_knn_wrapper.is_available(),
                    reason="requires the faiss library")
def test_faiss_knne_inds():
    X, y, knne = setup_test(n_neighbors=6, knn_classifier='faiss')
    inds = knne.kneighbors(X, return_distance=False)
    assert inds.shape == (15, 6)
