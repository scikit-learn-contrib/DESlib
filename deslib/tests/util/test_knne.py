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
