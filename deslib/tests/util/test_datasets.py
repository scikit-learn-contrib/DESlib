import pytest
import numpy as np
from deslib.util.datasets import make_P2
from deslib.util.datasets import make_banana
from deslib.util.datasets import make_banana2
from deslib.util.datasets import make_circle_square
from deslib.util.datasets import make_xor


def setup_class_sizes():

    size_class0 = np.random.randint(1, 1000)
    size_class1 = np.random.randint(1, 1000)
    return size_class0, size_class1


def test_P2_class_distribution():

    s0, s1 = setup_class_sizes()
    _, y = make_P2(size_classes=[s0, s1])
    assert np.sum(y == 0) == s0 and np.sum(y == 1) == s1


def test_banana2_class_distribution():
    s0, s1 = setup_class_sizes()
    _, y = make_banana2(size_classes=[s0, s1])
    assert np.sum(y == 0) == s0 and np.sum(y == 1) == s1


def test_banana_class_distribution():
    s0, s1 = setup_class_sizes()
    _, y = make_banana(size_classes=[s0, s1])
    assert np.sum(y == 0) == s0 and np.sum(y == 1) == s1


def test_circle_square_class_distribution():
    s0, s1 = setup_class_sizes()
    _, y = make_circle_square(size_classes=[s0, s1])
    assert np.sum(y == 0) == s0 and np.sum(y == 1) == s1


def test_xor_size():
    n_samples = np.random.randint(100, 2000)
    X, y = make_xor(n_samples)
    assert y.size == n_samples


def test_xor():
    n_samples = np.random.randint(100, 2000)
    X, y = make_xor(n_samples)
    X_0, X_1 = X[y == 0], X[y == 1]
    for x in X_0:
        assert np.all(x[0] < 0.5 and x[1] < 0.5) or (x[0] > 0.5 and x[1] > 0.5)


def test_banana_n_higher_than_one():
    s0, s1 = setup_class_sizes()
    na = np.random.rand() + 1
    with pytest.raises(ValueError):
        make_banana([s0, s1], na)
